//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <limits>
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "../HTKMLFReader/htkfeatio.h"
#include "../HTKMLFReader/msra_mgram.h"
#include "latticearchive.h"

#undef max // max is defined in minwindef.h

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

static float s_oneFloat = 1.0;
static double s_oneDouble = 1.0;

// Currently we only have a single mlf chunk that contains a vector of all labels.
// TODO: In the future MLF should be converted to a more compact format that is amenable to chunking.
class MLFDataDeserializer::MLFChunk : public Chunk
{
    MLFDataDeserializer* m_parent;
public:
    MLFChunk(MLFDataDeserializer* parent) : m_parent(parent)
    {}

    virtual void GetSequence(size_t sequenceId, vector<SequenceDataPtr>& result) override
    {
        m_parent->GetSequenceById(sequenceId, result);
    }
};

// Inner class for an utterance.
struct MLFUtterance : SequenceDescription
{
    size_t m_sequenceStart;
};

MLFDataDeserializer::MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
{
    // TODO: This should be read in one place, potentially given by SGD.
    m_frameMode = (ConfigValue)cfg("frameMode", "true");

    // MLF cannot control chunking.
    if (primary)
    {
        LogicError("Mlf deserializer does not support primary mode - it cannot control chunking.");
    }

    argvector<ConfigValue> inputs = cfg("inputs");
    if (inputs.size() != 1)
    {
        LogicError("MLFDataDeserializer supports a single input stream only.");
    }

    ConfigParameters input = inputs.front();
    auto inputName = input.GetMemberIds().front();

    ConfigParameters streamConfig = input(inputName);
    ConfigHelper config(streamConfig);

    size_t dimension = config.GetLabelDimension();

    wstring labelMappingFile = streamConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, dimension);
    InitializeStream(inputName, dimension);
}

MLFDataDeserializer::MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& labelConfig, const wstring& name)
{
    // The frame mode is currently specified once per configuration,
    // not in the configuration of a particular deserializer, but on a higher level in the configuration.
    // Because of that we are using find method below.
    m_frameMode = labelConfig.Find("frameMode", "true");

    ConfigHelper config(labelConfig);

    config.CheckLabelType();
    size_t dimension = config.GetLabelDimension();

    if (dimension > numeric_limits<IndexType>::max())
    {
        RuntimeError("Label dimension (%" PRIu64 ") exceeds the maximum allowed "
            "value (%" PRIu64 ")\n", dimension, (size_t)numeric_limits<IndexType>::max());
    }

    wstring labelMappingFile = labelConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, dimension);
    InitializeStream(name, dimension);
}

// Currently we create a single chunk only.
void MLFDataDeserializer::InitializeChunkDescriptions(CorpusDescriptorPtr corpus, const ConfigHelper& config, const wstring& stateListPath, size_t dimension)
{
    // TODO: Similarly to the old reader, currently we assume all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files

    // TODO: currently we do not use symbol and word tables.
    const msra::lm::CSymbolSet* wordTable = nullptr;
    unordered_map<const char*, int>* symbolTable = nullptr;
    vector<wstring> mlfPaths = config.GetMlfPaths();

    // TODO: Currently we still use the old IO module. This will be refactored later.
    const double htkTimeToFrame = 100000.0; // default is 10ms
    msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence> labels(mlfPaths, set<wstring>(), stateListPath, wordTable, symbolTable, htkTimeToFrame);

    // Make sure 'msra::asr::htkmlfreader' type has a move constructor
    static_assert(
        is_move_constructible<
        msra::asr::htkmlfreader<msra::asr::htkmlfentry,
        msra::lattices::lattice::htkmlfwordsequence >> ::value,
        "Type 'msra::asr::htkmlfreader' should be move constructible!");

    m_elementType = config.GetElementType();

    size_t totalFrames = 0;

    auto& stringRegistry = corpus->GetStringRegistry();

    // TODO resize m_keyToSequence with number of IDs from string registry

    for (const auto& l : labels)
    {
        // Currently the string registry contains only utterances described in scp.
        // So here we skip all others.
        size_t id = 0;
        if (!stringRegistry.TryGet(l.first, id))
            continue;

        const auto& utterance = l.second;
        size_t numberOfFrames = 0;

        size_t lastFrameFromLastRange = 0;
        size_t firstRunIndex = m_classIdRuns.size();

        for (const auto& timespan : utterance)
        {
            if (timespan.firstframe != lastFrameFromLastRange)
            {
                RuntimeError("Labels are not in the consecutive order MLF in label set: %ls", l.first.c_str());
            }
            lastFrameFromLastRange += timespan.numframes;

            if (timespan.classid >= dimension)
            {
                RuntimeError("Class id %d exceeds the model output dimension %d.", (int)timespan.classid, (int)dimension);
            }

            if (timespan.classid != static_cast<msra::dbn::CLASSIDTYPE>(timespan.classid))
            {
                RuntimeError("CLASSIDTYPE has too few bits");
            }

            numberOfFrames += timespan.numframes;
            m_classIdRuns.push_back(make_pair(timespan.firstframe, timespan.classid));
        }
        m_classIdRuns.push_back(make_pair(lastFrameFromLastRange, static_cast<msra::dbn::CLASSIDTYPE>(-1)));
        // TODO consider changing to total, not relative frames

        m_utteranceInfo.push_back(make_pair(totalFrames, firstRunIndex));
        totalFrames += numberOfFrames;

        if (m_keyToSequence.size() <= id)
        {
            m_keyToSequence.resize(id + 1, SIZE_MAX);
        }
        assert(m_keyToSequence[id] == SIZE_MAX);
        m_keyToSequence[id] = m_utteranceInfo.size() - 1;
        m_numberOfSequences++;
    }
    m_utteranceInfo.push_back(make_pair(totalFrames, m_classIdRuns.size()));

    m_totalNumberOfFrames = totalFrames;

    fprintf(stderr, "MLFDataDeserializer::MLFDataDeserializer: read %d frames\n", (int)m_totalNumberOfFrames);
    fprintf(stderr, "MLFDataDeserializer::MLFDataDeserializer: read %d utterances\n", (int)m_numberOfSequences);

    // Initializing array of labels.
    m_categories.reserve(dimension);
    m_categoryIndices.reserve(dimension);
    for (size_t i = 0; i < dimension; ++i)
    {
        SparseSequenceDataPtr category = make_shared<SparseSequenceData>();
        m_categoryIndices.push_back(static_cast<IndexType>(i));
        category->m_indices = &(m_categoryIndices[i]);
        category->m_nnzCounts.resize(1);
        category->m_nnzCounts[0] = 1;
        category->m_totalNnzCount = 1;
        category->m_numberOfSamples = 1;
        if (m_elementType == ElementType::tfloat)
        {
            category->m_data = &s_oneFloat;
        }
        else
        {
            assert(m_elementType == ElementType::tdouble);
            category->m_data = &s_oneDouble;
        }
        m_categories.push_back(category);
    }
}

void MLFDataDeserializer::InitializeStream(const wstring& name, size_t dimension)
{
    // Initializing stream description - a single stream of MLF data.
    StreamDescriptionPtr stream = make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = name;
    stream->m_sampleLayout = make_shared<TensorShape>(dimension);
    stream->m_storageType = StorageType::sparse_csc;
    stream->m_elementType = m_elementType;
    m_streams.push_back(stream);
}

void InitializeFeatureInformation();
void InitializeAugmentationWindow(ConfigHelper& config);

// Currently MLF has a single chunk.
// TODO: This will be changed when the deserializer properly supports chunking.
ChunkDescriptions MLFDataDeserializer::GetChunkDescriptions()
{
    auto cd = make_shared<ChunkDescription>();
    cd->m_id = 0;
    cd->m_numberOfSequences = m_frameMode ? m_totalNumberOfFrames : m_numberOfSequences;
    cd->m_numberOfSamples = m_totalNumberOfFrames;
    return ChunkDescriptions{cd};
}

// Gets sequences for a particular chunk.
void MLFDataDeserializer::GetSequencesForChunk(size_t, vector<SequenceDescription>& result)
{
    UNUSED(result);
    LogicError("Mlf deserializer does not support primary mode - it cannot control chunking.");
}

ChunkPtr MLFDataDeserializer::GetChunk(size_t chunkId)
{
    UNUSED(chunkId);
    assert(chunkId == 0);
    return make_shared<MLFChunk>(this);
};

// Sparse labels for an utterance.
template <class ElemType>
struct MLFSequenceData : SparseSequenceData
{
    vector<ElemType> m_values;
    unique_ptr<IndexType[]> m_indicesPtr;

    MLFSequenceData(size_t numberOfSamples) :
        m_values(numberOfSamples, 1),
        m_indicesPtr(new IndexType[numberOfSamples])
    {
        if (numberOfSamples > numeric_limits<IndexType>::max())
        {
            RuntimeError("Number of samples in an MLFSequence (%" PRIu64 ") "
                "exceeds the maximum allowed value (%" PRIu64 ")\n",
                numberOfSamples, (size_t)numeric_limits<IndexType>::max());
        }

        m_nnzCounts.resize(numberOfSamples, static_cast<IndexType>(1));
        m_numberOfSamples = numberOfSamples;
        m_totalNnzCount = static_cast<IndexType>(numberOfSamples);
        m_indices = m_indicesPtr.get();
        m_data = m_values.data();
    }
};

void MLFDataDeserializer::GetSequenceById(size_t sequenceId, vector<SequenceDataPtr>& result)
{
    if (m_frameMode)
    {
        const auto & next = std::upper_bound(
            m_utteranceInfo.begin(),
            m_utteranceInfo.end(),
            sequenceId,
            [](size_t fi, const pair<size_t, size_t> &a)
        {
            return fi < a.first;
        });
        const auto & prev = next - 1;

        const auto & run = std::upper_bound(
            &m_classIdRuns[prev->second],
            &m_classIdRuns[next->second - 1],
            sequenceId - prev->first,
            [](size_t fi, const pair<size_t, msra::dbn::CLASSIDTYPE> &a)
        {
            return fi < a.first;
        });

        // TODO more checks
        size_t label = (run - 1)->second;
        assert(label < m_categories.size());
        result.push_back(m_categories[label]);
    }
    else
    {
        // Packing labels for the utterance into sparse sequence.
        size_t startFrameIndex = m_utteranceInfo[sequenceId].first;
        size_t startRunIndex = m_utteranceInfo[sequenceId].second;
        size_t numberOfSamples = m_utteranceInfo[sequenceId + 1].first - startFrameIndex;
        size_t endRunIndex = m_utteranceInfo[sequenceId + 1].second - 1;
        SparseSequenceDataPtr s;
        if (m_elementType == ElementType::tfloat)
        {
            s = make_shared<MLFSequenceData<float>>(numberOfSamples);
        }
        else
        {
            assert(m_elementType == ElementType::tdouble);
            s = make_shared<MLFSequenceData<double>>(numberOfSamples);
        }

        for (size_t i = 0, j = startRunIndex; j < endRunIndex; j++)
        {
            size_t runLength = m_classIdRuns[j + 1].first - m_classIdRuns[j].first;
            size_t label = m_classIdRuns[j].second;

            while (runLength--)
            {
                s->m_indices[i++] = static_cast<IndexType>(label);
            }
        }
        result.push_back(s);
    }
}

static SequenceDescription s_InvalidSequence { 0, 0, 0, false, { 0, 0 } };

void MLFDataDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
{

    auto sequenceId = key.m_sequence < m_keyToSequence.size() ? m_keyToSequence[key.m_sequence] : SIZE_MAX;

    if (sequenceId == SIZE_MAX)
    {
        result = s_InvalidSequence;
        return;
    }

    result.m_chunkId = 0;
    result.m_key = key;
    result.m_isValid = true;

    if (m_frameMode)
    {
        size_t index = m_utteranceInfo[sequenceId].first + key.m_sample;
        assert(index < m_utteranceInfo[sequenceId + 1].first);
        result.m_id = index;
        result.m_numberOfSamples = 1;
    }
    else
    {
        assert(result.m_key.m_sample == 0);
        result.m_id = sequenceId;
        result.m_numberOfSamples = m_utteranceInfo[sequenceId + 1].first - m_utteranceInfo[sequenceId].first;
    }
}

}}}
