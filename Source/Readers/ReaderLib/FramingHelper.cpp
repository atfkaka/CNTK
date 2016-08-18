//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "FramingHelper.h"
#include "Bundler.h"
#define __STDC_FORMAT_MACROS

namespace Microsoft { namespace MSR { namespace CNTK {

FramingHelper::FramingHelper(std::shared_ptr<Bundler> bundler)
    : m_bundler(bundler)
{
    auto chunks = m_bundler->GetChunkDescriptions();
    
    m_chunks.reserve(chunks.size());

    for (ChunkIdType i = 0; i < chunks.size(); ++i)
    {
        auto cd = std::make_shared<ChunkDescription>();
        cd->m_numberOfSamples = chunks[i]->m_numberOfSamples;
        cd->m_numberOfSequences = chunks[i]->m_numberOfSamples;
        cd->m_id = chunks[i]->m_id;        
        m_chunks.push_back(cd);
    }

    InitializeStreams();
}

void FramingHelper::InitializeStreams()
{
    for (auto i : m_bundler->GetStreamDescriptions())
    {
        StreamDescriptionPtr stream = std::make_shared<StreamDescription>(*i);
        stream->m_id = m_streams.size();
        m_streams.push_back(stream);
    }
}

ChunkDescriptions FramingHelper::GetChunkDescriptions()
{        
    return ChunkDescriptions(m_chunks.begin(), m_chunks.end());
}

void FramingHelper::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& sequences)
{       
    ChunkDescriptionPtr chunk = m_chunks[chunkId];

    std::vector<SequenceDescription> result;
    m_bundler->GetSequencesForChunk(chunkId, sequences);
    result.reserve(chunk->m_numberOfSamples);    

    size_t offsetInChunk = 0;

    for (auto & seq : sequences)
    {
        for (size_t k = 0; k < seq.m_numberOfSamples; ++k)
        {
            SequenceDescription f;
            f.m_chunkId = chunkId;
            f.m_key.m_sequence = seq.m_key.m_sequence;
            f.m_key.m_sample = k;
            f.m_id = offsetInChunk++;
            f.m_numberOfSamples = 1;            
            result.push_back(f);
        }
    }

    std::swap(sequences, result);
}

class FramingHelper::FramingChunk : public Chunk
{
    Bundler::BundlingChunkPtr m_bundlingChunk;
    
    std::vector<std::pair<size_t, size_t>> m_frameToSequenceMap;
    //TODO: storageType;
    
    size_t m_numberOfInputs;

    DISABLE_COPY_AND_MOVE(FramingChunk);

public:
    FramingChunk(size_t numberOfInputs, Bundler* bundler, ChunkIdType chunkId)
        : m_numberOfInputs(numberOfInputs)
    {
        InitializeFrameToSequenceMap(bundler, chunkId);
        m_bundlingChunk = std::make_shared<Bundler::BundlingChunk>(numberOfInputs, bundler, chunkId);
    }

    void InitializeFrameToSequenceMap(Bundler * bundler, ChunkIdType chunkId)
    {
        std::vector<SequenceDescription> bundlingSequenceDescriptions;
        bundler->GetSequencesForChunk(chunkId, bundlingSequenceDescriptions);

        std::vector<ChunkDescriptionPtr> chunkDescription = bundler->GetChunkDescriptions();
        m_frameToSequenceMap.reserve(chunkDescription[chunkId]->m_numberOfSamples);
        for (auto & description : bundlingSequenceDescriptions)
        {
            for (size_t k = 0; k < description.m_numberOfSamples; ++k)
            {
                m_frameToSequenceMap.push_back(std::pair<size_t, size_t>(description.m_id, k));
            }
        }
    }
    
    virtual void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
    {
        size_t originalId = m_frameToSequenceMap[sequenceId].first;

        result.reserve(m_numberOfInputs);
        m_bundlingChunk->GetSequence(originalId, result);

        for (auto & streamResult : result)
        {
            streamResult.get()->m_numberOfSamples = 1;
            //size_t dimensions = streamResult.get()->m_sampleLayout.get()->GetNumElements();
            
            //streamResult.get()->m_data += (void *) (dimensions + () streamResult.get()->m_data);
        }
        
    }
};

ChunkPtr FramingHelper::GetChunk(ChunkIdType chunkId)
{
    return std::make_shared<FramingChunk>(m_streams.size(), m_bundler.get(), chunkId);
}

}}}