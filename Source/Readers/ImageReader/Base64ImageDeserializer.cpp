//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <limits>
#include "Base64ImageDeserializer.h"
#include "ImageConfigHelper.h"
#include "StringUtil.h"
#include "ConfigUtil.h"
#include "TimerUtility.h"
#include "ImageTransformers.h"
#include "SequenceData.h"
#include "ImageUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class Base64ImageDeserializer::ImageChunk : public Chunk, public std::enable_shared_from_this<ImageChunk>
    {
        ChunkDescriptor m_descriptor;
        size_t m_chunkOffset;
        Base64ImageDeserializer& m_parent;
        std::vector<char> m_buffer; // TODO: Could possibly use memory mapped IO.

    public:
        ImageChunk(const ChunkDescriptor& descriptor, Base64ImageDeserializer& parent)
            : m_descriptor(descriptor), m_parent(parent)
        {
            // Let's see if the open descriptor has problems.
            if (ferror(m_parent.m_dataFile.get()) != 0)
                m_parent.m_dataFile.reset(fopenOrDie(m_parent.m_fileName.c_str(), L"rbS"), [](FILE* f) { if (f) fclose(f); });

            if (descriptor.m_sequences.empty() || !descriptor.m_byteSize)
                LogicError("Empty chunks are not supported.");

            m_buffer.resize(descriptor.m_byteSize + 1);
            // Makesure we always have 0 at the end for safety.
            m_buffer[descriptor.m_byteSize] = 0;
            m_chunkOffset = descriptor.m_sequences.front().m_fileOffsetBytes;

            int rc = _fseeki64(m_parent.m_dataFile.get(), m_chunkOffset, SEEK_SET);
            if (rc)
                RuntimeError("Error seeking to position %" PRId64 " in the input file (%ls), error %d", m_chunkOffset, m_parent.m_fileName.c_str(), rc);

            freadOrDie(m_buffer.data(), descriptor.m_byteSize, 1, m_parent.m_dataFile.get());
        }

        virtual void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
        {
            auto resultingImage = std::make_shared<ImageSequenceData>();

            size_t innerSequenceId = m_parent.m_multiViewCrop ? sequenceId / 10 : sequenceId;
            const auto& sequence = m_descriptor.m_sequences[innerSequenceId];
            size_t offset = sequence.m_fileOffsetBytes - m_chunkOffset;

            // Let's parse the string
            char* next_token = nullptr;
            char* token = strtok_s(&m_buffer[0] + offset, "\t", &next_token);
            bool hasSequenceKey = m_parent.m_indexer->HasSequenceIds();
            if (hasSequenceKey) // Skip sequence key.
                token = strtok_s(nullptr, "\t", &next_token);

            // Let's get the label.
            if (!token)
                RuntimeError("Empty label value for sequence %" PRIu64, sequence.m_key.m_sequence);

            char* eptr = nullptr;
            errno = 0;
            size_t classId = strtoull(token, &eptr, 10);
            if (token == eptr || errno == ERANGE)
                RuntimeError("Cannot parse label value for sequence %" PRIu64, sequence.m_key.m_sequence);

            if (classId >= m_parent.m_labelDimension)
                RuntimeError(
                    "Image with id '%" PRIu64 "' has invalid class id '%" PRIu64 "'. It is exceeding the label dimension of '%" PRIu64,
                    sequence.m_key.m_sequence, classId, m_parent.m_labelDimension);

            // Let's get the image.
            token = strtok_s(nullptr, "\n", &next_token);
            if (!token)
                RuntimeError("Empty image for sequence %" PRIu64, sequence.m_key.m_sequence);

            // Find line end or end of buffer.
            char* endToken = strchr(token, 0);
            if (!endToken)
                RuntimeError("Cannot find the end of the image for sequence %" PRIu64, sequence.m_key.m_sequence);

            // Remove non Base64 characters at the end of the string (tabs/spaces/)
            while (endToken > token &&  !IsBase64Char(*(endToken - 1)))
                endToken--;

            std::vector<char> decodedImage;
            if (!Decode64BitImage(token, endToken, decodedImage))
            {
                fprintf(stderr, "Cannot decode sequence with id '%d'\n", (int)sequence.m_key.m_sequence);
                resultingImage->m_isValid = false;
            }
            else
            {
                cv::Mat img = cv::imdecode(decodedImage, m_parent.m_grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
                resultingImage->m_image = std::move(img);
                auto& cvImage = resultingImage->m_image;
                if (!cvImage.data)
                {
                    fprintf(stderr, "Cannot decode sequence with id '%d'\n", (int)sequence.m_key.m_sequence);
                    resultingImage->m_isValid = false;
                }
                else
                {
                    // Convert element type.
                    ElementType dataType = ConvertImageToSupportedDataType(cvImage);
                    if (!cvImage.isContinuous())
                        cvImage = cvImage.clone();
                    assert(cvImage.isContinuous());
                    resultingImage->m_elementType = dataType;
                }

                ImageDimensions dimensions(cvImage.cols, cvImage.rows, cvImage.channels());
                resultingImage->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));
                resultingImage->m_id = sequenceId;
                resultingImage->m_numberOfSamples = 1;
                resultingImage->m_chunk = shared_from_this();
            }

            result.push_back(resultingImage);

            auto label = std::make_shared<CategorySequenceData>();
            label->m_chunk = shared_from_this();
            m_parent.m_labelGenerator->CreateLabelFor(classId, *label);
            label->m_numberOfSamples = 1;
            result.push_back(label);
        }

    private:
        ElementType ConvertImageToSupportedDataType(cv::Mat& image)
        {
            ElementType resultType;
            if (!IdentifyElementTypeFromOpenCVType(image.depth(), resultType))
            {
                // Could not identify element type.
                // Natively unsupported image type. Let's convert it to required precision.
                int requiredType = m_parent.m_precision == ElementType::tfloat ? CV_32F : CV_64F;
                image.convertTo(image, requiredType);
                resultType = m_parent.m_precision;
            }
            return resultType;
        }
    };

    // A constructor to support compositional configuration
    Base64ImageDeserializer::Base64ImageDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config)
    {
        ConfigParameters inputs = config("input");
        std::vector<std::string> featureNames = GetSectionsWithParameter("Base64ImageDeserializer", inputs, "transforms");
        std::vector<std::string> labelNames = GetSectionsWithParameter("Base64ImageDeserializer", inputs, "labelDim");

        // TODO: currently support only one feature and label section.
        if (featureNames.size() != 1 || labelNames.size() != 1)
        {
            RuntimeError(
                "ImageReader currently supports a single feature and label stream. '%d' features , '%d' labels found.",
                static_cast<int>(featureNames.size()),
                static_cast<int>(labelNames.size()));
        }

        string precision = (ConfigValue)config("precision", "float");
        m_precision = AreEqualIgnoreCase(precision, "float") ? ElementType::tfloat : ElementType::tdouble;

        m_verbosity = config(L"verbosity", 0);

        // Feature stream.
        ConfigParameters featureSection = inputs(featureNames[0]);
        auto features = std::make_shared<StreamDescription>();
        features->m_id = 0;
        features->m_name = msra::strfun::utf16(featureSection.ConfigName());
        features->m_storageType = StorageType::dense;

        // Due to performance, now we support images of different types.
        features->m_elementType = ElementType::tvariant;
        m_streams.push_back(features);

        // Label stream.
        ConfigParameters label = inputs(labelNames[0]);
        m_labelDimension = label("labelDim");
        auto labels = std::make_shared<StreamDescription>();
        labels->m_id = 1;
        labels->m_name = msra::strfun::utf16(label.ConfigName());
        labels->m_sampleLayout = std::make_shared<TensorShape>(m_labelDimension);
        labels->m_storageType = StorageType::sparse_csc;
        labels->m_elementType = m_precision;
        m_streams.push_back(labels);

        m_labelGenerator = labels->m_elementType == ElementType::tfloat ?
            (LabelGeneratorPtr)std::make_shared<TypedLabelGenerator<float>>(m_labelDimension) :
                               std::make_shared<TypedLabelGenerator<double>>(m_labelDimension);

        m_grayscale = config(L"grayscale", false);

        // TODO: multiview should be done on the level of randomizer/transformers - it is responsiblity of the
        // TODO: randomizer to collect how many copies each transform needs and request same sequence several times.
        m_multiViewCrop = config(L"multiViewCrop", false);
        CreateSequenceDescriptions(corpus, config(L"file"));
    }

    // Descriptions of chunks exposed by the image reader.
    ChunkDescriptions Base64ImageDeserializer::GetChunkDescriptions()
    {
        const auto& index = m_indexer->GetIndex();
        size_t sequencesPerInitialSequence = m_multiViewCrop ? 10 : 1;
        ChunkDescriptions result;
        result.reserve(index.m_chunks.size() * sequencesPerInitialSequence);
        for (auto const& chunk : index.m_chunks)
        {
            auto c = std::make_shared<ChunkDescription>();
            c->m_id = chunk.m_id;
            c->m_numberOfSamples = chunk.m_numberOfSamples;
            c->m_numberOfSequences = chunk.m_numberOfSequences * sequencesPerInitialSequence;
            result.push_back(c);
        }
        return result;
    }

    void Base64ImageDeserializer::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result)
    {
        // TODO: Multicrop should be done in the randomizer.
        const auto& index = m_indexer->GetIndex();
        const auto& chunk = index.m_chunks[chunkId];
        size_t sequencesPerInitialSequence = m_multiViewCrop ? 10 : 1;
        result.reserve(sequencesPerInitialSequence * chunk.m_sequences.size());
        size_t currentId = 0;
        for (auto const& s : chunk.m_sequences)
        {
            assert(currentId / sequencesPerInitialSequence == s.m_id);
            for (size_t i = 0; i < sequencesPerInitialSequence; ++i)
            {
                result.push_back(
                {
                    currentId,
                    s.m_numberOfSamples,
                    s.m_chunkId,
                    s.m_key
                });
                currentId++;
            }
        }
    }

    static bool HasSequenceKeys(const std::string& mapPath)
    {
        std::ifstream mapFile(mapPath);
        if (!mapFile)
            RuntimeError("Could not open '%s' for reading.", mapPath.c_str());

        string line;
        if (!std::getline(mapFile, line))
            RuntimeError("Could not read the file '%s'.", mapPath.c_str());

        // Try to parse sequence id, file path and label.
        std::string image, classId, sequenceKey;
        std::stringstream ss(line);
        if (!std::getline(ss, sequenceKey, '\t') || !std::getline(ss, classId, '\t') || !std::getline(ss, image, '\t'))
        {
            return false;
        }
        return true;
    }

    void Base64ImageDeserializer::CreateSequenceDescriptions(CorpusDescriptorPtr corpus, std::string mapPath)
    {
        Timer timer;
        timer.Start();

        bool hasSequenceKeys = HasSequenceKeys(mapPath);
        m_fileName.assign(mapPath.begin(), mapPath.end());

        attempt(5, [this, hasSequenceKeys, corpus]()
        {
            if (!m_dataFile || ferror(m_dataFile.get()) != 0)
                m_dataFile.reset(fopenOrDie(m_fileName, L"rbS"), [](FILE* f) { if (f) fclose(f); });

            m_indexer = make_unique<Indexer>(m_dataFile.get(), !hasSequenceKeys);
            m_indexer->Build(corpus);
        });

        timer.Stop();
        if (m_verbosity > 1)
        {
            fprintf(stderr, "ImageDeserializer: Read information about %d images in %.6g seconds\n", (int)m_imageSequences.size(), timer.ElapsedSeconds());
        }
    }

    ChunkPtr Base64ImageDeserializer::GetChunk(ChunkIdType chunkId)
    {
        const auto& chunkDescriptor = m_indexer->GetIndex().m_chunks[chunkId];
        return make_shared<ImageChunk>(chunkDescriptor, *this);
    }

    bool Base64ImageDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
    {
        const auto& index = m_indexer->GetIndex();

        const auto& keys = index.m_keyToSequenceInChunk;
        auto sequenceLocation = keys.find(key.m_sequence);
        if (sequenceLocation == keys.end())
        {
            return false;
        }

        const auto& chunks = index.m_chunks;
        result = chunks[sequenceLocation->second.first].m_sequences[sequenceLocation->second.second];
        return true;
    }
}}}
