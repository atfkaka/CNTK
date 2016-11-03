//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <opencv2/core/mat.hpp>
#include "DataDeserializerBase.h"
#include "Config.h"
#include "ByteReader.h"
#include <unordered_map>
#include "CorpusDescriptor.h"
#include "ImageUtil.h"
#include "Indexer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Image data deserializer based on the OpenCV library.
    // The deserializer currently supports two output streams only: a feature and a label stream.
    // All sequences consist only of a single sample (image/label).
    // For features it uses dense storage format with different layout (dimensions) per sequence.
    // For labels it uses the csc sparse storage format.
    class Base64ImageDeserializer : public DataDeserializerBase
    {
    public:
        // A new constructor to support new compositional configuration,
        // that allows composition of deserializers and transforms on inputs.
        Base64ImageDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config);

        // Gets sequences by specified ids. Order of returned sequences corresponds to the order of provided ids.
        virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;

        // Gets chunk descriptions.
        virtual ChunkDescriptions GetChunkDescriptions() override;

        // Gets sequence descriptions for the chunk.
        virtual void GetSequencesForChunk(ChunkIdType, std::vector<SequenceDescription>&) override;

        // Gets sequence description by key.
        bool GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&) override;

    private:
        // Creates a set of sequence descriptions.
        void CreateSequenceDescriptions(CorpusDescriptorPtr corpus, std::string mapPath);

        // Image sequence descriptions. Currently, a sequence contains a single sample only.
        struct ImageSequenceDescription : public SequenceDescription
        {
            std::string m_path;
            size_t m_classId;
        };

        class ImageChunk;

        // A helper class for generation of type specific labels.
        LabelGeneratorPtr m_labelGenerator;

        // Sequence descriptions for all input data.
        std::vector<ImageSequenceDescription> m_imageSequences;

        // Mapping of logical sequence key into sequence description.
        std::map<size_t, size_t> m_keyToSequence;

        size_t m_labelDimension;

        // Precision required by the network.
        ElementType m_precision;

        // whether images shall be loaded in grayscale 
        bool m_grayscale;

        bool m_multiViewCrop;
        int m_verbosity;

        bool m_hasSequenceKeys;

        CorpusDescriptorPtr m_corpus;

        std::unique_ptr<Indexer> m_indexer;
        std::shared_ptr<FILE> m_dataFile;
        std::wstring m_fileName;
    };

}}}
