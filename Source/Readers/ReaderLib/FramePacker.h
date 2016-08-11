//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SequencePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

//A packer optimized for the case of single-frame sequences.
class FramePacker : public SequencePacker
{
public:
    FramePacker(
        MemoryProviderPtr memoryProvider,
        SequenceEnumeratorPtr sequenceEnumerator,
        const std::vector<StreamDescriptionPtr>& streams) :
        SequencePacker(memoryProvider, sequenceEnumerator, streams)
    {}

    void StartEpoch(const EpochConfiguration& config) override
    {
        SequencePacker::StartEpoch(config);

        // Warm up of the buffers.
        size_t numberOfSamples = config.m_minibatchSizeInSamples / config.m_numberOfWorkers;
        // Add ten percent of variation
        numberOfSamples += numberOfSamples / 10;

        // Do warm up of the buffers for the dense streams.
        for (size_t i = 0; i < m_outputStreamDescriptions.size(); ++i)
        {
            const auto& stream = m_outputStreamDescriptions[i];
            if (stream->m_storageType != StorageType::dense)
            {
                continue;
            }

            size_t sampleSize = GetSampleSize(stream);
            size_t requiredSize = numberOfSamples * sampleSize;
            m_streamBuffers[i].Resize(requiredSize);
        }
    }

private:

    MBLayoutPtr CreateMBLayout(const StreamBatch& batch) override;
};

typedef std::shared_ptr<FramePacker> FramePackerPtr;
} } }
