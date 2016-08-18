//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "DataDeserializerBase.h"
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class Bundler;

// Class takes bundled deserializers and performs framing if
// specified in the configuration file
class FramingHelper : public DataDeserializerBase
{
public:
    FramingHelper(std::shared_ptr<Bundler> bundler);

    // Gets chunk descriptions.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Gets sequence descriptions for a particular chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& sequences) override;

    // Gets a chunk with data.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;
    
private:
    class FramingChunk;    
    
    void InitializeStreams();

    DISABLE_COPY_AND_MOVE(FramingHelper);
    
    std::shared_ptr<Bundler> m_bundler;
    std::vector<ChunkDescriptionPtr> m_chunks;
};

}}}