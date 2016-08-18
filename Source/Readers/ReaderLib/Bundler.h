//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "DataDeserializerBase.h"
#include "Config.h"
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents an bundler of several deserializers.
// In case when only a single deserializer is used, the bundler can be omitted and 
// no performance penalty is paid.
class Bundler : public DataDeserializerBase
{
public:
    Bundler(const ConfigParameters& readerConfig, IDataDeserializerPtr driver, std::vector<IDataDeserializerPtr> deserializers, bool cleanse);

    // Gets chunk descriptions.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Gets sequence descriptions for a particular chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result) override;

    // Gets a chunk with data.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;

    class BundlingChunk;
    typedef std::shared_ptr<BundlingChunk> BundlingChunkPtr;

    struct BundlerChunkDescription;
    typedef std::shared_ptr<BundlerChunkDescription>BundlerChunkDescriptionPtr;

private:
    DISABLE_COPY_AND_MOVE(Bundler);

    // Creates chunk descriptions based on chunks of underlying deserializers.
    void CreateChunkDescriptions();

    // Underlying deserializers.
    std::vector<IDataDeserializerPtr> m_deserializers;

    // Driving deserializer that defines chunks.
    IDataDeserializerPtr m_driver;

    // Chunk descriptions.
    std::vector<BundlerChunkDescriptionPtr> m_chunks;

    // A flag that indicates whether there is a need to clean data between different deserializers.
    // It is possible that some sequence is valid in one deserializer but invalid in another. This sequences should be removed.
    // At the same time this introduces unnecessary overhead when the data is clean, because all chunks should be checked in advance to expose
    // correct number of samples/sequences they contain.
    // If this flag is set to false, no cleaning will be done, so additional overhead.
    bool m_cleanse;

    // If flag is set to true the sequence length is counted by the primary deserializer only.
    // Used for optimization when sequences between different deserializers are of the same length
    // (i.e. often in speech)
    bool m_takePrimarySequenceLength;

    // A table of loaded chunks to make sure we do not load same chunk twice.
    // Inner vector is the table of chunk id into weak pointer, the outer vector has an element per deserializer.
    std::vector<std::vector<std::weak_ptr<Chunk>>> m_weakChunkTable;

    // General configuration
    int m_verbosity;
};

// Represents a chunk that has pointers to the underlying deserializer chunks.
class Bundler::BundlingChunk : public Chunk
{
	size_t m_numberOfInputs;
	Bundler* m_parent;
	ChunkIdType m_chunkId;

	// A mapping between exposed sequence id and inner chunk for each deserializer.
	// Index i of the vector maps to the chunk of inner sequence (i / number of deserializers) of
	// deserializer (i % number of deserializers).
	std::vector<ChunkPtr> m_innerChunks;
	// A mapping between exposed sequence id and inner sequence id for each deserializer.
	// Indices as above.
	std::vector<size_t> m_sequenceToSequence;

	DISABLE_COPY_AND_MOVE(BundlingChunk);

public:
	BundlingChunk(size_t numberOfInputs, Bundler* parent, ChunkIdType chunkId);

	// Gets sequence by its id.
	virtual void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override;
};

// Represents bundled chunk description with possible cleansed data.
struct Bundler::BundlerChunkDescription : public ChunkDescription
{
    ChunkDescriptionPtr m_original;

    // Sequences that are invalid in at least one deserializer.
    std::set<size_t> m_invalid;
};

}}}
