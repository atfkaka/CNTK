#pragma once

#include "reader_interface.h"
#include "inner_interfaces.h"

#include <string>

class EpochImplementation : public Epoch
{
public:
    virtual Minibatch readMinibatch() override
    {
        throw std::logic_error("The method or operation is not implemented.");
        return Minibatch();
    };
    virtual ~EpochImplementation() {};
};


// TODO we don't use this yet
struct PhysicalTimeline : Timeline
{
    // Specific physical location per file format Sequence
};

class FileReader : public BlockReader
{
public:
    FileReader(std::string fileName);
    virtual ~FileReader() override
    {
    }

    virtual void get(char* buffer, size_t offset, size_t size) override
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

};

typedef std::shared_ptr<BlockReader> BlockReaderPtr;
typedef std::shared_ptr<FileReader> FileReaderPtr;

class ScpDataDerserializer
{
public:
    ScpDataDerserializer(BlockReaderPtr scp);
    PhysicalTimeline getTimeline();
};

typedef std::shared_ptr<ScpDataDerserializer> ScpDataDeserializerPtr;

class HtkDataDeserializer : public DataDeserializer
{
public:
    HtkDataDeserializer(BlockReaderPtr features, AugmentationDescriptor augmentationDescriptor, const PhysicalTimeline& timeline);
};

typedef std::shared_ptr<HtkDataDeserializer> HtkDataDeserializerPtr;

class MlfDataDeserializer : public DataDeserializer
{
public:
    MlfDataDeserializer(BlockReaderPtr lables, BlockReaderPtr states, const PhysicalTimeline& timeline);
};

typedef std::shared_ptr<MlfDataDeserializer> MlfDataDeserializerPtr;

class HtkMlfBundler : public Sequencer
{
public:
    HtkMlfBundler(HtkDataDeserializerPtr, MlfDataDeserializerPtr, ScpDataDeserializerPtr);

    virtual Timeline& getTimeline() const override;
    virtual std::vector<InputDescriptionPtr> getInputs() const override;
    virtual std::map<size_t, Sequence> getSequenceById(size_t id) override;
};

typedef std::shared_ptr<HtkMlfBundler> HtkMlfBundlerPtr;

class ChunkRandomizer : public Randomizer
{
public:
    ChunkRandomizer(SequencerPtr, size_t chunkSize, int seed);

    virtual std::vector<InputDescriptionPtr> getInputs() const override;
    virtual std::map<size_t, Sequence> getNextSequence() override;
};

class NormalPacker : public Packer
{
public:
    NormalPacker(MemoryProviderPtr memoryProvider, TransformerPtr transformer, const ConfigParameters& config) {}

    virtual std::vector<InputDescriptionPtr> getInputs() override;
    virtual EpochPtr startNextEpoch(const EpochConfiguration& config) override;
};

class BpttPacker : public Packer
{
};

class ReaderFacade : public Reader
{
public:
    ReaderFacade(PackerPtr packer) {}
    virtual std::vector<InputDescriptionPtr> getInputs()
    {
        std::vector<InputDescriptionPtr> result;
        return result;
    };
    virtual EpochPtr startNextEpoch(const EpochConfiguration& config)
    {
        return std::make_unique<EpochImplementation>();
    };
    virtual ~ReaderFacade() { }
};

ReaderPtr createReader(ConfigParameters& parameters, MemoryProviderPtr memoryProvider)
{
    // The code below will be split between the corresponding factory
    // methods with appropriate extraction of required parameters
    // from the config. Parameters will also be combined in the
    // appropriate structures when needed.

    // Read parameters from config
    const int chunkSize = std::stoi(parameters["..."]);
    const int seed = std::stoi(parameters["..."]);

    // Read scp and form initial Timeline
    auto scpFilename = parameters["scpFilename"];
    auto scp = std::make_shared<FileReader>(scpFilename);
    auto t = std::make_shared<ScpDataDerserializer>(scp);

    // Create Sequence readers to be combined by the Sequencer.
    auto featureFilename = parameters["featureFilename"];
    auto featureReader =
        std::make_shared<FileReader>(featureFilename);
    auto feature =
        std::make_shared<HtkDataDeserializer>(featureReader,
            AugmentationDescriptor(), t->getTimeline());

    auto labelsFilename = parameters["labelsFilename"];
    auto labelReader = std::make_shared<FileReader>(labelsFilename);
    auto statesFilename = parameters["statesFilename"];
    auto statesReader = std::make_shared<FileReader>(statesFilename);
    auto labels =std::make_shared<MlfDataDeserializer>(labelReader,
        statesReader, t->getTimeline());
    auto sequencer =
        std::make_shared<HtkMlfBundler>(feature, labels, t);

    // Create Randomizer and form randomized Timeline.
    auto randomizer =std::make_shared<ChunkRandomizer>(sequencer,
        chunkSize, seed);

    // Create the Packer that will consume the sequences from the
    // Randomizer and will pack them into efficient representation
    // using the memory provider.
    auto packer = std::make_unique<NormalPacker>(memoryProvider,
        randomizer, parameters);

    return std::make_unique<ReaderFacade>(std::move(packer));
}
