#pragma once

#include <vector>
#include <memory>
#include <map>

class ConfigParameters : public std::map<std::string, std::string>
{
};

// Epoch configuration.
struct EpochConfiguration
{
    size_t workerRank;
    size_t numberOfWorkers;

    size_t minibatchSize;
    size_t totalSize;

    size_t numberOfSequences;
};

typedef size_t InputId;

// Input description.
struct InputDescription
{
    std::string name;
    InputId id;
    std::string targetLayoutType;
    std::map<std::string, std::string> properties;
};
typedef std::shared_ptr<InputDescription> InputDescriptionPtr;

class MbLayout
{
};

class SampleLayout
{
};

struct Layout
{
    MbLayout columns;
    SampleLayout rows;
};

typedef std::shared_ptr<Layout> LayoutPtr;

// Input data.
class Input
{
    char* data_;
    size_t data_size_;
    LayoutPtr layout_;

public:
    Input(char* data, size_t dataSize, LayoutPtr layout)
        : data_(data)
        , data_size_(dataSize)
        , layout_(layout)
    {
    }

    const char* getData() const
    {
        return data_;
    }

    size_t getDataSize() const
    {
        return data_size_;
    }

    LayoutPtr getLayout() const
    {
        return layout_;
    }
};
typedef std::shared_ptr<Input> InputPtr;

// Memory provider. Should be used for allocating storage according to the Layout.
class MemoryProvider
{
public:
    void* alloc(size_t element, size_t numberOfElements);
    void free(void* ptr);
};
typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;

// Represents a single minibatch.
struct Minibatch
{
    bool notAtEndOfEpoch;
    std::map<size_t /*id from the Input description*/, InputPtr> minibatch;

    operator bool() const
    {
        return notAtEndOfEpoch;
    }
};

class Epoch
{
public:
    virtual Minibatch readMinibatch() = 0;
    virtual ~Epoch() = 0 {};
};
typedef std::unique_ptr<Epoch> EpochPtr;

class Packer
{
public:
    virtual std::vector<InputDescriptionPtr> getInputs() = 0;
    virtual EpochPtr startNextEpoch(const EpochConfiguration& config) = 0;
    virtual ~Packer() = 0 {};
};
typedef std::unique_ptr<Packer> PackerPtr;

// Main Reader interface. The border interface between the CNTK and Reader.
class Reader
{
public:
    virtual std::vector<InputDescriptionPtr> getInputs() = 0;
    virtual EpochPtr startNextEpoch(const EpochConfiguration& config) = 0;
    virtual ~Reader() = 0 {};
};
typedef std::unique_ptr<Reader> ReaderPtr;

// Factory function for creating a Reader.
ReaderPtr createReader(const ConfigParameters& parameters, MemoryProviderPtr memory_provider);
