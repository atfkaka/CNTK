#pragma once

#include <vector>
#include "reader_interface.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Defines the encoding of a frame.
    struct FrameDescription
    {
        std::vector<size_t> dimensions;
        size_t elementSize;

        size_t Size() const
        {
            size_t result = 1;
            for (auto d: dimensions)
            {
                result *= d;
            }

            return result;
        }
    };


    // Defines identifier and length of a Sequence.
    struct SequenceDescription
    {
        size_t id;
        size_t length;
    };

    // Defines a sequences, which consists of sequences description and a number
    // of frames, which have the same encoding and are layed out in memory contiguously.
    struct Sequence
    {
        SequenceDescription* description;
        FrameDescription* frameDescription;
        void* data;
        size_t numberOfFrames;
    };

    // Low-level input interface (for file, network, etc.).
    // Memory buffers to fill data into are provided by the caller.
    class BlockReader
    {
    public:
        virtual void get(char* buffer, size_t offset, size_t size) = 0;
        virtual ~BlockReader() = 0 {}
    };

    // Interface to for structured reading from a single datasource
    class DataDeserializer
    {
    };

    // Timeline specifies a vector of Sequence IDs and lengths.
    // This information is exposed by a Sequencer, e.g., to be used by the Randomizer.
    typedef std::vector<SequenceDescription> Timeline;

    // Timeline offsets specify file offset of sequences. These are used internally
    // of a Sequence Reader or a Sequencer.
    typedef std::vector<size_t> TimelineOffsets;

    // A Sequencer composes Timeline information and a number of Sequence readers, providing
    // random-access to the Timeline as well as the composed Sequence readers.
    class Sequencer
    {
    public:
        virtual Timeline& getTimeline() const = 0;
        virtual std::vector<InputDescriptionPtr> getInputs() const = 0;
        virtual std::map<InputId, Sequence> getSequenceById(size_t id) = 0;
        virtual ~Sequencer() = 0 {};
    };

    typedef std::shared_ptr<Sequencer> SequencerPtr;

    // Defines context augmentation (to the left and to the right).
    // This will be specified as a construction parameter to Sequence Reader.
    struct AugmentationDescriptor
    {
        size_t contextLeft;
        size_t contextRight;
    };

    // Provides Input descriptions and sequential access to sequences.
    class Transformer
    {
    public:
        virtual void SetEpochConfiguration(const EpochConfiguration& config) = 0;
        virtual std::vector<InputDescriptionPtr> getInputs() const = 0;
        virtual ~Transformer() = 0 {}
        virtual std::map<InputId, Sequence> getNextSequence() = 0;
    };

    typedef std::shared_ptr<Transformer> TransformerPtr;

    // A Randomizer implements Sequence randomization for a Sequencer and
    // additional parameters given at construction time.
    // Note: chunk-level randomization can be implemented based on Sequence lengths
    // exposed through the Sequencer's Timeline method.
    class Randomizer : public Transformer
    {
    };

    class ImageCropper : public Transformer
    {
    };
}}}