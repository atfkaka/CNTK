#pragma once

#include <vector>
#include <memory>

#include "reader_interface.h"
#include "commandArgUtil.h"

#include "biggrowablevectors.h"
#include "utterancesourcemulti.h"
#include "minibatchiterator.h"
#include <inner_interfaces.h>

namespace Microsoft { namespace MSR { namespace CNTK {

    class MonolithicTransformer : public Transformer
    {
    public:
        MonolithicTransformer(const ConfigParameters & readerConfig, size_t elementSize);

        virtual void SetEpochConfiguration(const EpochConfiguration& config);
        virtual std::vector<InputDescriptionPtr> getInputs() const override;
        virtual std::map<InputId, Sequence> getNextSequence() override;

        virtual ~MonolithicTransformer()
        {}

    private:
        enum InputOutputTypes
        {
            real,
            category,
        };

        std::vector<FrameDescription> m_featureFrameDescriptions;
        std::vector<FrameDescription> m_labelFrameDescriptions;

        /*not used by necessary to initialize the source*/
        msra::asr::simplesenonehmm m_hset;
        unique_ptr<msra::dbn::latticesource> m_lattices;
        map<wstring, msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;

        size_t m_elementSize; // size of the element, should go away probably and be taken from the layout?
        MemoryProviderPtr m_memoryProvider;
        intargvector m_numSeqsPerMBForAllEpochs;
        size_t m_numSeqsPerMB;                  // requested number of parallel sequences
        bool m_noData;
        std::vector<size_t> m_featDims;
        std::map<std::wstring, size_t> m_nameToTypeMap;
        std::map<std::wstring, size_t> m_featureNameToIdMap;
        std::map<std::wstring, size_t> m_featureNameToDimMap;
        std::vector<size_t> m_labelDims;
        std::map<std::wstring, size_t> m_labelNameToIdMap;
        std::map<std::wstring, size_t> m_labelNameToDimMap;
        int m_verbosity;
        bool m_partialMinibatch;
        unique_ptr<msra::dbn::minibatchiterator> m_mbiter;
        unique_ptr<msra::dbn::minibatchsource> m_frameSource;
        size_t m_mbNumTimeSteps;                // number of time steps  to fill/filled (note: for frame randomization, this the #frames, and not 1 as later reported)
        vector<size_t> m_numFramesToProcess;    // [seq index] number of frames available (left to return) in each parallel sequence
        vector<size_t> m_numValidFrames;        // [seq index] valid #frames in each parallel sequence. Frames (s, t) with t >= m_numValidFrames[s] are NoInput.

        std::map<std::wstring, size_t> m_nameToId;
    };

    typedef std::shared_ptr<MonolithicTransformer> MonolithicTransformerPtr;
}}}