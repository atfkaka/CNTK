#pragma once

#include <vector>
#include <memory>

#include "reader_interface.h"
#include "commandArgUtil.h"

#include "biggrowablevectors.h"
#include "utterancesourcemulti.h"
#include "minibatchiterator.h"


namespace Microsoft { namespace MSR { namespace CNTK {
    class FrameModePacker : public Reader
    {
    public:
        FrameModePacker(const ConfigParameters & config, MemoryProviderPtr memoryProvider, size_t elementSize);

        virtual std::vector<InputDescriptionPtr> getInputs() override;
        virtual EpochPtr startNextEpoch(const EpochConfiguration& config) override;

    private:
        class EpochImplementation : public Epoch
        {
            FrameModePacker* m_parent;

        public:
            EpochImplementation(FrameModePacker* parent);
            virtual Minibatch readMinibatch() override;
            virtual ~EpochImplementation();
        };

        void InitFromConfig(const ConfigParameters& config);
        void PrepareForTrainingOrTesting(const ConfigParameters& config);
        void StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/);
        void StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples);
        Minibatch GetMinibatch();
        void FillOneUttDataforParallelmode(size_t startFr,
            size_t framenum, size_t channelIndex, size_t sourceChannelIndex);
        bool ReNewBufferForMultiIO(size_t i);
        void GetDataNamesFromConfig(
            const ConfigParameters& readerConfig,
            std::vector<std::wstring>& features,
            std::vector<std::wstring>& labels,
            std::vector<std::wstring>& hmms,
            std::vector<std::wstring>& lattices);
        void ExpandDotDotDot(wstring & featPath, const wstring & scpPath, wstring & scpDirCached);
        std::shared_ptr<void> AllocateIntermediateBuffer(size_t numElements, size_t elementSize);

        enum InputOutputTypes
        {
            real,
            category,
        };
    private:
        size_t m_elementSize; // size of the element, should go away probably and be taken from the layout?
        MemoryProviderPtr m_memoryProvider;
        intargvector m_numSeqsPerMBForAllEpochs;
        size_t m_numSeqsPerMB;                  // requested number of parallel sequences
        bool m_noData;
        MBLayoutPtr m_pMBLayout;
        std::vector<size_t> m_featDims;
        std::map<std::wstring, size_t> m_nameToTypeMap;
        std::map<std::wstring, size_t> m_featureNameToIdMap;
        std::map<std::wstring, size_t> m_featureNameToDimMap;
        std::vector<std::shared_ptr<void>> m_featuresBufferMultiIO;
        std::vector<size_t> m_featuresBufferAllocatedMultiIO;
        std::vector<size_t> m_labelDims;
        std::map<std::wstring, size_t> m_labelNameToIdMap;
        std::map<std::wstring, size_t> m_labelNameToDimMap;
        std::vector<std::shared_ptr<void>> m_labelsBufferMultiIO;
        std::vector<size_t> m_labelsBufferAllocatedMultiIO;
        int m_verbosity;
        bool m_partialMinibatch;
        unique_ptr<msra::dbn::minibatchiterator> m_mbiter;
        unique_ptr<msra::dbn::minibatchsource> m_frameSource;
        size_t m_mbNumTimeSteps;                // number of time steps  to fill/filled (note: for frame randomization, this the #frames, and not 1 as later reported)
        vector<bool> m_sentenceEnd;
        vector<size_t> m_processedFrame;
        vector<size_t> m_numFramesToProcess;    // [seq index] number of frames available (left to return) in each parallel sequence
        vector<size_t> m_switchFrame;           /// TODO: something like the position where a new sequence starts; still supported?
        vector<size_t> m_numValidFrames;        // [seq index] valid #frames in each parallel sequence. Frames (s, t) with t >= m_numValidFrames[s] are NoInput.

        std::vector<std::shared_ptr<void>> m_featuresBufferMultiUtt;
        std::vector<size_t> m_featuresBufferAllocatedMultiUtt;
        std::vector<std::shared_ptr<void>> m_labelsBufferMultiUtt;
        std::vector<size_t> m_labelsBufferAllocatedMultiUtt;
        std::vector<size_t> m_featuresStartIndexMultiUtt;
        std::vector<size_t> m_labelsStartIndexMultiUtt;
        std::map<std::wstring, size_t> m_nameToId;

        //for lattice uids and phoneboundaries
        std::vector<shared_ptr<const msra::dbn::latticepair>>  m_latticeBufferMultiUtt;
        std::vector<std::vector<size_t>> m_labelsIDBufferMultiUtt;
        std::vector<std::vector<size_t>> m_phoneboundaryIDBufferMultiUtt;
        std::vector<shared_ptr<const msra::dbn::latticepair>>  m_extraLatticeBufferMultiUtt;
        std::vector<std::vector<size_t>> m_extraLabelsIDBufferMultiUtt;
        std::vector<std::vector<size_t>> m_extraPhoneboundaryIDBufferMultiUtt;
        vector<size_t> m_extraSeqsPerMB;
        size_t m_extraNumSeqs;
    };

    typedef std::shared_ptr<FrameModePacker> FrameModePackerPtr;
}}}