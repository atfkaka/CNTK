//
// <copyright file="HTKMLFReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReaderShim.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "DataReader.h"
#include "commandArgUtil.h"
#include "FrameModePacker.h"
#include "SubstitutingMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class HTKMLFReaderShim : public IDataReader<ElemType>
    {
        FrameModePackerPtr m_packer;
        EpochPtr m_epoch;
        MBLayoutPtr m_layout;
        SubstitutingMemoryProviderPtr m_memoryProvider;
        bool m_providerSet;

    public:
        virtual void Init(const ConfigParameters & config) override;

        virtual void Init(const ScriptableObjects::IConfigRecord & /*config*/) override { assert(false); }
        virtual void Destroy() { delete this; }
        virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);

        virtual bool SupportsDistributedMBRead() const override
        {
            return true;
        }

        virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize) override;

        virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);

        virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& /*sectionName*/)
        {
            assert(false);
            throw std::runtime_error("Not supported");
        }

        virtual void SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<LabelIdType, LabelType>& /*labelMapping*/)
        {
            assert(false);
            throw std::runtime_error("Not supported");
        }

        virtual bool GetData(const std::wstring& /*sectionName*/, size_t /*numRecords*/, void* /*data*/, size_t& /*dataBufferSize*/, size_t /*recordStart*/ = 0)
        {
            assert(false);
            throw std::runtime_error("Not supported");
        }

        virtual bool GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>> & /*latticeinput*/, vector<size_t> &/*uids*/, vector<size_t> &/*boundaries*/, vector<size_t> &/*extrauttmap*/)
        {
            assert(false);
            throw std::runtime_error("Not supported");
        }

        virtual bool GetHmmData(msra::asr::simplesenonehmm * /*hmm*/)
        {
            assert(false);
            throw std::runtime_error("Not supported");
        }

        virtual bool DataEnd(EndDataType endDataType);
        void CopyMBLayoutTo(MBLayoutPtr);
        void SetSentenceEndInBatch(vector<size_t> &/*sentenceEnd*/)
        {
            assert(false);
            throw std::runtime_error("Not supported");
        }

        void SetSentenceEnd(int /*actualMbSize*/){};

        void SetRandomSeed(int){ assert(false); };

        bool RequireSentenceSeg() const override 
        {
            assert(false);
            throw std::runtime_error("Not supported");
        };

        size_t GetNumParallelSequences();
    };
}}}
