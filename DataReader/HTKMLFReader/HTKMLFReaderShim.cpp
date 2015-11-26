//
// <copyright file="HTKMLFReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)

#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "commandArgUtil.h"
#include "HTKMLFReaderShim.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

#ifdef __unix__
#include <limits.h>
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::Init(const ConfigParameters & config)
    {
        m_packer = std::make_shared<FrameModePacker>(config);
    }

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize)
    {
        return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
    }

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
    {
        EpochConfiguration config;
        config.workerRank = subsetNum;
        config.numberOfWorkers = numSubsets;
        config.minibatchSize = requestedMBSize;
        config.totalSize = requestedEpochSamples;
        config.index = epoch;

        m_epoch = m_packer->startNextEpoch(config);
    }

    template<class ElemType>
    bool HTKMLFReaderShim<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
    {
        Minibatch m = m_epoch->readMinibatch();
        assert(false);

        auto inputs = m_packer->getInputs();
        std::map<size_t, wstring> idToName;
        for (auto i: inputs)
        {
            idToName.insert(std::make_pair(i->id, i->name));
        }

        for(auto input : m.minibatch)
        {
            const std::wstring& name = idToName[input.first];
            if (matrices.find(name) == matrices.end())
            {
                continue;
            }

            auto layout = input.second->getLayout();
            int rowNumber = 0;// layout->rows->Size();
            int columnNumber = 0; //layout->columns->Size();

            auto data = reinterpret_cast<const ElemType*>(input.second->getData());
            matrices[name]->SetValue(rowNumber, columnNumber, matrices[name]->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal);
        }

        return m;
    }

    template<class ElemType>
    bool HTKMLFReaderShim<ElemType>::DataEnd(EndDataType /*endDataType*/)
    {
        return false;
    }

    template<class ElemType>
    void HTKMLFReaderShim<ElemType>::CopyMBLayoutTo(MBLayoutPtr layout)
    {
        layout->CopyFrom(m_layout);
    }

    template<class ElemType>
    size_t HTKMLFReaderShim<ElemType>::GetNumParallelSequences()
    {
        return m_layout->GetNumParallelSequences();  // (this function is only used for validation anyway)
    }

    template class HTKMLFReaderShim<float>;
    template class HTKMLFReaderShim<double>;
}}}
