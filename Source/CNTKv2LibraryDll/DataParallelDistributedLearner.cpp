//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataParallelDistributedLearner.h"
#include "DistributedCommunicator.h"

#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
#include "QuantizedDistributedCommunicator.h"
#include "QuantizedDataParallelDistributedLearner.h"
#include "BlockMomentumDistributedLearner.h"
#endif

namespace CNTK
{
#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits)
    {
        return MakeSharedObject<QuantizedMPICommunicatorImpl>(zeroThresholdFor1Bit, useQuantizationForSelfStripe, numQuantizationBits);
    }

    DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(
        QuantizedDistributedCommunicatorPtr communicator,
        const std::vector<LearnerPtr>& learners,
        bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<QuantizedDataParallelDistributedLearner>(communicator, useAsyncBufferedParameterUpdate, learners);
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        const std::vector<LearnerPtr>& learners,
        size_t blockSize,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedLearner>(
            communicator,
            learners,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate);
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        const std::vector<LearnerPtr>& learners,
        size_t blockSize,
        double blockMomentumAsTimeConstant,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedLearner>(
            communicator,
            learners,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate,
            blockMomentumAsTimeConstant);
    }

#else
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool, bool, size_t)
    {
        LogicError("Quantized MPI Communicator is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateQuantizedDataParallelDistributedTrainer(QuantizedDistributedCommunicatorPtr, bool, size_t)
    {
        LogicError("Quantized Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr /*communicator*/,
        size_t /*blockSize*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/,
        size_t /*distributedAfterSampleCount*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr /*communicator*/,
        size_t /*blockSize*/,
        double /*blockMomentumAsTimeConstant*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/,
        size_t /*distributedAfterSampleCount*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }
#endif

    DistributedLearnerPtr CreateDataParallelDistributedLearner(DistributedCommunicatorPtr communicator, const std::vector<LearnerPtr>& learners, bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<DataParallelDistributedLearner>(communicator, useAsyncBufferedParameterUpdate, learners);
    }

    DataParallelDistributedLearner::DataParallelDistributedLearner(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate, const std::vector<LearnerPtr>& learners)
        : DistributedLearnerBase(communicator, learners)
    {
        if (useAsyncBufferedParameterUpdate)
            LogicError("Asynchronous parameter update is not yet supported.");
    }

    bool DataParallelDistributedLearner::Update(std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info, size_t& totalNumberOfSampleSeen)
    {
        if (info.IsEmpty())
            PrepaireZeroGradients(gradientValues, info);

        std::vector<NDArrayViewPtr> valuesToAggregate;
        for (const auto& i : gradientValues)
            valuesToAggregate.push_back(i.second);
        valuesToAggregate.push_back(info.evalCriterionValue);
        valuesToAggregate.push_back(info.trainingLossValue);

        auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{1}, DeviceDescriptor::CPUDevice());
        valuesToAggregate.push_back(value);

        m_communicator->AggregateInPlace(valuesToAggregate, m_communicator->Workers());

        info.numberOfSamples = static_cast<size_t>(*valuesToAggregate.back()->WritableDataBuffer<double>());
        totalNumberOfSampleSeen += info.numberOfSamples;
        return info.IsEmpty();
    }
}
