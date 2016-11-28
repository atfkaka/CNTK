//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DistributedLearnerBase.h"
#include "DistributedCommunicator.h"

namespace CNTK
{
    DistributedLearnerBase::DistributedLearnerBase(DistributedCommunicatorPtr communicator, const std::vector<LearnerPtr>& learners)
        : DistributedLearner(),
          m_communicator(communicator),
          m_learner(std::make_shared<CompositeLearner>(learners))
    {
    }

    // Get checkpoint state associated with distributed trainer
    Dictionary DistributedLearnerBase::CreateCheckpoint()
    {
        return m_learner->CreateCheckpoint();
    }

    // Restores the state associated with distributed trainer
    void DistributedLearnerBase::RestoreFromCheckpoint(const Dictionary& checkpoint)
    {
        return m_learner->RestoreFromCheckpoint(checkpoint);
    }

    void DistributedLearnerBase::PrepaireZeroGradients(std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info)
    {
        // Need to intialize gradients to 0 in case when it is an empty minibatch.
        for (auto& g : gradientValues)
        {
            auto weights = g.first.Value();
            g.second = MakeSharedObject<NDArrayView>(0, weights->GetDataType(), weights->Shape(), weights->Device());
        }

        auto dataType = gradientValues.front().first.GetDataType();
        info.evalCriterionValue = MakeSharedObject<NDArrayView>(0, dataType, NDShape{ 1 }, DeviceDescriptor::CPUDevice());
        info.trainingLossValue = MakeSharedObject<NDArrayView>(0, dataType, NDShape{ 1 }, DeviceDescriptor::CPUDevice());
    }
}
