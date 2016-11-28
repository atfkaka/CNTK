//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include "CNTKLibrary.h"
#include "DistributedLearnerBase.h"

namespace CNTK
{
    ///
    /// Distributed Trainer.
    ///
    class DataParallelDistributedLearner : public DistributedLearnerBase
    {
    public:
        DataParallelDistributedLearner(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate, const std::vector<LearnerPtr>& learners);

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool Update(std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& trainingSampleCount, size_t& totalNumberOfSampleSeen) override;
    };
}