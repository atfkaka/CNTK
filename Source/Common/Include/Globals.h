//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <atomic>
#include <string>
#include "Basics.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Class containing global configuration for CNTK.
    class Globals
    {
    public:
        static void ForceDeterministicAlgorithms()
        {
            m_forceDeterministicAlgorithms = true;
        }

        static bool ShouldForceDeterministicAlgorithms()
        {
            return m_forceDeterministicAlgorithms;
        }

        enum cudnnAutotunePolicy : int {
            OPTIMISTIC = 0,
            PESSIMISTIC,
            MEMORY_AWARE
        };

        static void SetCudnnAutotunePolicy(std::string cudnnAutotunePolicy)
        {
            if (cudnnAutotunePolicy == "optimistic")
            {
                m_cudnnAutotunePolicy = OPTIMISTIC;
            }
            else if (cudnnAutotunePolicy == "pessimistic")
            {
                m_cudnnAutotunePolicy = PESSIMISTIC;
            }
            else if (cudnnAutotunePolicy == "memoryAware")
            {
                m_cudnnAutotunePolicy = MEMORY_AWARE;
            }
            else
                RuntimeError("Unknown cudnnAutotunePolicy: %s\n", cudnnAutotunePolicy);
        }

        static int GetCudnnAutotunePolicy()
        {
            return m_cudnnAutotunePolicy;
        }

    private:
        static std::atomic<bool> m_forceDeterministicAlgorithms;
        static int m_cudnnAutotunePolicy;
    };
}}}
