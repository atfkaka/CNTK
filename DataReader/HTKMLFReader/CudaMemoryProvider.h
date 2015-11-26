#pragma once

#include <memory>
#include <CUDAPageLockedMemAllocator.h>

#include "reader_interface.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class CudaMemoryProvider : public MemoryProvider
    {
        std::unique_ptr<CUDAPageLockedMemAllocator> m_allocator;

    public:
        CudaMemoryProvider(int deviceId)
        {
            m_allocator = std::make_unique<CUDAPageLockedMemAllocator>(deviceId);
        }

        virtual void* alloc(size_t elementSize, size_t numberOfElements) override
        {
            size_t totalSize = elementSize * numberOfElements;
            return m_allocator->Malloc(totalSize);
        }

        virtual void free(void* p) override
        {
            if (!p)
            {
                return;
            }

            m_allocator->Free(reinterpret_cast<char*>(p));
        }
    };

}}}
