#pragma once

#include "reader_interface.h"
#include <memory>
#include <algorithm>

namespace Microsoft { namespace MSR { namespace CNTK {

    class HeapMemoryProvider : public MemoryProvider
    {
        static const size_t size_of_first_pointer = sizeof(void*);

    public:
        virtual void* alloc(size_t elementSize, size_t numberOfElements) override
        {
            size_t alignment = max(elementSize, size_of_first_pointer);
            size_t request_size = elementSize * numberOfElements + alignment;
            size_t needed = size_of_first_pointer + request_size;

            void* allocated = ::operator new(needed);
            void* allowed_space = reinterpret_cast<char*>(allocated) + size_of_first_pointer;
            void* p = std::align(alignment, elementSize, allowed_space, request_size);

            // save for delete calls to use
            (reinterpret_cast<void**>(p))[-1] = allocated;
            return p;
        }

        virtual void free(void* p) override
        {
            if (!p)
            {
                return;
            }

            void * alloc = reinterpret_cast<void**>(p)[-1];
            ::operator delete(alloc);
        }
    };
}}}
