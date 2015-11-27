#pragma once

#include "reader_interface.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Currently a hack because we do not know the device id in advance.
    class SubstitutingMemoryProvider : public MemoryProvider
    {
        MemoryProviderPtr m_provider;

    public:
        SubstitutingMemoryProvider() {}

        void SetMemoryProvider(MemoryProviderPtr p)
        {
            m_provider = p;
        }

        virtual void* alloc(size_t elementSize, size_t numberOfElements) override
        {
            return m_provider->alloc(elementSize, numberOfElements);
        }

        virtual void free(void* p) override
        {
            return m_provider->free(p);
        }
    };

    typedef std::shared_ptr<SubstitutingMemoryProvider> SubstitutingMemoryProviderPtr;
}}}
