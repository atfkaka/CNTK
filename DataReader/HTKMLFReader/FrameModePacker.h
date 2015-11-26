#pragma once

#include "reader_interface.h"
#include <vector>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {
    class FrameModePacker : public Reader
    {
    public:
        virtual std::vector<InputDescriptionPtr> getInputs() override;
        virtual EpochPtr startNextEpoch(const EpochConfiguration& config) override;

    public:
        FrameModePacker(const ConfigParameters & /*config*/) {}
    };

    typedef std::shared_ptr<FrameModePacker> FrameModePackerPtr;
}}}