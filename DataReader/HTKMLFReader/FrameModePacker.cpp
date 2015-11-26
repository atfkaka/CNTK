#include "stdafx.h"
#include "FrameModePacker.h"

std::vector<InputDescriptionPtr> Microsoft::MSR::CNTK::FrameModePacker::getInputs()
{
    throw std::logic_error("The method or operation is not implemented.");
}

EpochPtr Microsoft::MSR::CNTK::FrameModePacker::startNextEpoch(const EpochConfiguration& /*config*/)
{
    throw std::logic_error("The method or operation is not implemented.");
}
