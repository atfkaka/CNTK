//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "BestGpu.h"

namespace CNTK
{
    namespace Internal
    {
        size_t NewUniqueId()
        {
            static std::atomic<unsigned long long> s_nextUniqueId(0);
            return s_nextUniqueId++;
        }
    }

    /*static*/ std::atomic<bool> DeviceDescriptor::s_defaultDeviceFrozen(false);
    /*static*/ std::shared_ptr<DeviceDescriptor> DeviceDescriptor::s_defaultDevice(new DeviceDescriptor(DeviceDescriptor::BestDevice()));

    /*static*/ DeviceDescriptor DeviceDescriptor::DefaultDevice()
    {
        return *s_defaultDevice;
    }

    /*static*/ DeviceDescriptor DeviceDescriptor::UseDefaultDevice()
    {
        bool alreadyFrozen = s_defaultDeviceFrozen.exchange(true);
        auto selectedDevice = DefaultDevice();
        if (!alreadyFrozen)
        {
            Microsoft::MSR::CNTK::OnDeviceSelected(selectedDevice.Id());
        }
        return selectedDevice;
    }

    /*static*/ void DeviceDescriptor::SetDefaultDevice(const DeviceDescriptor& newDefaultDevice)
    {
        if (s_defaultDeviceFrozen.load())
            RuntimeError("Process wide default device cannot be changed since it has been frozen by being implicitly used as the default device in a CNTK API call");

        s_defaultDevice.reset(new DeviceDescriptor(newDefaultDevice));
    }
    
    /*static*/ DeviceDescriptor DeviceDescriptor::BestDevice()
    {
        // TODO: add unit tests for this.
        auto id = Microsoft::MSR::CNTK::GetBestDevice();
        return id >= 0 ? DeviceDescriptor::GPUDevice(id) : DeviceDescriptor::CPUDevice();
    }


    /*static*/ const std::wstring Axis::StaticAxisNamePrefix = L"staticAxis_";

    /*static*/ std::unordered_set<std::wstring> Axis::s_allKnownDynamicAxisNames;

    /*static*/ const std::vector<Axis> Axis::DefaultInputVariableDynamicAxes = { Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() };

    /*static*/ const Axis& Axis::DefaultDynamicAxis()
    {
        static const Axis s_defaultDynamicAxis(L"defaultDynamicAxis");
        return s_defaultDynamicAxis;
    }

    /*static*/ const Axis& Axis::DefaultBatchAxis()
    {
        static const Axis s_defaultBatchAxis(L"defaultBatchAxis", false);
        return s_defaultBatchAxis;
    }

    /*static*/ Axis Axis::NewUniqueDynamicAxis(const std::wstring& axisNamePrefix, bool isOrderedDynamicAxis /*= true*/)
    {
        if (s_allKnownDynamicAxisNames.find(axisNamePrefix) == s_allKnownDynamicAxisNames.end())
            return Axis(axisNamePrefix, isOrderedDynamicAxis);

        for (size_t i = 1;; i++)
        {
            auto newDynamicAxisName = axisNamePrefix + std::to_wstring(i);
            if (s_allKnownDynamicAxisNames.find(newDynamicAxisName) == s_allKnownDynamicAxisNames.end())
                return Axis(newDynamicAxisName, isOrderedDynamicAxis);
        }
    }

    void Axis::RegisterAxisName(const std::wstring& axisName)
    {
        s_allKnownDynamicAxisNames.insert(axisName);
    }
}
