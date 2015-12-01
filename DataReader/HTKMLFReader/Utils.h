#pragma once

#include <vector>
#include <memory>

#include "reader_interface.h"
#include "commandArgUtil.h"

#include "biggrowablevectors.h"
#include "utterancesourcemulti.h"
#include "minibatchiterator.h"


namespace Microsoft { namespace MSR { namespace CNTK {

    class Utils
    {
    public:
        static void ExpandDotDotDot(std::wstring& featPath, const std::wstring& scpPath, std::wstring& scpDirCached);

        static void GetDataNamesFromConfig(
            const ConfigParameters& readerConfig,
            std::vector<std::wstring>& features,
            std::vector<std::wstring>& labels,
            std::vector<std::wstring>& hmms,
            std::vector<std::wstring>& lattices);

        static void CheckMinibatchSizes(const intargvector& numberOfuttsPerMinibatchForAllEpochs);
    };
}}}