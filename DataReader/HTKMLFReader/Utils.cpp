#include "stdafx.h"
#include "Utils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    void Utils::ExpandDotDotDot(std::wstring& featPath, const std::wstring& scpPath, std::wstring& scpDirCached)
    {
        wstring delim = L"/\\";

        if (scpDirCached.empty())
        {
            scpDirCached = scpPath;
            wstring tail;
            auto pos = scpDirCached.find_last_of(delim);
            if (pos != wstring::npos)
            {
                tail = scpDirCached.substr(pos + 1);
                scpDirCached.resize(pos);
            }
            if (tail.empty()) // nothing was split off: no dir given, 'dir' contains the filename
                scpDirCached.swap(tail);
        }
        size_t pos = featPath.find(L"...");
        if (pos != featPath.npos)
            featPath = featPath.substr(0, pos) + scpDirCached + featPath.substr(pos + 3);
    }

    // GetFileConfigNames - determine the names of the features and labels sections in the config file
    // features - [in,out] a vector of feature name strings
    // labels - [in,out] a vector of label name strings
    void Utils::GetDataNamesFromConfig(
        const ConfigParameters& readerConfig,
        std::vector<std::wstring>& features,
        std::vector<std::wstring>& labels,
        std::vector<std::wstring>& hmms,
        std::vector<std::wstring>& lattices)
    {
        for (const auto & id : readerConfig.GetMemberIds())
        {
            if (!readerConfig.CanBeConfigRecord(id))
                continue;
            const ConfigParameters& temp = readerConfig(id);
            // see if we have a config parameters that contains a "file" element, it's a sub key, use it
            if (temp.ExistsCurrent(L"scpFile"))
            {
                features.push_back(id);
            }
            else if (temp.ExistsCurrent(L"mlfFile") || temp.ExistsCurrent(L"mlfFileList"))
            {
                labels.push_back(id);
            }
            else if (temp.ExistsCurrent(L"phoneFile"))
            {
                hmms.push_back(id);
            }
            else if (temp.ExistsCurrent(L"denlatTocFile"))
            {
                lattices.push_back(id);
            }
        }
    }

}}}