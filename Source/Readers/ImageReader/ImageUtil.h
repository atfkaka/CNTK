//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "SequenceData.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    inline bool IdentifyElementTypeFromOpenCVType(int openCvType, ElementType& type)
    {
        type = ElementType::tvariant;
        switch (openCvType)
        {
        case CV_64F:
            type = ElementType::tdouble;
            return true;
        case CV_32F:
            type = ElementType::tfloat;
            return true;
        case CV_8U:
            type = ElementType::tuchar;
            return true;
        default:
            return false;
        }
    }

    inline ElementType GetElementTypeFromOpenCVType(int openCvType)
    {
        ElementType result;
        if (!IdentifyElementTypeFromOpenCVType(openCvType, result))
            RuntimeError("Unsupported OpenCV type '%d'", openCvType);
        return result;
    }

    // A helper interface to generate a typed label in a sparse format for categories.
    // It is represented as a array indexed by the category, containing zero values for all categories the sequence does not belong to,
    // and a single one for a category it belongs to: [ 0 .. 0.. 1 .. 0 ]
    class LabelGenerator
    {
    public:
        virtual void CreateLabelFor(size_t classId, CategorySequenceData& data) = 0;
        virtual ~LabelGenerator() { }
    };
    typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;

    // Simple implementation of the LabelGenerator.
    // The class is parameterized because the representation of 1 is type specific.
    template <class TElement>
    class TypedLabelGenerator : public LabelGenerator
    {
    public:
        TypedLabelGenerator(size_t labelDimension) : m_value(1), m_indices(labelDimension)
        {
            if (labelDimension > numeric_limits<IndexType>::max())
            {
                RuntimeError("Label dimension (%d) exceeds the maximum allowed "
                    "value (%d)\n", (int)labelDimension, (int)numeric_limits<IndexType>::max());
            }
            iota(m_indices.begin(), m_indices.end(), 0);
        }

        virtual void CreateLabelFor(size_t classId, CategorySequenceData& data) override
        {
            data.m_nnzCounts.resize(1);
            data.m_nnzCounts[0] = 1;
            data.m_totalNnzCount = 1;
            data.m_data = &m_value;
            data.m_indices = &(m_indices[classId]);
        }

    private:
        TElement m_value;
        vector<IndexType> m_indices;
    };

    static std::vector<char> FillIndexTable()
    {
        std::vector<char> indexTable;
        indexTable.resize(std::numeric_limits<char>().max());
        char value = 0;
        for (char i = 'A'; i <= 'Z'; i++)
            indexTable[i] = value++;
        assert(value == 26);

        for (char i = 'a'; i <= 'z'; i++)
            indexTable[i] = value++;
        assert(value == 52);

        for (char i = '0'; i <= '9'; i++)
            indexTable[i] = value++;
        assert(value == 62);
        indexTable['+'] = value++;
        indexTable['/'] = value++;
        assert(value == 64);
        return indexTable;
    }

    const static char* base64IndexTable = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static std::vector<char> base64DecodeTable = FillIndexTable();

    inline std::vector<char> Decode64BitImage(const char* begin, const char* end)
    {
        size_t length = end - begin;
        if ((end - begin) % 4 != 0)
            RuntimeError("Invalid base64 data");
        std::vector<char> result;
        result.resize((length * 3) / 4 - (*(end - 2) == '=') ? 2 : ((*(end - 1) == '=') ? 1 : 0));
        size_t currentDecodedIndex = 0;
        for (size_t i = 0; i < length; i += 4)
        {
            result[currentDecodedIndex++] = base64DecodeTable[*begin] << 2 | base64DecodeTable[*(begin + 1)] >> 4;
            result[currentDecodedIndex++] = base64DecodeTable[*begin + 1] << 4 | base64DecodeTable[*(begin + 2)] >> 2;
            result[currentDecodedIndex++] = base64DecodeTable[*begin + 2] << 6 | base64DecodeTable[*(begin + 3)];
        }
        return result;
    }

}}}
