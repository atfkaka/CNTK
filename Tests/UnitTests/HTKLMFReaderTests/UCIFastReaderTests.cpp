//
// <copyright file="UCIFastReaderTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "Matrix.h"
#include "commandArgUtil.h"
#include "DataReader.h"
#include "common/ReaderTestHelper.h"
#include "boost/filesystem.hpp"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            namespace Test
            {
                const int deviceId = 0;
                const float epsilon = 0.0001f;

                BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, F)

                BOOST_AUTO_TEST_CASE(UCIFastReaderSimpleDataLoop)
                {

                    HelperRunReaderTest(
                        L"UCIFastReaderSimpleDataLoop_Config.txt",
                        L"UCIFastReaderSimpleDataLoop_Control.txt",
                        L"UCIFastReaderSimpleDataLoop_Data.txt",
                        "Simple_Test",
                        "reader",
                        500,
                        250,
                        2,
                        2,
                        2);

                   BOOST_AUTO_TEST_SUITE_END()
                }
            }
        }
    }
}