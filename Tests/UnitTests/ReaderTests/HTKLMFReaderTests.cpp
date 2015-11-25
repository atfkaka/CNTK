//
// <copyright file="HTKMLFReaderTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            namespace Test
            {
                struct HTKMLFixture : F
                {
                    HTKMLFixture() : F(L"HTKML")
                    {}
                };

                BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, HTKMLFixture)

                BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop)
                {
                    
                    HelperRunReaderTest(
                        L"HTKMLFReaderSimpleDataLoop_Config.txt",
                        L"HTKMLFReaderSimpleDataLoop_Control.txt",
                        L"HTKMLFReaderSimpleDataLoop_Data.txt",
                        "Simple_Test",
                        "reader",
                        500,
                        250,
                        2,
                        363,
                        132);
                };

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}

