//
// <copyright file="HTKMLFReaderTests.cpp" company="Microsoft">
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

                BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, F)

                BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop)
                {
                    
                    HelperRunReaderTest(
                        L"HTKMLFReaderSimple_Config.txt",
                        L"HTKMLFReaderSimple_Control.txt",
                        L"HTKMLFReaderSimple_Data.txt",
                        "Simple_Test",
                        "reader",
                        500,
                        250,
                        2,
                        363,
                        132);
                    /*
                    const size_t epochSize = 500;
                    const size_t mbSize = 250;
                    const size_t epochs = 2;

                    ConfigParameters config;
                    std::wstring configFilePath(L"configFile=HTKMLFReaderSimple_Config.txt");
                    const std::wstring controlDataFilePath(L"HTKMLFReaderSimple_Control.txt");
                    const std::wstring testDataFilePath(L"HTKMLFReaderSimple_Data.txt");

                    wchar_t* arg[2] { L"CNTK", &configFilePath[0] };
                    const std::string rawConfigString = ConfigParameters::ParseCommandLine(2, arg, config);

                    config.ResolveVariables(rawConfigString);
                    const ConfigParameters simpleDemoConfig = config("Simple_Test");
                    const ConfigParameters readerConfig = simpleDemoConfig("reader");

                    DataReader<float> dataReader(readerConfig);

                    std::map<std::wstring, Matrix<float>*> map;
                    Matrix<float> features;
                    Matrix<float> labels;
                    map.insert(std::pair<wstring, Matrix<float>*>(L"features", &features));
                    map.insert(std::pair<wstring, Matrix<float>*>(L"labels", &labels));

                    // Setup output file
                    boost::filesystem::remove(testDataFilePath);
                    ofstream outputFile(testDataFilePath, ios::out);

                    // Perform the data reading
                    HelperWriteReaderContentToFile(outputFile, dataReader, map, epochs, mbSize, epochSize, 363, 132);

                    outputFile.close();

                    std::ifstream ifstream1(controlDataFilePath);
                    std::ifstream ifstream2(testDataFilePath);

                    std::istream_iterator<char> beginStream1(ifstream1);
                    std::istream_iterator<char> endStream1;
                    std::istream_iterator<char> beginStream2(ifstream2);
                    std::istream_iterator<char> endStream2;

                    BOOST_CHECK_EQUAL_COLLECTIONS(beginStream1, endStream1, beginStream2, endStream2);
                    */
                };

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}

