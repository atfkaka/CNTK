//
// <copyright file="ReaderTestHelper.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "DataReader.h"
#include "boost/filesystem.hpp"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft {  namespace MSR {  namespace CNTK
{
    namespace Test
    {
        struct F
        {
            F(wstring dataPath)
            {
                m_dataPath = dataPath;

                BOOST_TEST_MESSAGE("setup fixture");
                BOOST_TEST_MESSAGE("Current working directory:");
                m_initialWorkingPath = boost::filesystem::current_path().c_str();
                BOOST_TEST_MESSAGE(m_initialWorkingPath.c_str());

                boost::filesystem::path path(boost::unit_test::framework::master_test_suite().argv[0]);
                m_parentPath = path.parent_path().c_str();
                m_parentPath += L"/../../../Tests/UnitTests/ReaderTests/Data";

                // TODO: Setup the CWD based on the data path
                BOOST_TEST_MESSAGE("Setting current path to:");
                BOOST_TEST_MESSAGE(m_parentPath.c_str());

                boost::filesystem::current_path(m_parentPath);

                BOOST_TEST_MESSAGE("Current working directory is now:");
                BOOST_TEST_MESSAGE(boost::filesystem::current_path());
            }

            ~F()
            {
                BOOST_TEST_MESSAGE("teardown fixture");
            }

            wstring m_initialWorkingPath;
            wstring m_parentPath;
            wstring m_dataPath;

            // Helper function to write the Reader's content to a file.
            // outputFile : the file stream to output to.
            // dataReader : the DataReader to get minibatches from
            // map        : the map containing the feature and label matrices
            // epochs     : the number of epochs to read
            // mbSize     : the minibatch size
            // epochSize  : the epoch size
            // expectedFeatureRowsCount : the expected number of rows in the feature matrix
            // expectedLabelRowsCount   : the expected number of rows in the label matrix
            void HelperWriteReaderContentToFile(
                ofstream& outputFile,
                DataReader<float>& dataReader,
                std::map<std::wstring, Matrix<float>*>& map,
                size_t epochs,
                size_t mbSize,
                size_t epochSize,
                size_t expectedFeatureRowsCount,
                size_t expectedLabelsRowsCount)
            {
                Matrix<float>& features = *map.at(L"features");
                Matrix<float>& labels = *map.at(L"labels");

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);

                    for (int cnt = 0; dataReader.GetMinibatch(map); cnt++)
                    {
                        size_t numLabelsRows = labels.GetNumRows();
                        size_t numLabelsCols = labels.GetNumCols();
                        size_t numFeaturesRows = features.GetNumRows();
                        size_t numFeaturesCols = features.GetNumCols();

                        BOOST_CHECK_EQUAL(expectedLabelsRowsCount, numLabelsRows);
                        BOOST_CHECK_EQUAL(mbSize, numLabelsCols);
                        BOOST_CHECK_EQUAL(expectedFeatureRowsCount, numFeaturesRows);
                        BOOST_CHECK_EQUAL(mbSize, numFeaturesCols);

                        std::unique_ptr<float[]> pFeature{ features.CopyToArray() };
                        std::unique_ptr<float[]> pLabel{ labels.CopyToArray() };

                        size_t numFeatures = numFeaturesRows * numFeaturesCols;
                        size_t numLabels = numLabelsRows * numLabelsCols;

                        for (int i = 0; i < numFeatures; i++)
                        {
                            outputFile << pFeature[i] << (i % numFeaturesRows ? "\n" : " ");
                        }

                        for (int i = 0; i < numLabels; i++)
                        {
                            outputFile << pLabel[i] << (i % numLabelsRows ? "\n" : " ");
                        }
                    }
                }
            }

            // Helper function to run a Reader test.
            // configFileName       : the file name for the config file
            // controlDataFilePath  : the file path for the control data to verify against
            // testDataFilePath     : the file path for writing the minibatch data (used for comparing against control data)
            // testSectionName      : the section name for the test inside the config file
            // readerSectionName    : the reader field name in the test section
            // epochSize            : the epoch size
            // mbSize               : the minibatch size
            // epochs               : the number of epochs to read
            // expectedFeatureRowsCount : the expected number of rows in the feature matrix
            // expectedLabelRowsCount   : the expected number of rows in the label matrix
            void HelperRunReaderTest(
                wstring configFileName, 
                const wstring controlDataFilePath, 
                const wstring testDataFilePath,
                string testSectionName,
                string readerSectionName,
                size_t epochSize,
                size_t mbSize,
                size_t epochs,
                size_t expectedFeatureRowsCount,
                size_t expectedLabelsRowsCount)
            {
                ConfigParameters config;
                std::wstring configFileCommand(L"configFile=" + configFileName);

                wchar_t* arg[2] { L"CNTK", &configFileCommand[0] };
                const std::string rawConfigString = ConfigParameters::ParseCommandLine(2, arg, config);

                config.ResolveVariables(rawConfigString);
                const ConfigParameters simpleDemoConfig = config(testSectionName);
                const ConfigParameters readerConfig = simpleDemoConfig(readerSectionName);

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
                HelperWriteReaderContentToFile(outputFile, dataReader, map, epochs, mbSize, epochSize, expectedFeatureRowsCount, expectedLabelsRowsCount); // 363, 132);

                outputFile.close();

                std::ifstream ifstream1(controlDataFilePath);
                std::ifstream ifstream2(testDataFilePath);

                std::istream_iterator<char> beginStream1(ifstream1);
                std::istream_iterator<char> endStream1;
                std::istream_iterator<char> beginStream2(ifstream2);
                std::istream_iterator<char> endStream2;

                BOOST_CHECK_EQUAL_COLLECTIONS(beginStream1, endStream1, beginStream2, endStream2);
            }
        };
    }
}}}