#include "stdafx.h"
#include "FrameModePacker.h"

#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "msra_mgram.h"                 // for unigram scores of ground-truth path in sequence training

#include "rollingwindowsource.h"        // minibatch sources
#include "utterancesourcemulti.h"
#include "chunkevalsource.h"
#include "minibatchiterator.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "commandArgUtil.h"
#include "ScriptableObjects.h"
#include "HTKMLFReader.h"
#include "TimerUtility.h"

#ifdef __unix__
#include <limits.h>
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
#endif
#pragma warning (disable: 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this


namespace Microsoft { namespace MSR { namespace CNTK {

    FrameModePacker::EpochImplementation::EpochImplementation(FrameModePacker* parent)
        : m_parent(parent)
    {}

    Minibatch FrameModePacker::EpochImplementation::readMinibatch()
    {
        return m_parent->GetMinibatch();
    }

    FrameModePacker::EpochImplementation::~EpochImplementation()
    {}

    FrameModePacker::FrameModePacker(const ConfigParameters & config, MemoryProviderPtr memoryProvider, size_t elementSize)
    {
        m_memoryProvider = memoryProvider;
        m_elementSize = elementSize;
        InitFromConfig(config);
    }

    std::vector<InputDescriptionPtr> FrameModePacker::getInputs()
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    EpochPtr FrameModePacker::startNextEpoch(const EpochConfiguration& config)
    {
        StartDistributedMinibatchLoop(config.minibatchSize, config.index, config.workerRank, config.numberOfWorkers, config.totalSize);
        return std::make_unique<EpochImplementation>(this);
    }

    void FrameModePacker::InitFromConfig(const ConfigParameters & readerConfig)
    {
        intargvector numberOfuttsPerMinibatchForAllEpochs =
            readerConfig(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{ 1 })));

        m_numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;

        for (int i = 0; i < m_numSeqsPerMBForAllEpochs.size(); i++)
        {
            if (m_numSeqsPerMBForAllEpochs[i] < 1)
            {
                LogicError("nbrUttsInEachRecurrentIter cannot be less than 1.");
            }
        }

        m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[0];
        m_pMBLayout->Init(m_numSeqsPerMB, 0, true); // (SGD will ask before entering actual reading --TODO: This is hacky.)

        m_noData = false;

        wstring command(readerConfig(L"action", L"")); //look up in the config for the master command to determine whether we're writing output (inputs only) or training/evaluating (inputs and outputs)

        if (readerConfig.Exists(L"legacyMode"))
            RuntimeError("legacy mode has been deprecated\n");
        PrepareForTrainingOrTesting(readerConfig);
    }

    void FrameModePacker::PrepareForTrainingOrTesting(const ConfigParameters& readerConfig)
    {
        vector<wstring> scriptpaths;
        vector<wstring> RootPathInScripts;
        vector<wstring> mlfpaths;
        vector<vector<wstring>>mlfpathsmulti;
        size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
        vector<vector<wstring>> infilesmulti;
        size_t numFiles;
        wstring unigrampath(L"");
        //wstring statelistpath(L"");
        size_t randomize = randomizeAuto;
        size_t iFeat, iLabel;
        iFeat = iLabel = 0;
        vector<wstring> statelistpaths;
        vector<size_t> numContextLeft;
        vector<size_t> numContextRight;

        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;
        // for hmm and lattice 
        std::vector<std::wstring> hmmNames;
        std::vector<std::wstring> latticeNames;
        GetDataNamesFromConfig(readerConfig, featureNames, labelNames, hmmNames, latticeNames);
        if (featureNames.size() + labelNames.size() <= 1)
        {
            InvalidArgument("network needs at least 1 input and 1 output specified!");
        }

        //load data for all real-valued inputs (features)
        foreach_index(i, featureNames)
        {
            const ConfigParameters & thisFeature = readerConfig(featureNames[i]);
            m_featDims.push_back(thisFeature(L"dim"));
            intargvector contextWindow = thisFeature(L"contextWindow", ConfigParameters::Array(intargvector(vector<int>{ 1 })));
            if (contextWindow.size() == 1) // symmetric
            {
                size_t windowFrames = contextWindow[0];
                if (windowFrames % 2 == 0)
                    InvalidArgument("augmentationextent: neighbor expansion of input features to %d not symmetrical", (int)windowFrames);
                size_t context = windowFrames / 2;           // extend each side by this
                numContextLeft.push_back(context);
                numContextRight.push_back(context);

            }
            else if (contextWindow.size() == 2) // left context, right context
            {
                numContextLeft.push_back(contextWindow[0]);
                numContextRight.push_back(contextWindow[1]);
            }
            else
            {
                InvalidArgument("contextFrames must have 1 or 2 values specified, found %d", (int)contextWindow.size());
            }
            // update m_featDims to reflect the total input dimension (featDim x contextWindow), not the native feature dimension
            // that is what the lower level feature readers expect
            m_featDims[i] = m_featDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

            wstring type = thisFeature(L"type", L"real");
            if (!_wcsicmp(type.c_str(), L"real"))
            {
                m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
            }
            else
            {
                InvalidArgument("feature type must be 'real'");
            }

            m_featureNameToIdMap[featureNames[i]] = iFeat;
            scriptpaths.push_back(thisFeature(L"scpFile"));
            RootPathInScripts.push_back(thisFeature(L"prefixPathInSCP", L""));
            m_featureNameToDimMap[featureNames[i]] = m_featDims[i];

            m_featuresBufferMultiIO.push_back(nullptr);
            m_featuresBufferAllocatedMultiIO.push_back(0);

            iFeat++;
        }

        foreach_index(i, labelNames)
        {
            const ConfigParameters & thisLabel = readerConfig(labelNames[i]);
            if (thisLabel.Exists(L"labelDim"))
                m_labelDims.push_back(thisLabel(L"labelDim"));
            else if (thisLabel.Exists(L"dim"))
                m_labelDims.push_back(thisLabel(L"dim"));
            else
                InvalidArgument("labels must specify dim or labelDim");

            wstring type;
            if (thisLabel.Exists(L"labelType"))
                type = (const wstring &)thisLabel(L"labelType"); // let's deprecate this eventually and just use "type"...
            else
                type = (const wstring &)thisLabel(L"type", L"category"); // outputs should default to category

            if (!_wcsicmp(type.c_str(), L"category"))
                m_nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
            else
                InvalidArgument("label type must be 'category'");

            statelistpaths.push_back(thisLabel(L"labelMappingFile", L""));

            m_labelNameToIdMap[labelNames[i]] = iLabel;
            m_labelNameToDimMap[labelNames[i]] = m_labelDims[i];
            mlfpaths.clear();
            if (thisLabel.ExistsCurrent(L"mlfFile"))
            {
                mlfpaths.push_back(thisLabel(L"mlfFile"));
            }
            else
            {
                if (!thisLabel.ExistsCurrent(L"mlfFileList"))
                {
                    InvalidArgument("Either mlfFile or mlfFileList must exist in HTKMLFReder");
                }
                wstring list = thisLabel(L"mlfFileList");
                for (msra::files::textreader r(list); r;)
                {
                    mlfpaths.push_back(r.wgetline());
                }
            }
            mlfpathsmulti.push_back(mlfpaths);

            m_labelsBufferMultiIO.push_back(nullptr);
            m_labelsBufferAllocatedMultiIO.push_back(0);

            iLabel++;
        }

        //get lattice toc file names 
        std::pair<std::vector<wstring>, std::vector<wstring>> latticetocs;
        foreach_index(i, latticeNames)  //only support one set of lattice now
        {
            const ConfigParameters & thisLattice = readerConfig(latticeNames[i]);

            vector<wstring> paths;
            expand_wildcards(thisLattice(L"denLatTocFile"), paths);
            latticetocs.second.insert(latticetocs.second.end(), paths.begin(), paths.end());

            if (thisLattice.Exists(L"numLatTocFile"))
            {
                paths.clear();
                expand_wildcards(thisLattice(L"numLatTocFile"), paths);
                latticetocs.first.insert(latticetocs.first.end(), paths.begin(), paths.end());
            }

        }

        //get HMM related file names
        vector<wstring> cdphonetyingpaths, transPspaths;
        foreach_index(i, hmmNames)
        {
            const ConfigParameters & thisHMM = readerConfig(hmmNames[i]);

            cdphonetyingpaths.push_back(thisHMM(L"phoneFile"));
            transPspaths.push_back(thisHMM(L"transPFile", L""));
        }

        if (iFeat != scriptpaths.size() || iLabel != mlfpathsmulti.size())
            RuntimeError("# of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n");

        if (readerConfig.Exists(L"randomize"))
        {
            wstring randomizeString = readerConfig.CanBeString(L"randomize") ? readerConfig(L"randomize") : wstring();
            if (!_wcsicmp(randomizeString.c_str(), L"none"))
                randomize = randomizeNone;
            else if (!_wcsicmp(randomizeString.c_str(), L"auto"))
                randomize = randomizeAuto;
            else
                randomize = readerConfig(L"randomize");
        }

        m_verbosity = readerConfig(L"verbosity", 2);

        // determine if we partial minibatches are desired
        wstring minibatchMode(readerConfig(L"minibatchMode", L"partial"));
        m_partialMinibatch = !_wcsicmp(minibatchMode.c_str(), L"partial");

        // get the read method, defaults to "blockRandomize" other option is "rollingWindow"
        wstring readMethod(readerConfig(L"readMethod", L"blockRandomize"));

        if (readMethod == L"blockRandomize" && randomize == randomizeNone)
            InvalidArgument("'randomize' cannot be 'none' when 'readMethod' is 'blockRandomize'.");

        // read all input files (from multiple inputs)
        // TO DO: check for consistency (same number of files in each script file)
        numFiles = 0;
        foreach_index(i, scriptpaths)
        {
            vector<wstring> filelist;
            std::wstring scriptpath = scriptpaths[i];
            fprintf(stderr, "reading script file %ls ...", scriptpath.c_str());
            size_t n = 0;
            for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/;)
            {
                filelist.push_back(reader.wgetline());
                n++;
            }

            fprintf(stderr, " %lu entries\n", n);

            if (i == 0)
                numFiles = n;
            else
                if (n != numFiles)
                    RuntimeError("number of files in each scriptfile inconsistent (%d vs. %d)", (int)numFiles, (int)n);

            // post processing file list : 
            //  - if users specified PrefixPath, add the prefix to each of path in filelist
            //  - else do the dotdotdot expansion if necessary 
            wstring rootpath = RootPathInScripts[i];
            if (!rootpath.empty()) // use has specified a path prefix for this  feature 
            {
                // first make slash consistent (sorry for linux users:this is not necessary for you)
                std::replace(rootpath.begin(), rootpath.end(), L'\\', L'/');
                // second, remove trailling slash if there is any 
                std::wregex trailer(L"/+$");
                rootpath = std::regex_replace(rootpath, trailer, wstring());
                // third, join the rootpath with each entry in filelist 
                if (!rootpath.empty())
                {
                    for (wstring & path : filelist)
                    {
                        if (path.find_first_of(L'=') != wstring::npos)
                        {
                            vector<wstring> strarr = msra::strfun::split(path, L"=");
#ifdef WIN32
                            replace(strarr[1].begin(), strarr[1].end(), L'\\', L'/');
#endif 

                            path = strarr[0] + L"=" + rootpath + L"/" + strarr[1];
                        }
                        else
                        {
#ifdef WIN32
                            replace(path.begin(), path.end(), L'\\', L'/');
#endif 
                            path = rootpath + L"/" + path;
                        }
                    }
                }
            }
            else
            {
                /*
                do "..." expansion if SCP uses relative path names
                "..." in the SCP means full path is the same as the SCP file
                for example, if scp file is "//aaa/bbb/ccc/ddd.scp"
                and contains entry like
                .../file1.feat
                .../file2.feat
                etc.
                the features will be read from
                //aaa/bbb/ccc/file1.feat
                //aaa/bbb/ccc/file2.feat
                etc.
                This works well if you store the scp file with the features but
                do not want different scp files everytime you move or create new features
                */
                wstring scpdircached;
                for (auto & entry : filelist)
                    ExpandDotDotDot(entry, scriptpath, scpdircached);
            }

            infilesmulti.push_back(std::move(filelist));
        }

        if (readerConfig.Exists(L"unigram"))
            unigrampath = (const wstring &)readerConfig(L"unigram");

        // load a unigram if needed (this is used for MMI training)
        msra::lm::CSymbolSet unigramsymbols;
        std::unique_ptr<msra::lm::CMGramLM> unigram;
        size_t silencewordid = SIZE_MAX;
        size_t startwordid = SIZE_MAX;
        size_t endwordid = SIZE_MAX;
        if (unigrampath != L"")
        {
            unigram.reset(new msra::lm::CMGramLM());
            unigram->read(unigrampath, unigramsymbols, false/*filterVocabulary--false will build the symbol map*/, 1/*maxM--unigram only*/);
            silencewordid = unigramsymbols["!silence"];     // give this an id (even if not in the LM vocabulary)
            startwordid = unigramsymbols["<s>"];
            endwordid = unigramsymbols["</s>"];
        }

        if (!unigram && latticetocs.second.size() > 0)
            fprintf(stderr, "trainlayer: OOV-exclusion code enabled, but no unigram specified to derive the word set from, so you won't get OOV exclusion\n");

        // currently assumes all mlfs will have same root name (key)
        set<wstring> restrictmlftokeys;     // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
        if (infilesmulti[0].size() <= 100)
        {
            foreach_index(i, infilesmulti[0])
            {
                msra::asr::htkfeatreader::parsedpath ppath(infilesmulti[0][i]);
                const wstring key = regex_replace((wstring)ppath, wregex(L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
                restrictmlftokeys.insert(key);
            }
        }
        // get labels

        //if (readerConfig.Exists(L"statelist"))
        //    statelistpath = readerConfig(L"statelist");

        double htktimetoframe = 100000.0;           // default is 10ms 
        //std::vector<msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>> labelsmulti;
        std::vector<std::map<std::wstring, std::vector<msra::asr::htkmlfentry>>> labelsmulti;
        //std::vector<std::wstring> pagepath;
        foreach_index(i, mlfpathsmulti)
        {
            const msra::lm::CSymbolSet* wordmap = unigram ? &unigramsymbols : NULL;
            msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>
                labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], wordmap, (map<string, size_t>*) NULL, htktimetoframe);      // label MLF
            // get the temp file name for the page file

            // Make sure 'msra::asr::htkmlfreader' type has a move constructor
            static_assert(std::is_move_constructible<msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>>::value,
                "Type 'msra::asr::htkmlfreader' should be move constructible!");

            labelsmulti.push_back(std::move(labels));
        }

        if (!_wcsicmp(readMethod.c_str(), L"blockRandomize"))
        {
            // construct all the parameters we don't need, but need to be passed to the constructor...

            // now get the frame source. This has better randomization and doesn't create temp files
            m_frameSource.reset(new msra::dbn::minibatchutterancesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomize, msra::dbn::latticesource(latticetocs, std::unordered_map<std::string, size_t>()), map<wstring, msra::lattices::lattice::htkmlfwordsequence>(), true));
            m_frameSource->setverbosity(m_verbosity);
        }
        else if (!_wcsicmp(readMethod.c_str(), L"rollingWindow"))
        {
            std::wstring pageFilePath;
            std::vector<std::wstring> pagePaths;
            if (readerConfig.Exists(L"pageFilePath"))
            {
                pageFilePath = (const wstring &)readerConfig(L"pageFilePath");

                // replace any '/' with '\' for compat with default path
                std::replace(pageFilePath.begin(), pageFilePath.end(), '/', '\\');
#ifdef _WIN32               
                // verify path exists
                DWORD attrib = GetFileAttributes(pageFilePath.c_str());
                if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY))
                    RuntimeError("pageFilePath does not exist");
#endif
#ifdef __unix__
                struct stat statbuf;
                if (stat(wtocharpath(pageFilePath).c_str(), &statbuf) == -1)
                {
                    RuntimeError("pageFilePath does not exist");
                }
#endif
            }
            else  // using default temporary path
            {
#ifdef _WIN32
                pageFilePath.reserve(MAX_PATH);
                GetTempPath(MAX_PATH, &pageFilePath[0]);
#endif
#ifdef __unix__
                pageFilePath = L"/tmp/temp.CNTK.XXXXXX";
#endif
            }

#ifdef _WIN32
            if (pageFilePath.size()>MAX_PATH - 14) // max length of input to GetTempFileName is MAX_PATH-14
                RuntimeError("pageFilePath must be less than %d characters", MAX_PATH - 14);
#else
            if (pageFilePath.size()>PATH_MAX - 14) // max length of input to GetTempFileName is PATH_MAX-14
                RuntimeError("pageFilePath must be less than %d characters", PATH_MAX - 14);
#endif
            foreach_index(i, infilesmulti)
            {
#ifdef _WIN32
                wchar_t tempFile[MAX_PATH];
                GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
                pagePaths.push_back(tempFile);
#endif
#ifdef __unix__
                char tempFile[PATH_MAX];
                strcpy(tempFile, msra::strfun::utf8(pageFilePath).c_str());
                int fid = mkstemp(tempFile);
                unlink(tempFile);
                close(fid);
                pagePaths.push_back(GetWC(tempFile));
#endif
            }

            const bool mayhavenoframe = false;
            int addEnergy = 0;

            m_frameSource.reset(new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomize, pagePaths, mayhavenoframe, addEnergy));
            m_frameSource->setverbosity(m_verbosity);
        }
        else
        {
            RuntimeError("readMethod must be 'rollingWindow' or 'blockRandomize'");
        }
    }

    //StartMinibatchLoop - Startup a minibatch loop 
    // requestedMBSize - [in] size of the minibatch (number of frames, etc.)
    // epoch - [in] epoch number for this loop
    // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
    void FrameModePacker::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
    {
        assert(subsetNum < numSubsets);
        assert((subsetNum == 0) && (numSubsets == 1));

        m_mbNumTimeSteps = requestedMBSize;       // note: ignored in frame mode and full-sequence mode

        m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[epoch];

        m_pMBLayout->Init(m_numSeqsPerMB, 0, false); // (SGD will ask before entering actual reading --TODO: This is hacky.)

        // resize the arrays
        // These are sized to the requested number. If not all can be filled, it will still return this many, just with gaps.
        // In frame mode, m_numSeqsPerMB must be 1. However, the returned layout has one 1-frame sequence per frame.
        m_sentenceEnd.assign(m_numSeqsPerMB, true);
        m_processedFrame.assign(m_numSeqsPerMB, 0);
        m_numFramesToProcess.assign(m_numSeqsPerMB, 0);
        m_switchFrame.assign(m_numSeqsPerMB, 0);
        m_numValidFrames.assign(m_numSeqsPerMB, 0);


        // for the multi-utterance process
        m_featuresBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_featuresBufferAllocatedMultiUtt.assign(m_numSeqsPerMB, 0);
        m_labelsBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_labelsBufferAllocatedMultiUtt.assign(m_numSeqsPerMB, 0);

        // for the multi-utterance process for lattice and phone boundary
        m_latticeBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_labelsIDBufferMultiUtt.resize(m_numSeqsPerMB);
        m_phoneboundaryIDBufferMultiUtt.resize(m_numSeqsPerMB);

        if ((m_numSeqsPerMB > 1))
        {
            LogicError("nbrUttsInEachRecurrentIter cannot be more than 1 in frame mode reading.");
        }

        // BUGBUG: in BPTT and sequence mode, we should pass 1 or 2 instead of requestedMBSize to ensure we only get one utterance back at a time
        StartMinibatchLoopToTrainOrTest(requestedMBSize, epoch, subsetNum, numSubsets, requestedEpochSamples);
    }

    void FrameModePacker::StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples)
    {
        size_t datapasses = 1;
        size_t totalFrames = m_frameSource->totalframes();

        size_t extraFrames = totalFrames%mbSize;
        size_t minibatches = totalFrames / mbSize;

        // if we are allowing partial minibatches, do nothing, and let it go through
        if (!m_partialMinibatch)
        {
            // we don't want any partial frames, so round total frames to be an even multiple of our mbSize
            if (totalFrames > mbSize)
                totalFrames -= extraFrames;

            if (requestedEpochSamples == requestDataSize)
            {
                requestedEpochSamples = totalFrames;
            }
            else if (minibatches > 0)   // if we have any full minibatches
            {
                // since we skip the extraFrames, we need to add them to the total to get the actual number of frames requested
                size_t sweeps = (requestedEpochSamples - 1) / totalFrames; // want the number of sweeps we will skip the extra, so subtract 1 and divide
                requestedEpochSamples += extraFrames*sweeps;
            }
        }
        else if (requestedEpochSamples == requestDataSize)
        {
            requestedEpochSamples = totalFrames;
        }

        m_mbiter.reset(new msra::dbn::minibatchiterator(*m_frameSource, epoch, requestedEpochSamples, mbSize, subsetNum, numSubsets, datapasses));
        // Advance the MB iterator until we find some data or reach the end of epoch
        while ((m_mbiter->currentmbframes() == 0) && *m_mbiter)
        {
            (*m_mbiter)++;
        }

        m_noData = false;
        if (!(*m_mbiter))
            m_noData = true;

        if (!m_featuresBufferMultiIO.empty())
        {
            if (m_featuresBufferMultiIO[0] != nullptr) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
            {
                foreach_index(i, m_featuresBufferMultiIO)
                {
                    m_featuresBufferMultiIO[i] = nullptr;
                    m_featuresBufferAllocatedMultiIO[i] = 0;
                }
            }

            m_featuresStartIndexMultiUtt.assign(m_featuresBufferMultiIO.size()*m_numSeqsPerMB, 0);

        }

        if (!m_labelsBufferMultiIO.empty())
        {
            if (m_labelsBufferMultiIO[0] != nullptr)
            {
                foreach_index(i, m_labelsBufferMultiIO)
                {
                    m_labelsBufferMultiIO[i] = nullptr;
                    m_labelsBufferAllocatedMultiIO[i] = 0;
                }
            }

            m_labelsStartIndexMultiUtt.assign(m_labelsBufferMultiIO.size()*m_numSeqsPerMB, 0);
        }

        for (size_t u = 0; u < m_numSeqsPerMB; u++)
        {
            if (m_featuresBufferMultiUtt[u] != NULL)
            {
                m_featuresBufferMultiUtt[u] = NULL;
                m_featuresBufferAllocatedMultiUtt[u] = 0;
            }

            if (m_labelsBufferMultiUtt[u] != NULL)
            {
                m_labelsBufferMultiUtt[u] = NULL;
                m_labelsBufferAllocatedMultiUtt[u] = 0;
            }

            if (m_latticeBufferMultiUtt[u] != NULL)
            {
                m_latticeBufferMultiUtt[u].reset();
            }

            ReNewBufferForMultiIO(u);
        }
    }

    Minibatch FrameModePacker::GetMinibatch()
    {
        size_t id;
        size_t dim;
        bool skip = false;
        
        Minibatch mb;


        // on first minibatch, make sure we can supply data for requested nodes
        std::map<std::wstring, size_t>::iterator iter;

        Timer aggregateTimer;
        if (m_verbosity > 2)
            aggregateTimer.Start();

        do
        {
            m_extraLatticeBufferMultiUtt.clear();
                m_extraLabelsIDBufferMultiUtt.clear();
                m_extraPhoneboundaryIDBufferMultiUtt.clear();
                m_extraSeqsPerMB.clear();
                if (m_noData && m_numFramesToProcess[0] == 0)    //no data left for the first channel of this minibatch, 
                {
                    mb.notAtEndOfEpoch = false;
                    return mb;
                }

                // BUGBUG: We should decide how many utterances we are going to take, until the desired number of frames has been filled.
                //         Currently it seems to fill a fixed number of utterances, regardless of their length.

                // decide the m_mbNumTimeSteps
                // The number of columns is determined by the longest utterance amongst the desired set.
                // I.e. whatever is user-specified as the MB size, will be ignored here (that value is, however, passed down to the underlying reader).  BUGBUG: That is even more wrong.
                // BUGBUG: We should honor the mbSize parameter and fill up to the requested number of samples, using the requested #parallel sequences.
                m_mbNumTimeSteps = m_numFramesToProcess[0];
                for (size_t i = 1; i < m_numSeqsPerMB; i++)
                {
                    if (m_mbNumTimeSteps < m_numFramesToProcess[i])
                        m_mbNumTimeSteps = m_numFramesToProcess[i];
                }

                    assert(m_numSeqsPerMB == 1);
                    m_pMBLayout->Init(m_mbNumTimeSteps, 1, false/*means not sequential*/);

                // create a MB with the desired utterance
                // Each utterance become a separate parallel sequence.
                skip = (!m_partialMinibatch && (m_mbiter->requestedframes() != m_mbNumTimeSteps) && (m_frameSource->totalframes() > m_mbNumTimeSteps));
                for (size_t i = 0; i < m_numSeqsPerMB; i++)
                {
                    if (!skip)
                    {
                        m_numValidFrames[i] = m_numFramesToProcess[i];
                        if (m_numValidFrames[i] > 0)
                        {
                            m_extraSeqsPerMB.push_back(i);
                            FillOneUttDataforParallelmode(0, m_numValidFrames[i], i, i);
                            if (m_latticeBufferMultiUtt[i] != nullptr)
                            {
                                m_extraLatticeBufferMultiUtt.push_back(m_latticeBufferMultiUtt[i]);
                                m_extraLabelsIDBufferMultiUtt.push_back(m_labelsIDBufferMultiUtt[i]);
                                m_extraPhoneboundaryIDBufferMultiUtt.push_back(m_phoneboundaryIDBufferMultiUtt[i]);
                            }
                        }
                    }
                    ReNewBufferForMultiIO(i);
                }

                // insert extra utt to the channels with space
                // As long as there is a gap at the end of any parallel sequence that is large enough for another utterance, fill it in.
                if (!skip)
                {
                    size_t nextMinibatchUttnum = 0;
                    bool inserted;
                    m_extraNumSeqs = 0;
                    for (size_t i = 0; i < m_numSeqsPerMB; i++)
                    {
                        while (nextMinibatchUttnum <= i)
                        {
                            inserted = false;
                            if (!inserted)
                            {
                                nextMinibatchUttnum++;
                            }
                        }
                    }

                    // Filling in the minibatch.
                    for (auto name : m_nameToTypeMap)
                    {
                        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                        {
                            id = m_featureNameToIdMap[iter->first];
                            dim = m_featureNameToDimMap[iter->first];

                            auto layout = std::make_shared<Layout>();
                            layout->columns = m_pMBLayout;

                            mb.minibatch[m_nameToId[name.first]] =
                                std::make_shared<Input>(m_featuresBufferMultiIO[id].get(), dim * m_mbNumTimeSteps * m_numSeqsPerMB * m_elementSize, layout);
                        }
                        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                        {
                            id = m_labelNameToIdMap[iter->first];
                            dim = m_labelNameToDimMap[iter->first];
                            auto layout = std::make_shared<Layout>();
                            layout->columns = m_pMBLayout;

                            mb.minibatch[m_nameToId[name.first]] =
                                std::make_shared<Input>(m_labelsBufferMultiIO[id].get(), dim * m_mbNumTimeSteps * m_numSeqsPerMB * m_elementSize, layout);
                        }
                    }
                }
        } while (skip); // keep going if we didn't get the right size minibatch

        if (m_verbosity > 2)
        {
            aggregateTimer.Stop();
            double totalMBReadTime = aggregateTimer.ElapsedSeconds();
            fprintf(stderr, "Total Minibatch read time = %.8g\n", totalMBReadTime);
        }

        mb.notAtEndOfEpoch = true;
        return mb;
    }

    // copy an utterance into the minibatch given a location (parallel-sequence index, start frame)
    // TODO: This should use DataSlice(). But for that, DataSlice() will have to move out from ComputationNode.
    void FrameModePacker::FillOneUttDataforParallelmode(
        size_t startFr,
        size_t framenum,
        size_t channelIndex,
        size_t sourceChannelIndex)
    {
        size_t id;
        size_t dim;
        size_t numOfFea = m_featuresBufferMultiIO.size();

        for (auto name: m_nameToId)
        {
            if (m_nameToTypeMap[name.first] == InputOutputTypes::real)
            {
                id = m_featureNameToIdMap[name.first];
                dim = m_featureNameToDimMap[name.first];

                if (m_featuresBufferMultiIO[id] == nullptr || m_featuresBufferAllocatedMultiIO[id] < dim*m_mbNumTimeSteps*m_numSeqsPerMB)
                {
                    m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(dim*m_mbNumTimeSteps*m_numSeqsPerMB, m_elementSize);
                    memset(m_featuresBufferMultiIO[id].get(), 0, m_elementSize*dim*m_mbNumTimeSteps*m_numSeqsPerMB);
                    m_featuresBufferAllocatedMultiIO[id] = dim*m_mbNumTimeSteps*m_numSeqsPerMB;
                }

                for (size_t j = 0, k = startFr; j < framenum; j++, k++) // column major, so iterate columns
                {
                    // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                    memcpy_s(&((char*)m_featuresBufferMultiIO[id].get())[(k*m_numSeqsPerMB + channelIndex)*dim*m_elementSize], m_elementSize*dim, &((char*)m_featuresBufferMultiUtt[sourceChannelIndex].get())[(j*dim + m_featuresStartIndexMultiUtt[id + sourceChannelIndex*numOfFea])*m_elementSize], m_elementSize*dim);
                }
            }
            else if (m_nameToTypeMap[name.first] == InputOutputTypes::category)
            {
                id = m_labelNameToIdMap[name.first];
                dim = m_labelNameToDimMap[name.first];
                if (m_labelsBufferMultiIO[id] == nullptr || m_labelsBufferAllocatedMultiIO[id] < dim*m_mbNumTimeSteps*m_numSeqsPerMB)
                {
                    m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(dim*m_mbNumTimeSteps*m_numSeqsPerMB, m_elementSize);
                    memset(m_labelsBufferMultiIO[id].get(), 0, m_elementSize*dim*m_mbNumTimeSteps*m_numSeqsPerMB);
                    m_labelsBufferAllocatedMultiIO[id] = dim*m_mbNumTimeSteps*m_numSeqsPerMB;
                }

                for (size_t j = 0, k = startFr; j < framenum; j++, k++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        memcpy_s(&((char*)m_labelsBufferMultiIO[id].get())[(k*m_numSeqsPerMB + channelIndex)*dim*m_elementSize], m_elementSize*dim, &((char*)m_labelsBufferMultiUtt[sourceChannelIndex].get())[(j*dim + m_labelsStartIndexMultiUtt[id + sourceChannelIndex*numOfFea])*m_elementSize], m_elementSize*dim);
                    }
                }
            }
        }
    }

    bool FrameModePacker::ReNewBufferForMultiIO(size_t i)
    {
        if (m_noData)
        {
            if (i == 0)
                m_numFramesToProcess[i] = 0;
            return false;
        }

        size_t numOfFea = m_featuresBufferMultiIO.size();
        size_t numOfLabel = m_labelsBufferMultiIO.size();

        size_t totalFeatNum = 0;
        foreach_index(id, m_featuresBufferAllocatedMultiIO)
        {
            const msra::dbn::matrixstripe featOri = m_mbiter->frames(id);
            size_t fdim = featOri.rows();
            const size_t actualmbsizeOri = featOri.cols();
            m_featuresStartIndexMultiUtt[id + i*numOfFea] = totalFeatNum;
            totalFeatNum = fdim * actualmbsizeOri + m_featuresStartIndexMultiUtt[id + i*numOfFea];
        }
        if ((m_featuresBufferMultiUtt[i] == NULL) || (m_featuresBufferAllocatedMultiUtt[i] < totalFeatNum))
        {
            // eldak : should use simple new
            m_featuresBufferMultiUtt[i] = AllocateIntermediateBuffer(totalFeatNum, m_elementSize);
            m_featuresBufferAllocatedMultiUtt[i] = totalFeatNum;
        }

        size_t totalLabelsNum = 0;
        for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it)
        {
            size_t id = m_labelNameToIdMap[it->first];
            size_t dim = m_labelNameToDimMap[it->first];

            const vector<size_t> & uids = m_mbiter->labels(id);
            size_t actualmbsizeOri = uids.size();
            m_labelsStartIndexMultiUtt[id + i*numOfLabel] = totalLabelsNum;
            totalLabelsNum = m_labelsStartIndexMultiUtt[id + i*numOfLabel] + dim * actualmbsizeOri;
        }

        if ((m_labelsBufferMultiUtt[i] == NULL) || (m_labelsBufferAllocatedMultiUtt[i] < totalLabelsNum))
        {
            // eldak: should use simple new.
            m_labelsBufferMultiUtt[i] = AllocateIntermediateBuffer(totalLabelsNum, m_elementSize);
            m_labelsBufferAllocatedMultiUtt[i] = totalLabelsNum;
        }

        memset(m_labelsBufferMultiUtt[i].get(), 0, m_elementSize*totalLabelsNum);

        bool first = true;
        foreach_index(id, m_featuresBufferMultiIO)
        {
            const msra::dbn::matrixstripe featOri = m_mbiter->frames(id);
            const size_t actualmbsizeOri = featOri.cols();
            size_t fdim = featOri.rows();
            if (first)
            {
                m_numFramesToProcess[i] = actualmbsizeOri;
                first = false;
            }
            else
            {
                if (m_numFramesToProcess[i] != actualmbsizeOri)
                {
                    RuntimeError("The multi-IO features has inconsistent number of frames!");
                }
            }
            assert(actualmbsizeOri == m_mbiter->currentmbframes());


            for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns
            {
                // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                memcpy_s(&((char*)m_featuresBufferMultiUtt[i].get())[(k*fdim + m_featuresStartIndexMultiUtt[id + i*numOfFea]) * m_elementSize], m_elementSize*fdim, &featOri(0, k), m_elementSize*fdim);
            }
        }

        for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it)
        {
            size_t id = m_labelNameToIdMap[it->first];
            size_t dim = m_labelNameToDimMap[it->first];

            const vector<size_t> & uids = m_mbiter->labels(id);
            size_t actualmbsizeOri = uids.size();


            // loop through the columns and set one value to 1
            // in the future we want to use a sparse matrix here
            for (int k = 0; k < actualmbsizeOri; k++)
            {
                assert(uids[k] < dim);
                // eldak: dirty hack for now - the values should come from the underlying layer, not from the packer.
                if (m_elementSize == sizeof(float))
                {
                    ((float*)m_labelsBufferMultiUtt[i].get())[k*dim + uids[k] + m_labelsStartIndexMultiUtt[id + i*numOfLabel]] = 1;
                }
                else
                {
                    ((double*)m_labelsBufferMultiUtt[i].get())[k*dim + uids[k] + m_labelsStartIndexMultiUtt[id + i*numOfLabel]] = 1;
                }
            }
        }
        //lattice
        if (m_latticeBufferMultiUtt[i] != NULL)
        {
            m_latticeBufferMultiUtt[i].reset();
        }

        if (m_mbiter->haslattice())
        {
            m_latticeBufferMultiUtt[i] = std::move(m_mbiter->lattice(0));
            m_phoneboundaryIDBufferMultiUtt[i].clear();
            m_phoneboundaryIDBufferMultiUtt[i] = m_mbiter->bounds();
            m_labelsIDBufferMultiUtt[i].clear();
            m_labelsIDBufferMultiUtt[i] = m_mbiter->labels();
        }

        m_processedFrame[i] = 0;

        Timer mbIterAdvancementTimer;
        if (m_verbosity > 2)
            mbIterAdvancementTimer.Start();

        // Advance the MB iterator until we find some data or reach the end of epoch
        do
        {
            (*m_mbiter)++;
        } while ((m_mbiter->currentmbframes() == 0) && *m_mbiter);

        if (m_verbosity > 2)
        {
            mbIterAdvancementTimer.Stop();
            double advancementTime = mbIterAdvancementTimer.ElapsedSeconds();
            fprintf(stderr, "Time to advance mbiter = %.8g\n", advancementTime);
        }

        if (!(*m_mbiter))
            m_noData = true;

        return true;
    }

    // GetFileConfigNames - determine the names of the features and labels sections in the config file
    // features - [in,out] a vector of feature name strings
    // labels - [in,out] a vector of label name strings
    void FrameModePacker::GetDataNamesFromConfig(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels,
        std::vector<std::wstring>& hmms, std::vector<std::wstring>& lattices)
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

    void FrameModePacker::ExpandDotDotDot(wstring & featPath, const wstring & scpPath, wstring & scpDirCached)
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

    std::shared_ptr<void> FrameModePacker::AllocateIntermediateBuffer(size_t numElements, size_t elementSize)
    {
        return std::shared_ptr<void>(m_memoryProvider->alloc(elementSize, numElements), [this](void* p) {
            this->m_memoryProvider->free(p);
        });
    }
}}}