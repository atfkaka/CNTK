//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Serialization.h"

namespace
{
    const std::wstring learnersPropertyName = L"Learners";
    const std::wstring externalStatePropertyName = L"ExternalState";
    const std::wstring totalSeenSamplesPropertyName = L"TotalSeenSamples";
}

namespace CNTK
{
    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners)
        : Trainer(model, lossFunction, nullptr, parameterLearners)
    {}

    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners)
        : Trainer(model, lossFunction, evaluationFunction, std::make_shared<CompositeLearner>(parameterLearners), 0)
    {}

    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, LearnerPtr learner, size_t)
        : m_model(model),
          m_lossFunction(lossFunction),
          m_evaluationFunction(evaluationFunction),
          m_learner(learner),
          m_prevMinibatchNumSamples(1),
          m_totalSamplesSeen(0)
    {
        std::vector<Variable> combinedFunctionArgs = { m_model, m_lossFunction };
        if (!m_lossFunction->Output().DynamicAxes().empty())
        {
            m_aggregatedLossFunction = ReduceSum(lossFunction);
            combinedFunctionArgs.push_back(m_aggregatedLossFunction);
            m_trainingSampleCountVar = m_lossFunction;
        }
        else
        {
            m_aggregatedLossFunction = m_lossFunction;
            m_trainingSampleCountVar = m_lossFunction->RootFunction()->Inputs()[0];
            if (model->Output() != m_trainingSampleCountVar)
                combinedFunctionArgs.push_back(m_trainingSampleCountVar);
        }

        if (m_evaluationFunction)
        {
            combinedFunctionArgs.push_back(m_evaluationFunction);

            if (!m_evaluationFunction->Output().DynamicAxes().empty())
            {
                m_aggregatedEvaluationFunction = ReduceSum(m_evaluationFunction);
                combinedFunctionArgs.push_back(m_aggregatedEvaluationFunction);
                m_testSampleCountVar = m_evaluationFunction;
            }
            else
            {
                m_aggregatedEvaluationFunction = m_evaluationFunction;
                m_testSampleCountVar = m_evaluationFunction->RootFunction()->Inputs()[0];
                if ((m_testSampleCountVar != m_trainingSampleCountVar) && (model->Output() != m_testSampleCountVar))
                    combinedFunctionArgs.push_back(m_testSampleCountVar);
            }
        }

        m_combinedTrainingFunction = Combine(combinedFunctionArgs);

        auto modelParameters = m_combinedTrainingFunction->Parameters();
        auto learnerParameters = m_learner->Parameters();
        std::unordered_set<Parameter> modelParametersSet(modelParameters.begin(), modelParameters.end());
        std::unordered_set<Parameter> learnerParametersSet(learnerParameters.begin(), learnerParameters.end());
        if (modelParametersSet != learnerParametersSet)
            InvalidArgument("Trainer ctor: Union of the parameters covered by the specified parameterLearners should match the specified model's parameters");
    }

    CNTK_API Trainer::~Trainer()
    {
    }

    static double GetScalarValue(const ValuePtr& value)
    {
        if (value->Mask())
            LogicError("Scalar Value object cannot have an associated mask");

        auto scalarData = value->Data();
        if (scalarData->Shape().TotalSize() != 1)
            LogicError("Scalar Value object's has a size > 1");

        double scalar = std::numeric_limits<double>::quiet_NaN();
        NDArrayViewPtr cpuData;
        if (scalarData->Device() == DeviceDescriptor::CPUDevice())
            cpuData = scalarData;
        else
        {
            cpuData = std::make_shared<NDArrayView>(scalarData->GetDataType(), scalarData->Shape(), CNTK::DeviceDescriptor::CPUDevice());
            cpuData->CopyFrom(*scalarData);
        }

        if (scalarData->GetDataType() == DataType::Float)
            scalar = *(cpuData->DataBuffer<float>());
        else if (scalarData->GetDataType() == DataType::Double)
            scalar = *(cpuData->DataBuffer<double>());
        else
            LogicError("Unsupported DataType of training loss value");

        return scalar;
    }

    static size_t GetSampleCount(const Variable& var, const ValuePtr& value)
    {
        auto valueDataShape = value->Shape();
        size_t numMaskedSamples = value->MaskedCount();
        size_t numSamplesInDataArrayView = valueDataShape.SubShape(var.Shape().Rank()).TotalSize();
        if (numMaskedSamples > numSamplesInDataArrayView)
            LogicError("Number of masked values cannot exceed the number of samples that the Value object's Data NDArrayView can hold");

        return (numSamplesInDataArrayView - numMaskedSamples);
    }

    double Trainer::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        if (!m_aggregatedEvaluationFunction)
            InvalidArgument("Trainer::TestMinibatch: Cannot test when no evaluation function was specified during 'this' trainer's construction");

        // TODO: Should we refactor this code that is somewhat similar to the prologue of the TrainMinibatch function
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedEvaluationFunction, nullptr }, { m_testSampleCountVar, nullptr } };
        m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice);

        auto sampleCount = GetSampleCount(m_testSampleCountVar, outputs[m_testSampleCountVar]);
        return (GetScalarValue(outputs[m_aggregatedEvaluationFunction]) / sampleCount);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TrainMinibatch(arguments, outputsToFetch, computeDevice);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        // TODO: We should reconsider the interface
        // Probably passing the flag that the minibatch is the last, and empty arguments in case of empty minibatch.

        std::vector<std::pair<Parameter, NDArrayViewPtr>> gradients;
        auto modelParameters = m_combinedTrainingFunction->Parameters();
        gradients.reserve(modelParameters.size());

        bool emptyMinibatch = arguments.empty() || (arguments.begin()->second == nullptr);
        if (emptyMinibatch)
        {
            m_prevMinibatchNumSamples = 0;
            // Gradients are not existing.
            for (const auto& parameter : modelParameters)
                gradients.push_back(std::make_pair(parameter, nullptr));
        }
        else
        {
            // Get gradients after forward/backward pass.
            std::unordered_map<Variable, ValuePtr> parameterGradients;
            ExecuteForwardBackward(arguments, outputsToFetch, computeDevice, parameterGradients);
            for (const auto& parameter : modelParameters)
                gradients.push_back(std::make_pair(parameter, parameterGradients[parameter]->Data()));
        }

        // Update parameters.
        MinibatchInfo info
        {
            arguments.empty(),
            m_prevMinibatchNumSamples,
            m_prevMinibatchAggregateTrainingLossValue->Data(),
            m_prevMinibatchAggregateEvalCriterionValue->Data()
        };

        auto updated = m_learner->Update(gradients, info, m_totalSamplesSeen);
        m_prevMinibatchNumSamples = info.numberOfSamples;
        return updated;
    }

    void Trainer::ExecuteForwardBackward(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice, std::unordered_map<Variable, ValuePtr>& parameterGradients)
    {
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedLossFunction, nullptr }, { m_trainingSampleCountVar, nullptr } };
        if (m_aggregatedEvaluationFunction)
            outputs.insert({ m_aggregatedEvaluationFunction, nullptr });

        outputs.insert(outputsToFetch.begin(), outputsToFetch.end());

        auto backPropSate = m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice, { m_aggregatedLossFunction });
        m_prevMinibatchAggregateTrainingLossValue = outputs[m_aggregatedLossFunction];
        if (m_aggregatedEvaluationFunction)
            m_prevMinibatchAggregateEvalCriterionValue = outputs[m_aggregatedEvaluationFunction];

        for (auto outputToFetch : outputsToFetch)
        {
            if (outputToFetch.second == nullptr)
                outputsToFetch[outputToFetch.first] = outputs[outputToFetch.first];
        }

        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(m_aggregatedLossFunction->Output().GetDataType(), m_prevMinibatchAggregateTrainingLossValue->Shape(), computeDevice), outputs.at(m_aggregatedLossFunction)->Mask());
        if (m_aggregatedLossFunction->Output().GetDataType() == DataType::Float)
            rootGradientValue->Data()->SetValue(1.0f);
        else
            rootGradientValue->Data()->SetValue(1.0);

        auto modelParameters = m_combinedTrainingFunction->Parameters();
        for (const auto& parameter : modelParameters)
        {
            parameterGradients[parameter] = nullptr;
        }

        // TODO: Why Backward signature does not take Parameter instead of Variable for gradients?
        m_combinedTrainingFunction->Backward(backPropSate, { { m_aggregatedLossFunction, rootGradientValue } }, parameterGradients);
        m_prevMinibatchNumSamples = GetSampleCount(m_trainingSampleCountVar, outputs[m_trainingSampleCountVar]);
    }

    static std::wstring GetTrainerStateCheckpointFilePath(const std::wstring& modelFilePath)
    {
        const wchar_t* checkpointExt = L".ckp";
        return modelFilePath + checkpointExt;
    }

    void Trainer::SaveCheckpoint(const std::wstring& modelFilePath, bool usinglegacyModelFormat, Dictionary externalState)
    {
        // TODO: Need to pass currect state of the minibatch source in here.
        Dictionary learnerState = m_learner->CreateCheckpoint();

        if (!m_learner->IsDistributed())
            return Save(modelFilePath, usinglegacyModelFormat, learnerState, externalState);

        // Collect distrbuted external state.
        DistributedCommunicatorPtr communicator = MPICommunicator();
        communicator->Barrier();

        std::vector<DictionaryPtr> remoteState;
        communicator->Gather(externalState, remoteState, communicator->Workers());

        Dictionary aggregatedState;
        for (const auto& w : communicator->Workers())
        {
            aggregatedState[std::to_wstring(w.m_globalRank)] = *remoteState[w.m_globalRank];
        }

        if (communicator->CurrentWorker().IsMain())
            Save(modelFilePath, usinglegacyModelFormat, learnerState, aggregatedState);

        // all workers need to sync up after saving model to avoid read-after-write hazard
        // i.e. one worker is in the middle of write while another tries to read
        communicator->Barrier();
    }

    void Trainer::Save(const std::wstring& modelFilePath, bool usinglegacyModelFormat, const Dictionary& learnerState, const Dictionary& externalState)
    {
        Dictionary state;
        state[learnersPropertyName] = learnerState;
        state[externalStatePropertyName] = externalState;
        state[totalSeenSamplesPropertyName] = m_totalSamplesSeen;

        m_combinedTrainingFunction->SaveModel(modelFilePath, usinglegacyModelFormat);
        std::wstring trainerStateCheckpointFilePath = GetTrainerStateCheckpointFilePath(modelFilePath);
        auto ckpStream = GetFstream(trainerStateCheckpointFilePath, false);
        *ckpStream << state;
        ckpStream->flush();
    }

    Dictionary Trainer::RestoreFromCheckpoint(const std::wstring& modelFilePath)
    {
        // Restore the model's parameters
        m_combinedTrainingFunction->RestoreModel(modelFilePath);

        std::wstring trainerStateCheckpointFilePath = GetTrainerStateCheckpointFilePath(modelFilePath);
        auto ckpStream = GetFstream(trainerStateCheckpointFilePath, true);
        Dictionary checkpoint;
        *ckpStream >> checkpoint;

        m_totalSamplesSeen = checkpoint[totalSeenSamplesPropertyName].Value<size_t>();
        const DictionaryValue& learnerState = checkpoint[learnersPropertyName];
        m_learner->RestoreFromCheckpoint(learnerState.Value<Dictionary>());
        auto externalState = checkpoint[externalStatePropertyName].Value<Dictionary>();

        if (!m_learner->IsDistributed())
            return externalState;

        DistributedCommunicatorPtr communicator = MPICommunicator();
        communicator->Barrier();

        auto key = std::to_wstring(communicator->CurrentWorker().m_globalRank);

        if (externalState.Contains(key))
            return externalState[key].Value<Dictionary>();
        else
            return externalState[std::to_wstring(0)].Value<Dictionary>();
    }

    double Trainer::PreviousMinibatchLossAverage() const
    {
        return (GetScalarValue(m_prevMinibatchAggregateTrainingLossValue) / m_prevMinibatchNumSamples);
    }

    double Trainer::PreviousMinibatchEvaluationAverage() const
    {
        if (!m_evaluationFunction)
            InvalidArgument("Trainer::PreviousMinibatchEvaluationAverage: Cannot get evaluation criterion value when no evaluation function was specified during 'this' trainer's construction");

        return (GetScalarValue(m_prevMinibatchAggregateEvalCriterionValue) / m_prevMinibatchNumSamples);
    }
}
