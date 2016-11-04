#include <stdio.h>
#include <vector>
#include <string>
#include "CppHeader.h"

using std::vector;


    ///
    /// Load a model for evaluation
    ///
    /// dataType: Data type: float or double.
    /// modelFile: the path of the model to be loaded.
    /// computeDevice: the device to run evaluation.
    void Eval::LoadModel(const DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice)
    {
        printf("LoadModel: dataType=%d, modelfile=%ls, computDevice=%d\n", dataType, modelFile.c_str(), computeDevice);
    }

    ///
    /// Get variable according to name
    ///
    /// name: the node name
    /// Return: the variable having the name
    Variable Eval::GetVariableByName(std::wstring name)
    {
        printf("GetVariableByName: name=%ls\n", name.c_str());
        return Variable(name);
    }

    ///
    /// Retrieve output variables of the model.
    ///
    /// outputVariables: a list of output variables of the model. 
    void Eval::GetModelOutputs(std::vector<Variable>& outputVariables)
    {
        Variable var1(L"labels");
        outputVariables.push_back(var1);
        printf("GetModelOutputs: var[0]=%ls\n", outputVariables[0].m_name.c_str());
    }


    ///
    /// Retrieve required input variables based on output variables 
    ///
    /// inputVariables: a list of input variables which are required in order to create output variables.
    void Eval::GetModelInputs(std::vector<Variable>& inputVariables, std::vector<Variable>& outputVariables)
    {
        Variable var1(L"features");
        inputVariables.push_back(var1);
        printf("GetModelInputs: output[0]=%ls, input[0]=%ls\n", outputVariables[0].m_name.c_str(), inputVariables[0].m_name.c_str());
    }

    ///
    /// Evaluate with single input. 
    ///
    /// inputs: map from node name to input data.
    /// outputs - map from node name to output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    void Eval::Evaluate(std::unordered_map<std::wstring, ValueBuffer *>& inputs, std::unordered_map<std::wstring, ValueBuffer *>& outputs)
    {
        float *data = new float[10];
        int i;
        for (i = 0; i < 10; i++)
        {
            data[i] = (float)i;
        }
        std::wstring name = L"labels";
        ValueBuffer* valp = new ValueBuffer();
        valp->m_type = BufferType::DenstInput;
        valp->m_buffer = data;
        outputs.insert({name, valp});

        ValueBuffer* inputValp = inputs[L"features"];

        printf("Evaluate single element: input features matrix type=%d, input value[0]=%f\n", inputValp->m_type, ((float *)(inputValp->m_buffer))[0]);
    }

    ///
    /// Evaluate with sequence input.
    ///
    /// inputs: map from node name to data of sequence input.
    /// outputs - map from node name to vector of output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    void Eval::Evaluate(std::unordered_map<std::wstring, SequenceOfValueBuffer*>& inputs, std::unordered_map<std::wstring, SequenceOfValueBuffer *>& outputs)
    {

    }

    ///
    /// Evaluate with batch input of sequence.
    ///
    /// inputs: map from node name to data of batch input.
    /// outputs - map from node name to vector of output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    void Eval::Evaluate(std::unordered_map<std::wstring, BatchOfSequenceOfValueBuffer*>& inputs, std::unordered_map<std::wstring, BatchOfSequenceOfValueBuffer *>& outputs)
    {

    }


int main()
{
    Eval myEval;
    vector<Variable> outputVarList;
    vector<Variable> inputVarList;
    myEval.LoadModel(DataType::Float, L"mypath", DeviceDescriptor::AUTO);
    myEval.GetModelOutputs(outputVarList);
    printf("Eval: outputVarList[0]=%ls\n", outputVarList[0].m_name.c_str());
    std::wstring outputName = outputVarList[0].m_name;
    myEval.GetModelInputs(inputVarList, outputVarList);
    printf("Eval: inputVarList[0]=%ls\n", inputVarList[0].m_name.c_str());
    
    std::unordered_map<std::wstring, ValueBuffer*> inputs;
    std::unordered_map<std::wstring, ValueBuffer*> outputs;
    
    float data[5] = {(float)1.1, (float)1.2, (float)1.3, (float)1.4, (float)1.5};
    ValueBuffer inputVal;
    inputVal.m_type = BufferType::DenstInput;
    inputVal.m_buffer = data;
    inputs.insert({L"features", &inputVal});
    myEval.Evaluate(inputs, outputs);
    float *p = (float *)(outputs[outputName]->m_buffer);
    printf("Eval: output: name=%ls, val=", outputName.c_str());
    int i;
    for (i = 0; i < 10; i++)
    {
        printf("%f ", p[i]);
    }
    printf("\n");

    return 0;

}