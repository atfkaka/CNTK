#pragma once

#include <vector>
#include <unordered_map>
#include <string>

// just a placeholder.
class Variable
{
public:
    Variable(std::wstring name) : m_name(name) {}

    std::wstring m_name;

};

// just a placeholder.
enum class DeviceDescriptor : int
{
    CPUOnly = -1,
    AUTO = 0,
    GPU = 1
};

// just a placeholder
enum class DataType : unsigned int
{
    Unknown = 0,
    Float = 1,
    Double = 2
};

enum BufferType
{
    DenstInput = 1,
    SparseInput
};


/// 
/// Represent input data of single element which could be dense or sparse.
///
struct ValueBuffer
{
    ValueBuffer() {}

    // Dense or sparse input.
    enum BufferType m_type;

    // data buffer, both for sparse and dense.
    void* m_buffer;

    // 
    // Only for sparse input
    // every element in buffer, an entry in this array gives its position.
    // For every vector the entries must be ascending.
    //
    std::vector<int> m_indices;

    //
    // Only for sparse input.
    // Contains row+1 indices into the buffer. The first entry
    // is always 0. The last entry points after the last element.
    // See http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc
    //
    std::vector<int> m_colIndicies;
};

/// 
/// Represent a sequence of input elements.
///
using SequenceOfValueBuffer = std::vector<ValueBuffer>;

///
/// Represent batch input of sequences of input elements.
///
using BatchOfSequenceOfValueBuffer = std::vector<SequenceOfValueBuffer>;

// Not sure whether we should have a separate class Eval or distributed the member functions into existing V2 API?
class IEvalModel
{
public:
    ///
    /// Load a model for evaluation
    ///
    /// dataType: Data type: float or double.
    /// modelFile: the path of the model to be loaded.
    /// computeDevice: the device to run evaluation.
    virtual void LoadModel(const DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice) = 0;

    ///
    /// Get variable according to name
    ///
    /// name: the node name
    /// Return: the variable having the name
    virtual Variable GetVariableByName(std::wstring name) = 0;

    ///
    /// Retrieve output variables of the model.
    ///
    /// outputVariables: a list of output variables of the model. 
    virtual void GetModelOutputs(std::vector<Variable>& outputVariables) = 0;

    ///
    /// Retrieve required input variables based on output variables 
    ///
    /// inputVariables: a list of input variables which are required in order to create output variables.
    virtual void GetModelInputs(std::vector<Variable>& inputVariables, std::vector<Variable>& outputVariables) = 0;

    ///
    /// Evaluate with single input. 
    ///
    /// inputs: map from node name to input data.
    /// outputs - map from node name to output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    virtual void Evaluate(std::unordered_map<std::wstring, ValueBuffer *>& inputs, std::unordered_map<std::wstring, ValueBuffer *>& outputs) = 0;

    ///
    /// Evaluate with sequence input.
    ///
    /// inputs: map from node name to data of sequence input.
    /// outputs - map from node name to vector of output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    virtual void Evaluate(std::unordered_map<std::wstring, SequenceOfValueBuffer*>& inputs, std::unordered_map<std::wstring, SequenceOfValueBuffer *>& outputs) = 0;

    ///
    /// Evaluate with batch input of sequence.
    ///
    /// inputs: map from node name to data of batch input.
    /// outputs - map from node name to vector of output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    virtual void Evaluate(std::unordered_map<std::wstring, BatchOfSequenceOfValueBuffer*>& inputs, std::unordered_map<std::wstring, BatchOfSequenceOfValueBuffer *>& outputs) = 0;
};


class Eval : public IEvalModel 
{
public:
    ///
    /// Load a model for evaluation
    ///
    /// dataType: Data type: float or double.
    /// modelFile: the path of the model to be loaded.
    /// computeDevice: the device to run evaluation.
    virtual void LoadModel(const DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice) override;

    ///
    /// Get variable according to name
    ///
    /// name: the node name
    /// Return: the variable having the name
    virtual Variable GetVariableByName(std::wstring name) override;

    ///
    /// Retrieve output variables of the model.
    ///
    /// outputVariables: a list of output variables of the model. 
    virtual void GetModelOutputs(std::vector<Variable>& outputVariables) override;

    ///
    /// Retrieve required input variables based on output variables 
    ///
    /// inputVariables: a list of input variables which are required in order to create output variables.
    virtual void GetModelInputs(std::vector<Variable>& inputVariables, std::vector<Variable>& outputVariables) override;

    ///
    /// Evaluate with single input. 
    ///
    /// inputs: map from node name to input data.
    /// outputs - map from node name to output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    virtual void Evaluate(std::unordered_map<std::wstring, ValueBuffer *>& inputs, std::unordered_map<std::wstring, ValueBuffer *>& outputs) override;

    ///
    /// Evaluate with sequence input.
    ///
    /// inputs: map from node name to data of sequence input.
    /// outputs - map from node name to vector of output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    virtual void Evaluate(std::unordered_map<std::wstring, SequenceOfValueBuffer*>& inputs, std::unordered_map<std::wstring, SequenceOfValueBuffer *>& outputs) override;

    ///
    /// Evaluate with batch input of sequence.
    ///
    /// inputs: map from node name to data of batch input.
    /// outputs - map from node name to vector of output buffer. The storage of output buffer can be pre-allocated by the caller or 
    /// if by the implementation if the buffer is passed with null. 
    virtual void Evaluate(std::unordered_map<std::wstring, BatchOfSequenceOfValueBuffer*>& inputs, std::unordered_map<std::wstring, BatchOfSequenceOfValueBuffer *>& outputs) override;
};
