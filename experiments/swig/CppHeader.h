#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>

#define CNTK_API //_declspec(dllimport)

typedef int SparseIndexType;

///// 
///// The struct defines layout of samples of a sparse sequence (csc)
///// Each sample is represented by 1-dimentional array. N-dimential tensor needs to be flatted into the 1-dimentional array first.
/////
///// This struct can also be represented by std::vector<std::tuple<std::vector<ElementType>, std::vector<SparseIndexType>, std::vector<SparseIndexType>>>.
///// However, the struct is easier to understand and use.
//template <typename struct SparseSequenceData
//ElementType>
//{
//    // Todo: use pointer to vector should be more efficient, but need to check how to work with C#.
//    SparseSequenceData(std::vector<ElementType> data, std::vector<SparseIndexType> indices, std::vector<SparseIndexType> nnzCounts) : m_data(data), m_indices(indices), m_nnzCounts(nnzCounts) {}
//    std::vector<ElementType> m_data;            // All non-zero data in the sequence.
//    std::vector<SparseIndexType> m_indices;     // The index of each non-zero data in m_data.
//    std::vector<SparseIndexType> m_nnzCounts;   // The number of non-zero data of each sample.
//};


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

enum class DataType : unsigned int
{
    Unknown = 0,
    Float = 1,
    Double = 2,
};

///
/// Denotes a multi-dimensional rectangular shape.
///
class NDShape final
{
   /* friend bool operator==(const NDShape& first, const NDShape& second);
    friend class PrimitiveFunction;*/

public:

    ///
    /// A placeholder value to use for an axis whose dimension is unknown and is to be inferred by the system.
    ///
    static const size_t InferredDimension = (size_t)-1;

    ///
    /// A placeholder shape to use to denote an unknown shape
    ///
    CNTK_API static const NDShape Unknown;

public:
    ///
    /// Construct a NDShape with 0 axes, which denotes a scalar.
    ///
    NDShape() {}

    ///
    /// Returns the total size of the rectangular shape that 'this' shape denotes.
    ///
    size_t TotalSize() const
    {
        return 0;
    }

    ///
    /// Returns the rank of 'this' shape.
    ///
    size_t Rank() const {
        return m_shapeDims.size();
    }

private:
    std::vector<size_t> m_shapeDims;
};

class Value;
typedef std::shared_ptr<Value> ValuePtr;

class Value : public std::enable_shared_from_this<Value>
{
public:
    ///
    /// Create a new Value object containing a collection of variable length sequences.
    /// The created Value object contains a copy of the specified 'sequences' data.
    ///
   /* template <typename ElementType>*/
    CNTK_API static ValuePtr Create(const NDShape& sampleShape, const std::vector<float>& sequences, const DeviceDescriptor& device, bool readOnly = false);
    CNTK_API static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<float>>& sequences, const DeviceDescriptor& device, bool readOnly = false);

    ///
    /// Create a new Value object containing a collection of variable length sequences of one hot vectors
    /// The created Value object contains a copy of the specified 'sequences' data.
    ///
    CNTK_API static ValuePtr Create(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly = false);


    //template <typename ElementType>
    //CNTK_API static ValuePtr Create(const NDShape& sampleShape, const std::vector<SparseSequenceData<ElementType>>& sparseSequences, const DeviceDescriptor& device, bool readOnly = false);


    ///
    /// Destruct 'this' Value object.
    ///
    virtual ~Value();
};


class Function;
typedef std::shared_ptr<Function> FunctionPtr;

class BackPropState : public std::enable_shared_from_this<BackPropState>
{
public:
    ///
    /// Returns the Function that 'this' BackPropState belongs to
    ///
    FunctionPtr Function() const {
        return m_function;
    }
    virtual ~BackPropState() {}

protected:
    BackPropState(const FunctionPtr& function) : m_function(function) {}

protected:
    FunctionPtr m_function;
};
typedef std::shared_ptr<BackPropState> BackPropStatePtr;

class Function : public std::enable_shared_from_this<Function>
{
public:
    /*virtual BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                     std::unordered_map<Variable, ValuePtr>& outputs,
                                     const DeviceDescriptor& computeDevice = DeviceDescriptor::CPUOnly,
                                     const std::unordered_set<Variable>& outputsToRetainBackwardStateFor = {}) = 0;*/
    virtual BackPropStatePtr Forward(const std::unordered_map<std::wstring, ValuePtr>& arguments,
                                     std::unordered_map<std::wstring, ValuePtr>& outputs,
                                     const DeviceDescriptor& computeDevice = DeviceDescriptor::CPUOnly,
                                     const std::unordered_set<std::wstring>& outputsToRetainBackwardStateFor = {}) = 0;

    ///
    /// Load a function from a model file
    ///
    CNTK_API static FunctionPtr LoadModel(const DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice = DeviceDescriptor::CPUOnly);

};

class CompositeFunciton : public Function
{
public:
    std::wstring name;
    int getName();
    virtual BackPropStatePtr Forward(const std::unordered_map<std::wstring, ValuePtr>& arguments,
                                     std::unordered_map<std::wstring, ValuePtr>& outputs,
                                     const DeviceDescriptor& computeDevice = DeviceDescriptor::CPUOnly,
                                     const std::unordered_set<std::wstring>& outputsToRetainBackwardStateFor = {}) override;
};