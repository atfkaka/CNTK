#include <stdio.h>
#include <vector>
#include <string>
#include "CppHeader.h"

using std::vector;


//template <typename ElementType>
///*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<std::tuple<std::vector<ElementType>, std::vector<SparseIndexType>, std::vector<SparseIndexType>>>& SparseSequences, const DeviceDescriptor& device, bool readOnly /*= false*/)
template <typename ElementType>
/*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<SparseSequenceData<ElementType>>& SparseSequences, const DeviceDescriptor& device, bool readOnly /*= false*/)
{
    // Todo: add implementation.
    return nullptr;
}

template <typename ElementType>
/*static*/ ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly /* = false */)
{
    reutn nullptr;
}

///
/// Create a new Value object containing a collection of variable length sequences of one hot vectors
/// The created Value object contains a copy of the specified 'sequences' data.
///
template <typename ElementType>
/*static*/ ValuePtr Create(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly /*= false*/)
{
    return nullptr;
}

/*virtual*/ BackPropStatePtr CompositeFunciton::Forward(const std::unordered_map<std::wstring, ValuePtr>& arguments,
                                                        std::unordered_map<std::wstring, ValuePtr>& outputs,
                                                        const DeviceDescriptor& computeDevice,
                                                        const std::unordered_set<std::wstring>& outputsToRetainBackwardStateFor)
{
    return nullptr;
}

///
/// Load a model for evaluation
///
/// dataType: Data type: float or double.
/// modelFile: the path of the model to be loaded.
/// computeDevice: the device to run evaluation.
FunctionPtr Function::LoadModel(const DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice)
{
    printf("LoadModel: dataType=%d, modelfile=%ls, computDevice=%d\n", dataType, modelFile.c_str(), computeDevice);
    return nullptr;
}


int main()
{

    return 0;

}