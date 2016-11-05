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
void LoadModel(const DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice)
{
    printf("LoadModel: dataType=%d, modelfile=%ls, computDevice=%d\n", dataType, modelFile.c_str(), computeDevice);
}


int main()
{

    return 0;

}