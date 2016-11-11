%module CSEval
%{
#include "CppHeader.h"
%}

%include "stl.i"
%include "std_wstring.i"
%include <std_vector.i>
%include <std_map.i>
//%include <std_set.i>
%include <std_pair.i>
%include <windows.i>
%include <attribute.i>
%include <std_shared_ptr.i>


// %template() std::vector<SparseSequenceData<float>>;
// %template() std::vector<SparseSequenceData<float>>;

%template() std::vector<size_t>;
%template() std::vector<bool>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(SizeTVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;

%template() std::vector<CNTK::Variable>;
%template() std::vector<CNTK::Parameter>;
%template() std::vector<CNTK::Constant>;
%template() std::vector<CNTK::Axis>;
%template() std::vector<CNTK::DeviceDescriptor>;
%template() std::vector<CNTK::StreamConfiguration>;
//%template() std::vector<CNTK::DictionaryValue>;
%template() std::vector<std::shared_ptr<CNTK::Function>>;
%template() std::vector<std::shared_ptr<CNTK::Learner>>;
%template() std::pair<size_t, double>;
%template() std::vector<std::pair<size_t, double>>;
%shared_ptr(BackPropState);
%shared_ptr(Function);
%shared_ptr(CompositeFunction);
%shared_ptr(Value);
%shared_ptr(NDShape);
%shared_ptr(std::vector<float>);

// %feature ("noabstract") Function

%include "CppHeader.h" 

//%template(SparseSequenceDataFloat) SparseSequenceData<float>;
//%template(SparseSequenceDataDouble) SparseSequenceData<double>;

