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


%template() std::vector<SparseSequenceData<float>>;
%template() std::vector<SparseSequenceData<float>>;

%template() std::vector<size_t>;
%template() std::vector<bool>;
%template() std::vector<double>;
%template() std::vector<float>;
%template() std::vector<std::vector<size_t>>;
%template() std::vector<std::vector<float>>;
%template() std::vector<std::vector<double>>;

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
%shared_ptr(BackPropStatePtr);
%shared_ptr(FunctionPtr);
%shared_ptr(ValuePtr);

// %feature ("noabstract") Function

%include "CppHeader.h" 

%template(SparseSequenceDataFloat) SparseSequenceData<float>;
%template(SparseSequenceDataDouble) SparseSequenceData<double>;

%shared_ptr(BackPropStatePtr);
%shared_ptr(FunctionPtr);
%shared_ptr(ValuePtr);


