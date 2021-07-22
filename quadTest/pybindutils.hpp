#ifndef PYBINDUTILS_HPP_

#include <vector>
#include <tuple>
#include <functional>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "diffusion.hpp"
#include <boost/multiprecision/float128.hpp>

namespace py = pybind11;

typedef boost::multiprecision::float128 RealType;

//Quad input scalars are handled as singleton arrays
typedef struct quadArrayIn {unsigned char* data; long size;} quadArrayIn;
using intIn = long;
typedef struct intArrayIn {long* data; long size;} intArrayIn;
//Quad output scalars are handled as singleton arrays
using quadArrayOut = std::vector<unsigned char>;
using intOut = long;
using intArrayOut = std::vector<long>;

static const long quadSize = 16;

class Diffusion{
public:
  Diffusion(){};
  ~Diffusion(){};
  quadArrayOut takesQuad(quadArrayIn x){
		RealType* y = RealType*(&x);
    
  }
};

// Boilerplate for wrapper functions ------------------------------------------
/*
   Fundamentally, the wrapper functions all have the form
   f(args...) --> parseOut(f(parseIn(args...)))
   However, a nontrivial amount of metaprogramming is required to implement this
   pattern in a type safe manner. The following code is evaluated at compile time
   and allows the parse functions to automatically interpret the signature of
   a C++ function and allow it to interact with python types.
 */

// Implementation of index_sequence_for from C++14 stdlib
template<typename T, T ... Ints>
struct integer_sequence
{
	typedef T value_type;
	static constexpr std::size_t size() {
		return sizeof ... (Ints);
	}
};

template<std::size_t ... Ints>
using index_sequence = integer_sequence<std::size_t, Ints ...>;

template<typename T, std::size_t N, T ... Is>
struct make_integer_sequence : make_integer_sequence<T, N-1, N-1, Is ...> {};

template<typename T, T ... Is>
struct make_integer_sequence<T, 0, Is ...> : integer_sequence<T, Is ...> {};

template<std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

template<typename ... T>
using index_sequence_for = make_index_sequence<sizeof ... (T)>;

// Get the initial input types associated with each data type
template <class T> struct getRawInType; // default case unused
template <> struct getRawInType<intIn> {typedef long type;};
template <> struct getRawInType<intArrayIn> {typedef py::array_t<long, py::array::c_style | py::array::forcecast> type;};
template <> struct getRawInType<quadArrayIn> {typedef py::array_t<unsigned char, py::array::c_style | py::array::forcecast> type;};

// Get the final input types associated with each data type
template <class T> struct getRawOutType; // default case unused
template <> struct getRawOutType<intOut> {typedef long type;};
template <> struct getRawOutType<intArrayOut> {typedef py::array_t<long> type;};
template <> struct getRawOutType<quadArrayOut> {typedef py::array_t<unsigned char> type;};

// Composes std::function types from bare templates
template<class ... T> struct makeFunctionType;
template<typename outType, typename ... inTypes>
struct makeFunctionType<outType, inTypes ...>
{
	typedef std::function<outType(inTypes ...)> type;
};

// parseIn functions convert native python objects to C++ input types.
// The default case is set up to throuw a compile time error if an
// unimplemented template is used.
// Generic pointers are needed as input to parseIn prevent input type matching
// from interfering with template matching
// These functions are implementation specific, and should be refactored should
// any change be made to the data types.
template<typename retType>
retType parseIn(void* ptr)
{
	// This method should not be implemented. Attempting to call a case that has not been specialized
	// will result in a compile time error
	return retType::unimplemented_function();
}
// Each acceptable input type is specialized explicitly
template <>
intIn parseIn<intIn> (void* data)
{
	return *((long*) data);
}
template <>
intArrayIn parseIn<intArrayIn> (void* data)
{
	auto proxy = ((py::array_t<long>*)data)->mutable_unchecked<>();
	return {proxy.mutable_data(), proxy.size()};
}
template <>
quadArrayIn parseIn<quadArrayIn> (void* data)
{
	auto proxy = ((py::array_t<unsigned char>*)data)->mutable_unchecked<>();
	return {proxy.mutable_data(), proxy.size() / quadSize};
}

// parseOut functions convert C++ output types to python objects
// Implemented identically to parseIn
template<typename argType>
inline typename getRawOutType<argType>::type parseOut(argType data)
{
	return argType::unimplemented_function();
}
template<>
inline typename getRawOutType<intOut>::type parseOut<intOut> (intOut data)
{
	return data;
}
template <>
inline typename getRawOutType<intArrayOut>::type parseOut<intArrayOut> (intArrayOut data)
{
	auto v = new std::vector<long>(data);
	auto capsule = py::capsule(v, [](void *v) {
		delete reinterpret_cast<std::vector<long>*>(v);
	});
	return py::array(v->size(), v->data(), capsule);
}
template <>
inline typename getRawOutType<quadArrayOut>::type parseOut<quadArrayOut> (quadArrayOut data)
{
	auto v = new std::vector<unsigned char>(data);
	auto capsule = py::capsule(v, [](void *v) {
		delete reinterpret_cast<std::vector<unsigned char>*>(v);
	});
	return py::array_t<unsigned char>(v->size(), v->data(), capsule);
}

template <typename ... types, size_t ... I>
py::tuple parseTupleHelper(std::tuple<types ...> t, index_sequence<I ...> seq)
{
	return py::make_tuple(parseOut<types>(std::get<I>(t)) ...);
}
template <typename ... types>
py::tuple parseTuple(std::tuple<types ...> t)
{
	return parseTupleHelper<types ...>(t, index_sequence_for<types ...>());
}

// Wrapper functions ----------------------------------------------------------

// For wrapping Diffusion methods with no outputs
template <typename ... inTypes>
typename makeFunctionType<void, Diffusion*, typename getRawInType<inTypes>::type ... >::type
wrapVoidDiffusionMethod(void (Diffusion::*fptr)(inTypes ...))
{
	typename makeFunctionType<void, Diffusion*, typename getRawInType<inTypes>::type ... >::type
	        newFunc = [fptr](Diffusion* p, typename getRawInType<inTypes>::type ... xs)
			  {
				  (p->*fptr)(parseIn<inTypes>(&xs) ...);
			  };
	return newFunc;
}

// For wrapping functions with a single output
template <typename outType, typename ... inTypes>
typename makeFunctionType<typename getRawOutType<outType>::type, Diffusion*, typename getRawInType<inTypes>::type ... >::type
wrapUnaryDiffusionMethod(outType (Diffusion::*fptr)(inTypes ...))
{
	typename makeFunctionType<typename getRawOutType<outType>::type, Diffusion*, typename getRawInType<inTypes>::type ... >::type
	        newFunc = [fptr](Diffusion* p, typename getRawInType<inTypes>::type ... xs)
			  {
				  return parseOut<outType>((p->*fptr)(parseIn<inTypes>(&xs) ...));
			  };
	return newFunc;
}

// For wrapping functions which a tuple of data objects as output
template <typename ... inTypes, typename ... outTypes>
typename makeFunctionType<py::tuple, Diffusion*, typename getRawInType<inTypes>::type ... >::type
wrapTupleDiffusionMethod(std::tuple<outTypes ...> (Diffusion::*fptr)(inTypes ...))
{
	typename makeFunctionType<py::tuple, Diffusion*, typename getRawInType<inTypes>::type ... >::type
	        newFunc = [fptr](Diffusion* p, typename getRawInType<inTypes>::type ... xs)
			  {
				  return parseTuple((p->*fptr)(parseIn<inTypes>(&xs) ...));
			  };
	return newFunc;
}

#endif
