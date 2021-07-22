#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#include <boost/multiprecision/float128.hpp>
#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

namespace pybind11 { namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 256;

// Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <>
struct npy_format_descriptor<RealType> {
  static constexpr auto name = _("RealType");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <>
struct type_caster<RealType> : npy_scalar_caster<RealType> {
  static constexpr auto name = _("RealType");
};

}}  // namespace pybind11::detail

void print_scalar(RealType x){
  std::cout << x << std::endl;
}

py::array_t<RealType> return_vector(){
  std::vector<RealType> v = {1, 3, 5};
  return py::array(v.size(), v.data());
}

// Input vector can be numpy array or list
//
std::vector<RealType> double_vector(std::vector<RealType> x){

  for (unsigned long int i = 0; i < x.size(); i++){
    x[i] = 2 * x[i];
  }
  return x;
}

PYBIND11_MODULE(quadTest, m) {
  m.def("print_scalar", &print_scalar);
  m.def("make_scalar", []() { return RealType{2.}; });
  m.def("make_array", []() {
    py::array_t<RealType> x({2});
    x.mutable_at(0) = RealType{1.};
    x.mutable_at(1) = RealType{10.};
    return x;
  });
  m.def("return_vector", &return_vector);
  m.def("double_vector", &double_vector);

}
