#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>

#include "half.hpp"
#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");

namespace pybind11 { namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

// Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <>
struct type_caster<float16> : npy_scalar_caster<float16> {
  static constexpr auto name = _("float16");
};

}}  // namespace pybind11::detail

void print_scalar(float16 x){
  std::cout << x << std::endl;
}

PYBIND11_MODULE(halfTest, m) {
  m.def("print_scalar", &print_scalar);
  m.def("make_scalar", []() { return float16{2.}; });
  m.def("make_array", []() {
    py::array_t<float16> x({2});
    x.mutable_at(0) = float16{1.};
    x.mutable_at(1) = float16{10.};
    return x;
  });

}
