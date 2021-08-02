#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <iostream>

#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

// Boilerplate to get PyBind11 to cast to a npquad precision.
namespace pybind11 {
namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 256;

// Kinda following:
// https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <> struct npy_format_descriptor<RealType> {
  static constexpr auto name = _("RealType");
  static pybind11::dtype dtype()
  {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <> struct type_caster<RealType> : npy_scalar_caster<RealType> {
  static constexpr auto name = _("RealType");
};

} // namespace detail
} // namespace pybind11

std::vector<std::vector<RealType> > makeRec(unsigned long int tmax){
  std::vector<std::vector<RealType> > zB(tmax); // Number of columns set to tmax
  for (unsigned long int n = 0; n < zB.size(); n++){
    zB.at(n) = std::vector<RealType>(tmax); // Number of rows set to tmax
    for (unsigned long int t = n; t < tmax; t++){
      double bias = 0.5; // Some random beta distributed variable
      std::cout << "At: (" << n << "," << t << ")" << std::endl;
      if (n == t){
        zB.at(n).at(t) = 1;
      }
      else if (n==0){
        zB.at(n).at(t) = zB.at(n).at(t-1) * bias;
      }
      else{
        zB.at(n).at(t) = zB.at(n).at(t-1) * bias + zB.at(n-1).at(t-1) * (1-bias);
      }
    }
  }
  return zB;
}

std::vector<unsigned long int> findQuintile(
  std::vector<std::vector<RealType> >zB, RealType N)
{
  unsigned long int tMax = zB.size();
  std::vector<unsigned long int> quintile(tMax);
  for (unsigned long int t = 0; t < tMax; t++){
    for (unsigned long int n = 0; n < tMax; t++){
      if (zB[n][t] > 1. / N){
        quintile[t] = t - 2 * (n) + 2;
        break;
      }
    }
  }
  return quintile;
}

PYBIND11_MODULE(recurrance, m)
{
  m.doc() = "Diffusion recurrance relation";
  m.def("makeRec", &makeRec, py::arg("time"));
  m.def("findQuintile", &findQuintile, py::arg("zB"), py::arg("N"));

}
