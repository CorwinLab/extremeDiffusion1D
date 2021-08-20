#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>

#include "pybind11_numpy_scalar.h"
#include "diffusionCDF.hpp"

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


DiffusionCDF::DiffusionCDF(const double _beta, const unsigned long int _tMax)
{
  beta = _beta;
  t = 0;
  tMax = _tMax;
  zB.resize(tMax+1); // Initialize so that zB(n=0, t=0) = 1
  zB[0] = 1;

  if (_beta != 0) {
    boost::random::beta_distribution<>::param_type params(_beta, _beta);
    betaParams = params;
  }

  std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
  dis.param(unifParams);
  gen.seed(rd());
}

double DiffusionCDF::generateBeta()
{
  // If beta = 0 return either 0 or 1
  if (beta == 0.0) {
    return round(dis(gen));
  }
  // If beta = 1 use random uniform distribution
  else if (beta == 1) {
    return dis(gen);
  }
  // If beta = inf return 0.5
  else if (isinf(beta)) {
    return 0.5;
  }
  else {
    return beta_dist(gen, betaParams);
  }
}

void DiffusionCDF::iterateTimeStep()
{
  std::vector<RealType> zB_next(tMax+1);
  for (unsigned long int n = 0; n <= t+1; n++){
    if (n == 0){
      zB_next[n] = 1; // Need zB(n=0, t) = 1
    }
    else if (n == t+1){
      RealType beta = RealType(generateBeta());
      zB_next[n] = beta * zB[n-1];
    }
    else{
      RealType beta = RealType(generateBeta());
      zB_next[n] = beta * zB[n-1] + (1 - beta) * zB[n];
    }
  }
  zB = zB_next;
  t += 1;
}

unsigned long int DiffusionCDF::findQuantile(RealType quantile)
{
  unsigned long int quantilePosition;
  for (unsigned long int n = t; n >= 0; n--){
    if (zB[n] > 1 / quantile){
      quantilePosition = 2* n + 2 - t;
      break;
    }
  }
  return quantilePosition;
}

std::vector<unsigned long int> DiffusionCDF::findQuantiles(
  std::vector<RealType> quantiles)
{
  // Sort incoming quantiles b/c we need them to be in descending order for
  // algorithm to work
  std::sort(quantiles.begin(), quantiles.end(), std::greater<RealType>());

  // Initialize to have same number of vectors as in Ns
  std::vector<unsigned long int> quantilePositions(quantiles.size());
  unsigned long int quantile_idx = 0;
  for (unsigned long int n = t; n >= 0; n--){
    while(zB[n] > 1 / quantiles[quantile_idx]){
      quantilePositions[quantile_idx] = 2 * n + 2 - t;
      quantile_idx += 1;

      //Break while loop if past last position
      if (quantile_idx == quantiles.size()){
        break;
      }
    }
    // Also need to break for loop if in last position b/c we are done searching
    if (quantile_idx == quantiles.size()){
      break;
    }
  }
  return quantilePositions;
}

PYBIND11_MODULE(diffusionCDF, m)
{
  m.doc() = "Diffusion recurrance relation";
  py::class_<DiffusionCDF>(m, "DiffusionCDF")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getBeta", &DiffusionCDF::getBeta)
      .def("getzB", &DiffusionCDF::getzB)
      .def("gettMax", &DiffusionCDF::gettMax)
      .def("setBetaSeed", &DiffusionCDF::setBetaSeed, py::arg("seed"))
      .def("getTime", &DiffusionCDF::getTime)
      .def("iterateTimeStep", &DiffusionCDF::iterateTimeStep)
      .def("findQuantile", &DiffusionCDF::findQuantile, py::arg("quantile"))
      .def("findQuantiles", &DiffusionCDF::findQuantiles, py::arg("quantiles"));
}
