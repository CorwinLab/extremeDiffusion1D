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
  tMax = _tMax;

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

DiffusionTimeCDF::DiffusionTimeCDF(const double _beta, const unsigned long int _tMax) : DiffusionCDF(_beta, _tMax)
{
  CDF.resize(tMax + 1);
  CDF[0] = 1;
}

void DiffusionTimeCDF::iterateTimeStep()
{
  std::vector<RealType> CDF_next(tMax+1);
  for (unsigned long int n = 0; n <= t+1; n++){
    if (n == 0){
      CDF_next[n] = 1; // Need CDF(n=0, t) = 1
    }
    else if (n == t+1){
      RealType beta = RealType(generateBeta());
      CDF_next[n] = beta * CDF[n-1];
    }
    else{
      RealType beta = RealType(generateBeta());
      CDF_next[n] = beta * CDF[n-1] + (1 - beta) * CDF[n];
    }
  }
  CDF = CDF_next;
  t += 1;
}

unsigned long int DiffusionTimeCDF::findQuantile(RealType quantile)
{
  unsigned long int quantilePosition;
  for (unsigned long int n = t; n >= 0; n--){
    if (CDF[n] > 1 / quantile){
      quantilePosition = 2* n + 2 - t;
      break;
    }
  }
  return quantilePosition;
}

std::vector<unsigned long int> DiffusionTimeCDF::findQuantiles(
  std::vector<RealType> quantiles)
{
  // Sort incoming quantiles b/c we need them to be in descending order for
  // algorithm to work
  std::sort(quantiles.begin(), quantiles.end(), std::greater<RealType>());

  // Initialize to have same number of vectors as in Ns
  std::vector<unsigned long int> quantilePositions(quantiles.size());
  unsigned long int quantile_idx = 0;
  for (unsigned long int n = t; n >= 0; n--){
    while(CDF[n] > 1 / quantiles[quantile_idx]){
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

DiffusionPositionCDF::DiffusionPositionCDF(const double _beta, const unsigned long int _tMax) : DiffusionCDF(_beta, _tMax)
{
  CDF.resize(tMax+1); // Initialize so that CDF(n=0, t) = 1
  position = 0;
}

void DiffusionPositionCDF::stepPosition()
{ int whatever = 0; }

unsigned long int DiffusionPositionCDF::findQuantile(RealType quantile)
{ std::cout << "Do some shit";
  return 0; }

std::vector<unsigned long int> DiffusionPositionCDF::findQuantiles(std::vector<RealType> quantiles)
{
  std::vector<unsigned long int> quantilePositions(quantiles.size());
  return quantilePositions;
}

PYBIND11_MODULE(diffusionCDF, m)
{
  m.doc() = "Diffusion recurrance relation";
  py::class_<DiffusionCDF>(m, "DiffusionCDF")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getBeta", &DiffusionCDF::getBeta)
      .def("getCDF", &DiffusionCDF::getCDF)
      .def("gettMax", &DiffusionCDF::gettMax)
      .def("setBetaSeed", &DiffusionCDF::setBetaSeed, py::arg("seed"));

  py::class_<DiffusionTimeCDF, DiffusionCDF>(m, "DiffusionTimeCDF")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getTime", &DiffusionTimeCDF::getTime)
      .def("iterateTimeStep", &DiffusionTimeCDF::iterateTimeStep)
      .def("findQuantile", &DiffusionTimeCDF::findQuantile, py::arg("quantile"))
      .def("findQuantiles", &DiffusionTimeCDF::findQuantiles, py::arg("quantiles"));

  py::class_<DiffusionPositionCDF, DiffusionCDF>(m, "DiffusionPositionCDF")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getPosition", &DiffusionPositionCDF::getPosition)
      .def("stepPosition", &DiffusionPositionCDF::stepPosition)
      .def("findQuantile", &DiffusionPositionCDF::findQuantile, py::arg("quantile"))
      .def("findQuantiles", &DiffusionPositionCDF::findQuantiles, py::arg("quantiles"));
}
