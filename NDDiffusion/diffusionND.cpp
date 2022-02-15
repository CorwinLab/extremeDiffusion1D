#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>

#include "pybind11_numpy_scalar.h"
#include "diffusionND.hpp"
#define _USE_MATH_DEFINES

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

DiffusionND::DiffusionND(const double _beta, const unsigned long int _tMax)
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

  biases = {0.25, 0.25, 0.25, 0.25};

  CDF.resize(tMax+1, std::vector<RealType>(tMax+1));
  CDF[0][0] = 1;

  t = 0;
  maxDist.resize(tMax+1);
  maxTheta.resize(tMax+1);
}

void DiffusionND::iterateTimestep(){
  std::vector<std::vector<RealType> > CDF_new( CDF.size(), std::vector<RealType> (CDF[0].size()));
  double maxDistance = 0;
  double theta = 0;
  double distance;
  double ratio;
  for (unsigned long int i = 0; i < CDF.size()-1; i++){ // i is the columns
    for (unsigned long int j=0; j < CDF[i].size()-1; j++){ // j is the row
      RealType currentPos = CDF[i][j];
      if (currentPos == 0){
        continue;
      }
      CDF_new[i][j] += currentPos * biases[0];
      CDF_new[i][j+1] += currentPos * biases[1];
      CDF_new[i+1][j] += currentPos * biases[2];
      CDF_new[i+1][j+1] += currentPos * biases[3];
      float row = i;
      float col = j;
      float i_pos = row - t / 2;
      float j_pos = col - t / 2;
      distance = sqrt((pow(i_pos, 2) + pow(j_pos, 2)));
      if (distance > maxDistance){
        maxDistance = distance;
        ratio = i_pos / j_pos;
        theta = atan(ratio) * 180 / M_PI;
      }
    }
  }
  maxDist[t] = maxDistance;
  maxTheta[t] = theta;
  CDF = CDF_new;
  t += 1;
}

PYBIND11_MODULE(diffusionND, m)
{
  m.doc() = "Diffusion in multiple dimensions";
  py::class_<DiffusionND>(m, "DiffusionND")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getBiases", &DiffusionND::getBiases)
      .def("getCDF", &DiffusionND::getCDF)
      .def("setCDF", &DiffusionND::setCDF, py::arg("CDF"))
      .def("iterateTimestep", &DiffusionND::iterateTimestep)
      .def("getDistance", &DiffusionND::getDistance)
      .def("getTheta", &DiffusionND::getTheta);
}
