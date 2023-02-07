#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#include "pybind11_numpy_scalar.h"
#include "diffusionND.hpp"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

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

RandDistribution::RandDistribution(const std::vector<double> _alpha){
  alpha = _alpha;
  gsl_rng_set(gen,(unsigned)time(NULL));
  K = 4;
  theta.resize(4);
}

std::vector<RealType> RandDistribution::getRandomNumbers(){
  if (isinf(alpha[0])){
    return {(RealType)0.25, (RealType)0.25, (RealType)0.25, (RealType)0.25};
  }
  else{
    gsl_ran_dirichlet(gen, K, &alpha[0], &theta[0]);
    /* Cast double precision to quad precision by dividing by sum of double precision rand numbers*/
    std::vector<RealType> biases(theta.begin(), theta.end());
    RealType sum = 0;
    for (unsigned int i=0; i < biases.size(); i++){
      sum += biases[i];
    }
    for (unsigned int i=0; i<biases.size(); i++){
      biases[i] /= sum;
    }
    return biases;
  }
}

DiffusionND::DiffusionND(const std::vector<double> _alpha, const unsigned long int _tMax, int _L) : RandDistribution(_alpha)
{
  tMax = _tMax;
  L = _L;
  absorbedProb = 0;

  std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
  dis.param(unifParams);
  gen.seed(rd());

  std::vector<double> biases;

  CDF.resize(tMax+1, std::vector<RealType>(tMax+1));
  CDF[0][0] = 1;

  t = 0;
}

void DiffusionND::iterateTimestep(){
  std::cout << std::setprecision(50);
  biases = getRandomNumbers();
  std::vector<std::vector<RealType> > CDF_new( CDF.size(), std::vector<RealType> (CDF[0].size()));
  RealType cdf_new_sum = 0;

  for (unsigned long int i = 0; i < CDF.size()-1; i++){ // i is the columns
    for (unsigned long int j=0; j < CDF[i].size()-1; j++){ // j is the row    
      RealType currentPos = CDF[i][j];
      if (currentPos == 0){
        continue;
      }

      int yval = j-i;
      
      CDF_new[i][j] += currentPos * (RealType)biases[0];
      CDF_new[i][j+1] += currentPos * (RealType)biases[1];
      CDF_new[i+1][j+1] += currentPos * (RealType)biases[3];
      
      cdf_new_sum += currentPos * ((RealType)biases[0] + (RealType)biases[1] + (RealType)biases[3]);

      if ((yval-1) <= -L){
        absorbedProb += currentPos * ((RealType)biases[2]);
      }
      else {
        CDF_new[i+1][j] += currentPos * (RealType)biases[2];
        cdf_new_sum += currentPos * ((RealType)biases[2]);
      }
    }
  }
  /* Ensure we aren't losing/gaining probability */
  if ((cdf_new_sum + absorbedProb) < ((RealType)1.-(RealType)pow(10, -25)) || (cdf_new_sum + absorbedProb) > ((RealType)1.+(RealType)pow(10, -25))){
    std::cout << "CDF total: " << cdf_new_sum + absorbedProb << std::endl;
    throw std::runtime_error("Total probability not within tolerance of 10^-25");
  }
  CDF = CDF_new;
  t += 1;
}

PYBIND11_MODULE(diffusionND, m)
{
  m.doc() = "Diffusion in multiple dimensions";
  py::class_<RandDistribution>(m, "RandDistribution")
      .def(py::init<const std::vector<double>>())
      .def("getRandomNumbers", &RandDistribution::getRandomNumbers);
  py::class_<DiffusionND, RandDistribution>(m, "DiffusionND")
      .def(py::init<const std::vector<double>, const unsigned long int, int>())
      .def("getBiases", &DiffusionND::getBiases)
      .def("getCDF", &DiffusionND::getCDF)
      .def("setCDF", &DiffusionND::setCDF, py::arg("CDF"))
      .def("iterateTimestep", &DiffusionND::iterateTimestep)
      .def("getRandomNumbers", &DiffusionND::getRandomNumbers)
      .def("getTime", &DiffusionND::getTime)
      .def("getAbsorbedProb", &DiffusionND::getAbsorbedProb)
      .def("getL");
}
