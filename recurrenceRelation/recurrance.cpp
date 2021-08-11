#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>

#include "pybind11_numpy_scalar.h"
#include "recurrance.hpp"

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


Recurrance::Recurrance(const double _beta, const unsigned long int _tMax)
{
  beta = _beta;
  t = 0;
  tMax = _tMax;
  zB.resize(tMax+1); // Initialize so that zB(n=0, t=0) = 1
  zB[0] = 1.0;

  if (_beta != 0) {
    boost::random::beta_distribution<>::param_type params(_beta, _beta);
    betaParams = params;
  }

  std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
  dis.param(unifParams);
  gen.seed(rd());
}

double Recurrance::generateBeta()
{
  // If beta = 0 return either 0 or 1
  if (beta == 0.0) {
    return round(dis(gen));
  }
  // If beta = 1 use random uniform distribution
  else if (beta == 1.0) {
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

void Recurrance::iterateTimeStep()
{
  std::vector<RealType> zB_next(tMax+1); // Initialize next zb(n, t+1) to size zb(n, t-1) + 1
  for (unsigned long int n = 0; n <= t+1; n++){
    if (n == 0){
      zB_next[n] = 1.0; // Need zB(n=0, t) = 0
    }
    else if (n == t+1){
      double double_beta = generateBeta();
      RealType beta = RealType(double_beta);
      zB_next[n] = beta * zB[n-1];
    }
    else{
      double double_beta = generateBeta();
      RealType beta = RealType(double_beta);
      // std::cout << "n=" << n << ", t=" << t+1 << std::endl;
      // std::cout << "zB(n-1, t-1)=" << zB[n-1] << ", zB(n, t-1)=" <<  zB[n] << "\n" << std::endl;
      zB_next[n] = beta * zB[n-1] + (1.0 - beta) * zB[n];
    }
  }
  zB = zB_next;
  t += 1;
}

unsigned long int Recurrance::findQuintile(RealType N)
{
  unsigned long int quintile;
  for (unsigned long int n = t; n >= 0; n--){
    if (zB[n] > 1.0 / N){
      n = t - n;
      quintile = t - 2 * n + 2;
      break;
    }
  }
  return quintile;
}

std::vector<unsigned long int> Recurrance::findQuintiles(
  std::vector<RealType> Ns)
{
  // Sort incoming Ns b/c we need them to be in decreasing order for algorithm
  // to work
  std::sort(Ns.begin(), Ns.end(), std::greater<RealType>());

  // Initialize to have same number of vectors as in Ns
  std::vector<unsigned long int> quintiles(Ns.size());
  unsigned long int Npos = 0;
  for (unsigned long int n = t; n >= 0; n--){
    while(zB[n] > 1.0 / Ns[Npos]){
      unsigned long int nval = t - n;
      quintiles[Npos] = t - 2 * nval + 2;
      Npos += 1;

      //Break while loop if past last position
      if (Npos == Ns.size()){
        break;
      }
    }
    // Also need to break for loop if in last position b/c we are done searching
    if (Npos == Ns.size()){
      break;
    }
  }
  return quintiles;
}

PYBIND11_MODULE(recurrance, m)
{
  m.doc() = "Diffusion recurrance relation";
  py::class_<Recurrance>(m, "Recurrance")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getBeta", &Recurrance::getBeta)
      .def("getzB", &Recurrance::getzB)
      .def("gettMax", &Recurrance::gettMax)
      .def("setBetaSeed", &Recurrance::setBetaSeed, py::arg("seed"))
      .def("getTime", &Recurrance::getTime)
      .def("iterateTimeStep", &Recurrance::iterateTimeStep)
      .def("findQuintile", &Recurrance::findQuintile, py::arg("N"))
      .def("findQuintiles", &Recurrance::findQuintiles, py::arg("Ns"));
}
