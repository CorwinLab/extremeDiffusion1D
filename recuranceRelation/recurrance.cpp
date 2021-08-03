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
  tMax = _tMax;
  zB.resize(tMax);

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

void Recurrance::makeRec()
{
  for (unsigned long int n = 0; n < tMax; n++) {
    zB.at(n) = std::vector<RealType>(tMax); // Number of rows set to tmax
    for (unsigned long int t = n; t < tMax; t++) {
      double doublebias =
          generateBeta(); // Some random beta distributed variable
      RealType bias = RealType(doublebias);
      if (n == t) {
        zB.at(n).at(t) = 1;
      }
      else if (n == 0) {
        zB.at(n).at(t) = zB.at(n).at(t - 1) * bias;
      }
      else {
        zB.at(n).at(t) =
            zB.at(n).at(t - 1) * bias + zB.at(n - 1).at(t - 1) * (1 - bias);
      }
    }
  }
}

std::vector<unsigned long int> Recurrance::findQuintile(RealType N)
{
  std::vector<unsigned long int> quintile(tMax);
  for (unsigned long int t = 0; t < tMax; t++) {
    for (unsigned long int n = 0; n < tMax; n++) {
      if (zB[n][t] > 1. / N) {
        quintile[t] = t - 2 * n + 2;
        break;
      }
    }
  }
  return quintile;
}

std::vector<std::vector<unsigned long int> > Recurrance::findQuintiles(
  std::vector<RealType> Ns)
{
  // Sort incoming Ns b/c we need them to be in decreasing order for algorithm
  // to work
  std::sort(Ns.begin(), Ns.end(), std::greater<RealType>());

  // Initialize to have same number of vectors as in Ns
  std::vector<std::vector<unsigned long int> > quintiles(Ns.size());

  // Initialize all vectors to have size tMax
  for (unsigned long int col = 0; col < Ns.size(); col++){
    quintiles[col] = std::vector<unsigned long int>(tMax);
  }
  for (unsigned long int t=0; t < tMax; t++){
    unsigned int pos = 0;
    for (unsigned long int n = 0; n < tMax; n++){
      // Could have multiple Ns with the same quintiles so need to loop through
      // each one
      while (zB[n][t] > 1. / Ns[pos]){
        quintiles[pos][t] = t - 2 * n + 2;
        pos += 1;

        // Break while loop if past last position
        if (pos == Ns.size()){
          break;
        }
      }

      // Also need to break for loop if in last position b/c we are done searching
      if (pos == Ns.size()){
        break;
      }
    }
  }
  return quintiles;
}

PYBIND11_MODULE(recurrance, m)
{
  m.doc() = "Diffusion recurrance relation";
  py::class_<Recurrance>(m, "Recurrance")
      .def(py::init<const double, const unsigned long int>(),
           py::arg("beta"),
           py::arg("tMax"))
      .def("getBeta", &Recurrance::getBeta)
      .def("getzB", &Recurrance::getzB)
      .def("setBetaSeed", &Recurrance::setBetaSeed, py::arg("seed"))
      .def("gettMax", &Recurrance::gettMax)
      .def("makeRec", &Recurrance::makeRec)
      .def("findQuintile", &Recurrance::findQuintile, py::arg("N"))
      .def("findQuintiles", &Recurrance::findQuintiles, py::arg("Ns"));
}
