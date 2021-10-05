#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <math.h>

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

RealType calculateMean(std::vector<long int> xvals, std::vector<RealType> cdf)
{
  RealType mean = 0;
  int H;
  for (unsigned long int i=0; i < cdf.size(); i++){
    if (xvals.at(i) <= 0){
      H = 0;
    }
    else{
      H = 1;
    }
    mean += H - cdf[i];
  }
  return mean;
}

std::pair<std::vector<long int>, std::vector<RealType> > DiffusionTimeCDF::getPDF(RealType nParticles){
  std::vector<long int> xvals(t);
  std::vector<RealType> PDF(t);
  RealType CDF_n, CDF_prev;
  for (unsigned long int n=1; n<=t; n++){
    xvals[n-1] = 2*n + 2 - t;
    CDF_n = exp(-CDF[n] * nParticles);
    CDF_prev = exp(-CDF[n-1] * nParticles);
    PDF[n-1] = CDF_n - CDF_prev;
  }
  return std::make_pair(xvals, PDF);
}

std::pair<std::vector<long int>, std::vector<RealType> > DiffusionTimeCDF::getCDF(RealType nParticles){
  std::vector<long int> xvals(t+1);
  std::vector<RealType> CDF_n(t+1);
  for (unsigned long int n=0; n<=t; n++){
    xvals[n] = 2*n + 2 - t;
    CDF_n[n] = exp(-CDF[n] * nParticles);
  }
  return std::make_pair(xvals, CDF_n);

}

// Note that this shifts the xvals by the mean.
RealType calculateVariance(std::vector<long int> xvals, std::vector<RealType> cdf)
{
  RealType var;
  // first shift so the mean is zero
  long int mean = static_cast<long int>(calculateMean(xvals, cdf));
  std::cout << "CDF Calculated Mean:" << mean << std::endl;
  std::transform(xvals.begin(), xvals.end(), xvals.begin(), std::bind2nd(std::minus<long int>(), mean));
  std::cout << "New Mean: " << calculateMean(xvals, cdf) << std::endl;
  RealType first_sum = 0;
  RealType second_sum = 0;
  int H;
  for (unsigned long int i=0; i < cdf.size(); i++){
    if (xvals[i] <=0){
      H = 0;
    }
    else{
      H = 1;
    }
    first_sum += 2 * xvals[i] * (H - cdf[i]);
    second_sum += (H - cdf[i]);
  }
  var = first_sum - pow(second_sum, 2); // second_sum should be 0 but isn't
  return var;
}

RealType DiffusionTimeCDF::getDiscreteVariance(RealType nParticles)
{
  std::pair<std::vector<long int>, std::vector<RealType> > pair = getCDF(nParticles);
  std::vector<long int> xvals = pair.first;
  std::vector<RealType> cdf = pair.second;
  return calculateVariance(xvals, cdf);
}

RealType calculateMeanPDF(std::vector<long int> xvals, std::vector<RealType> PDF){
  RealType mean = 0;
  for (unsigned long int i=0; i < PDF.size(); i++){
    mean += xvals[i] * PDF[i];
  }
  return mean;
}

RealType calculateVariancePDF(std::vector<long int> xvals, std::vector<RealType> PDF){
  RealType mean = calculateMeanPDF(xvals, PDF);
  RealType var = 0;
  for (unsigned long int i=1; i < PDF.size(); i++){
    var += PDF[i] * pow(xvals[i] - mean, 2);
  }
  return var;
}

RealType DiffusionTimeCDF::getDiscreteVarianceDiff(RealType nParticles)
{
  std::pair<std::vector<long int>, std::vector<RealType> > pair = getPDF(nParticles);
  std::vector<long int> xvals = pair.first;
  std::vector<RealType> PDF = pair.second;

  RealType mean = calculateMeanPDF(xvals, PDF);
  RealType var = calculateVariancePDF(xvals, PDF);
  std::cout << "PDF calculated Mean:" << mean << std::endl;
  std::cout << "PDF variance:" << var << std::endl;
  return var;
}

DiffusionPositionCDF::DiffusionPositionCDF(const double _beta, const unsigned long int _tMax, std::vector<RealType> _quantiles) : DiffusionCDF(_beta, _tMax)
{
  // Initialize such that CDF(n=0, t) = 1
  CDF.resize(tMax+1, 1);
  quantiles = _quantiles;

  // Create a measurment array such that each column is a different quantile
  // measurement and each row is a differeent time.
  quantilesMeasurement.resize(quantiles.size());
  for (unsigned long int i=0; i < quantilesMeasurement.size(); i++){
    quantilesMeasurement[i].resize(tMax+1, 0);
    // At t=0, all the quantiles will be 2 (2*n + 2 - t for n=t=0) - I think this is mainly for testing.
    quantilesMeasurement[i][0] = 2;
  }
}

void DiffusionPositionCDF::stepPosition()
{
  // Initialize all values to 0 since below CDF(n, n) is 0
  std::vector<RealType> CDF_next(tMax+1, 0);

  for (unsigned long int t = position+1; t < tMax + 1; t++){
    RealType beta = RealType(generateBeta());
    if (t == position+1){
      CDF_next[t] = beta * CDF[t-1];
    }
    else{
      CDF_next[t] = beta * CDF[t-1] + (1 - beta) * CDF_next[t-1];
    }

    // Now want to loop through every quantile vector and update positions of
    // each quantile at the current time.
    for (unsigned long int i=0; i < quantiles.size(); i++){
      if (CDF_next[t] > 1 / quantiles[i]){
        // quantilesMeasurement[i][t] is the ith quantile in the list at time t
        // Note: position will always be greater so don't need to check this.
        quantilesMeasurement[i][t] = 2 * (position + 1) + 2 - t;
      }
    }
  }
  CDF = CDF_next;
  position += 1;
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
      .def("getDiscreteVarianceDiff", &DiffusionTimeCDF::getDiscreteVarianceDiff, py::arg("nParticles"))
      .def("getDiscreteVariance", &DiffusionTimeCDF::getDiscreteVariance, py::arg("nParticles"))
      .def("getTime", &DiffusionTimeCDF::getTime)
      .def("iterateTimeStep", &DiffusionTimeCDF::iterateTimeStep)
      .def("findQuantile", &DiffusionTimeCDF::findQuantile, py::arg("quantile"))
      .def("findQuantiles", &DiffusionTimeCDF::findQuantiles, py::arg("quantiles"))
      .def("getPDF", &DiffusionTimeCDF::getPDF)
      .def("getCDF", &DiffusionTimeCDF::getCDF);

  py::class_<DiffusionPositionCDF, DiffusionCDF>(m, "DiffusionPositionCDF")
      .def(py::init<const double, const unsigned long int, std::vector<RealType> >(), py::arg("beta"), py::arg("tMax"), py::arg("quantiles"))
      .def("getPosition", &DiffusionPositionCDF::getPosition)
      .def("getQuantilesMeasurement", &DiffusionPositionCDF::getQuantilesMeasurement)
      .def("getQuantiles", &DiffusionPositionCDF::getQuantiles)
      .def("stepPosition", &DiffusionPositionCDF::stepPosition);

  m.def("calculateMean", &calculateMean, py::arg("xvals"), py::arg("cdf"));
  m.def("calculateVariance", &calculateVariance, py::arg("xvals"), py::arg("cdf"));
  m.def("calculateMeanPDF", &calculateMeanPDF, py::arg("xvals"), py::arg("pdf"));
  m.def("calculateVariancePDF", &calculateVariancePDF, py::arg("xvals"), py::arg("pdf"));
}
