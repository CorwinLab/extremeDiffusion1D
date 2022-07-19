#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <math.h>

#include "firstPassagePDF.hpp"
#include "pybind11_numpy_scalar.h"
#include "randomNumGenerator.hpp"

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

FirstPassagePDF::FirstPassagePDF(const double _beta,
                                 const unsigned long int _maxPosition)
                                : RandomNumGenerator(_beta)
{
  maxPosition = _maxPosition;
  PDF.resize(2 * maxPosition + 1);

  // Set middle element of array to 1
  PDF[maxPosition] = 1;
}

void FirstPassagePDF::iterateTimeStep()
{
  std::vector<RealType> newPDF(PDF.size(), 0);

  // Need to keep track of the transition probabilities to make sure
  // that the PDF stays normalized.
  std::vector<double> transitionProbabilities(PDF.size(), 0);
  for (unsigned int i = 0; i < transitionProbabilities.size(); i++) {
    if (PDF[i] != 0) {
      transitionProbabilities[i] = generateBeta();
    }
  }

  for (unsigned int i = 0; i < PDF.size(); i++) {
    if (i == 0) {
      newPDF.at(0) = PDF.at(0) + transitionProbabilities[1] * PDF.at(1);
    }
    else if (i == 1) {
      newPDF.at(1) = transitionProbabilities[2] * PDF.at(2);
    }
    else if (i == PDF.size() - 1) {
      newPDF.at(i) =
          PDF.at(i) + (1 - transitionProbabilities[i - 1]) * PDF.at(i - 1);
    }
    else if (i == PDF.size() - 2) {
      newPDF.at(i) = (1 - transitionProbabilities[i - 1]) * PDF.at(i - 1);
    }
    else {
      newPDF.at(i) = (1 - transitionProbabilities[i - 1]) * PDF.at(i - 1) +
                     transitionProbabilities[i + 1] * PDF.at(i + 1);
    }
  }

  firstPassageProbability =
      transitionProbabilities[1] * PDF.at(1) +
      (1 - transitionProbabilities[PDF.size() - 2]) * PDF.at(PDF.size() - 2);
  PDF = newPDF;
  t += 1;
}

std::tuple<std::vector<unsigned int long>,
           std::vector<RealType>,
           std::vector<RealType>,
           std::vector<RealType>>
FirstPassagePDF::evolveToCutoff(RealType cutOff, RealType nParticles)
{
  std::vector<RealType> pdf;
  std::vector<RealType> cdf;
  std::vector<RealType> cdfN;
  std::vector<unsigned int long> times;

  // Not sure how big these arrays will need to be but they should be at least
  // the size of maxPosition So I'll allocate that much memory
  pdf.reserve(maxPosition);
  cdf.reserve(maxPosition);
  cdfN.reserve(maxPosition);
  times.reserve(maxPosition);

  RealType cdf_sum = 0;
  RealType cdf_sumN = 0; 

  while ((cdf_sumN < cutOff) || (cdf_sum < 1/nParticles)) {
    iterateTimeStep();
    pdf.push_back(firstPassageProbability);
    cdf_sum += firstPassageProbability;
    cdf.push_back(cdf_sum);
    cdf_sumN = 1 - exp(-cdf_sum * nParticles);
    cdfN.push_back(cdf_sumN);
    times.push_back(t);
  }

  return std::make_tuple(times, pdf, cdf, cdfN);
}

PYBIND11_MODULE(firstPassagePDF, m)
{
  py::class_<FirstPassagePDF, RandomNumGenerator>(m, "FirstPassagePDF")
      .def(py::init<const double, const unsigned long int>(),
           py::arg("beta"),
           py::arg("maxPosition"))
      .def("getBeta", &FirstPassagePDF::getBeta)
      .def("setBeta", &FirstPassagePDF::setBeta)
      .def("getTime", &FirstPassagePDF::getTime)
      .def("setTime", &FirstPassagePDF::setTime)
      .def("getPDF", &FirstPassagePDF::getPDF)
      .def("setPDF", &FirstPassagePDF::setPDF)
      .def("getMaxPosition", &FirstPassagePDF::getMaxPosition)
      .def("setMaxPosition", &FirstPassagePDF::setMaxPosition)
      .def("iterateTimeStep", &FirstPassagePDF::iterateTimeStep)
      .def("getFirstPassageProbability",
           &FirstPassagePDF::getFirstPassageProbability)
      .def("evolveToCutoff", &FirstPassagePDF::evolveToCutoff);
}
