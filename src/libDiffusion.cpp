#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "firstPassagePDF.hpp"
#include "randomNumGenerator.hpp"
#include "diffusionCDFBase.hpp"
#include "diffusionTimeCDF.hpp"
#include "diffusionPositionCDF.hpp"
#include "diffusionPDF.hpp"
#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

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

std::vector<RealType> iterateTimestep(std::vector<RealType> pdf, unsigned int maxPosition, unsigned int t)
{
  std::vector<RealType> pdf_new(pdf.size());
  // Okay this stuff is good
  if (t <= maxPosition) {
    for (unsigned int i = 0; i < pdf.size() - 1; i++) {
      pdf_new.at(i) += pdf.at(i) * 1 / 2;
      pdf_new.at(i + 1) += pdf.at(i) * 1 / 2;
    }
  }

  // This stuff needs the work
  else {
    for (unsigned int i = 0; i < pdf.size(); i++) {
      if (i == 0) {
        pdf_new.at(i) += pdf.at(i);
      }
      else if (pdf.back() != 0) {
        if (i == (pdf.size() - 1)) {
          pdf_new.at(i - 1) += pdf.at(i);
        }
        else {
          pdf_new.at(i) += pdf.at(i) * 1 / 2;
          pdf_new.at(i - 1) += pdf.at(i) * 1 / 2;
        }
      }
      else {
        if (i == (pdf.size() - 1)) {
          continue;
        }
        else if (i == (pdf.size() - 2)) {
          pdf_new.at(i + 1) += pdf.at(i);
        }
        else {
          pdf_new.at(i) += pdf.at(i) * 1 / 2;
          pdf_new.at(i + 1) += pdf.at(i) * 1 / 2;
        }
      }
    }
  }
  return pdf_new;
}

PYBIND11_MODULE(libDiffusion, m)
{
     m.doc() = "Random walk library";
     m.def("iteratePDF", iterateTimestep);
     py::class_<RandomNumGenerator>(m, "RandomNumGenerator")
          .def(py::init<const double>())
          .def("getBeta", &RandomNumGenerator::getBeta)
          .def("setBeta", &RandomNumGenerator::setBeta)
          .def("generateBeta", &RandomNumGenerator::generateBeta)
          .def("setBetaSeed", &RandomNumGenerator::setBetaSeed);
          
     py::class_<FirstPassagePDF, RandomNumGenerator>(m, "FirstPassagePDF")
          .def(py::init<const double, const unsigned long int, const bool>(),
               py::arg("beta"),
               py::arg("maxPosition"),
               py::arg("staticEnvironment"))
          .def("getBeta", &FirstPassagePDF::getBeta)
          .def("setBeta", &FirstPassagePDF::setBeta)
          .def("getTime", &FirstPassagePDF::getTime)
          .def("setTime", &FirstPassagePDF::setTime)
          .def("getPDF", &FirstPassagePDF::getPDF)
          .def("setPDF", &FirstPassagePDF::setPDF)
          .def("getTransitionProbabilities", &FirstPassagePDF::getTransitionProbabilities)
          .def("getMaxPosition", &FirstPassagePDF::getMaxPosition)
          .def("setMaxPosition", &FirstPassagePDF::setMaxPosition)
          .def("iterateTimeStep", &FirstPassagePDF::iterateTimeStep)
          .def("getFirstPassageProbability",
               &FirstPassagePDF::getFirstPassageProbability)
          .def("evolveToCutoff", &FirstPassagePDF::evolveToCutoff)
          .def("evolveToCutoffMultiple", &FirstPassagePDF::evolveToCutoffMultiple);

     py::class_<DiffusionPDF, RandomNumGenerator>(m, "DiffusionPDF")
          .def(py::init<const RealType,
                         const double,
                         const unsigned long int,
                         const bool,
                         const bool>(),
               py::arg("numberOfParticles"),
               py::arg("beta"),
               py::arg("occupancySize"),
               py::arg("ProbDistFlag") = true,
               py::arg("staticEnvironment") = false)
          .def("getOccupancy", &DiffusionPDF::getOccupancy)
          .def("setOccupancy", &DiffusionPDF::setOccupancy, py::arg("occupancy"))
          .def("getOccupancySize", &DiffusionPDF::getOccupancySize)
          .def("getTransitionProbabilities", &DiffusionPDF::getTransitionProbabilities)
          .def("getSaveOccupancy", &DiffusionPDF::getSaveOccupancy)
          .def("getStaticEnvironment", &DiffusionPDF::getStaticEnvironment)
          .def("setStaticEnvironment", &DiffusionPDF::setStaticEnvironment)
          .def("resizeOccupancy",
               &DiffusionPDF::resizeOccupancy,
               py::arg("size"))
          .def("getNParticles", &DiffusionPDF::getNParticles)
          .def("getBeta", &DiffusionPDF::getBeta)
          .def("setProbDistFlag",
               &DiffusionPDF::setProbDistFlag,
               py::arg("ProbDistFlag"))
          .def("getProbDistFlag", &DiffusionPDF::getProbDistFlag)
          .def("getSmallCutoff", &DiffusionPDF::getSmallCutoff)
          .def("setSmallCutoff",
               &DiffusionPDF::setSmallCutoff,
               py::arg("smallCutoff"))
          .def("getLargeCutoff", &DiffusionPDF::getLargeCutoff)
          .def("setLargeCutoff",
               &DiffusionPDF::setLargeCutoff,
               py::arg("largeCutoff"))
          .def("getEdges", &DiffusionPDF::getEdges)
          .def("setEdges", &DiffusionPDF::setEdges)
          .def("getMaxIdx", &DiffusionPDF::getMaxIdx)
          .def("getMinIdx", &DiffusionPDF::getMinIdx)
          .def("getTime", &DiffusionPDF::getTime)
          .def("setTime", &DiffusionPDF::setTime)
          .def("iterateTimestep", &DiffusionPDF::iterateTimestep)
          .def("findQuantile", &DiffusionPDF::findQuantile, py::arg("quantile"))
          .def("findQuantiles", &DiffusionPDF::findQuantiles, py::arg("quantiles"))
          .def("pGreaterThanX", &DiffusionPDF::pGreaterThanX, py::arg("idx"))
          .def("calcVsAndPb", &DiffusionPDF::calcVsAndPb, py::arg("num"))
          .def("VsAndPb", &DiffusionPDF::VsAndPb, py::arg("v"))
          .def("getGumbelVariance",
               &DiffusionPDF::getGumbelVariance,
               py::arg("nParticles"))
          .def("getCDF", &DiffusionPDF::getCDF);
     
     py::class_<DiffusionCDF, RandomNumGenerator>(m, "DiffusionCDF")
          .def(py::init<const double, const unsigned long int>(),
               py::arg("beta"),
               py::arg("tMax"))
          .def("getBeta", &DiffusionCDF::getBeta)
          .def("getCDF", &DiffusionCDF::getCDF)
          .def("setCDF", &DiffusionCDF::setCDF, py::arg("CDF"))
          .def("gettMax", &DiffusionCDF::gettMax)
          .def("settMax", &DiffusionCDF::settMax)
          .def("setBetaSeed", &DiffusionCDF::setBetaSeed, py::arg("seed"));

     py::class_<DiffusionTimeCDF, DiffusionCDF>(m, "DiffusionTimeCDF")
          .def(py::init<const double, const unsigned long int>(),
               py::arg("beta"),
               py::arg("tMax"))
          .def("getGumbelVariance",
               static_cast<RealType (DiffusionTimeCDF::*)(RealType)>(
                    &DiffusionTimeCDF::getGumbelVariance),
               py::arg("nParticles"))
          .def("getGumbelVariance",
               static_cast<std::vector<RealType> (DiffusionTimeCDF::*)(
                    std::vector<RealType>)>(&DiffusionTimeCDF::getGumbelVariance),
               py::arg("nParticles"))
          .def("getTime", &DiffusionTimeCDF::getTime)
          .def("setTime", &DiffusionTimeCDF::setTime)
          .def("iterateTimeStep", &DiffusionTimeCDF::iterateTimeStep)
          .def("findQuantile", &DiffusionTimeCDF::findQuantile, py::arg("quantile"))
          .def("findQuantiles",
               &DiffusionTimeCDF::findQuantiles,
               py::arg("quantiles"))
          .def("findLowerQuantile",
               &DiffusionTimeCDF::findLowerQuantile,
               py::arg("quantile"))
          .def("getSaveCDF", &DiffusionTimeCDF::getSaveCDF)
          .def("getxvals", &DiffusionTimeCDF::getxvals)
          .def("getProbandV", &DiffusionTimeCDF::getProbandV, py::arg("quantile"))
          .def("generateBeta", &DiffusionTimeCDF::generateBeta);

     py::class_<DiffusionPositionCDF, DiffusionCDF>(m, "DiffusionPositionCDF")
          .def(py::init<const double,
                         const unsigned long int,
                         std::vector<RealType>>(),
               py::arg("beta"),
               py::arg("tMax"),
               py::arg("quantiles"))
          .def("getPosition", &DiffusionPositionCDF::getPosition)
          .def("getQuantilePositions", &DiffusionPositionCDF::getQuantilePositions)
          .def("getQuantiles", &DiffusionPositionCDF::getQuantiles)
          .def("stepPosition", &DiffusionPositionCDF::stepPosition);
}
