#include <boost/multiprecision/float128.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "diffusionPDF.hpp"
#include "diffusionPositionCDF.hpp"
#include "diffusionTimeCDF.hpp"
#include "firstPassageBase.hpp"
#include "firstPassageDriver.hpp"
#include "firstPassageEvolve.hpp"
#include "firstPassagePDF.hpp"
#include "particleData.hpp"
#include "pybind11_numpy_scalar.h"
#include "randomDistribution.hpp"
#include "randomNumGenerator.hpp"

typedef boost::multiprecision::float128 RealType;

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

PYBIND11_MODULE(libDiffusion, m)
{
  m.doc() = "Random walk library";
  py::class_<ParticleData>(m, "ParticleData")
      .def(py::init<const RealType>())
      .def(py::self == py::self)
      .def("push_back_cdf", &ParticleData::push_back_cdf)
      .def("calculateVariance", &ParticleData::calculateVariance)
      .def_readwrite("quantileTime", &ParticleData::quantileTime)
      .def_readwrite("variance", &ParticleData::variance)
      .def_readwrite("quantileSet", &ParticleData::quantileSet)
      .def_readwrite("varianceSet", &ParticleData::varianceSet)
      .def_readwrite("cdfPrev", &ParticleData::cdfPrev)
      .def_readwrite("runningSumSquared", &ParticleData::runningSumSquared)
      .def_readwrite("runningSum", &ParticleData::runningSum);

  py::class_<RandomNumGenerator>(m, "RandomNumGenerator")
      .def(py::init<const double>())
      .def("getBeta", &RandomNumGenerator::getBeta)
      .def("setBeta", &RandomNumGenerator::setBeta)
      .def("generateBeta", &RandomNumGenerator::generateBeta)
      .def("setBetaSeed", &RandomNumGenerator::setBetaSeed)
      .def("getBetaSeed", &RandomNumGenerator::getBetaSeed);

  py::class_<RandomDistribution>(m, "RandomDistribution")
      .def(py::init<std::string, std::vector<double>>())
      .def("getDistributionName", &RandomDistribution::getDistributionName)
      .def("setDistributionName", &RandomDistribution::setDistributionName)
      .def("getParameters", &RandomDistribution::getParameters)
      .def("setParameters", &RandomDistribution::setParameters)
      .def("setGeneratorSeed", &RandomDistribution::setGeneratorSeed)
      .def("generateRandomVariable",
           &RandomDistribution::generateRandomVariable);

  py::class_<FirstPassageBase>(m, "FirstPassageBase")
      .def(py::init<const unsigned long int>(), py::arg("maxPosition"))
      .def(py::self == py::self)
      .def("getTime", &FirstPassageBase::getTime)
      .def("setTime", &FirstPassageBase::setTime)
      .def("getPDF", &FirstPassageBase::getPDF)
      .def("setPDF", &FirstPassageBase::setPDF)
      .def("getHaltingFlag", &FirstPassageBase::getHaltingFlag)
      .def("setHaltingFlag", &FirstPassageBase::setHaltingFlag)
      .def("getFirstPassageCDF", &FirstPassageBase::getFirstPassageCDF)
      .def("setFirstPassageCDF", &FirstPassageBase::setFirstPassageCDF)
      .def("getMaxPosition", &FirstPassageBase::getMaxPosition)
      .def("setMaxPosition", &FirstPassageBase::setMaxPosition)
      .def("iterateTimeStep", &FirstPassageBase::iterateTimeStep);

  py::class_<FirstPassageDriver, RandomNumGenerator>(m, "FirstPassageDriver")
      .def(py::init<const double, std::vector<unsigned int long>>())
      .def("iterateTimeStep", &FirstPassageDriver::iterateTimeStep)
      .def("getMaxPositions", &FirstPassageDriver::getMaxPositions)
      .def("setMaxPositions", &FirstPassageDriver::setMaxPositions)
      .def("getBiases", &FirstPassageDriver::getBiases)
      .def("getTime", &FirstPassageDriver::getTime)
      .def("setTime", &FirstPassageDriver::setTime)
      .def("getPDFs", &FirstPassageDriver::getPDFs)
      .def("setPDFs", &FirstPassageDriver::setPDFs)
      .def("evolveToCutoff", &FirstPassageDriver::evolveToCutoff);

  py::class_<FirstPassageEvolve, FirstPassageDriver>(m, "FirstPassageEvolve")
      .def(py::init<const double, std::vector<unsigned int long>, RealType>())
      .def("getParticleData", &FirstPassageEvolve::getParticleData)
      .def("setParticleData", &FirstPassageEvolve::setParticleData)
      .def("getNumberHalted", &FirstPassageEvolve::getNumberHalted)
      .def("setNumberHalted", &FirstPassageEvolve::setNumberHalted)
      .def("getNParticles", &FirstPassageEvolve::getNParticles)
      .def("setNParticles", &FirstPassageEvolve::setNParticles)
      .def("checkParticleData", &FirstPassageEvolve::checkParticleData)
      .def("getNumberOfPositions", &FirstPassageEvolve::getNumberOfPositions);

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
      .def("getTransitionProbabilities",
           &FirstPassagePDF::getTransitionProbabilities)
      .def("getFirstPassageCDF", &FirstPassagePDF::getFirstPassageCDF)
      .def("getMaxPosition", &FirstPassagePDF::getMaxPosition)
      .def("setMaxPosition", &FirstPassagePDF::setMaxPosition)
      .def("iterateTimeStep", &FirstPassagePDF::iterateTimeStep)
      .def("evolveToCutoff", &FirstPassagePDF::evolveToCutoff)
      .def("evolveToCutoffMultiple", &FirstPassagePDF::evolveToCutoffMultiple);

  py::class_<DiffusionPDF, RandomDistribution>(m, "DiffusionPDF")
      .def(py::init<const RealType,
                    std::string,
                    std::vector<double>,
                    const unsigned long int,
                    const bool,
                    const bool>())
      .def("getOccupancy", &DiffusionPDF::getOccupancy)
      .def("setOccupancy", &DiffusionPDF::setOccupancy, py::arg("occupancy"))
      .def("getOccupancySize", &DiffusionPDF::getOccupancySize)
      .def("getTransitionProbabilities",
           &DiffusionPDF::getTransitionProbabilities)
      .def("getSaveOccupancy", &DiffusionPDF::getSaveOccupancy)
      .def("getStaticEnvironment", &DiffusionPDF::getStaticEnvironment)
      .def("setStaticEnvironment", &DiffusionPDF::setStaticEnvironment)
      .def("resizeOccupancy", &DiffusionPDF::resizeOccupancy, py::arg("size"))
      .def("getNParticles", &DiffusionPDF::getNParticles)
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
      .def("removeParticle", &DiffusionPDF::removeParticle)
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

  py::class_<DiffusionTimeCDF, RandomDistribution>(m, "DiffusionTimeCDF")
      .def(py::init<std::string, std::vector<double>, const unsigned long int>())
      .def("getCDF", &DiffusionTimeCDF::getCDF)
      .def("setCDF", &DiffusionTimeCDF::setCDF)
      .def("gettMax", &DiffusionTimeCDF::gettMax)
      .def("settMax", &DiffusionTimeCDF::settMax)
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
      .def("getProbOutsidePositions", &DiffusionTimeCDF::getProbOutsidePositions);
      
  py::class_<DiffusionPositionCDF, RandomDistribution>(m, "DiffusionPositionCDF")
      .def(py::init<std::string,
                    std::vector<double>,
                    const unsigned long int,
                    std::vector<RealType>>())
      .def("getCDF", &DiffusionPositionCDF::getCDF)
      .def("setCDF", &DiffusionPositionCDF::setCDF)
      .def("gettMax", &DiffusionPositionCDF::gettMax)
      .def("settMax", &DiffusionPositionCDF::settMax)
      .def("getPosition", &DiffusionPositionCDF::getPosition)
      .def("getQuantilePositions", &DiffusionPositionCDF::getQuantilePositions)
      .def("getQuantiles", &DiffusionPositionCDF::getQuantiles)
      .def("stepPosition", &DiffusionPositionCDF::stepPosition);
}
