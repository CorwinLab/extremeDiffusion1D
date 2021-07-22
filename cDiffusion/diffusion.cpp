#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "diffusion.hpp"
#include "pybindutils.hpp"
#include <boost/multiprecision/float128.hpp>
#include <limits>
#include <cmath>

typedef boost::multiprecision::float128 RealType;
namespace py = pybind11;

//Constuctor
Diffusion::Diffusion(
  const double _nParticles,
  const double _beta,
  const unsigned long int occupancySize,
  const bool _probDistFlag)
  : nParticles(RealType(_nParticles)),
    ProbDistFlag(_probDistFlag),
    beta(_beta)
{
  edges.first.resize(occupancySize+1), edges.second.resize(occupancySize+1);
  edges.first[0] = 0, edges.second[0] = 0;

  occupancy.resize(occupancySize+1);
  occupancy[0] = nParticles;

  if (_beta != 0 ){
    boost::random::beta_distribution<>::param_type params(_beta, _beta);
    betaParams = params;
  }

  std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
  dis.param(unifParams);
  gen.seed(rd());

  time = 0;
}

RealType Diffusion::toNextSite(RealType currentSite, RealType bias){
  return (currentSite * bias);
}

double Diffusion::generateBeta(){
	// If beta = 0 return either 0 or 1
	if (beta == 0.0){
		return round(dis(gen));
	}
	// If beta = 1 use random uniform distribution
	else if (beta == 1.0){
		return dis(gen);
	}
  // If beta = inf return 0.5
  else if (isinf(beta)){
    return 0.5;
  }
	else{
		return beta_dist(gen, betaParams);
	}
}

void Diffusion::iterateTimestep()
{
  unsigned long int prevMinIndex = edges.first[time];
  unsigned long int prevMaxIndex = edges.second[time];
	if (prevMinIndex > prevMaxIndex) {
		throw std::runtime_error("Minimum edge must be greater than maximum edge: (" + std::to_string(prevMinIndex) + ", " + std::to_string(prevMaxIndex) + ")");
	}

	// If iterating over the whole array extend the occupancy.
	if ((prevMaxIndex+1) == occupancy.size()){
		occupancy.push_back(0);
		std::cout << "Warning: pushing back occupancy size. If this happens a lot it may effect performance." << std::endl;
	}

	RealType fromLastSite = 0;
	RealType toNextSite = 0;
	unsigned long int minEdge = 0;
	unsigned long int maxEdge = 0;
	bool firstNonzero = true;

	for (auto i = prevMinIndex; i < prevMaxIndex+2; i++) {

		RealType* occ = &occupancy.at(i);

    double bias;
    RealType floatbias;
		if (*occ != 0) {
			bias = Diffusion::generateBeta();
      floatbias = RealType(bias);
			toNextSite = Diffusion::toNextSite(*occ, floatbias);
      if (!ProbDistFlag){
        toNextSite = round(toNextSite);
      }
		}
		else{
			toNextSite = 0;
		}

		RealType prevOcc = *occ; // For error checking below
		*occ += fromLastSite - toNextSite;
		fromLastSite = toNextSite;

		if (*occ != 0) {
			maxEdge = i;
			if (firstNonzero) {
				minEdge = i;
				firstNonzero = false;
			}
		}

		if (toNextSite < 0 || toNextSite > prevOcc || bias < 0.0 || bias > 1.0 || *occ < 0 || *occ > nParticles || isnan(*occ)){
			throw std::runtime_error("One or more variables out of bounds: ");
		}
	}

  edges.first[time+1] = minEdge;
  edges.second[time+1] = maxEdge;
  time += 1;
}

double Diffusion::NthquartileSingleSided(const RealType NQuart)
{
  unsigned long int maxIdx = edges.second[time];
  double centerIdx = time * 0.5;

	double dist = maxIdx - centerIdx;
	RealType sum = occupancy.at(maxIdx);
	while (sum < NQuart){
		maxIdx -= 1;
		dist -= 1;
		sum += occupancy.at(maxIdx);
	}
	return dist;
}

RealType Diffusion::pGreaterThanX(const unsigned long int idx)
{
  RealType Nabove = 0.0;
  for (unsigned long int j = idx; j <= time; j++){
    Nabove += occupancy.at(j);
  }
  return Nabove / nParticles;
}

std::pair<std::vector<double>, std::vector<RealType> > Diffusion::calcVsAndPb(const unsigned long int num)
{
  std::vector<double> vs;
  std::vector<RealType> Pbs;
  unsigned long int maxIdx = edges.second[time];
  RealType Nabove = 0.0;
  for (unsigned long int i = maxIdx; i > (maxIdx - num); i--){
    Nabove += occupancy.at(i);
    RealType probAbove = Nabove / nParticles;
    double v = (2. * i - time) / time;
    vs.push_back(v);
    Pbs.push_back(probAbove);
  }
  std::pair<std::vector<double>, std::vector<RealType> > returnTuple(vs, Pbs);
  return returnTuple;
}

std::pair<std::vector<double>, std::vector<RealType> > Diffusion::VsAndPb(const double v)
{
  std::vector<double> vs;
  std::vector<RealType> Pbs;
  unsigned long int idx = edges.second[time];
  RealType Nabove = 0.0;
  double currentV = (2. * idx - time) / time;
  while (currentV >= v){
    Nabove += occupancy.at(idx);
    RealType probAbove = Nabove / nParticles;
    vs.push_back(currentV);
    Pbs.push_back(probAbove);

    idx -= 1;
    currentV = (2. * idx - time) / time;
  }
  std::pair<std::vector<double>, std::vector<RealType> > returnTuple(vs, Pbs);
  return returnTuple;
}

PYBIND11_MODULE(diffusion, m){
	m.doc() = "C++ diffusion";

	py::class_<Diffusion>(m, "Diffusion")
		.def(py::init<
      const double,
      const double,
      const unsigned long int,
      const bool>(),
			py::arg("numberOfParticles"),
      py::arg("beta"),
      py::arg("occupancySize"),
      py::arg("probDistFlag")=true)

		.def("getOccupancy", &Diffusion::getOccupancy)
		.def("setOccupancy", &Diffusion::setOccupancy, py::arg("occupancy"))
		.def("getNParticles", &Diffusion::getNParticles)
		.def("getBeta", &Diffusion::getBeta)
    .def("setProbDistFlag", &Diffusion::setProbDistFlag, py::arg("ProbDistFlag"))
    .def("getProbDistFlat", &Diffusion::getProbDistFlag)
		.def("getEdges", &Diffusion::getEdges)
    .def("getTime", &Diffusion::getTime)
		.def("iterateTimestep", &Diffusion::iterateTimestep)
    .def("NthquartileSingleSided", &Diffusion::NthquartileSingleSided)
    .def("pGreaterThanX", &Diffusion::pGreaterThanX, py::arg("idx"))
    .def("calcVsAndPb", &Diffusion::calcVsAndPb, py::arg("num"))
    .def("VsAndPb", &Diffusion::VsAndPb, py::arg("v"));
}
