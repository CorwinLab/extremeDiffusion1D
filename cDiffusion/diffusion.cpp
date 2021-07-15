#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "diffusion.hpp"

namespace py = pybind11;

std::random_device rd;
boost::random::mt19937_64 gen(rd());

//Constuctor
Diffusion::Diffusion(
  const double _nParticles,
  const double _beta,
  const unsigned long int occupancySize,
  const double _smallCutoff,
  const double _largeCutoff,
  const bool _probDistFlag)
  : nParticles(_nParticles),
    smallCutoff(_smallCutoff),
    largeCutoff(_largeCutoff),
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

double Diffusion::toNextSite(double currentSite, double bias){
  if (bias >= 0.999999){
		return (currentSite * bias);
	}

	if (bias <= 0.000001){
		return (currentSite * bias);
	}

	if (currentSite < smallCutoff) {
		// If small enough to use integer representations use binomial distribution
	  return binomial(gen, boost::random::binomial_distribution<>::param_type(currentSite, bias));
	}
	else if (currentSite > largeCutoff) {
		// If so large that sqrt(N) is less than precision just use occupancy
		return (currentSite * bias);
	}
	else {
		// If in sqrt(N) precision use gaussian approximation.
		double mediumVariance = sqrt(currentSite * bias * (1 - bias));
		return normal(gen, boost::random::normal_distribution<>::param_type(currentSite * bias, mediumVariance));
	}
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

	// If we keep the occupancy the same throughout the whole experiment we probably
	// only need to construct this distribution once but w/e

	double fromLastSite = 0;
	double toNextSite = 0;
	unsigned long int minEdge = 0;
	unsigned long int maxEdge = 0;
	bool firstNonzero = true;

	for (auto i = prevMinIndex; i < prevMaxIndex+2; i++) {

		double* occ = &occupancy.at(i);

		double bias = 0;
		if (*occ != 0) {
			bias = Diffusion::generateBeta();
			toNextSite = Diffusion::toNextSite(*occ, bias);
      if (!ProbDistFlag){
        toNextSite = round(toNextSite);
      }
		}
		else{
			toNextSite = 0;
		}

		double prevOcc = *occ; // For error checking below
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
			throw std::runtime_error("One or more variables out of bounds: Right shift= "
			+ std::to_string(toNextSite) + ", occupancy=" + std::to_string(*occ)
			+ ", bias=" + std::to_string(bias) + ", smallCutoff=" + std::to_string(smallCutoff)
			+ ", N=" + std::to_string(nParticles));
		}
	}

  /*
	if (minEdge == maxEdge){
		// Only one state occupied so add 1 to maxEdge to ensure loop runs next step.
		maxEdge += 1;
	}
  */

  edges.first[time+1] = minEdge;
  edges.second[time+1] = maxEdge;
  time += 1;
}

double Diffusion::NthquartileSingleSided(const double NQuart)
{
  unsigned long int maxIdx = edges.second[time];
  double centerIdx = time * 0.5;

	double dist = maxIdx - centerIdx;
	double sum = occupancy.at(maxIdx);
	while (sum < NQuart){
		maxIdx -= 1;
		dist -= 1;
		sum += occupancy.at(maxIdx);
	}
	return dist;
}

double Diffusion::pGreaterThanX(const unsigned long int idx)
{
  unsigned long int i = time;
  double Nabove = 0.0;
  for (unsigned long int j = idx; j <= i; j++){
    Nabove += occupancy.at(j);
  }
  return Nabove; 
}

PYBIND11_MODULE(diffusion, m){
	m.doc() = "C++ diffusion";

	py::class_<Diffusion>(m, "Diffusion")
		.def(py::init<
      const double,
      const double,
      const unsigned long int,
      const double,
      const double,
      const bool>(),
			py::arg("numberOfParticles"),
      py::arg("beta"),
      py::arg("occupancySize"),
      py::arg("smallCutoff")=pow(2,31)-2,
			py::arg("largeCutoff")=1e31,
      py::arg("probDistFlag")=true)

		.def("getOccupancy", &Diffusion::getOccupancy)
		.def("setOccupancy", &Diffusion::setOccupancy, py::arg("occupancy"))
		.def("getNParticles", &Diffusion::getNParticles)
		.def("getBeta", &Diffusion::getBeta)
		.def("getsmallCutoff", &Diffusion::getSmallCutoff)
		.def("setsmallCutoff", &Diffusion::setSmallCutoff, py::arg("smallCutoff"))
		.def("getlargeCutoff", &Diffusion::getLargeCutoff)
		.def("setlargeCutoff", &Diffusion::setLargeCutoff, py::arg("largeCutoff"))
    .def("setProbDistFlag", &Diffusion::setProbDistFlag, py::arg("ProbDistFlag"))
    .def("getProbDistFlat", &Diffusion::getProbDistFlag)
		.def("getEdges", &Diffusion::getEdges)
    .def("getTime", &Diffusion::getTime)
		.def("iterateTimestep", &Diffusion::iterateTimestep)
    .def("NthquartileSingleSided", &Diffusion::NthquartileSingleSided)
    .def("pGreaterThanX", &Diffusion::pGreaterThanX, py::arg("idx"));
}
