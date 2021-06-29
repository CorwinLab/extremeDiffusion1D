#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <time.h>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <random>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <assert.h>

namespace py = pybind11;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
boost::random::binomial_distribution<> binomial;
boost::random::normal_distribution<> normal;
boost::random::beta_distribution<> beta_dist;
const double smallCutoff = pow(2, 31) - 2;
const double largeCutoff = 1e31;

template<class Temp>
void print_generic(Temp vec) {
	// Prints an object with an interator member.
	// Should cut this off at like 10 items but w/e
	std::cout << "[";
	for (auto i = vec.begin(); i != vec.end(); i++) {
		std::cout << *i << ", ";
	}
	std::cout << "] \n";
};

// Given the number of particles and probability of going right, generate the number
// of particles that move to the right. If the number of walkers are small enough
// draw from a binomial distribution. If they are huge we can just assume the number
// moving right = number of particles * prob to move right. If they are somewhere
// in the middle we can get away with approximating the binomial distribution as
// a normal distribution. The smallCutoff is used to determine what is small,
// medium and large. Note, if the smallCutoff is too small, the normal approximation
// may breakdown causing the (# particles right shifted) > (# of particles).
// Note that the boost C++ binomial distribution breaks for occupancies bigger than
// a long integer. It returns the negative bound of a long integer.
double gettoNextSite(const double occ, const double bias,
																const double smallCutoff=smallCutoff,
																const double largeCutoff=largeCutoff) {
	if (bias >= 0.999999){
		return round(occ * bias);
	}

	if (bias <= 0.000001){
	       return round(occ * bias);
	}

	if (occ < smallCutoff) {
		// If small enough to use integer representations use binomial distribution
	  return binomial(gen, boost::random::binomial_distribution<>::param_type(occ, bias));
	}
	else if (occ > largeCutoff) {
		// If so large that sqrt(N) is less than precision just use occupancy
		return round(occ * bias);
	}
	else {
		// If in sqrt(N) precision use gaussian approximation.
		double mediumVariance = sqrt(occ * bias * (1 - bias));
		return round(normal(gen, boost::random::normal_distribution<>::param_type(occ * bias, mediumVariance)));
	}
}

// Iterate one time step according to Barraquad/Corwin model
// Note: (alpha=1, beta=1) gives uniform distribution.
std::pair<unsigned long int, unsigned long int> floatEvolveTimeStep(
	std::vector<double>& occupancy,
	const boost::random::beta_distribution<>::param_type betaParams,
	const unsigned long int prevMinIndex,
	const unsigned long int prevMaxIndex,
	const double N,
	const double smallCutoff=smallCutoff,
	const double largeCutoff=largeCutoff
)
{
	if (prevMinIndex >= prevMaxIndex) {
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
			bias = beta_dist(gen, betaParams);
			toNextSite = gettoNextSite(*occ, bias, smallCutoff, largeCutoff);
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

		if (toNextSite < 0 || toNextSite > prevOcc || bias < 0.0 || bias > 1.0 || *occ < 0 || *occ > N || isnan(*occ)){
			throw std::runtime_error("One or more variables out of bounds: Right shift= "
			+ std::to_string(toNextSite) + ", occupancy=" + std::to_string(*occ)
			+ ", bias=" + std::to_string(bias) + ", smallCutoff=" + std::to_string(smallCutoff)
			+ ", N=" + std::to_string(N));
		}
	}

	if (minEdge == maxEdge){
		// Only one state occupied so add 1 to maxEdge to ensure loop runs next step.
		maxEdge += 1;
	}

	std::pair<unsigned long int, unsigned long int> edges(minEdge, maxEdge);

	return edges;
}

// Iterate one time step according to Einstein Diffusion or bias=0.5
std::pair<unsigned long int, unsigned long int> floatEvolveEinstein(
	std::vector<double>& occupancy,
	const unsigned long int prevMinIndex,
	const unsigned long int prevMaxIndex,
	const double N,
	const double smallCutoff=smallCutoff,
	const double largeCutoff=largeCutoff
)
{
	if (prevMinIndex >= prevMaxIndex) {
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

		double bias = 0.5;
		if (*occ != 0) {
			toNextSite = gettoNextSite(*occ, bias, smallCutoff, largeCutoff);
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

		if (toNextSite < 0 || toNextSite > prevOcc || bias < 0.0 || bias > 1.0 || *occ < 0 || *occ > N || isnan(*occ)){
			throw std::runtime_error("One or more variables out of bounds: Right shift= "
			+ std::to_string(toNextSite) + ", occupancy=" + std::to_string(*occ)
			+ ", bias=" + std::to_string(bias) + ", smallCutoff=" + std::to_string(smallCutoff)
			+ ", N=" + std::to_string(N));
		}
	}

	if (minEdge == maxEdge){
		// Only one state occupied so add 1 to maxEdge to ensure loop runs next step.
		maxEdge += 1;
	}

	std::pair<unsigned long int, unsigned long int> edges(minEdge, maxEdge);

	return edges;
}

// Does the same thing as floatEvolveTimeStep but returns the occupancy.
// This is because passing by reference doesn't work easily with Pybind11.
// I think every time we pass a vector between C++ and Python it's copied.
std::pair<std::pair<unsigned long int, unsigned long int>, std::vector<double> > pyfloatEvolveTimestep(
	std::vector<double> occupancy,
	const double beta,
	const unsigned long int prevMinIndex,
	const unsigned long int prevMaxIndex,
	const double N,
	const double smallCutoff=smallCutoff,
	const double largeCutoff=largeCutoff
)
{
	boost::random::beta_distribution<>::param_type betaParams(beta, beta);
	std::pair<unsigned long int, unsigned long int> edges = floatEvolveTimeStep(occupancy, betaParams, prevMinIndex, prevMaxIndex, N, smallCutoff, largeCutoff);
	std::pair<std::pair<unsigned long int, unsigned long int>, std::vector<double> > returnVal(edges, occupancy);
	return returnVal;
}

// Class to take make a diffusion experiment easier. All the date is handled on
// the C++ side of things so that the occupancy array is only called when python
// calls for an array.
class Diffusion{
	private:
		std::vector<double> occupancy;
		double N;
		boost::random::beta_distribution<>::param_type betaParams;
		double smallCutoff;
		double largeCutoff;
		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edges;
		unsigned long int time;

	public:
		Diffusion(
			const double numberOfParticles,
			const double b,
			const double scutoff=pow(2, 31)-2,
			const double lcutoff=1e31){
			N = numberOfParticles;
			smallCutoff = scutoff;
			largeCutoff = lcutoff;
			edges.first.push_back(0), edges.second.push_back(1);
			boost::random::beta_distribution<>::param_type params(b, b);
			betaParams = params;
			time = 0;
		}

		void initializeOccupationAndEdges(unsigned long int size=0){
			if (size==0){
				size = getLogN52();
			}
			occupancy.resize(size);
			occupancy[0] = N;
			edges.first.resize(size);
			edges.second.resize(size);
		}

		unsigned long int getLogN52(){
			double logN = log(N);
			double numTimeSteps = pow(logN, 5.0/2.0);
			return llround(numTimeSteps);
		}

		std::vector<double> getOccupancy(){
			return occupancy;
		}

		void setOccupancy(const std::vector<double> occ){
			occupancy = occ;
		}

		double getN(){
			return N;
		}

		void setN(const double Number){
			N = Number;
		}

		double getBeta(){
			return betaParams.beta();
		}

		void setBeta(const double b){
			boost::random::beta_distribution<>::param_type params(b, b);
			betaParams = params;
		}

		double getsmallCutoff(){
			return smallCutoff;
		}

		void setsmallCutoff(const double s){
			smallCutoff = s;
		}

		double getlargeCutoff(){
			return largeCutoff;
		}

		void setlargeCutoff(const double l){
			largeCutoff = l;
		}

		unsigned long int getTime(){
			return time;
		}

		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > getEdges(){
			return edges;
		}

		// Move the occupancy forward one step in time. Requires push_back for the edges
		// which might slow it down a lot.
		void iterateTimestep(bool inplace=false){
			if (!inplace){
				edges.first.push_back(0);
				edges.second.push_back(0);
				occupancy.push_back(0);
			}
			auto newEdges = floatEvolveTimeStep(
				occupancy,
				betaParams,
				edges.first[time],
				edges.second[time],
				N,
				smallCutoff,
				largeCutoff);
			edges.first[time+1] = newEdges.first;
			edges.second[time+1] = newEdges.second;
			time += 1;
		}

		// Move the ocupancy forward N steps in time.
		void evolveTimesteps(const unsigned long int iterations, bool inplace=false){
			if (!inplace){
				unsigned int edgesLength = edges.first.size();
				edges.first.resize(iterations + edgesLength);
				edges.second.resize(iterations + edgesLength);
				occupancy.resize(iterations + occupancy.size());
			}
			if (iterations == 1){
				iterateTimestep(inplace=true);
			}
			else{
				unsigned long int max = time+iterations;
				for (unsigned long int i = time; i < max-1; i++){
					iterateTimestep(inplace=true);
				}
			}
		}

		void evolveEinstein(const unsigned int iterations, bool inplace=false){
			if (!inplace){
				unsigned int edgesLength = edges.first.size();
				edges.first.resize(iterations + edgesLength);
				edges.second.resize(iterations + edgesLength);
				occupancy.resize(iterations + occupancy.size());
			}
			unsigned long int max = time + iterations;
			for (unsigned long int i = time; i < max - 1; i++){
				auto newEdges = floatEvolveEinstein(
					occupancy,
					edges.first[i],
					edges.second[i],
					N,
					smallCutoff,
					largeCutoff);
				edges.first[i+1] = newEdges.first;
				edges.second[i+1] = newEdges.second;
				time += 1;
			}
		}

		unsigned int findNumberParticles(){
			double sum = std::accumulate(occupancy.begin(), occupancy.end(), 0);
			return sum;
		}

		void addNumElemEdges(unsigned long int num){
			edges.first.resize(num + edges.first.size());
			edges.second.resize(num + edges.second.size());
		}
};

double generateBeta(double beta){
	boost::random::beta_distribution<>::param_type params(beta, beta);
	return beta_dist(gen, params);
}

PYBIND11_MODULE(cDiffusion, m){
	m.doc() = "C++ diffusion";
	m.def("generateRandomBeta", &generateBeta, py::arg("beta"));

	const char * pyfloatdoc = R"V0G0N(
Evolve the occupancy forward one timestep drawing from the provided beta distribution.

Parameters
----------
occupancy : numpy array or list
	Number of walkers at each location

beta : float
	Value of beta for the beta distribution to draw biases from

prevMinIndex : int
	Index of the first occupied, or nonzero, index in occupation

prevMaxIndex : int
  Index of last occupied, or nonzer, index in occupation

smallCutoff : int (2^53)
	Precision of double to use for normal and binomial approximation.

Returns
-------
edges : tuple
	Index of first and last occupied, or nonzero, index in new occupation

occupancy : numpy array
	New number of walkers at each loation. Size should be preserved from the input
	occupancy array.
)V0G0N";

	m.def("floatEvolveTimeStep", &pyfloatEvolveTimestep, pyfloatdoc,
				py::arg("occupancy"), py::arg("beta"), py::arg("prevMinIndex"),
				py::arg("prevMaxIndex"), py::arg("N"), py::arg("smallCutoff")=smallCutoff,
				py::arg("largeCutoff")=largeCutoff);

const char * initdoc = R"V0G0N(
Helper class to run diffusion experiments with. Stores the occoupancy and edges
in C++ only when they are called from Python. This makes it a lot faster than passing
variables between C++ and Python which is a lot slower.

Examples
--------
>>> from cDiffusion import Diffusion
>>> d = Diffusion(10, 1.0)
>>> d.initializeOccupationAndEdges(size=5)
>>> print(d.getOccupancy())
[10.0, 0.0, 0.0, 0.0, 0.0]
>>> d.iterateTimestep(inplace=True)
>>> print(d.getOccupancy())
[3.0, 7.0, 0.0, 0.0, 0.0]

)V0G0N";

	const char * iterateTimestepdoc = R"V0G0N(
Move the occupancy forward through one timestep. Appends the new edge positions
to the edges vector.

Parameters
----------
inplace : bool (False)
	Whether or not to push back the edges/occupancy or not. If true, it will push back
	the edges & occupancy by 1. If false doesn't push back the edges & occupancy so
	the edges and occupancy must be large enough to support one timestep.

Example
-------
>>> from cDiffusion import Diffusion
>>> d = Diffusion(10, 1.0)
>>> d.setOccupancy([10])
>>> d.iterateTimestep()
>>> print(d.getOccupancy())
Warning: pushing back occupancy size. If this happens a lot it may effect performance.
[2.0, 8.0, 0.0]

Note
----
Use evolveTimesteps to move forward multiple timesteps as evolveTimesteps preallocates
needed space in edges vector for new edge positions whereas iterateTimestep just
push_back the edges vector every iteration.
)V0G0N";

	const char * classevolveTimestepsdoc = R"V0G0N(
Evolve the occupancy forward N number of iterations.

Parameters
----------
iterations : int
	Number of iterations to move the system forward int time

inplace : bool (False)
	Whether or not to extend the occupancy and edges by iterations. If true, occupancy
	and edges remain the same length so they must be at least size=len(iterations).
	If false, changes the length of edges/ocupancy to length=original length + iterations.

Examples
--------
>>> from cDiffusion import Diffusion
>>> d = Diffusion(10, 1.0)
>>> d.setOccupancy([10])
>>> d.evolveTimesteps(10)
>>> print(d.getOccupancy())
[0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0]

Note
----
Much faster than iterateTimestep method because it preallocates needed space in
edges vector.
)V0G0N";

	const char * findNumberParticlesdoc = R"V0G0N(
Sum over occupancy to find the current number of particles. Could probably just sum
from minEdge to maxEdge.
)V0G0N";

const char * initializeOccupationAndEdgesdoc = R"V0G0N(
Resizes the occupancy and edges to a specified size and puts all N particles in the
first occupancy position. Default size is Log(N) ** (5/2)

Parameters
----------
size : int
	Size to change the occupantion and edges to. Defaults to Log(N) ** (5/2)

Examples
--------
>>> from cDiffusion import Diffusion
>>> d = Diffusion(10, 1.0)
>>> d.initializeOccupationAndEdges(size=5)
>>> print(d.getOccupancy())
[10.0, 0.0, 0.0, 0.0, 0.0]
)V0G0N";

const char * evolveEinsteinddoc = R"V0G0N(
Evolve the occupancy forward N times according to Einstein diffusion or bias=0.5.

Parameters
----------
iterations : int
	Number of time steps to evolve the system

inplace : bool
	Whether or not to extend the occupancy and edges by iterations. If true, occupancy
	and edges remain the same length so they must be at least size=len(iterations).
	If false, changes the length of edges/ocupancy to length=original length + iterations.

Examples
--------
>>> from cDiffusion import Diffusion
>>> d = Diffusion(10, 1.0)
>>> d.setOccupancy([10])
>>> d.evolveEinstein(10)
>>> print(d.getOccupancy())
[0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 3.0, 1.0, 0.0, 0.0]

)V0G0N";

	py::class_<Diffusion>(m, "Diffusion")
		.def(py::init<const double, const double, const double, const double>(), initdoc,
					py::arg("numberOfParticles"), py::arg("beta"), py::arg("smallCutoff")=smallCutoff,
					py::arg("largeCutoff")=largeCutoff)
		.def("initializeOccupationAndEdges", &Diffusion::initializeOccupationAndEdges, initializeOccupationAndEdgesdoc, py::arg("size")=0)
		.def("getLogN52", &Diffusion::getLogN52)
		.def("getOccupancy", &Diffusion::getOccupancy)
		.def("setOccupancy", &Diffusion::setOccupancy, py::arg("occupancy"))
		.def("getN", &Diffusion::getN)
		.def("getBeta", &Diffusion::getBeta)
		.def("setBeta", &Diffusion::setBeta, py::arg("beta"))
		.def("getsmallCutoff", &Diffusion::getsmallCutoff)
		.def("setsmallCutoff", &Diffusion::setsmallCutoff, py::arg("smallCutoff"))
		.def("getlargeCutoff", &Diffusion::getlargeCutoff)
		.def("setlargeCutoff", &Diffusion::setlargeCutoff, py::arg("largeCutoff"))
		.def("getEdges", &Diffusion::getEdges)
		.def("iterateTimestep", &Diffusion::iterateTimestep, iterateTimestepdoc, py::arg("inplace")=false)
		.def("evolveTimesteps", &Diffusion::evolveTimesteps, classevolveTimestepsdoc, py::arg("iterations"), py::arg("inplace")=false)
		.def("evolveEinstein", &Diffusion::evolveEinstein, evolveEinsteinddoc, py::arg("iterations"), py::arg("inplace")=false)
		.def("findNumberParticles", &Diffusion::findNumberParticles, findNumberParticlesdoc);
}
