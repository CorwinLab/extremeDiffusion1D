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

namespace py = pybind11;

std::random_device rd;
// Take out random seed an initialize generator on import - make sure
// that this is initialized on import into python
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
boost::random::binomial_distribution<> binomial;
boost::random::normal_distribution<> normal;
boost::random::beta_distribution<> beta_dist;

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
double getRightShift(const double occ, const double bias,
																const double smallCutoff=pow(2,53)-2,
																const double largeCutoff=pow(10, 31)) {
	double rightShift = 0;
	if (occ < smallCutoff) {
		// If small enough to use integer representations use binomial distribution
		boost::random::binomial_distribution<>::param_type params(occ, bias);
	  rightShift = binomial(gen, params);
	}
	else if (occ > largeCutoff) {
		// If so large that sqrt(N) is less than precision just use occupancy
		rightShift = round(occ * bias);
	}
	else {
		// If in sqrt(N) precision use gaussian approximation.
		double mediumVariance = occ * bias * (1 - bias);
		mediumVariance = sqrt(mediumVariance);
		boost::random::normal_distribution<>::param_type params(occ * bias, mediumVariance);
		rightShift = round(normal(gen, params));
	}
	if (rightShift < 0){
		throw std::runtime_error("Right shift = " + std::to_string(rightShift) + " for occupancy=" + std::to_string(occ) + ", bias=" + std::to_string(bias) + ", smallCutoff=" + std::to_string(smallCutoff));
	}
	return rightShift;
}

// Iterate one time step according to Barraquad/Corwin model
// Note: (alpha=1, beta=1) gives uniform distribution.
std::pair<unsigned long int, unsigned long int> floatEvolveTimeStep(
	std::vector<double> &occupancy,
	const double beta,
	const unsigned long int prevMinIndex,
	const unsigned long int prevMaxIndex,
	const double N,
	const double smallCutoff=pow(2,53)-2,
	const double largeCutoff=pow(10, 31)
)
{
	if (prevMinIndex >= prevMaxIndex) {
		throw std::runtime_error("Minimum edge must be greater than maximum edge: (" + std::to_string(prevMinIndex) + ", " + std::to_string(prevMaxIndex) + ")");
	}

	// Need to check when prevMaxIndex breaks the for loop.
	if ((prevMaxIndex+1) > occupancy.size()){
		throw std::runtime_error("Maximum edge exceeds size of vector");
	}

	// If iterating over the whole array extend the occupancy.
	if ((prevMaxIndex+1) == occupancy.size()){
		occupancy.push_back(0);
	}

	// If we keep the occupancy the same throughout the whole experiment we probably
	// only need to construct this distribution once but w/e
	boost::random::beta_distribution<>::param_type params(beta, beta);

	double leftShift = 0;
	double rightShift = 0;
	unsigned long int minEdge = 0;
	unsigned long int maxEdge = 0;
	bool firstNonzero = true;

	// Now can use an iterator?
	// Check out range operator
	for (auto i = prevMinIndex; i < prevMaxIndex+1; i++) {

		// Skip over occupation value if ocuppancy=0 and not moving any walkers
		// to the position
		if (occupancy[i] == 0 && leftShift == 0) {
			continue;
		}

		if (occupancy[i] < 0) {
			throw std::runtime_error("Occupancy must be > 0 but Occupancy[" + std::to_string(i) + "]=" + std::to_string(occupancy[i]));
		}

		if (occupancy[i] > N){
			throw std::runtime_error("Occupancy greater than total number of walkers N=" + std::to_string(N) + ", but occupancy[" + std::to_string(i) + "]=" + std::to_string(occupancy[i]));
		}

		// Generate a random bias (the expensive part in this algorithm)
		double bias = beta_dist(gen, params);

		// Only keeping this b/c once we accept functions need to check limits
		if (bias < 0.0 || bias > 1.0) {
			throw std::runtime_error("Biases must satisfy 0 <= biases <= 1");
		}

		rightShift = getRightShift(occupancy[i], bias, smallCutoff, largeCutoff);

		if (rightShift > occupancy[i]){
			throw std::runtime_error("Right shift cannot be larger than occupancy, but " + std::to_string(rightShift) + " > " + std::to_string(occupancy[i]));
		}

		if (rightShift < 0){
			throw std::runtime_error("Right shift cannot be less than zero, but rightShift = " + std::to_string(rightShift));
		}

		if (i == prevMinIndex) {
			occupancy[i] = occupancy[i] - rightShift;
			leftShift = rightShift;
			if (occupancy[i] != 0) {
				minEdge = i;
				firstNonzero = false;
			}
			continue;
		}

		occupancy[i] = occupancy[i] - rightShift + leftShift;
		leftShift = rightShift;

		if (occupancy[i] != 0) {
			if (firstNonzero) {
				minEdge = i;
				firstNonzero = false;
			}
			maxEdge = i;
		}

		if (i == prevMaxIndex) {
			occupancy[i + 1] = rightShift;
			if (occupancy[i + 1] != 0) {
				maxEdge = i + 1;
			}
		}
	}

	if (minEdge > maxEdge) {
		throw std::runtime_error("Minimum edge is greater than maximum edge. Something went wrong.");
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

// So this needs to be fixed ASAP. Currently have N=0 in the function call b/c
// As the overloaded function floatEvolveTimeStep is written it's ambigious which
// function to defer to. This is b/c smallCutoff and N are now both doubles
// so they're ambigious?
std::pair<std::pair<unsigned long int, unsigned long int>, std::vector<double> > pyfloatEvolveTimestep(
	std::vector<double> occupancy,
	const double beta,
	const unsigned long int prevMinIndex,
	const unsigned long int prevMaxIndex,
	const double N,
	const double smallCutoff=pow(2, 53)-2,
	const double largeCutoff=pow(10, 31)
)
{

	std::pair<unsigned long int, unsigned long int> edges = floatEvolveTimeStep(occupancy, beta, prevMinIndex, prevMaxIndex, N, smallCutoff, largeCutoff);
	std::pair<std::pair<unsigned long int, unsigned long int>, std::vector<double> > returnVal(edges, occupancy);
	return returnVal;
}

// Evolve an occupancy through multiple timesteps and return history of the edges.
std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > evolveTimesteps(
	const unsigned long int timesteps,
	std::vector<double> & occupancy,
	const double beta,
	const double prevMinIndex,
	const double prevMaxIndex,
	const double N,
	const double smallCutoff=pow(2,53)-2,
	const double largeCutoff=pow(10, 31)
)
{
	std::vector<unsigned long int> minEdges(N);
	std::vector<unsigned long int> maxEdges(N);
	minEdges[0] = prevMinIndex;
	maxEdges[0] = prevMaxIndex;

	for (unsigned long int i = 0; i<timesteps; i++){
		std::pair<unsigned int, unsigned int> edges = floatEvolveTimeStep(
			occupancy,
			beta,
			minEdges[i],
			maxEdges[i],
			N,
			smallCutoff,
			largeCutoff);
		minEdges[i] = edges.first;
		maxEdges[i] = edges.second;
	}

	std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edgesHistory(minEdges, maxEdges);
	return edgesHistory;
}

// Initializes an array of size N and initializes the occupancy to have N walkers
// at index 0. Then evolves the occupancy for N timesteps.
std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > initializeAndEvolveTimesteps(
	const unsigned long int N,
	const double beta,
	const double smallCutoff=pow(2,53)-2,
	const double largeCutoff=pow(10, 31)
)
{
	std::vector<unsigned long int> minEdge(N);
	std::vector<unsigned long int> maxEdge(N);
	std::vector<double> occ(N);
	occ[0] = N;

	std::pair<unsigned int, unsigned int> edges(0, 1);
	for (unsigned long int i=0; i != N; i++){
		edges = floatEvolveTimeStep(occ, beta, edges.first, edges.second, N, smallCutoff, largeCutoff);
		minEdge[i] = edges.first;
		maxEdge[i] = edges.second;
	}

	std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edgesHistory(minEdge, maxEdge);
	return edgesHistory;
}

// Class to take make a diffusion experiment easier. All the date is handled on
// the C++ side of things so that the occupancy array is only called when python
// calls for an array.
class Diffusion{
	private:
		std::vector<double> occupancy;
		double N;
		double beta;
		double smallCutoff;
		double largeCutoff;
		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edges;

	public:
		Diffusion(const double numberOfParticles, const double b, const double scutoff, const double lcutoff){
			N = numberOfParticles;
			beta = b;
			smallCutoff = scutoff;
			largeCutoff = lcutoff;
			edges.first.push_back(0), edges.second.push_back(1);
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
			return beta;
		}

		void setBeta(const double b){
			beta = b;
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

		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > getEdges(){
			return edges;
		}

		// Move the occupancy forward one step in time. Requires push_back for the edges
		// which might slow it down a lot.
		void iterateTimestep(){
			unsigned long int minIndex = edges.first.back();
			unsigned long int maxIndex = edges.second.back();
			std::pair<unsigned long int, unsigned long int> newEdges = floatEvolveTimeStep(occupancy, beta, minIndex, maxIndex, N, smallCutoff, largeCutoff);
			edges.first.push_back(newEdges.first);
			edges.second.push_back(newEdges.second);
		}

		// Move the ocupancy forward N steps in time. Edges extended initially so
		// we avoid push_back one by one so it should be faster than running
		// iterateTimestep N times.
		void evolveTimesteps(const unsigned int iterations){
			unsigned int edgesLength = edges.first.size();
			edges.first.resize(iterations + edgesLength);
			edges.second.resize(iterations + edgesLength);

			for (unsigned long int i = edgesLength-1; i < edges.first.size()-1; i++){
				unsigned long int minIndex = edges.first[i];
				unsigned long int maxIndex = edges.second[i];
				std::pair<unsigned long int, unsigned long int> newEdges = floatEvolveTimeStep(occupancy, beta, minIndex, maxIndex, N, smallCutoff, largeCutoff);
				edges.first[i+1] = newEdges.first;
				edges.second[i+1] = newEdges.second;
			}
		}

		unsigned int findNumberParticles(){
			double sum = std::accumulate(occupancy.begin(), occupancy.end(), 0);
			return sum;
		}
};

PYBIND11_MODULE(cDiffusion, m){
	m.doc() = "C++ diffusion";

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
				py::arg("prevMaxIndex"), py::arg("N"), py::arg("smallCutoff")=pow(2, 53)-2,
				py::arg("largeCutoff")=pow(10, 31));

	const char * evolveTimestepsdoc = R"V0G0N(
Evolve the occupancy forward through N numbers of timesteps.
)V0G0N";

	m.def("evolveTimesteps", &evolveTimesteps, evolveTimestepsdoc,
				py::arg("timesteps"), py::arg("occupancy"), py::arg("beta"),
				py::arg("prevMinIndex"), py::arg("prevMaxIndex"), py::arg("N"),
				py::arg("smallCutoff")=pow(2, 53)-2, py::arg("largeCutoff")=pow(10, 31));

	const char * iterateTimestepdoc = R"V0G0N(
Move the occupancy forward through one timestep. Appends the new edge positions
to the edges vector.

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

Note
----
Much faster than iterateTimestep method because it preallocates needed space in
edges vector.
)V0G0N";

	const char * findNumberParticlesdoc = R"V0G0N(
Sum over occupancy to find the current number of particles.
)V0G0N";

	py::class_<Diffusion>(m, "Diffusion")
		.def(py::init<const double, const double, const double, const double>())
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
		.def("iterateTimestep", &Diffusion::iterateTimestep, iterateTimestepdoc)
		.def("evolveTimesteps", &Diffusion::evolveTimesteps, classevolveTimestepsdoc, py::arg("iterations"))
		.def("findNumberParticles", &Diffusion::findNumberParticles, findNumberParticlesdoc);
}
