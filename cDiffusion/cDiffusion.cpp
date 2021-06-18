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
// may breakdown causing the (# particles right shifted) > (# of particles)
unsigned long int getRightShift(const double occ, const double bias,
																const double smallCutoff) {
	double rightShift = 0;
	if (occ < smallCutoff) {
		// If small enough to use integer representations use binomial distribution
		boost::random::binomial_distribution<int> distribution(occ, bias);
		rightShift = distribution(gen);
	}
	else if (occ > pow(smallCutoff, 2)) {
		// If so large that sqrt(N) is less than precision just use occupancy
		rightShift = lround(occ * bias);
	}
	else {
		// If in sqrt(N) precision use gaussian approximation.
		double mediumVariance = occ * bias * (1 - bias);
		mediumVariance = sqrt(mediumVariance);
		boost::random::normal_distribution<> distribution(occ * bias, mediumVariance);
		rightShift = lround(distribution(gen));
	}
	return rightShift;
}

// Iterate one time step according to Barraquad/Corwin model
// Note: (alpha=1, beta=1) gives uniform distribution.
std::pair<double, double> floatEvolveTimeStep(
	std::vector<double> &occupancy,
	const double beta,
	const double minEdgeIndex,
	const double maxEdgeIndex,
	const double N,
	const double smallCutoff=pow(2,53)
)
{
	if (minEdgeIndex >= maxEdgeIndex) {
		throw std::runtime_error("Minimum edge must be greater than maximum edge: (" + std::to_string(minEdgeIndex) + ", " + std::to_string(maxEdgeIndex) + ")");
	}

	// Need to check when maxEdgeIndex breaks the for loop.
	if ((maxEdgeIndex+1) > occupancy.size()){
		throw std::runtime_error("Maximum edge exceeds size of vector");
	}

	// If iterating over the whole array extend the occupancy.
	if ((maxEdgeIndex+1) == occupancy.size()){
		occupancy.push_back(0);
	}

	// If we keep the occupancy the same throughout the whole experiment we probably
	// only need to construct this distribution once but w/e
	boost::random::beta_distribution<> betaDist(1, beta);

	double leftShift = 0;
	double rightShift = 0;
	double minEdge = 0;
	double maxEdge = 0;
	bool firstNonzero = true;

	// Now can use an iterator?
	// Check out range operator
	for (auto i = minEdgeIndex; i < maxEdgeIndex+1; i++) {

		// Skip over occupation value if ocuppancy=0 and not moving any walkers
		// to the position
		if (occupancy[i] == 0 && leftShift == 0) {
			continue;
		}

		if (occupancy[i] < 0) {
			throw std::runtime_error("Occupancy must be > 0");
		}

		if (occupancy[i] > N){
			throw std::runtime_error("Occupancy greater than total number of walkers N=" + std::to_string(N));
		}

		// Generate a random bias (the expensive part in this algorithm)
		double bias = betaDist(gen);

		// Only keeping this b/c once we accept functions need to check limits
		if (bias < 0.0 || bias > 1.0) {
			throw std::runtime_error("Biases must satisfy 0 <= biases <= 1");
		}

		rightShift = getRightShift(occupancy[i], bias, smallCutoff);

		if (rightShift > occupancy[i]){
			throw std::runtime_error("Right shift cannot be larger than occupancy, but " + std::to_string(rightShift) + " > " + std::to_string(occupancy[i]));
		}

		if (i == minEdgeIndex) {
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

		if (i == maxEdgeIndex) {
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

	std::pair<double, double> edges(minEdge, maxEdge);

	return edges;
}

// Overloaded so that N can be obtained by simply summing the occupancy. This may
// take a bit longer depending on the size of the occupancy vector due to summing
// the array beforehand.
std::pair<unsigned int, unsigned int> floatEvolveTimeStep(
	std::vector<double> &occupancy,
	const double beta,
	const unsigned int minEdgeIndex,
	const unsigned int maxEdgeIndex,
	const double smallCutoff=pow(2,53)
)
{
	std::cout << "Warning: No number of walkers N provided so summing over occupancy. This may take a lot longer \n";
	double N = accumulate(occupancy.begin(), occupancy.end(), 0);
	return floatEvolveTimeStep(occupancy, beta, minEdgeIndex, maxEdgeIndex, N, smallCutoff);
}

// Does the same thing as floatEvolveTimeStep but returns the occupancy.
// This is because passing by reference doesn't work easily with Pybind11.
// I think every time we pass a vector between C++ and Python it's copied.
std::pair<std::pair<unsigned int, unsigned int>, std::vector<double> > pyfloatEvolveTimestep(
	std::vector<double> occupancy,
	const double beta,
	const unsigned int minEdgeIndex,
	const unsigned int maxEdgeIndex,
	const double smallCutoff=pow(2, 53)
)
{
	std::pair<unsigned int, unsigned int> edges = floatEvolveTimeStep(occupancy, beta, minEdgeIndex, maxEdgeIndex, smallCutoff);
	std::pair<std::pair<unsigned int, unsigned int>, std::vector<double> > returnVal(edges, occupancy);
	return returnVal;
}

// Evolve an occupancy through multiple timesteps and return history of the edges.
std::pair<std::vector<double>, std::vector<double> > evolveTimesteps(
	const unsigned long int timesteps,
	std::vector<double> & occupancy,
	const double beta,
	const double minEdgeIndex,
	const double maxEdgeIndex,
	const double N,
	const double smallCutoff=pow(2,53)
)
{
	std::vector<double> minEdges(N);
	std::vector<double> maxEdges(N);
	minEdges[0] = minEdgeIndex;
	maxEdges[0] = maxEdgeIndex;

	for (unsigned long int i = 0; i<timesteps; i++){
		std::pair<unsigned int, unsigned int> edges = floatEvolveTimeStep(
			occupancy,
			beta,
			minEdges[i],
			maxEdges[i],
			N,
			smallCutoff);
		minEdges[i] = edges.first;
		maxEdges[i] = edges.second;
	}

	std::pair<std::vector<double>, std::vector<double> > edgesHistory(minEdges, maxEdges);
	return edgesHistory;
}

// Initializes an array of size N and initializes the occupancy to have N walkers
// at index 0. Then evolves the occupancy for N timesteps.
std::pair<std::vector<int>, std::vector<int> > initializeAndEvolveTimesteps(
	const unsigned long int N,
	const double beta,
	const double smallCutoff=pow(2,53)
)
{
	std::vector<int> minEdge(N);
	std::vector<int> maxEdge(N);
	std::vector<double> occ(N);
	occ[0] = N;

	std::pair<unsigned int, unsigned int> edges(0, 1);
	for (unsigned long int i=0; i != N; i++){
		edges = floatEvolveTimeStep(occ, beta, edges.first, edges.second, N, smallCutoff);
		minEdge[i] = edges.first;
		maxEdge[i] = edges.second;
	}

	std::pair<std::vector<int>, std::vector<int> > edgesHistory(minEdge, maxEdge);
	return edgesHistory;
}

// Class to take make a diffusion experiment easier. All the date is handled on
// the C++ side of things so that the occupancy array is only called when python
// calls for an array.
class Diffusion{
	private:
		std::vector<double> occupancy;
		unsigned int N;
		double beta;
		double smallCutoff;
		std::pair<unsigned int, unsigned int> edges;

	public:
		Diffusion(const unsigned int numberOfParticles, const double b, const unsigned int cutoff){
			N = numberOfParticles;
			beta = b;
			smallCutoff = cutoff;
			edges.first = 0, edges.second = 1;
		}

		std::vector<double> getOccupancy(){
			return occupancy;
		}

		void setOccupancy(const std::vector<double> occ){
			occupancy = occ;
		}

		unsigned int getN(){
			return N;
		}
		void setN(const unsigned int Number){
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

		std::pair<unsigned int, unsigned int> getEdges(){
			return edges;
		}

		void iterateTimestep(){
			edges = floatEvolveTimeStep(occupancy, beta, edges.first, edges.second, smallCutoff);
		}

		void evolveTimesteps(const unsigned int iterations){
			for (unsigned int i=0; i < iterations; i++){
				Diffusion::iterateTimestep();
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

minEdgeIndex : int
	Index of the first occupied, or nonzero, index in occupation

maxEdgeIndex : int
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
				py::arg("occupancy"), py::arg("beta"), py::arg("minEdgeIndex"),
				py::arg("maxEdgeIndex"), py::arg("smallCutoff")=pow(2, 53));

const char * evolveTimestepsdoc = R"V0G0N(
Evolve the occupancy forward through N numbers of timesteps.
)V0G0N";

	m.def("evolveTimesteps", &evolveTimesteps, evolveTimestepsdoc,
				py::arg("timesteps"), py::arg("occupancy"), py::arg("beta"),
				py::arg("minEdgeIndex"), py::arg("maxEdgeIndex"), py::arg("N"),
				py::arg("smallCutoff")=pow(2, 53));

	py::class_<Diffusion>(m, "Diffusion")
		.def(py::init<const unsigned int, const double, const unsigned int>())
		.def("getOccupancy", &Diffusion::getOccupancy)
		.def("setOccupancy", &Diffusion::setOccupancy)
		.def("getN", &Diffusion::getN)
		.def("getBeta", &Diffusion::getBeta)
		.def("setBeta", &Diffusion::setBeta)
		.def("getsmallCutoff", &Diffusion::getsmallCutoff)
		.def("setsmallCutoff", &Diffusion::setsmallCutoff)
		.def("getEdges", &Diffusion::getEdges);
}
