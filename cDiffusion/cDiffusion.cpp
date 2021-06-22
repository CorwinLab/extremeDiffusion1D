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
// EC: Do you need this real distribution?
std::uniform_real_distribution<> dis(0.0, 1.0);
boost::random::binomial_distribution<> binomial;
boost::random::normal_distribution<> normal;
boost::random::beta_distribution<> beta_dist;

// EC: I think that there's a better way to print a vector: https://stackoverflow.com/questions/10750057/how-do-i-print-out-the-contents-of-a-vector/11335634#11335634
// std::copy(vec.begin(), vec.end(), std::ostream_iterator<char>(std::cout, " "));
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
double getRightShift(
	const double occ,
	const double bias,
	const double smallCutoff,
	const double largeCutoff,
)
{

	if (occ < smallCutoff) {
		// If small enough to use integer representations use binomial distribution
	  return binomial(gen, boost::random::binomial_distribution<>::param_type params(occ, bias) );
	} else if (occ < largeCutoff) {
		// If sqrt(N) is within our precision
		double mediumStd = sqrt(occ * bias * (1 - bias));
		return round( normal( gen, boost::random::normal_distribution<>::param_type params(occ * bias, mediumStd) ) )
	} else {
		// If N is so large the sqrt(N) is less than precision
		return round(occ*bias);
	}

	// I would put this error check on the output of getRightShift, this prevents needing the temporary variable
	// if (rightShift < 0){
	// 	throw std::runtime_error("Right shift = " + std::to_string(rightShift) + " for occupancy=" + std::to_string(occ) + ", bias=" + std::to_string(bias) + ", smallCutoff=" + std::to_string(smallCutoff));
	// }

}

// Iterate one time step according to Barraquad/Corwin model
// Note: (alpha=1, beta=1) gives uniform distribution.
std::pair<unsigned long int, unsigned long int> floatEvolveTimeStep(
	std::vector<double>& occupancy,
	const double beta,
	const unsigned long int minEdgeIndex, //Call this prevMinEdge
	const unsigned long int maxEdgeIndex,
	const double N,
	const double smallCutoff = 2147483646, // This is 2^31-2, which seemed to be the largest number that would work for me?
	const double largeCutoff = 1e31,
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
		//EC: Does this ever actually happen?  It would probably be faster to simply pre-allocate the occupancy array to be the right length
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
	for (auto i = minEdgeIndex; i < maxEdgeIndex+1; i++) {

		// Skip over occupation value if ocuppancy=0 and not moving any walkers
		// to the position
		if (occupancy[i] == 0 && leftShift == 0) {
			continue;
		}

		//Too much error checking
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

		rightShift = getRightShift(occupancy[i], bias, smallCutoff);

		if (rightShift > occupancy[i]){
			throw std::runtime_error("Right shift cannot be larger than occupancy, but " + std::to_string(rightShift) + " > " + std::to_string(occupancy[i]));
		}

		if (rightShift < 0){
			throw std::runtime_error("Right shift cannot be less than zero, but rightShift = " + std::to_string(rightShift));
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

	std::pair<unsigned long int, unsigned long int> edges(minEdge, maxEdge);

	return edges;
}

// Overloaded so that N can be obtained by simply summing the occupancy. This may
// take a bit longer depending on the size of the occupancy vector due to summing
// the array beforehand.
std::pair<unsigned long int, unsigned long int> floatEvolveTimeStep(
	std::vector<double> &occupancy,
	const double beta,
	const unsigned long int minEdgeIndex,
	const unsigned long int maxEdgeIndex,
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

// So this needs to be fixed ASAP. Currently have N=0 in the function call b/c
// As the overloaded function floatEvolveTimeStep is written it's ambigious which
// function to defer to. This is b/c smallCutoff and N are now both doubles
// so they're ambigious?
std::pair<std::pair<unsigned long int, unsigned long int>, std::vector<double> > pyfloatEvolveTimestep(
	std::vector<double> occupancy,
	const double beta,
	const unsigned long int minEdgeIndex,
	const unsigned long int maxEdgeIndex,
	const double smallCutoff=pow(2, 53)
)
{

	auto edges = floatEvolveTimeStep(occupancy, beta, minEdgeIndex, maxEdgeIndex, 0, smallCutoff);
	std::pair<std::pair<unsigned long int, unsigned long int>, std::vector<double> > returnVal(edges, occupancy);
	return returnVal;
}

// Evolve an occupancy through multiple timesteps and return history of the edges.
std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > evolveTimesteps(
	const unsigned long int timesteps,
	std::vector<double> & occupancy,
	const double beta,
	const double minEdgeIndex,
	const double maxEdgeIndex,
	const double N,
	const double smallCutoff=pow(2,53)
)
{
	std::vector<unsigned long int> minEdges(N);
	std::vector<unsigned long int> maxEdges(N);
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

	std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edgesHistory(minEdges, maxEdges);
	return edgesHistory;
}

// Initializes an array of size N and initializes the occupancy to have N walkers
// at index 0. Then evolves the occupancy for N timesteps.
std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > initializeAndEvolveTimesteps(
	const unsigned long int N,
	const double beta,
	const double smallCutoff=pow(2,53)
)
{
	std::vector<unsigned long int> minEdge(N);
	std::vector<unsigned long int> maxEdge(N);
	std::vector<double> occ(N);
	occ[0] = N;

	std::pair<unsigned int, unsigned int> edges(0, 1);
	for (unsigned long int i=0; i != N; i++){
		edges = floatEvolveTimeStep(occ, beta, edges.first, edges.second, N, smallCutoff);
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
		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edges;

	public:
		Diffusion(const double numberOfParticles, const double b, const double cutoff){
			N = numberOfParticles;
			beta = b;
			smallCutoff = cutoff;
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

		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > getEdges(){
			return edges;
		}

		// Move the occupancy forward one step in time. Requires push_back for the edges
		// which might slow it down a lot.
		void iterateTimestep(){
			unsigned long int minIndex = edges.first.back();
			unsigned long int maxIndex = edges.second.back();
			std::pair<unsigned long int, unsigned long int> newEdges = floatEvolveTimeStep(occupancy, beta, minIndex, maxIndex, N, smallCutoff);
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

			for (unsigned long int i = 0edgesLength-1; i < edges.first.size()-1; i++){
				unsigned long int minIndex = edges.first[i];
				unsigned long int maxIndex = edges.second[i];
				std::pair<unsigned long int, unsigned long int> newEdges = floatEvolveTimeStep(occupancy, beta, minIndex, maxIndex, N, smallCutoff);
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

	)V0G0N"

	py::class_<Diffusion>(m, "Diffusion")
		.def(py::init<const double, const double, const double>())
		.def("getOccupancy", &Diffusion::getOccupancy)
		.def("setOccupancy", &Diffusion::setOccupancy, py::arg("occupancy"))
		.def("getN", &Diffusion::getN)
		.def("getBeta", &Diffusion::getBeta)
		.def("setBeta", &Diffusion::setBeta, py::arg("beta"))
		.def("getsmallCutoff", &Diffusion::getsmallCutoff)
		.def("setsmallCutoff", &Diffusion::setsmallCutoff, py::arg("smallCutoff"))
		.def("getEdges", &Diffusion::getEdges)
		.def("iterateTimestep", &Diffusion::iterateTimestep, iterateTimestepdoc)
		.def("evolveTimesteps", &Diffusion::evolveTimesteps, classevolveTimestepsdoc, py::arg("iterations"))
		.def("findNumberParticles", &Diffusion::findNumberParticles);
}
