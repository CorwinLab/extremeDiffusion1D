// cDiffusion.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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

unsigned long int getRightShift(const unsigned int occ, const double bias,
																const unsigned int smallCutoff) {
	// Generate left and right shift based on small cutoff
	unsigned long int rightShift = 0;
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

std::pair<std::vector<int>, std::vector<int> > evolveTimesteps(int N){
	// Returns list of edges
}

// Does the same thing as floatEvolveTimeStep but returns the occupancy
// This is because passing by reference doesn't work easily with Pybind11.
// I think every time we pass a vector between C++ and Python it's copied.

std::pair<unsigned int, unsigned int> floatEvolveTimeStep(
	std::vector<int> &occupancy,
	const double beta,
	const unsigned int minEdgeIndex,
	const unsigned int maxEdgeIndex,
	const unsigned long int smallCutoff=pow(2,53)
)
{
	// Iterate one time step according to Barraquad/Corwin model
	// Pass in a beta value or even function to generate biases
	// Changes occupancy vector itself rather than creating a new vector

	// Need to set smallCutoff to double precision cutoff = 2^53

	// Throws: Error if occupancy is less than 0, Error if biases does not satisfy 0 <= biases <= 1,
	// Error if minEdge > maxEdge -> Could we make this a hard cutoff or no?

	// Note: (alpha=1, beta=1) gives uniform distribution.

	if (minEdgeIndex > maxEdgeIndex) {
		throw std::runtime_error("Minimum edge must be greater than maximum edge");
	}

	// Need to check when maxEdgeIndex breaks the for loop.
	if ((maxEdgeIndex+1) > occupancy.size()){
		throw std::runtime_error("Maximum edge exceeds size of array");
	}

	// Pushback the occupancy if we're iterating through the whole array
	if ((maxEdgeIndex+1) == occupancy.size()){
		occupancy.push_back(0);
	}

	// If we keep the occupancy the same throughout the whole experiment we probably
	// only need to construct this distribution once but w/e
	boost::random::beta_distribution<> betaDist(1, beta);

	unsigned long int leftShift = 0;
	unsigned long int rightShift = 0;
	unsigned int minEdge = 0;
	unsigned int maxEdge = 0;
	bool firstNonzero = true;

	// Now can use an iterator?
	// Check out range operator
	for (auto i = minEdgeIndex; i != maxEdgeIndex+1; i++) {

		// Skip over occupation value if ocuppancy=0 and not moving any walkers
		// to the position
		if (occupancy[i] == 0 && leftShift == 0) {
			continue;
		}

		if (occupancy[i] < 0) {
			throw std::runtime_error("Occupancy must be > 0");
		}

		// Generate a random bias (the expensive part in this algorithm)
		double bias = betaDist(gen);

		// Only keeping this b/c once we accept functions need to check limits
		if (bias < 0.0 || bias > 1.0) {
			throw std::runtime_error("Biases must satisfy 0 <= biases <= 1");
		}

		rightShift = getRightShift(occupancy[i], bias, smallCutoff);

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

	std::pair<unsigned int, unsigned int> edges(minEdge, maxEdge);

	return edges;
}

std::pair<std::pair<unsigned int, unsigned int>, std::vector<int> > pyfloatEvolveTimestep(
	std::vector<int> occupancy,
	const double beta,
	const unsigned int minEdgeIndex,
	const unsigned int maxEdgeIndex,
	const unsigned long int smallCutoff=pow(2, 53)
)
{
	std::pair<unsigned int, unsigned int> edges = floatEvolveTimeStep(occupancy, beta, minEdgeIndex, maxEdgeIndex, smallCutoff);
	std::pair<std::pair<unsigned int, unsigned int>, std::vector<int> > returnVal(edges, occupancy);
	return returnVal;
}

bool checksum(std::vector<int> arr1, std::vector<int> arr2) {
	int sum1 = 0;
	std::accumulate(arr1.begin(), arr1.end(), sum1);

	int sum2 = 0;
	std::accumulate(arr2.begin(), arr2.end(), sum2);
	return sum1 == sum2;
}

PYBIND11_MODULE(cDiffusion, m){
	m.doc() = "C++ diffusion";
	m.def("floatEvolveTimeStep", &pyfloatEvolveTimestep, "Iterate a step",
				py::arg("occupancy"), py::arg("beta"), py::arg("minEdgeIndex"),
				py::arg("maxEdgeIndex"), py::arg("smallCutoff"));
}

int main()
{
	std::vector<int> occupation = { 10, 12, 20, 0, 0, 5, 0, 0 };
	int occSum = 10 + 12 + 20 + 5;
	std::pair<size_t, size_t> edges = floatEvolveTimeStep(occupation, 1, 0, 0, occupation.size());
	print_generic(occupation);
	int sum = 0;
	sum = accumulate(occupation.begin(), occupation.end(), sum);
	std::cout << sum << "\n";
	std::cout << edges.first << " " << edges.second << "\n";
	return 0;
}
