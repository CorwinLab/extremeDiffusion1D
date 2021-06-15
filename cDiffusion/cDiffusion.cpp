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

std::pair<std::pair<unsigned int, unsigned int>, std::vector<int>> floatEvolveTimeStep(std::vector<int> occupancy, double beta,
																						const int smallCutoff) 
{
	// Iterate one time step according to Barraquad/Corwin model

	// Pass in a beta value or even function to generate biases

	// Use single occupancy vector rather than doubling the size of the occupancy
	// Pre-allocate the occupancy vector to size N

	// Look up best practices w/ standard vectors.

	// Throws: Error if occupancy is less than 0, Error if biases does not satisfy 0 <= biases <= 1

	boost::random::beta_distribution<> betaDist(1, beta);
	std::vector<int> newOccupancy(occupancy.size() + 1);
	unsigned long int leftShift = 0;
	unsigned long int rightShift = 0;
	
	// Use unsigned long instead of size_t to be consistent
	unsigned int minEdge = 0;
	unsigned int maxEdge = 0;
	bool firstNonzero = true;

	// Now can use an iterator? 
	// Check out range operator
	// Use a switch statement instead of all the tests
	for (auto i = 0; i != occupancy.size(); i++) {

		// So, this becomes a problem later on if the occupancy has a bunch of zeros at the end.
		// Need to make sure that the left shift gets set to zero at the end of the "line".
		// I think it does - it's just going to run one extra time to set it to zero. 
		if (occupancy[i] == 0 && leftShift == 0) {
			continue;
		}

		double bias = betaDist(gen);

		if (occupancy[i] < 0) {
			throw std::runtime_error("Occupancy must be > 0");
		}

		if (bias < 0.0 || bias > 1.0) {
			throw std::runtime_error("Biases must satisfy 0 <= biases <= 1");
		}

		if (occupancy[i] < smallCutoff) {
			// If small enough to use integer representations use binomial distribution
			boost::random::binomial_distribution<int> distribution(occupancy[i], bias);
			rightShift = distribution(gen);
		}
		else if (occupancy[i] > pow(smallCutoff, 2)) {
			// If so large that sqrt(N) is less than precision just use occupancy
			rightShift = lround(occupancy[i] * bias);
		}
		else {
			// If in sqrt(N) precision use gaussian approximation. 
			double mediumVariance = occupancy[i] * bias * (1 - bias);
			mediumVariance = sqrt(mediumVariance);
			boost::random::normal_distribution<> distribution(occupancy[i] * bias, mediumVariance);
			rightShift = lround(distribution(gen));
		}

		if (i == 0) {
			newOccupancy[i] = occupancy[i] - rightShift;
			leftShift = rightShift;
			if (newOccupancy[i] != 0) {
				minEdge = i;
				firstNonzero = false;
			}
			continue;
		}

		newOccupancy[i] = occupancy[i] - rightShift + leftShift;
		leftShift = rightShift;

		if (newOccupancy[i] != 0) {
			if (firstNonzero) {
				minEdge = i;
				firstNonzero = false;
			}
			maxEdge = i;
		}

		if (i == (occupancy.size() - 1)) {
			newOccupancy[i + 1] = rightShift;
			if (newOccupancy[i + 1] != 0) {
				maxEdge = i + 1;
			}
		}
	}

	std::pair<unsigned int, unsigned int> edges(minEdge, maxEdge);
	std::pair<std::pair<unsigned int, unsigned int>, std::vector<int>> returnVal(edges, newOccupancy);

	return returnVal;
}

int main()
{
	std::vector<float> biases = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	std::vector<int> occupation = { 10, 12, 20, 0, 0, 5, 0, 0 };
	std::pair<std::pair<size_t, size_t>, std::vector<int>> newOccupation = floatEvolveTimeStep(occupation,5, 4);
	print_generic(newOccupation.second);
	std::pair<std::pair<size_t, size_t>, std::vector<int>> newOcc2 = floatEvolveTimeStep(occupation,5, 4);
	print_generic(newOcc2.second);

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

