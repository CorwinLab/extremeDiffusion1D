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

unsigned long int getRightShift(const unsigned int occ, const double bias, const unsigned int smallCutoff) {
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

std::pair<unsigned int, unsigned int> floatEvolveTimeStep(std::vector<int> &occupancy, 
															const double beta,
															const int smallCutoff,
															unsigned int minEdgeBound, 
															unsigned int maxEdgeBound)
{
	// Iterate one time step according to Barraquad/Corwin model
	// Pass in a beta value or even function to generate biases
	// Changes occupancy vector itself rather than creating a new vector
	
	// Need to pre-allocate size of occupancy and then only sum over min/max edges
	
	// Need to set smallCutoff to double precision cutoff = 2^53

	// Throws: Error if occupancy is less than 0, Error if biases does not satisfy 0 <= biases <= 1, 
	// Error if minEdge > maxEdge -> Could we make this a hard cutoff or no?

	if (minEdgeBound > maxEdgeBound) {
		throw std::runtime_error("Minimum edge must be greater than maximum edge");
	}

	occupancy.push_back(0);

	// If we keep the occupancy the same throughout the whole experiment we probably only need
	// to construct this distribution once but w/e
	boost::random::beta_distribution<> betaDist(1, beta);

	unsigned long int leftShift = 0;
	unsigned long int rightShift = 0;
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

		if (occupancy[i] < 0) {
			throw std::runtime_error("Occupancy must be > 0");
		}

		//Then generate a random bias (the expensive part in this algorithm)
		double bias = betaDist(gen);

		// Only keeping this b/c once we accept functions need to check limits
		if (bias < 0.0 || bias > 1.0) {
			throw std::runtime_error("Biases must satisfy 0 <= biases <= 1");
		}

		rightShift = getRightShift(occupancy[i], bias, smallCutoff);

		if (i == 0) {
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

		if (i == (occupancy.size() - 1)) {
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

bool checksum(std::vector<int> arr1, std::vector<int> arr2) {
	int sum1 = 0; 
	std::accumulate(arr1.begin(), arr1.end(), sum1);

	int sum2 = 0; 
	std::accumulate(arr2.begin(), arr2.end(), sum2);
	return sum1 == sum2; 
}

void editArray(std::vector<int> &occ) {
	// Pass by reference to change the actual values of the array
	for (int i = 0; i < occ.size(); i++) {
		occ[i] = 1; 
	}
}

int main()
{
	std::vector<float> biases = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	std::vector<int> occupation = { 10, 12, 20, 0, 0, 5, 0, 0 };
	int occSum = 10 + 12 + 20 + 5; 
	std::pair<size_t, size_t> edges = floatEvolveTimeStep(occupation, 5, 0, 1, 4);
	print_generic(occupation);
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

