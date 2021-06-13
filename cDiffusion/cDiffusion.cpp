// cDiffusion.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <time.h>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <random>

std::mt19937 gen(2);

std::vector<int> generateUniformRandom(int const N) {
	// Generate N random numbers between 0 and 100 according to uniform distribution. 
	std::vector<int> vec(N);
	for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); it++) {
		*it = rand() % 100;
	}
	return vec;
}

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

std::vector<int> floatEvolveTimeStep(std::vector<int> occupancy, std::vector<float> biases, int smallCutoff) {
	// Iterate one time step according to Barraquad/Corwin model
	// TODO: Return edges of the timestep as well. 
	// TODO: Handle shape error of biases and occupancy on python side of things
	// Throws: Error if occupancy is less than 0, Error if biases does not satisfy 0 <= biases <= 1

	std::vector<int> newOccupancy(occupancy.size() + 1);
	float leftShift = 0;
	float rightShift = 0;

	int minEdge = 0;
	int maxEdge = 0;
	bool firstNonzero = true;
	for (size_t i = 0; i != occupancy.size(); i++) {

		// So, this becomes a problem later on if the occupancy has a bunch of zeros at the end.
		// Need to make sure that the left shift gets set to zero at the end of the "line".
		// I think it does - it's just going to run one extra time to set it to zero. 
		if (occupancy[i] == 0 && leftShift == 0) {
			continue;
		}

		if (occupancy[i] < 0) {
			throw std::runtime_error("Occupancy must be > 0");
		}

		if (biases[i] < 0.0 || biases[i] > 1.0) {
			throw std::runtime_error("Biases must satisfy 0 <= biases <= 1");
		}

		if (occupancy[i] < smallCutoff) {
			// If small enough to use integer representations use binomial distribution
			boost::random::binomial_distribution<int> distribution(occupancy[i], biases[i]);
			rightShift = distribution(gen);
		}
		else if (occupancy[i] > pow(smallCutoff, 2)) {
			// If so large that sqrt(N) is less than precision just use occupancy
			rightShift = round(occupancy[i] * biases[i]);
		}
		else {
			// If in sqrt(N) precision use gaussian approximation. 
			float mediumVariance = occupancy[i] * biases[i] * (1 - biases[i]);
			mediumVariance = sqrt(mediumVariance);
			boost::random::normal_distribution<> distribution(occupancy[i] * biases[i], mediumVariance);
			rightShift = round(distribution(gen));
		}
		std::cout << rightShift << "\n";

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
	std::cout << minEdge << "\n" << maxEdge << "\n";
	return newOccupancy;
}

int main()
{
	std::vector<float> biases = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	std::vector<int> occupation = { 10, 12, 20, 0, 0, 5, 0, 0 };
	std::vector<int> newOccupation = floatEvolveTimeStep(occupation, biases, 4);
	print_generic(newOccupation);
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

