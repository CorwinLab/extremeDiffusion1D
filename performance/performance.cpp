#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <time.h>
#include <boost/random.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <random>
#include <utility>
#include <assert.h>

std::random_device rd;
boost::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
boost::random::binomial_distribution<> binomial;
boost::random::normal_distribution<> normal;
boost::random::beta_distribution<> beta_dist;
const double smallCutoff = pow(2, 31) - 2;
const double largeCutoff = 1e31;

double generateBeta(double beta){
	// If beta = 0 return either 0 or 1
	if (beta == 0.0){
		return round(dis(gen));
	}
	// If beta = 1 use random uniform distribution
	else if (beta == 1.0){
		return dis(gen);
	}
	else{
		boost::random::beta_distribution<>::param_type params(beta, beta);
		return beta_dist(gen, params);
	}
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
		return (occ * bias);
	}

	if (bias <= 0.000001){
		return (occ * bias);
	}

	if (occ < smallCutoff) {
		// If small enough to use integer representations use binomial distribution
	  return binomial(gen, boost::random::binomial_distribution<>::param_type(occ, bias));
	}
	else if (occ > largeCutoff) {
		// If so large that sqrt(N) is less than precision just use occupancy
		return (occ * bias);
	}
	else {
		// If in sqrt(N) precision use gaussian approximation.
		double mediumVariance = sqrt(occ * bias * (1 - bias));
		return (normal(gen, boost::random::normal_distribution<>::param_type(occ * bias, mediumVariance)));
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

// Get the Nth quartile of the occupancy vector. The minIdx and maxIdx help
// narrow down the search range.
double getNthquartileDoublesided(
		const std::vector<double>& occupancy,
		const double centerIdx,
		const unsigned long int minIdx,
		const unsigned long int maxIdx,
		const double N
)
{
	double right_dist = maxIdx - centerIdx;
	double left_dist = centerIdx - minIdx;
	double dist;

	unsigned long int right_idx = maxIdx;
	unsigned long int left_idx = minIdx;

	double sum = 0;

	while (sum < N){
		if (right_dist >= left_dist){
			dist = right_dist;
			sum += occupancy[right_idx];
			right_idx -= 1;
			right_dist -= 1.0;
		}
		else{
			dist = left_dist;
			sum += occupancy[left_idx];
			left_idx += 1;
			left_dist -= 1.0;
		}
	}
	return dist;
}

double getNthquartileSingleSided(
	const std::vector<double>& occupancy,
	const double centerIdx,
	unsigned long int maxIdx,
	const double N
)
{
	double dist = maxIdx - centerIdx;
	double sum = occupancy[maxIdx];
	while (sum < N){
		maxIdx -= 1;
		dist -= 1;
		sum += occupancy[maxIdx];
	}
	return dist;
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

		double getNthquartile(const double N){
			unsigned long int maxEdge = edges.second[time];
			return getNthquartileSingleSided(occupancy, time * 0.5, maxEdge, N);
		}
};

int main(){
	double N = 1e10;
	Diffusion d = Diffusion(N, 1, 0, 0);
	d.initializeOccupationAndEdges();
	unsigned long int num_of_steps = d.getLogN52();
	d.evolveTimesteps(num_of_steps, true);
	auto edges = d.getEdges();
}
