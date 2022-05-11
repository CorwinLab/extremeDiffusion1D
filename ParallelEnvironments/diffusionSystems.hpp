#include <vector>
#include <random>
#include <boost/random.hpp>

// So we can switch to quads if we want to
using RealType = double; 

class AllSystems{
private: 
    double beta;
    unsigned int numSystems;
    RealType numWalkers;
    unsigned int xMax;

    unsigned int time;
    unsigned int minPos;
    unsigned int maxPos;

    // Each element will be a specific system's occupancy
    std::vector<std::vector<RealType> > occupancy; 

    // Calculate number of particles that should move to next site 
    // at position x for system i
    double toNextSitePosition(RealType numWalkers, double bias);

    // Random number overhead 
    std::random_device rd;
    boost::random::mt19937_64 gen;
    std::uniform_real_distribution<> dis;
    boost::random::binomial_distribution<> binomial;

    // Set precision cutoffs
    double smallCutoff = pow(2, 31) - 2;
    double largeCutoff = 1e64;

public:
    AllSystems(const unsigned int _numSystems, const double _beta, const unsigned int xMax, const RealType numWalkers);
    ~AllSystems(){};

    double getBeta(){ return beta; };
    std::vector<std::vector<RealType> > getOccupancy(){ return occupancy; };

    // Move the system forward one step in time
    void iterateTimeStep();
    
    // Makes a random value from a generator. Could be const, uniform or beta
    double generateRandomNumber();

};