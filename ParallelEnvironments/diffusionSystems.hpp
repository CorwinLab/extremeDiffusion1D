#include <vector>
#include <random>
#include <boost/random.hpp>
#include <iostream>

// So we can switch to quads if we want to
using RealType = double; 

class System{
private: 
    unsigned int minPos; 
    unsigned int maxPos;
    RealType numMoveRight;


public:
    System(const unsigned int xMax, const RealType numWalkers);
    ~System(){};
    
    RealType getNumMoveRight(){ return numMoveRight; };
    void setNumMoveRight(RealType _numMoveRight){ numMoveRight = _numMoveRight;}

    // Just going to make occupancy public b/c going to need to access 
    // elements often
    std::vector<RealType> occupancy;
    unsigned int time;

    unsigned int getMinPos(){ return minPos; };
    unsigned int getMaxPos(){ return maxPos; }; 

    void setMinPos(unsigned int _minPos){ minPos = _minPos; };
    void setMaxPos(unsigned int _maxPos){ maxPos = _maxPos; };

    RealType& at(unsigned int i){ return occupancy.at(i); };
    unsigned int getTime(){ return time; };
    void setTime(unsigned int _time){ time = _time; };

    friend std::ostream& operator<<(std::ostream& os, const System& sys);
};

class AllSystems{
private: 
    double beta;
    unsigned int numSystems;
    RealType numWalkers;
    unsigned int xMax;

    unsigned int time;
    unsigned int minPos;
    unsigned int maxPos;

    // Each element will be a specific System
    std::vector<System> systems; 

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
    unsigned int getXMax(){ return xMax; };
    std::vector<System> getSystems(){ return systems; };
    unsigned int getMinPos(){ return minPos; };
    unsigned int getMaxPos(){ return maxPos; };
    unsigned int getTime(){ return time; };
    unsigned int getNumSystems(){ return numSystems; };

    // Move the system forward one step in time
    void iterateTimeStep(std::vector<unsigned int> sysIDs);
    
    // Makes a random value from a generator. Could be const, uniform or beta
    double generateRandomNumber();

    std::vector<std::vector<unsigned int> > measureFirstPassageTimes(std::vector<RealType> distances);
};
