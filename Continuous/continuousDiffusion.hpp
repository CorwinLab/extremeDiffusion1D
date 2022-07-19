#include <vector> 
#include <random>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

using RealType = double; 

class Particle{
private: 
    RealType x;

public:
    Particle(const RealType _x);
    ~Particle(){};

    RealType getX(){ return x; };
    void setX(RealType _x){ x = _x; };
};

class System{
private:
    std::vector<Particle> particles; 
    unsigned int time; 
    unsigned int numParticles;

    std::random_device rd;
    boost::random::mt19937_64 gen;

    boost::random::normal_distribution<> normal_dist;
    boost::random::normal_distribution<>::param_type normalParams;

public: 
    System(const unsigned int _numParticles);
    ~System(){};

    double generateRandomDistance();

    std::vector<Particle> getParticles(){ return particles; };
    std::vector<RealType> getParticlePositions();
    unsigned int getNumParticles(){ return numParticles; };
    RealType getMaximumParticle();

    void iterateTimeStep();
};
