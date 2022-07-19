#include "continuousDiffusion.hpp"
#include <vector> 
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using RealType = double;
namespace py = pybind11;

std::vector<RealType> arange(RealType minimum, RealType maximum, RealType spacing)
{
    unsigned int numberOfPoints = int((maximum - minimum + spacing) / spacing);
    std::vector<RealType> vec(numberOfPoints);
    for (unsigned int i=0; i<vec.size(); i++){
        vec[i] = minimum + spacing * i;
    }
    return vec;
} 

// Generate a vector of random values drawn uniformally between -1 and 1
std::vector<RealType> getRandomBiasingField(unsigned int numOfValues)
{ 
    std::random_device rd;
    boost::random::mt19937_64 gen;
    std::uniform_real_distribution<> dis;
    std::uniform_real_distribution<>::param_type unifParams(-1.0, 1.0);

    gen.seed(rd());
    dis.param(unifParams);

    std::vector<RealType> randomValues(numOfValues);
    for (unsigned int i=0; i < randomValues.size(); i++){
        randomValues[i] = dis(gen);
    }
    return randomValues;
}

std::vector<RealType> sampleBiasFromBiasingField(std::vector<RealType> positions, RealType corrLength, RealType gridSpacing){
    std::vector<RealType> biases(positions.size());
    sort(positions.begin(), positions.end());

    std::vector<RealType> fieldPositions = arange(positions[0], positions[positions.size()-1], gridSpacing);
    std::vector<RealType> biasingField = getRandomBiasingField(fieldPositions.size());

    unsigned int minIdx = 0; 
    unsigned int maxIdx = fieldPositions.size();

    // first sum over all input particle positions
    for (unsigned int i=0; i<positions.size(); i++){
        RealType pos = positions.at(i);
        RealType bias = 0;

        // Define the search range. Field points inside the search range 
        // will get summed up
        RealType search_min = pos - corrLength; 
        RealType search_max = pos + corrLength;

        // Now sum over all field positions to check if it's inside or out
        for (unsigned int j=minIdx; j<maxIdx; j++){
            RealType fieldPosition = fieldPositions.at(j);
            if (fieldPosition >= search_min && fieldPosition <= search_max){
                bias += biasingField.at(j);
            }
        }
        biases[i] = bias;
    }
    return biases;
}

Particle::Particle(const RealType _x) 
                : x(_x)
{
    // Don't actually do anything, but maybe later? 
};

System::System(const unsigned int _numParticles) : numParticles(_numParticles)
{
    particles.resize(_numParticles, Particle(0));
    gen.seed(rd());
    boost::random::normal_distribution<>::param_type params(0.0, 1.0);
    normalParams = params;
    time = 0;
};

double System::generateRandomDistance()
{
    return normal_dist(gen, normalParams);
}

void System::iterateTimeStep()
{
    for (unsigned int i=0; i < particles.size(); i++){
        double dist = generateRandomDistance();
        particles[i].setX(particles[i].getX() + dist);
    }
}

std::vector<RealType> System::getParticlePositions()
{
    std::vector<RealType> positions(particles.size());
    for (unsigned int i=0; i < particles.size(); i++){
        positions[i] = particles[i].getX();
    }
    return positions;
}

RealType System::getMaximumParticle()
{
    RealType maximum=0;
    for (unsigned int i=0; i < particles.size(); i++){
        RealType pos = particles[i].getX();
        if (pos > maximum){
            maximum = pos;
        }
    }
    return maximum;
}

PYBIND11_MODULE(continuousDiffusion, m)
{
    m.doc() = "C++ continuous Diffusion";

    py::class_<System>(m, "System")
        .def(py::init<const unsigned int>())
        .def("iterateTimeStep", &System::iterateTimeStep)
        .def("getParticlePositions", &System::getParticlePositions);

    m.def("arange", &arange);
    m.def("getRandomBiasingField", &getRandomBiasingField);
    m.def("sampleBiasFromBiasingField", &sampleBiasFromBiasingField);
}