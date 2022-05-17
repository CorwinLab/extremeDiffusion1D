#include "continuousDiffusion.hpp"
#include <vector> 
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using RealType = double;
namespace py = pybind11;

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

RealType System::getMaximumParticle(){
    
}

PYBIND11_MODULE(continuousDiffusion, m)
{
  m.doc() = "C++ continuous Diffusion";

  py::class_<System>(m, "System")
    .def(py::init<const unsigned int>())
    .def("iterateTimeStep", &System::iterateTimeStep)
    .def("getParticlePositions", &System::getParticlePositions);
}