#include <vector>
#include "diffusionSystems.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
using RealType = double;

AllSystems::AllSystems(const unsigned int _numSystems, 
                    const double _beta, 
                    const unsigned int _xMax, 
                    const RealType _numWalkers)
                    : beta(_beta), numSystems(_numSystems), numWalkers(_numWalkers), xMax(_xMax) {
    occupancy.resize(_numSystems, std::vector<RealType>(_xMax, 0));
    for (unsigned int i=0; i < occupancy.size(); i++){
        occupancy[i][0] = _numWalkers;
    }
    std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
    dis.param(unifParams);
    gen.seed(rd());
    minPos = 0; maxPos = 0;
}

double AllSystems::generateRandomNumber(){
    if (isinf(beta)){
        return 0.5;
    }
    else{
        return dis(gen);
    }
}

// Measuring distances: keep another array with currentTargetDistance
// Remove arrays that you've passed max distance with.
void AllSystems::iterateTimeStep(){
    // Just record minminIdx and maxMaxIdx and go through those
    // i and j could be like pos and sysID
    std::vector<RealType> numMoveRight(occupancy.size(), 0);
    // j runs through the position
    unsigned int prevMinPos = minPos;
    unsigned int prevMaxPos = maxPos;
    bool firstNonZero = true;
    for (unsigned int j=prevMinPos; j < prevMaxPos + 2; j++){
        // generate random number before going through all the systems so that 
        // the same random number is used for each position
        double bias = generateRandomNumber();
        // i is the system id - runs through all systems
        for (unsigned int i=0; i < occupancy.size(); i++){
            RealType numWalkers = occupancy.at(i).at(j) - numMoveRight.at(i);
            RealType moveRight = round(toNextSitePosition(numWalkers, bias));
            // Add number of particles to the next site
            occupancy.at(i).at(j+1) += moveRight;
            // Subtract number of particles at same site
            occupancy.at(i).at(j) -= moveRight;
            // Record how many particles moved to the right
            numMoveRight.at(i) = moveRight;
            // set new minPos and maxPos
            if (numWalkers != 0){
                maxPos = j; 
                if (firstNonZero){ 
                    minPos = j;
                    firstNonZero = false;
                }
            }
        }
    }
    time += 1;
}

RealType AllSystems::toNextSitePosition(RealType numWalkers, double bias){
    // The boost binomial can sometimes return negative numbers (or inf) for
    // large or small biases. So we default to number of particles * bias
    if (bias >= 0.99999 || bias <= 0.000001) {
        return (numWalkers * bias);
    }
    // For smallCutoff need to downcast numWalkers to double. And then cast
    // answer to RealType.
    if (numWalkers < smallCutoff) {
        return RealType(binomial(gen, boost::random::binomial_distribution<>::param_type(double(numWalkers), double(bias))));
    }
    else if (numWalkers > largeCutoff) {
        return (numWalkers * bias);
    }
    else {
        RealType mediumVariance = sqrt(numWalkers * bias * (1 - bias));
        return numWalkers * bias + mediumVariance * (2*RealType(dis(gen))-1);
    }
}

PYBIND11_MODULE(diffusionSystems, m)
{
    py::class_<AllSystems>(m, "AllSystems")
        .def(py::init<const unsigned int, const double, const unsigned int, const RealType>())
        .def("iterateTimeStep", &AllSystems::iterateTimeStep)
        .def("getBeta", &AllSystems::getBeta)
        .def("getOccupancy", &AllSystems::getOccupancy)
        .def("generateRandomNumber", &AllSystems::generateRandomNumber);
}