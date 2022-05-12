#include <vector>
#include "diffusionSystems.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace py = pybind11;
using RealType = double;

System::System(const unsigned int _xMax,
            const RealType _numWalkers)
{
    occupancy.resize(_xMax, 0);
    occupancy[0] = _numWalkers;
    minPos = 0; maxPos = 0;
    time = 0;
}

std::ostream& operator<<(std::ostream& os, const System& sys){
    os << "[";
    for (unsigned int i=0; i < sys.occupancy.size(); i++){
        os << sys.occupancy[i] << ", ";
    }
    os << "] Time=" << sys.time;
    return os;
} 

AllSystems::AllSystems(const unsigned int _numSystems, 
                    const double _beta, 
                    const unsigned int _xMax, 
                    const RealType _numWalkers)
                    : beta(_beta), numSystems(_numSystems), numWalkers(_numWalkers), xMax(_xMax) {
    systems.resize(_numSystems, System(_xMax, _numWalkers));

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
void AllSystems::iterateTimeStep(std::vector<unsigned int> sysIDs){
    // Just record minminIdx and maxMaxIdx and go through those
    // i and j could be like pos and sysID
    std::vector<RealType> numMoveRight(sysIDs.size(), 0);
    // j runs through the position
    unsigned int prevMinPos = minPos;
    unsigned int prevMaxPos = maxPos;
    bool firstNonZero = true;
    for (unsigned int pos=prevMinPos; pos < prevMaxPos + 2; pos++){
        // generate random number before going through all the systems so that 
        // the same random number is used for each position
        double bias = generateRandomNumber();
        // i is the system id - runs through all systems
        for (unsigned int i=0; i < sysIDs.size(); i++){
            unsigned int id = sysIDs[i];
            System *sys = &systems.at(id);

            RealType numWalkers = sys->at(pos) - numMoveRight.at(i);
            RealType moveRight = round(toNextSitePosition(numWalkers, bias));
            // Add number of particles to the next site
            sys->at(pos+1) += moveRight;
            // Subtract number of particles at same site
            sys->at(pos) -= moveRight;
            // Record how many particles moved to the right
            numMoveRight.at(i) = moveRight;
            // set new minPos and maxPos in system
            if (numWalkers != 0){
                maxPos = pos; 
                if (firstNonZero){ 
                    minPos = pos;
                    firstNonZero = false;
                }
                // set minPos and maxPos for Systems
                sys->setMaxPos(maxPos);
                if (pos == sys->getMinPos()+1 && sys->at(pos-1) == 0){
                    sys->setMinPos(pos);
                }
            }
        }
    }
    for (unsigned int j=0; j < sysIDs.size(); j++){
        systems.at(sysIDs[j]).time += 1;
    }
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

std::vector<std::vector<unsigned int> > AllSystems::measureFirstPassageTimes(std::vector<RealType> distances){
    std::vector<std::vector<unsigned int> > firstPassageTimes(systems.size(), std::vector<unsigned int>(distances.size(), 0));

    // Stores what systems we want to move forward
    std::vector<unsigned int> systemIdx(systems.size());
    for (unsigned int i=0; i < systems.size(); i++){
        systemIdx.at(i) = i;
    }
    // Stores the current distance we're measuring for each system
    std::vector<unsigned int> distanceIdx(systemIdx.size(), 0);

    while (systemIdx.size() > 0){
        // Move the system forward one timestep
        iterateTimeStep(systemIdx);
        // Now iterate over every system index to get max and min positions
        for (unsigned int idx = 0; idx < systemIdx.size(); idx++){
            unsigned int sysIdx = systemIdx[idx];
            System *sys = &systems.at(sysIdx);
            double currentMaxPos = 2 * double(sys->getMaxPos()) - double(sys->getTime());
            double currentMinPos = abs(2 * double(sys->getMinPos()) - double(sys->getTime()));
            
            unsigned int *distIdx = &distanceIdx.at(sysIdx);
            RealType dist = distances[*distIdx];
            if (currentMaxPos >= dist || currentMinPos >= dist){
                firstPassageTimes.at(sysIdx).at(*distIdx) = sys->getTime();
                /*std::cout << "System: " << systems.at(sysIdx) << " Distance " << dist << "\n"; For trouble shooting */
                *distIdx += 1;
                if (*distIdx == distances.size()){
                    sysIdx = *systemIdx.erase(systemIdx.begin() + idx);
                }
            }
        }
    }
    return firstPassageTimes;
}

PYBIND11_MODULE(diffusionSystems, m)
{
    py::class_<AllSystems>(m, "AllSystems")
        .def(py::init<const unsigned int, const double, const unsigned int, const RealType>())
        .def("iterateTimeStep", &AllSystems::iterateTimeStep)
        .def("getBeta", &AllSystems::getBeta)
        .def("getXMax", &AllSystems::getXMax)
        .def("getSystems", &AllSystems::getSystems)
        .def("getMaxPos", &AllSystems::getMaxPos)
        .def("getMinPos", &AllSystems::getMinPos)
        .def("getTime", &AllSystems::getTime)
        .def("getNumSystems", &AllSystems::getNumSystems)
        .def("generateRandomNumber", &AllSystems::generateRandomNumber)
        .def("measureFirstPassageTimes", &AllSystems::measureFirstPassageTimes);

    py::class_<System>(m, "System")
        .def(py::init<const unsigned int, const RealType>())
        .def_readwrite("occupancy", &System::occupancy)
        .def("getMinPos", &System::getMinPos)
        .def("getMaxPos", &System::getMaxPos);
}