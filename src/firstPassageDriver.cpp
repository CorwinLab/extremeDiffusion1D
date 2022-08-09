#include <vector>

#include "firstPassageDriver.hpp"
#include "firstPassagePDF.hpp"
#include "randomNumGenerator.hpp"


FirstPassageDriver::FirstPassageDriver(const double _beta, std::vector<unsigned int long> _maxPositions) : RandomNumGenerator(_beta) {
    t = 0;
    maxPositions = _maxPositions;
    std::sort(maxPositions.begin(), maxPositions.end());

    for (unsigned int i=0; i < maxPositions.size(); i++){
        pdfs.push_back(FirstPassagePDF(maxPositions[i]));
    }
}

std::vector<RealType> FirstPassageDriver::getBiases(){
    unsigned int long x = maxPositions.back();
    std::vector<RealType> biases(x, 0);

    for (unsigned int i=0; i < biases.size(); i++){
        biases.at(i) = generateBeta();
    }

    return biases;
}

void FirstPassageDriver::iterateTimeStep(){
    std::vector<RealType> biases = getBiases();

    for (unsigned int i=0; i < pdfs.size(); i++){
        pdfs[i].iterateTimeStep(biases);
    }
    t+=1;
}