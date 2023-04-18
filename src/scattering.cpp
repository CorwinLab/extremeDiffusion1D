#include "scattering.hpp"
#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <math.h>

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

Scattering::Scattering(std::string _distributionName,
                       std::vector<double> _parameters,
                       const unsigned long int _occupancySize)
    : RandomDistribution(_distributionName, _parameters),
      occupancySize(_occupancySize)
{
  edges.first = 0, edges.second = 0;

  pright.resize(_occupancySize + 1);
  pright[0] = 1;

  pleft.resize(_occupancySize + 1);

  time = 0;
}

void Scattering::iterateTimestep()
{
  unsigned long int prevMinIndex = edges.first;
  unsigned long int prevMaxIndex = edges.second;
  if (prevMinIndex > prevMaxIndex) {
    throw std::runtime_error(
        "Minimum edge must be greater than maximum edge: (" +
        std::to_string(prevMinIndex) + ", " + std::to_string(prevMaxIndex) +
        ")");
  }

  // If iterating over the whole array extend the occupancy.
  if ((prevMaxIndex + 1) == pright.size()) {
    pright.push_back(0);
    pleft.push_back(0);
    std::cout << "Warning: pushing back occupancy size. If this happens a lot "
                 "it may effect performance."
              << std::endl;
  }

  std::vector<RealType> pleftNew(pleft.size(), 0);
  std::vector<RealType> prightNew(pright.size(), 0);

  unsigned long int minEdge = 0;
  unsigned long int maxEdge = 0;
  bool firstNonzero = true;
  RealType prevBias = RealType(generateRandomVariable());

  for (auto i = prevMinIndex; i < prevMaxIndex + 2; i++) {
    RealType newBias = RealType(generateRandomVariable());
    if (i == prevMinIndex){
      prightNew.at(i) = 0;
    }
    else{
      prightNew.at(i) =
          pright.at(i - 1) * prevBias + pleft.at(i - 1) * (1 - prevBias);      
    }
    pleftNew.at(i) =
        pright.at(i) * (1 - newBias) + pleft.at(i) * newBias;

    if (prightNew.at(i) != 0 || pleftNew.at(i) != 0) {
      maxEdge = i;
      if (firstNonzero) {
        minEdge = i;
        firstNonzero = false;
      }
    }
    prevBias = newBias;
  }

  edges.first = minEdge;
  edges.second = maxEdge;
  pright = prightNew;
  pleft = pleftNew;
  time += 1;
}

RealType Scattering::getProbAbove(unsigned int idx){
  RealType CDF = 0;

  for (unsigned int i=idx; i < edges.second+1; i++){
    CDF += pright.at(i) + pleft.at(i);
  }
  return CDF;
}

RealType Scattering::getDeltaAt(unsigned int idx){
  return pright.at(idx) - pleft.at(idx);
}