#ifndef PARTICLESDATA_HPP_
#define PARTICLESDATA_HPP_

#include "stat.hpp"
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <vector>

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

class ParticleData {
public:
  ParticleData(const RealType _nParticles)
  {
    nParticles = _nParticles;
    quantileSet = false;
    varianceSet = false;
    variance = -1;
    quantileTime = -1;
    cdfPrev = 0;
  };
  ~ParticleData(){};

  RealType nParticles;

  // Parameters we want to set
  long int quantileTime;
  RealType variance;
  bool quantileSet;
  bool varianceSet;

  RealType cdfPrev;
  RealType runningSumSquared;
  RealType runningSum;

  void push_back_cdf(RealType singleParticleCDF, unsigned long int time)
  {
    RealType currentCDF = 1 - exp(-singleParticleCDF * nParticles);
    RealType pdf = currentCDF - cdfPrev;
    runningSumSquared += pdf * pow(time, 2);
    runningSum += pdf * time;
    cdfPrev = currentCDF;
  }

  RealType calculateVariance()
  {
    RealType otherVar = runningSumSquared - pow(runningSum, 2);
    return otherVar;
  }

  friend bool operator==(const ParticleData &lhs, const ParticleData &rhs)
  {
    return (lhs.nParticles == rhs.nParticles) &&
           (lhs.quantileTime == rhs.quantileTime) &&
           (lhs.variance == rhs.variance) &&
           (lhs.quantileSet == rhs.quantileSet) &&
           (lhs.varianceSet == rhs.varianceSet) && 
           (lhs.cdfPrev == rhs.cdfPrev) && 
           (lhs.runningSumSquared == rhs.runningSumSquared) &&
           (lhs.runningSum == rhs.runningSum);
  }
};

#endif /* PARTICLESDATA_HPP_ */