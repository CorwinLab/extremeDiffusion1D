#ifndef PARTICLESDATA_HPP_
#define PARTICLESDATA_HPP_

#include <boost/multiprecision/float128.hpp>
#include <vector>
#include <cmath>
#include <iomanip>
#include "stat.hpp"

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

class ParticleData {
public:
  ParticleData(const RealType _nParticles)
  {
    nParticles = _nParticles;
    quantileSet = false;
    varianceSet = false;
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
};

#endif /* PARTICLESDATA_HPP_ */