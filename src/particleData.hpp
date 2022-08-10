#ifndef PARTICLESDATA_HPP_
#define PARTICLESDATA_HPP_

#include <boost/multiprecision/float128.hpp>
#include <vector>
#include <cmath>
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
  };
  ~ParticleData(){};

  RealType nParticles;
  std::vector<RealType> cdf;
  std::vector<unsigned long int> times;
  long int quantileTime;
  RealType variance;
  bool quantileSet;
  bool varianceSet;
  void push_back_cdf(RealType singleParticleCDF)
  {
    cdf.push_back(1 - exp(-singleParticleCDF * nParticles));
  }

  void push_back_times(unsigned long int time){
    times.push_back(time);
  }

  RealType calculateVariance(unsigned int long t)
  {
    std::vector<RealType> pdf(cdf.size() - 1);
    for (unsigned int i = 0; i < cdf.size() - 1; i++) {
      pdf[i] = cdf[i + 1] - cdf[i];
    }
    std::vector<unsigned int long> pdfTimes = slice(times, 0, pdf.size() - 1);
    RealType var = calculateVarianceFromPDF(pdfTimes, pdf);
    return var;
  }
};

#endif /* PARTICLESDATA_HPP_ */