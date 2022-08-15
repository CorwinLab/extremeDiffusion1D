#ifndef FISRTPASSAGEEVOLVE_HPP_
#define FISRTPASSAGEEVOLVE_HPP_

#include "firstPassageDriver.hpp"
#include "particleData.hpp"
#include <boost/multiprecision/float128.hpp>
#include <vector>

typedef boost::multiprecision::float128 RealType;

class FirstPassageEvolve : public FirstPassageDriver {
protected:
  std::vector<ParticleData> particlesData;
  unsigned int numberHalted;
  RealType nParticles;
  unsigned int numberOfPositions;

public:
  FirstPassageEvolve(const double _beta,
                     std::vector<unsigned int long> _maxPositions,
                     RealType _nParticles);
  ~FirstPassageEvolve(){};

  std::vector<ParticleData> getParticleData(){ return particlesData; };
  void setParticleData(std::vector<ParticleData> _particlesData)
  {
    particlesData = _particlesData;
  };

  unsigned int getNumberHalted() { return numberHalted; };
  void setNumberHalted(unsigned int _numberHalted)
  {
    numberHalted = _numberHalted;
  };

  RealType getNParticles() { return nParticles; };
  void setNParticles(RealType _nParticles) {nParticles = _nParticles; };

  unsigned int getNumberOfPositions() {return pdfs.size(); };

  std::tuple<std::vector<unsigned int long>,
             std::vector<RealType>,
             std::vector<unsigned int long>>
  checkParticleData();
};

#endif