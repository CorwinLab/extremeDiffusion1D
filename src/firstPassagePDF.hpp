#ifndef FISRTPASSAGEPDF_HPP_
#define FISRTPASSAGEPDF_HPP_

#include <assert.h>
#include <boost/math/distributions.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/random.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <math.h>
#include <random>
#include <utility>
#include <vector>
#include "randomNumGenerator.hpp"

typedef boost::multiprecision::float128 RealType;

class FirstPassagePDF : public RandomNumGenerator {
protected:
  std::vector<RealType> PDF;
  unsigned long int maxPosition;
  unsigned long int t = 0;
  RealType firstPassageProbability;
  std::vector<double> transitionProbabilities;
  bool staticEnvironment;

public:
  FirstPassagePDF(const double _beta, const unsigned long int _maxPosition, const bool _staticEnvironment);
  ~FirstPassagePDF(){};

  unsigned long int getTime() { return t; };
  void setTime(unsigned long int _t) { t = _t; };

  std::vector<RealType> getPDF() { return PDF; };
  void setPDF(std::vector<RealType> _PDF) { PDF = _PDF; };

  std::vector<double> getTransitionProbabilities() { return transitionProbabilities; };

  unsigned long int getMaxPosition() { return maxPosition; };
  void setMaxPosition(unsigned long int _maxPosition)
  {
    maxPosition = _maxPosition;
  };

  void iterateTimeStep();

  RealType getFirstPassageProbability() { return firstPassageProbability; };

  std::tuple<unsigned int long, RealType>
  evolveToCutoff(RealType prob_cutOff, RealType nParticles);

  std::tuple<std::vector<std::vector<unsigned int long>, std::vector<RealType>>
  evolveToCutoffMultiple(RealType prob_cutOff, std::vector<RealType> nParticles);
};

#endif /* FISRTPASSAGEPDF_HPP_ */
