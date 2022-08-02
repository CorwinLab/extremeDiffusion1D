#ifndef FISRTPASSAGEPDFBASE_HPP_
#define FISRTPASSAGEPDFBASE_HPP_

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

class FirstPassagePDFBase {
protected:
  std::vector<RealType> PDF;
  unsigned long int maxPosition;
  RealType firstPassageProbability;

public:
  FirstPassagePDFBase(const unsigned long int _maxPosition);
  ~FirstPassagePDFBase(){};

  std::vector<RealType> getPDF() { return PDF; };
  void setPDF(std::vector<RealType> _PDF) { PDF = _PDF; };

  unsigned long int getMaxPosition() { return maxPosition; };
  void setMaxPosition(unsigned long int _maxPosition)
  {
    maxPosition = _maxPosition;
  };

  RealType getFirstPassageProbability() { return firstPassageProbability; };
  void setFirstPassageProbability(RealType _firstPassageProbability) { firstPassageProbability = _firstPassageProbability; };
};

#endif /* FISRTPASSAGEPDFBASE_HPP_ */
