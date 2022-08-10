#ifndef FISRTPASSAGEBASE_HPP_
#define FISRTPASSAGEBASE_HPP_

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

typedef boost::multiprecision::float128 RealType;

class FirstPassageBase {
protected:
  std::vector<RealType> PDF;
  unsigned long int maxPosition;
  unsigned long int t = 0;
  RealType firstPassageCDF;

public:
  FirstPassageBase(const unsigned long int _maxPosition);
  ~FirstPassageBase(){};

  unsigned long int getTime() { return t; };
  void setTime(unsigned long int _t) { t = _t; };

  std::vector<RealType> getPDF() { return PDF; };
  void setPDF(std::vector<RealType> _PDF) { PDF = _PDF; };

  unsigned long int getMaxPosition() { return maxPosition; };
  void setMaxPosition(unsigned long int _maxPosition)
  {
    maxPosition = _maxPosition;
  };

  void iterateTimeStep(std::vector<RealType> biases);

  RealType getFirstPassageCDF() { return firstPassageCDF; };
};

#endif /* FISRTPASSAGEBASE_HPP_ */
