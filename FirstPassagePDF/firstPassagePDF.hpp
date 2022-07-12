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

#ifndef FISRTPASSAGEPDF_HPP_
#define FISRTPASSAGEPDF_HPP_

class FirstPassagePDF {
protected:
  std::vector<RealType> PDF;
  double beta;
  unsigned long int maxPosition;
  unsigned long int t = 0;
  RealType firstPassageProbability;

  boost::random::beta_distribution<>::param_type betaParams;

  std::random_device rd;
  boost::random::mt19937_64 gen;

  std::uniform_real_distribution<> dis;
  boost::random::beta_distribution<> beta_dist;

  double generateBeta();

public:
  FirstPassagePDF(const double _beta, const unsigned long int _maxPosition);
  ~FirstPassagePDF(){};

  double getBeta() { return beta; };
  void setBeta(double _beta) { beta = _beta; };

  unsigned long int getTime() { return t; };
  void setTime(unsigned long int _t) { t = _t; };

  std::vector<RealType> getPDF() { return PDF; };
  void setPDF(std::vector<RealType> _PDF) { PDF = _PDF; };

  unsigned long int getMaxPosition() { return maxPosition; };
  void setMaxPosition(unsigned long int _maxPosition)
  {
    maxPosition = _maxPosition;
  };

  void iterateTimeStep();

  RealType getFirstPassageProbability() { return firstPassageProbability; };

  std::tuple<std::vector<unsigned int long>,
             std::vector<RealType>,
             std::vector<RealType>,
             std::vector<RealType>>
  evolveToCutoff(RealType prob_cutOff, RealType nParticles);
};

#endif /* FISRTPASSAGEPDF_HPP_ */
