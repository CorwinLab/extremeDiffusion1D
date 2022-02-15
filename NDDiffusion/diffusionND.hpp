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

#ifndef DIFFUSIONCDF_HPP_
#define DIFFUSIONCDF_HPP_

// Base Diffusion class
class DiffusionND{
protected:
  std::vector<std::vector<RealType> > CDF;
  double beta;
  unsigned long int tMax;
  unsigned long int t;

  // It would be nice if this could be a generic distribution as:
  // boost::random::distribution bias
  boost::random::beta_distribution<>::param_type betaParams;

  std::random_device rd;
  boost::random::mt19937_64 gen;

  std::uniform_real_distribution<> dis;
  boost::random::beta_distribution<> beta_dist;
  std::vector<float> biases;
  std::vector<double> maxDist;
  std::vector<double> maxTheta;

public:
  DiffusionND(const double _beta, const unsigned long int _tMax);
  ~DiffusionND(){};

  double getBeta() { return beta; };
  void setBeta(double _beta){ beta = _beta; };

  std::vector<std::vector<RealType> > getCDF() { return CDF; };
  void setCDF(std::vector<std::vector<RealType> > _CDF){ CDF = _CDF; };

  unsigned long int gettMax() { return tMax; };
  void settMax(unsigned long int _tMax){ tMax = _tMax; };

  void setBetaSeed(const unsigned int seed) { gen.seed(seed); };
  std::vector<float> getBiases() { return biases;}

  void iterateTimestep();
  std::vector<double> getDistance() { return maxDist;}
  std::vector<double> getTheta() {return maxTheta; }
};

#endif /* DIFFUSIONCDF_HPP_ */
