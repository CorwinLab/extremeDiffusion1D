// This is just a possible template for the recurance Relation object but I
// think that it might be overkill at this point
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

class DiffusionCDF {
private:
  std::vector<RealType> zB;
  unsigned long int t;
  double beta;
  unsigned long int tMax;

  // It would be nice if this could be a generic distribution as:
  // boost::random::distribution bias
  boost::random::beta_distribution<>::param_type betaParams;

  std::random_device rd;
  boost::random::mt19937_64 gen;

  std::uniform_real_distribution<> dis;
  boost::random::beta_distribution<> beta_dist;

  double generateBeta();

public:
  DiffusionCDF(const double _beta, const unsigned long int _tMax);
  ~DiffusionCDF(){};

  double getBeta() { return beta; };

  std::vector<RealType> getzB() { return zB; };

  unsigned long int gettMax() { return tMax; };

  void setBetaSeed(const unsigned int seed) { gen.seed(seed); };

  unsigned long int getTime() { return t; };

  // Functions that do things
  void iterateTimeStep();

  unsigned long int findQuantile(RealType quantile);

  std::vector<unsigned long int> findQuantiles(std::vector<RealType> quantiles);
};

#endif /* DIFFUSIONCDF_HPP_ */
