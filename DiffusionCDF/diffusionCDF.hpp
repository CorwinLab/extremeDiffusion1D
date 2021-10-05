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

// Base Diffusion class
class DiffusionCDF{
protected:
  std::vector<RealType> CDF;
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

  std::vector<RealType> getCDF() { return CDF; };

  unsigned long int gettMax() { return tMax; };

  void setBetaSeed(const unsigned int seed) { gen.seed(seed); };

};

class DiffusionTimeCDF: public DiffusionCDF {
private:
  unsigned long int t = 0;

public:
  DiffusionTimeCDF(const double _beta, const unsigned long int _tMax);

  unsigned long int getTime() { return t; };

  // Functions that do things
  void iterateTimeStep();

  unsigned long int findQuantile(RealType quantile);

  std::vector<unsigned long int> findQuantiles(std::vector<RealType> quantiles);

  RealType getDiscreteVarianceDiff(RealType nParticles);
  RealType getDiscreteVariance(RealType nParticles);
  std::pair<std::vector<long int>, std::vector<RealType> > getPDF(RealType nParticles);
  std::pair<std::vector<long int>, std::vector<RealType> > getCDF(RealType nParticles);
};

class DiffusionPositionCDF: public DiffusionCDF{
private:
  unsigned long int position = 0;
  std::vector<RealType> quantiles;
  std::vector<std::vector<unsigned long int> > quantilesMeasurement;

public:
  DiffusionPositionCDF(const double _beta, const unsigned long int _tMax, std::vector<RealType> _quantiles);
  unsigned long int getPosition() { return position; };
  std::vector<std::vector<unsigned long int> > getQuantilesMeasurement() { return quantilesMeasurement; };
  std::vector<RealType> getQuantiles() { return quantiles; };

  // Functions that do things
  void stepPosition();
};

#endif /* DIFFUSIONCDF_HPP_ */
