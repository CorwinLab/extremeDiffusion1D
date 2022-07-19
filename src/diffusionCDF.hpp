#ifndef DIFFUSIONCDF_HPP_
#define DIFFUSIONCDF_HPP_

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

// Base Diffusion class
class DiffusionCDF : public RandomNumGenerator {
protected:
  std::vector<RealType> CDF;
  unsigned long int tMax;

public:
  DiffusionCDF(const double _beta, const unsigned long int _tMax);
  ~DiffusionCDF(){};

  std::vector<RealType> getCDF() { return CDF; };
  void setCDF(std::vector<RealType> _CDF) { CDF = _CDF; };

  unsigned long int gettMax() { return tMax; };
  void settMax(unsigned long int _tMax) { tMax = _tMax; };

};

class DiffusionTimeCDF : public DiffusionCDF {
private:
  unsigned long int t = 0;

public:
  DiffusionTimeCDF(const double _beta, const unsigned long int _tMax);

  unsigned long int getTime() { return t; };
  void setTime(unsigned long int _t) { t = _t; };

  // Functions that do things
  void iterateTimeStep();

  long int findQuantile(RealType quantile);
  std::vector<long int> findQuantiles(std::vector<RealType> quantiles);

  long int findLowerQuantile(RealType quantile);

  RealType getGumbelVariance(RealType nParticles);
  std::vector<RealType> getGumbelVariance(std::vector<RealType> nParticles);
  std::vector<long int> getxvals();
  std::vector<RealType> getSaveCDF();
  std::pair<RealType, float> getProbandV(RealType quantile);
};

class DiffusionPositionCDF : public DiffusionCDF {
private:
  unsigned long int position = 0;
  std::vector<RealType> quantiles;
  std::vector<std::vector<unsigned long int>> quantilePositions;

public:
  DiffusionPositionCDF(const double _beta,
                       const unsigned long int _tMax,
                       std::vector<RealType> _quantiles);
  unsigned long int getPosition() { return position; };
  std::vector<std::vector<unsigned long int>> getQuantilePositions()
  {
    return quantilePositions;
  };
  std::vector<RealType> getQuantiles() { return quantiles; };

  // Functions that do things
  void stepPosition();
};

#endif /* DIFFUSIONCDF_HPP_ */
