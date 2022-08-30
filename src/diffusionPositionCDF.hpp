#ifndef DIFFUSIONPOSITIONCDF_HPP_
#define DIFFUSIONPOSITIONCDF_HPP_

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
#include "randomDistribution.hpp"

typedef boost::multiprecision::float128 RealType;

class DiffusionPositionCDF : public RandomDistribution {
private:
  unsigned long int position = 0;
  std::vector<RealType> quantiles;
  std::vector<std::vector<unsigned long int>> quantilePositions;
  unsigned long int tMax;
  std::vector<RealType> CDF;

public:
  DiffusionPositionCDF(std::string _distributionName,
                       std::vector<double> _parameters,
                       const unsigned long int _tMax,
                       std::vector<RealType> _quantiles);
  unsigned long int getPosition() { return position; };
  std::vector<std::vector<unsigned long int>> getQuantilePositions()
  {
    return quantilePositions;
  };
  std::vector<RealType> getQuantiles() { return quantiles; };

  std::vector<RealType> getCDF() { return CDF; };
  void setCDF(std::vector<RealType> _CDF) { CDF = _CDF; };
  
  unsigned long int gettMax() { return tMax; };
  void settMax(unsigned int long _tMax) { tMax = _tMax; };

  // Functions that do things
  void stepPosition();
};

#endif /* DIFFUSIONPOSITIONCDF_HPP_ */
