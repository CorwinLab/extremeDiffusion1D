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
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

typedef boost::multiprecision::float128 RealType;

#ifndef DIFFUSIONCDF_HPP_
#define DIFFUSIONCDF_HPP_

class RandDistribution{
  protected:
    std::vector<double> alpha;
    gsl_rng *gen = gsl_rng_alloc(gsl_rng_mt19937);
    size_t K; 
    std::vector<double> theta;

  public: 
    RandDistribution(const std::vector<double> _alpha);
    ~RandDistribution(){};

    std::vector<RealType> getRandomNumbers();
};

// Base Diffusion class
class DiffusionND : public RandDistribution {
protected:
  std::vector<std::vector<RealType> > CDF;
  std::vector<double> alpha;
  unsigned long int tMax;
  unsigned long int t;
  int L;
  RealType absorbedProb;
  RealType tol = pow(10, -4500);

  // It would be nice if this could be a generic distribution as:
  // boost::random::distribution bias
  boost::random::beta_distribution<>::param_type betaParams;

  std::random_device rd;
  boost::random::mt19937_64 gen;

  std::uniform_real_distribution<> dis;
  boost::random::beta_distribution<> beta_dist;
  std::vector<RealType> biases;

public:
  DiffusionND(const std::vector<double> _alpha, const unsigned long int _tMax, int _L);
  ~DiffusionND(){};

  std::vector<double> getAlpha() { return alpha; };
  void setAlpha(std::vector<double> _alpha){ alpha = _alpha; };

  std::vector<std::vector<RealType> > getCDF() { return CDF; };
  void setCDF(std::vector<std::vector<RealType> > _CDF){ CDF = _CDF; };

  unsigned long int gettMax() { return tMax; };
  void settMax(unsigned long int _tMax){ tMax = _tMax; };

  void setBetaSeed(const unsigned int seed) { gen.seed(seed); };
  std::vector<RealType> getBiases() { return biases;}

  unsigned long int getTime(){ return t; };
  int getL() { return L; }
  
  RealType getAbsorbedProb() { return absorbedProb; }

  void iterateTimestep();
};

#endif /* DIFFUSIONCDF_HPP_ */
