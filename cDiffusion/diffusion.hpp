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

#ifndef DIFFUSION_HPP_
#define DIFFUSION_HPP_

class Diffusion {
private:
  std::vector<RealType> occupancy;
  RealType nParticles;
  bool ProbDistFlag;
  double beta;

  // It would be nice if this could be a generic distribution as:
  // boost::random::distribution bias
  boost::random::beta_distribution<>::param_type betaParams;

  std::random_device rd;
  boost::random::mt19937_64 gen;

  // std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_real_distribution<> dis;
  boost::random::beta_distribution<> beta_dist;

  std::pair<std::vector<unsigned long int>, std::vector<unsigned long int>>
      edges;
  unsigned long int time;

  RealType toNextSite(RealType currentSite, RealType bias);
  double generateBeta();

public:
  Diffusion(const RealType _nParticles,
            const double _beta,
            const unsigned long int occupancySize,
            const bool _probDistFlag = true);
  ~Diffusion(){};

  RealType getNParticles() { return nParticles; };

  double getBeta() { return betaParams.beta(); };

  void setProbDistFlag(bool _probDistFlag) { ProbDistFlag = _probDistFlag; };
  bool getProbDistFlag() { return ProbDistFlag; };

  void setOccupancy(const std::vector<RealType> _occupancy)
  {
    occupancy = _occupancy;
  };
  std::vector<RealType> getOccupancy() { return occupancy; };

  void setBetaSeed(const unsigned int seed) { gen.seed(seed); };

  double getTime() { return time; };

  std::pair<std::vector<unsigned long int>, std::vector<unsigned long int>>
  getEdges()
  {
    return edges;
  };

  // Functions that do things

  void iterateTimestep();

  double NthquartileSingleSided(const RealType NQuart);

  RealType pGreaterThanX(const unsigned long int idx);

  std::pair<std::vector<double>, std::vector<RealType>>
  calcVsAndPb(const unsigned long int num);

  std::pair<std::vector<double>, std::vector<RealType>> VsAndPb(const double v);
};

#endif /* DIFFUSION_HPP_ */
