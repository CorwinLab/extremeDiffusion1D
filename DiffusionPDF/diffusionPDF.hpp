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

#ifndef DIFFUSIONPDF_HPP_
#define DIFFUSIONPDF_HPP_

class DiffusionPDF {
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
  boost::random::binomial_distribution<> binomial;
  boost::random::beta_distribution<> beta_dist; 

  std::pair<std::vector<unsigned long int>, std::vector<unsigned long int>>
      edges;
  unsigned long int time;

  RealType toNextSite(RealType currentSite, RealType bias);
  double generateBeta();

public:
  DiffusionPDF(const RealType _nParticles,
            const double _beta,
            const unsigned long int occupancySize,
            const bool _probDistFlag = true);
  ~DiffusionPDF(){};

  RealType getNParticles() { return nParticles; };

  double getBeta() { return betaParams.beta(); };

  void setProbDistFlag(bool _probDistFlag) { ProbDistFlag = _probDistFlag; };
  bool getProbDistFlag() { return ProbDistFlag; };

  void setOccupancy(const std::vector<RealType> _occupancy)
  {
    occupancy = _occupancy;
  };
  std::vector<RealType> getOccupancy() { return occupancy; };

  void resizeOccupancy(unsigned long int size) { occupancy.resize(size); };

  void setBetaSeed(const unsigned int seed) { gen.seed(seed); };

  unsigned long int getTime() { return time; };

  void setTime(const unsigned long int _time) { time = _time; };

  std::pair<std::vector<unsigned long int>, std::vector<unsigned long int>>
  getEdges()
  {
    return edges;
  };

  // Functions that do things

  void iterateTimestep();

  double findQuantile(const RealType quantile);
  std::vector<double> findQuantiles(std::vector<RealType> quantiles);

  RealType pGreaterThanX(const unsigned long int idx);

  std::pair<std::vector<double>, std::vector<RealType>>
  calcVsAndPb(const unsigned long int num);

  std::pair<std::vector<double>, std::vector<RealType>> VsAndPb(const double v);

};

#endif /* DIFFUSIONPDF_HPP_ */
