#ifndef SCATTERING_HPP_
#define SCATTERING_HPP_

#include "randomDistribution.hpp"
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

class Scattering : public RandomDistribution {
private:
  std::vector<RealType> pleft;
  std::vector<RealType> pright;
  RealType nParticles;
  unsigned long int occupancySize;
  bool ProbDistFlag;
  bool staticEnvironment;

  std::pair<unsigned long int, unsigned long int> edges;
  unsigned long int time;

public:
  Scattering(std::string _distributionName,
             std::vector<double> _parameters,
             const unsigned long int _occupancySize);
  ~Scattering(){};

  std::vector<RealType> getpright() { return pright; };
  std::vector<RealType> getpleft() { return pleft; };
  unsigned long int getOccupancySize() { return occupancySize; };

  unsigned long int getTime() { return time; };

  void setTime(const unsigned long int _time) { time = _time; };

  std::pair<unsigned long int, unsigned long int> getEdges() { return edges; };

  void setEdges(std::pair<unsigned long int, unsigned long int> _edges)
  {
    edges = _edges;
  }

  unsigned long int getMaxIdx() { return edges.second; };
  unsigned long int getMinIdx() { return edges.first; };

  // Functions that do things
  void iterateTimestep();
  RealType getProbAbove(unsigned int idx);
  RealType getDeltaAt(unsigned int idx);
  RealType getProbAt(unsigned int idx) {return pright.at(idx) + pleft.at(idx); }
};

#endif /* SCATTERING_HPP_ */