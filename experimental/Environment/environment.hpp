#include <vector>

#ifndef ENVIRONMENT_HPP_
#define ENVIRONMENT_HPP_

class Environment {
private:
  std::vector<std::vector<double> > biases;
  void extendBiases();
  void fillBiases();

  std::random_device rd;
  boost::random::mt19937_64 gen;

  // std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_real_distribution<> dis;
  boost::random::beta_distribution<> beta_dist;
  boost::random::binomial_distribution<> binomial;

public:
  Environment(const unsigned long int tMax, const unsigned long int xMax);
  ~Environment(){};

  double &operator[] (int i);
  std::vector<std::vector<double> > getBiases(){ return biases; };
};

#endif
