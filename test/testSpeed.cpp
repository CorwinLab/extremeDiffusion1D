#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <time.h>
#include <boost/random.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <random>
#include <utility>

std::random_device rd;
boost::mt19937 gen(rd());

int main(){
  boost::random::beta_distribution<double> distribution(1, 2);
  double a;
  int N = 100000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++){
    a = distribution(gen);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  std::cout << float(microseconds)/1e6 / N * 1e9 << std::endl;
}
