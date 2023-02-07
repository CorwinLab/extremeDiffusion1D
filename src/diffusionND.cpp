#include <math.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#include "diffusionND.hpp"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define _USE_MATH_DEFINES


using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

RandDistribution::RandDistribution(const std::vector<double> _alpha){
  alpha = _alpha;
  gsl_rng_set(gen,(unsigned)time(NULL));
  K = 4;
  theta.resize(4);
}

std::vector<RealType> RandDistribution::getRandomNumbers(){
  if (isinf(alpha[0])){
    return {(RealType)0.25, (RealType)0.25, (RealType)0.25, (RealType)0.25};
  }
  else{
    gsl_ran_dirichlet(gen, K, &alpha[0], &theta[0]);
    /* Cast double precision to quad precision by dividing by sum of double precision rand numbers*/
    std::vector<RealType> biases(theta.begin(), theta.end());
    RealType sum = 0;
    for (unsigned int i=0; i < biases.size(); i++){
      sum += biases[i];
    }
    for (unsigned int i=0; i<biases.size(); i++){
      biases[i] /= sum;
    }
    return biases;
  }
}

DiffusionND::DiffusionND(const std::vector<double> _alpha, const unsigned long int _tMax, int _L) : RandDistribution(_alpha)
{
  tMax = _tMax;
  L = _L;
  absorbedProb = 0;

  std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
  dis.param(unifParams);
  gen.seed(rd());

  std::vector<double> biases;

  CDF.resize(tMax+1, std::vector<RealType>(tMax+1));
  CDF[0][0] = 1;
  CDF_new.resize(CDF.size(), std::vector<RealType>(CDF[0].size()));

  t = 0;
}

void DiffusionND::iterateTimestep(){
  RealType cdf_new_sum = 0;

  for (unsigned long int i = 0; i < t+1; i++){ // i is the columns
    for (unsigned long int j = 0; j < t+1; j++){ // j is the row
      RealType currentPos = CDF[i][j];
      if (currentPos == 0){
        continue;
      }
      int yval = j-i;
      
      biases = getRandomNumbers();

      CDF_new[i][j] += currentPos * (RealType)biases[0];
      CDF_new[i][j+1] += currentPos * (RealType)biases[1];
      CDF_new[i+1][j+1] += currentPos * (RealType)biases[3];
      
      cdf_new_sum += currentPos * ((RealType)biases[0] + (RealType)biases[1] + (RealType)biases[3]);

      if ((yval-1) <= -L){
        absorbedProb += currentPos * ((RealType)biases[2]);
      }
      else {
        CDF_new[i+1][j] += currentPos * (RealType)biases[2];
        cdf_new_sum += currentPos * ((RealType)biases[2]);
      }
      CDF[i][j] = 0;
    }
  }
  /* Ensure we aren't losing/gaining probability */
  if ((cdf_new_sum + absorbedProb) < ((RealType)1.-(RealType)pow(10, -25)) || (cdf_new_sum + absorbedProb) > ((RealType)1.+(RealType)pow(10, -25))){
    std::cout << "CDF total: " << cdf_new_sum + absorbedProb << std::endl;
    throw std::runtime_error("Total probability not within tolerance of 10^-25");
  }

  /* Swap CDF_new and CDF */
  for (unsigned long int i=0; i < t+2; i++){
    CDF[i].swap(CDF_new[i]);
  }

  t += 1;
}
