#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <math.h>

#include "firstPassagePDF.hpp"
#include "randomNumGenerator.hpp"
#include "stat.h"

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

class ParticleData {
public:
  ParticleData(const RealType _nParticles)
  {
    nParticles = _nParticles;
    quantileSet = false;
    varianceSet = false;
  };
  ~ParticleData(){};

  RealType nParticles;
  std::vector<RealType> cdf;
  std::vector<unsigned long int> times;
  long int quantileTime;
  RealType variance;
  bool quantileSet;
  bool varianceSet;
  void push_back_cdf(RealType singleParticleCDF)
  {
    cdf.push_back(1 - exp(-singleParticleCDF * nParticles));
  }

  void push_back_times(unsigned long int time){
    times.push_back(time);
  }

  RealType calculateVariance(unsigned int long t)
  {
    std::vector<RealType> pdf(cdf.size() - 1);
    for (unsigned int i = 0; i < cdf.size() - 1; i++) {
      pdf[i] = cdf[i + 1] - cdf[i];
    }
    std::vector<unsigned int long> pdfTimes = slice(times, 0, pdf.size() - 1);
    RealType var = calculateVarianceFromPDF(pdfTimes, pdf);
    return var;
  }
};

FirstPassagePDF::FirstPassagePDF(const double _beta,
                                 const unsigned long int _maxPosition,
                                 const bool _staticEnvironment)
    : RandomNumGenerator(_beta)
{
  staticEnvironment = _staticEnvironment;
  maxPosition = _maxPosition;
  PDF.resize(maxPosition + 2);

  firstPassageCDF = 0;

  // Set first element of array to 1
  PDF[0] = 1;
  transitionProbabilities.resize(PDF.size(), 0);
  // If we are using a static environment generate transition probabilities
  if (staticEnvironment) {
    for (unsigned int i = 0; i < transitionProbabilities.size(); i++) {
      transitionProbabilities.at(i) = generateBeta();
    }
  }
}

void FirstPassagePDF::iterateTimeStep()
{
  std::vector<RealType> pdf_new(PDF.size());
  RealType bias;
  firstPassageCDF = 0; // Rest to zero

  if (t <= maxPosition) {
    for (unsigned int i = 0; i <= t; i++) {
      bias = generateBeta();
      pdf_new.at(i) += PDF.at(i) * bias;
      pdf_new.at(i + 1) += PDF.at(i) * (1 - bias);
    }
    if (t == maxPosition){
      firstPassageCDF += pdf_new.at(0) + pdf_new.back();
    }
    else if (t == maxPosition - 1){
      firstPassageCDF += pdf_new.at(0) + pdf_new.at(pdf_new.size()-2);
    }
  }

  else {
    for (unsigned int i = 0; i < PDF.size(); i++) {
      if (i == 0) {
        pdf_new.at(i) += PDF.at(i);
      }
      else if (PDF.back() != 0) {
        if (i == (PDF.size() - 1)) {
          pdf_new.at(i - 1) += PDF.at(i);
        }
        else {
          bias = generateBeta();
          pdf_new.at(i) += PDF.at(i) * bias;
          pdf_new.at(i - 1) += PDF.at(i) * (1 - bias);
        }
      }
      else {
        if (i == (PDF.size() - 1)) {
          continue;
        }
        else if (i == (PDF.size() - 2)) {
          pdf_new.at(i + 1) += PDF.at(i);
        }
        else {
          bias = generateBeta();
          pdf_new.at(i) += PDF.at(i) * bias;
          pdf_new.at(i + 1) += PDF.at(i) * (1 - bias);
        }
      }
    }
    firstPassageCDF += pdf_new.at(0);
    if (pdf_new.back() == 0) {
      firstPassageCDF += pdf_new.at(pdf_new.size() - 2);
    }
    else {
      firstPassageCDF += pdf_new.back();
    }
  }
  PDF = pdf_new;
  t += 1;
}

std::tuple<unsigned int long, RealType>
FirstPassagePDF::evolveToCutoff(RealType cutOff, RealType nParticles)
{
  std::vector<RealType> cdf;
  std::vector<RealType> cdfN;
  std::vector<unsigned int long> times;

  // Not sure how big these arrays will need to be but they should be at least
  // the size of maxPosition So I'll allocate that much memory
  cdf.reserve(maxPosition);
  cdfN.reserve(maxPosition);
  times.reserve(maxPosition);

  RealType cdf_sum = 0;
  RealType cdf_sumN = 0;

  while ((cdf_sumN < cutOff) || (cdf_sum < 1 / nParticles)) {
    iterateTimeStep();

    cdf_sum = firstPassageCDF;
    cdf.push_back(cdf_sum);

    cdf_sumN = 1 - exp(-cdf_sum * nParticles);
    cdfN.push_back(cdf_sumN);

    times.push_back(t);
  }

  // Find the lower Nth quantile of the system
  unsigned int long quantileTime;
  for (unsigned int i = 0; i < cdf.size(); i++) {
    if (cdf[i] >= 1 / nParticles) {
      quantileTime = times[i];
      break;
    }
  }

  // Find the variance from the CDF of nParticles
  // First get the PDF of the system
  std::vector<RealType> pdfN(cdfN.size() - 1);
  for (unsigned int i = 0; i < cdfN.size() - 1; i++) {
    pdfN[i] = cdfN[i + 1] - cdfN[i];
  }
  std::vector<unsigned int long> pdfTimes = slice(times, 0, times.size() - 2);
  RealType var = calculateVarianceFromPDF(pdfTimes, pdfN);

  return std::make_tuple(quantileTime, var);
}

// Could always require number of particles input to constructor.
std::tuple<std::vector<unsigned long int>,
           std::vector<RealType>,
           std::vector<RealType>>
FirstPassagePDF::evolveToCutoffMultiple(RealType cutoff,
                                        std::vector<RealType> nParticles)
{

  std::vector<unsigned long int> quantiles;
  std::vector<RealType> variance;
  std::vector<RealType> setNParticles;

  std::vector<ParticleData> particlesData;
  for (unsigned int i = 0; i < nParticles.size(); i++) {
    particlesData.push_back(ParticleData(nParticles[i]));
  }

  while (!particlesData.empty()) {
    iterateTimeStep(); // Okay this is all good
    for (auto it = particlesData.begin(); it != particlesData.end(); it++) {

      // Get the measurement of the quantile position
      if ((firstPassageCDF >= 1 / it->nParticles) && (!(it->quantileSet))) {
        // std::cout << "trying to set quantile time" << std::endl;
        it->quantileTime = t;
        it->quantileSet = true;
        // std::cout << "set quantile time" << std::endl;
      }

      // Calculate the nParticle variance
      if (!(it->varianceSet)) {
        // std::cout << "trying to set variance" << std::endl;
        it->push_back_cdf(firstPassageCDF);
        it->push_back_times(t);
        if (it->cdf.back() == 1) {
          // calculate variance here
          RealType var = it->calculateVariance(t);
          it->variance = var;
          it->varianceSet = true;
        }
      }

      // Now set quantile and variance if
      if (it->varianceSet && it->quantileSet) {
        quantiles.push_back(it->quantileTime);
        variance.push_back(it->variance);
        setNParticles.push_back(it->nParticles);
        // Need to erase data if quantile and variance have been saved
        particlesData.erase(it--);
      }
    }
  }
  return std::make_tuple(quantiles, variance, setNParticles);
}
