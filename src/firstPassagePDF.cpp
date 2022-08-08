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
  long int quantileTime;
  RealType variance;
  bool quantileSet;
  bool varianceSet;
  void push_back_cdf(RealType singleParticleCDF)
  {
    cdf.push_back(1 - exp(-singleParticleCDF * nParticles));
  }

  RealType calculateVariance(unsigned int long t)
  {
    std::vector<unsigned int long> times(t);
    for (unsigned int i = 1; i <= t; i++) {
      times.at(i - 1) = i;
    }
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
  PDF.resize(2 * maxPosition + 1);

  firstPassageCDF = 0;

  // Set middle element of array to 1
  PDF[maxPosition] = 1;
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
  std::vector<RealType> newPDF(PDF.size(), 0);

  if (!staticEnvironment) {
    for (unsigned int i = 0; i < transitionProbabilities.size(); i++) {
      if (PDF.at(i) != 0) {
        transitionProbabilities.at(i) = generateBeta();
      }
    }
  }

  for (unsigned int i = 0; i < PDF.size(); i++) {
    if (i == 0) {
      newPDF.at(0) = PDF.at(0) + transitionProbabilities[1] * PDF.at(1);
    }
    else if (i == 1) {
      newPDF.at(1) = transitionProbabilities[2] * PDF.at(2);
    }
    else if (i == PDF.size() - 1) {
      newPDF.at(i) =
          PDF.at(i) + (1 - transitionProbabilities[i - 1]) * PDF.at(i - 1);
    }
    else if (i == PDF.size() - 2) {
      newPDF.at(i) = (1 - transitionProbabilities[i - 1]) * PDF.at(i - 1);
    }
    else {
      newPDF.at(i) = (1 - transitionProbabilities[i - 1]) * PDF.at(i - 1) +
                     transitionProbabilities[i + 1] * PDF.at(i + 1);
    }
  }

  firstPassageProbability =
      transitionProbabilities.at(1) * PDF.at(1) +
      (1 - transitionProbabilities.at(PDF.size() - 2)) * PDF.at(PDF.size() - 2);
  PDF = newPDF;
  firstPassageCDF += firstPassageProbability;
  t += 1;
}

std::tuple<unsigned int long, RealType>
FirstPassagePDF::evolveToCutoff(RealType cutOff, RealType nParticles)
{
  std::vector<RealType> pdf;
  std::vector<RealType> cdf;
  std::vector<RealType> cdfN;
  std::vector<unsigned int long> times;

  // Not sure how big these arrays will need to be but they should be at least
  // the size of maxPosition So I'll allocate that much memory
  pdf.reserve(maxPosition);
  cdf.reserve(maxPosition);
  cdfN.reserve(maxPosition);
  times.reserve(maxPosition);

  RealType cdf_sum = 0;
  RealType cdf_sumN = 0;

  while ((cdf_sumN < cutOff) || (cdf_sum < 1 / nParticles)) {
    iterateTimeStep();
    pdf.push_back(firstPassageProbability);

    cdf_sum += firstPassageProbability;
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
  std::vector<unsigned int long> pdfTimes = slice(times, 0, pdf.size() - 2);
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
