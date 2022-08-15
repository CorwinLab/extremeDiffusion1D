#include "firstPassageEvolve.hpp"
#include "firstPassageDriver.hpp"
#include "particleData.hpp"
#include <vector>

FirstPassageEvolve::FirstPassageEvolve(
    const double _beta,
    std::vector<unsigned int long> _maxPositions,
    RealType _nParticles)
    : FirstPassageDriver(_beta, _maxPositions), nParticles(_nParticles)
{
  for (unsigned int i = 0; i < maxPositions.size(); i++) {
    particlesData.push_back(ParticleData(nParticles));
  }
  numberHalted = 0;
}

std::tuple<std::vector<unsigned int long>,
           std::vector<RealType>,
           std::vector<unsigned int long>>
FirstPassageEvolve::checkParticleData()
{
  std::vector<unsigned long int> quantileTimes;
  std::vector<RealType> variance;
  std::vector<unsigned int long> positions;
  RealType firstPassageCDF;
  for (unsigned int i = 0; i < pdfs.size(); i++) {
    // Get the measurement of the quantile position
    FirstPassageBase *pdf = &pdfs.at(i);
    ParticleData *it = &particlesData.at(i);
    firstPassageCDF = pdf->getFirstPassageCDF();

    // Get the measurement of the quantile position
    if ((firstPassageCDF >= 1 / nParticles) && (!(it->quantileSet))) {
      it->quantileTime = t;
      it->quantileSet = true;
    }

    // Calculate the nParticle variance
    if (!(it->varianceSet)) {
      it->push_back_cdf(firstPassageCDF, t);
      if (it->cdfPrev == 1) {
        // calculate variance here
        RealType var = it->calculateVariance();
        it->variance = var;
        it->varianceSet = true;
      }
    }

    // Now set quantile & variance plus delete elements
    if (it->varianceSet && it->quantileSet && !pdf->haltingFlag) {
      quantileTimes.push_back(it->quantileTime);
      variance.push_back(it->variance);
      positions.push_back(pdf->getMaxPosition());

      // now set flags to false
      pdf->haltingFlag = true;
      numberHalted += 1;
    }
  }
  return std::make_tuple(quantileTimes, variance, positions);
}