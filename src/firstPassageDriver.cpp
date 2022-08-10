#include <vector>

#include "firstPassageDriver.hpp"
#include "firstPassageBase.hpp"
#include "particleData.hpp"
#include "randomNumGenerator.hpp"

FirstPassageDriver::FirstPassageDriver(
    const double _beta, std::vector<unsigned int long> _maxPositions)
    : RandomNumGenerator(_beta)
{
  t = 0;
  maxPositions = _maxPositions;
  std::sort(maxPositions.begin(), maxPositions.end());

  for (unsigned int i = 0; i < maxPositions.size(); i++) {
    pdfs.push_back(FirstPassageBase(maxPositions[i]));
  }
}

std::vector<RealType> FirstPassageDriver::getBiases()
{
  unsigned int long x = maxPositions.back();
  std::vector<RealType> biases;
  unsigned int numberToGenerate;

  if (t < x){
    numberToGenerate = t+2;
  }
  else{
    numberToGenerate = x + 2;
  }

  for (unsigned int i = 0; i < numberToGenerate; i++) {
    biases.push_back(generateBeta());
  }

  return biases;
}

void FirstPassageDriver::iterateTimeStep()
{
  std::vector<RealType> biases = getBiases();
  for (unsigned int i = 0; i < pdfs.size(); i++) {
    pdfs[i].iterateTimeStep(biases);
  }
  t += 1;
}

std::tuple<std::vector<unsigned int long>,
           std::vector<RealType>,
           std::vector<unsigned int long>>
FirstPassageDriver::evolveToCutoff(RealType nParticles, RealType cutoff)
{
  std::vector<unsigned int long> quantileTimes;
  std::vector<RealType> variance;
  std::vector<unsigned int long> positions;
  std::vector<ParticleData> particlesData;
  for (unsigned int i = 0; i < maxPositions.size(); i++) {
    particlesData.push_back(ParticleData(nParticles));
  }
  unsigned int index;
  RealType firstPassageCDF;
  while (!particlesData.empty()) {
    iterateTimeStep();
    // Each ParticleData object corresponds to a different pdf.
    for (auto it = particlesData.begin(); it != particlesData.end(); it++) {
      // Get the measurement of the quantile position
      index = it - particlesData.begin();
      FirstPassageBase pdf = pdfs.at(index);
      firstPassageCDF = pdf.getFirstPassageCDF();

      // Get the measurement of the quantile position
      if ((firstPassageCDF >= 1 / nParticles) && (!(it->quantileSet))) {
        it->quantileTime = t;
        it->quantileSet = true;
      }

      // Calculate the nParticle variance
      if (!(it->varianceSet)) {
        it->push_back_cdf(firstPassageCDF);
        it->push_back_times(t);
        if (it->cdf.back() == 1) {
          // calculate variance here
          RealType var = it->calculateVariance();
          it->variance = var;
          it->varianceSet = true;
        }
      }

      // Now set quantile & variance plus delete elements
      if (it->varianceSet && it->quantileSet) {
        quantileTimes.push_back(it->quantileTime);
        variance.push_back(it->variance);
        positions.push_back(pdf.getMaxPosition());

        // now need to erase elements in particlesData and pdfs
        particlesData.erase(it--);
        pdfs.erase(pdfs.begin() + index);
      }
    }
  }
  return std::make_tuple(quantileTimes, variance, positions);
}
