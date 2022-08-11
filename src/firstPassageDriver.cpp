#include <boost/multiprecision/float128.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "firstPassageBase.hpp"
#include "firstPassageDriver.hpp"
#include "particleData.hpp"
#include "randomNumGenerator.hpp"

using RealType = boost::multiprecision::float128;

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

  if (t < x) {
    numberToGenerate = t + 2;
  }
  else {
    numberToGenerate = x + 2;
  }
  biases.resize(numberToGenerate);
  for (unsigned int i = 0; i < biases.size(); i++) {
    biases.at(i) = generateBeta();
  }
  return biases;
}

void FirstPassageDriver::iterateTimeStep()
{
  std::vector<RealType> biases = getBiases();
  for (unsigned int i = 0; i < pdfs.size(); i++) {
    if (pdfs.at(i).haltingFlag){
      continue;
    }
    pdfs[i].iterateTimeStep(biases);
  }
  t += 1;
}

std::tuple<std::vector<unsigned int long>,
           std::vector<RealType>,
           std::vector<unsigned int long>>
FirstPassageDriver::evolveToCutoff(RealType nParticles,
                                   RealType cutoff,
                                   std::string filePath,
                                   bool writeHeader)
{
  std::vector<unsigned int long> quantileTimes;
  std::vector<RealType> variance;
  std::vector<unsigned int long> positions;
  std::vector<ParticleData> particlesData;
  for (unsigned int i = 0; i < maxPositions.size(); i++) {
    particlesData.push_back(ParticleData(nParticles));
  }
  RealType firstPassageCDF;

  // Set up file writing
  std::ofstream myfile;
  myfile.open(filePath, std::ios::app);

  // This saves the data to double precision.
  // Could change to RealType but that gives too much info
  myfile << std::fixed
         << std::setprecision(std::numeric_limits<double>::max_digits10);
  if (writeHeader) {
    myfile << "distance,quantile,variance\n";
    myfile.flush();
  }

  unsigned int numberHalted = 0;
  while (numberHalted < pdfs.size()) {
    iterateTimeStep();
    // Each ParticleData object corresponds to a different pdf.
    for (unsigned int i = 0; i < pdfs.size(); i++) {
      // Get the measurement of the quantile position
      FirstPassageBase* pdf = &pdfs.at(i);
      ParticleData* it = &particlesData.at(i);
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
        myfile << pdf->getMaxPosition() << "," << it->quantileTime << ","
               << it->variance << "\n";
        myfile.flush();

        // now set flags to false
        pdf->haltingFlag = true;
        numberHalted += 1;
      }
    }
  }
  myfile.close();
  return std::make_tuple(quantileTimes, variance, positions);
}
