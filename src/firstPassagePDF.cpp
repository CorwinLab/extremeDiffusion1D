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

#include "firstPassagePDF.hpp"
#include "randomNumGenerator.hpp"

FirstPassagePDFMain::FirstPassagePDFMain(
    const double _beta,
    std::vector<unsigned long int> _maxPositions,
    RealType _nParticles)
    : RandomNumGenerator(_beta), maxPositions(_maxPositions), nParticles(_nParticles)
{
  sort(maxPositions.begin(), maxPositions.end());
  for (unsigned int i = 0; i < maxPositions.size(); i++) {
    pdfs.push_back(FirstPassagePDFBase(maxPositions[i]));
  }
  t = 0;

  CDFSum = 0;
}

std::vector<RealType>
FirstPassagePDFMain::iteratePDF(std::vector<RealType> PDF,
                                std::vector<double> transitionProbabilities, 
                                int transitionProbIndex)
{

  std::vector<RealType> newPDF(PDF.size(), 0);

  for (unsigned int i = 0; i < PDF.size(); i++) {
    if (i == 0) {
      newPDF.at(0) = PDF.at(0) + transitionProbabilities[1 + transitionProbIndex] * PDF.at(1);
    }
    else if (i == 1) {
      newPDF.at(1) = transitionProbabilities[2+ transitionProbIndex] * PDF.at(2);
    }
    else if (i == PDF.size() - 1) {
      newPDF.at(i) =
          PDF.at(i) + (1 - transitionProbabilities[i - 1+ transitionProbIndex]) * PDF.at(i - 1);
    }
    else if (i == PDF.size() - 2) {
      newPDF.at(i) = (1 - transitionProbabilities[i - 1+ transitionProbIndex]) * PDF.at(i - 1);
    }
    else {
      newPDF.at(i) = (1 - transitionProbabilities[i - 1+ transitionProbIndex]) * PDF.at(i - 1) +
                     transitionProbabilities[i + 1+ transitionProbIndex] * PDF.at(i + 1);
    }
  }
  return newPDF;
}

std::vector<double> FirstPassagePDFMain::generateTransitionProbabilities()
{
  std::vector<RealType> pdf = pdfs.back().getPDF();
  std::vector<double> _transitionProbabilities(pdf.size(), 0);
  for (unsigned int i = 0; i < _transitionProbabilities.size(); i++) {
      if (pdf.at(i) != 0){
        _transitionProbabilities.at(i) = generateBeta();
      }
  }
  return _transitionProbabilities;
}

void FirstPassagePDFMain::iterateTimeStep()
{
  std::vector<RealType> newPDF;
  std::vector<double> _transitionProbabilities =
      generateTransitionProbabilities();
  transitionProbabilities = _transitionProbabilities;

  for (unsigned long int i = 0; i < pdfs.size(); i++) {
    unsigned int maxSize = pdfs.back().getMaxPosition();
    unsigned int currentSize = pdfs[i].getMaxPosition();
    newPDF = iteratePDF(pdfs[i].getPDF(), transitionProbabilities, maxSize - currentSize);
    pdfs[i].setPDF(newPDF);
    RealType firstPassageProbability =
        transitionProbabilities.at(1) * newPDF.at(1) +
        (1 - transitionProbabilities.at(newPDF.size() - 2)) *
            newPDF.at(newPDF.size() - 2);

    // Set the single particle first passage probability and 
    // first passage cdf 
    pdfs[i].setFirstPassageProbability(firstPassageProbability);
    pdfs[i].setCDFSum(pdfs[i].getCDFSum() + firstPassageProbability);

    // Set the nParticle first passage cdf
    std::vector<RealType> currentCDFN = pdfs[i].getCDFN();
    currentCDFN.push_back(1-exp(-pdfs[i].getCDFSum() * nParticles));
    pdfs[i].setCDFN(currentCDFN);
  }
  t += 1;
}

// I think this may be far easier if you input nParticles into the original class :(
// Fuck this isn't nice at all.
std::tuple<std::vector<unsigned long int>, std::vector<RealType> >
FirstPassagePDFMain::evolveToCutoff(RealType cutoff, RealType nParticles){
  std::vector<unsigned long int> quantiles(pdfs.size()); 
  std::vector<RealType> variance(pdfs.size());

  std::vector<std::vector<RealType> > cdfN;
  std::vector<std::vector<unsigned long int> > times;

  std::vector<unsigned int> positionIndex(pdfs.size());
  for (unsigned int i=0; i<positionIndex.size(); i++){
    positionIndex[i] = i;
  }
  
  while (!positionIndex.empty()){
    iterateTimeStep();
    for (unsigned int i=0; i<positionIndex.size(); i++){
      unsigned int index = positionIndex[i];
      cdfN[index].push_back(1 - exp(-pdfs[index].getCDFSum() * nParticles));
      times.push_back(t);

      if ((cdfN[index].back() >= cutOff) && pdfs[index].getCDFSum() >= 1/nParticles){
        unsigned int timeIndex = findQuantile(cdf)
      }
    }
  }
  return std::make_tuple(quantiles, variance);
}

unsigned long int findQuantile(std::vector<RealType> cdf, RealType nParticles){
  unsigned long int index; 
  for (unsigned int i=0; i<cdf.size(); i++){
    if (cdf[i] >= 1/nParticles){
      index = i;
      break;
    }
  }
  return index;
}