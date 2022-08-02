#include <math.h>
#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>

#include "firstPassagePDFBase.hpp"
#include "randomNumGenerator.hpp"
#include "stat.h"

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

FirstPassagePDFBase::FirstPassagePDFBase(const unsigned long int _maxPosition)
{
  maxPosition = _maxPosition;
  PDF.resize(2 * maxPosition + 1);

  // Set middle element of array to 1
  PDF[maxPosition] = 1;
}

/*
std::tuple<unsigned int long, RealType>
FirstPassagePDFBase::evolveToCutoff(RealType cutOff, RealType nParticles)
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

  while ((cdf_sumN < cutOff) || (cdf_sum < 1/nParticles)) {
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
  for (unsigned int i=0; i<cdf.size(); i++){
    if (cdf[i] >= 1/nParticles){
      quantileTime = times[i];
      break;
    }
  }

  // Find the variance from the CDF of nParticles
  // First get the PDF of the system
  std::vector<RealType> pdfN(cdfN.size()-1);
  for (unsigned int i=0; i < cdfN.size()-1; i++){
    pdfN[i] = cdfN[i+1] - cdfN[i];
  }
  std::vector<unsigned int long> pdfTimes = slice(times, 0, pdf.size()-2);
  RealType var = calculateVarianceFromPDF(pdfTimes, pdfN);

  return std::make_tuple(quantileTime, var);
}
*/
