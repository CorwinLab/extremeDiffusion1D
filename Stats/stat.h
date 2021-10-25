// #include <math.h> I think this might be deprecated
#include <cmath>
#include <vector>
#include <iostream>
#include <utility>

template <class RealType, class x_numeric>
RealType calculateMeanFromPDF(std::vector<x_numeric> xvals, std::vector<RealType> PDF){
  RealType mean = 0;
  for (unsigned long int i=0; i < PDF.size(); i++){
    mean += xvals[i] * PDF[i];
  }
  return mean;
}

template <class RealType, class x_numeric>
RealType calculateVarianceFromPDF(std::vector<x_numeric> xvals, std::vector<RealType> PDF){
  RealType mean = calculateMeanFromPDF(xvals, PDF);
  RealType var = 0;
  for (unsigned long int i=0; i < PDF.size(); i++){
    var += PDF[i] * pow(xvals[i] - mean, 2);
  }
  return var;
}

template <class RealType, class x_numeric>
RealType calculateMeanFromCDF(std::vector<x_numeric> xvals, std::vector<RealType> cdf)
{
  RealType mean = 0;
  int H;
  for (unsigned long int i=0; i < cdf.size(); i++){
    if (xvals.at(i) <= 0){
      H = 0;
    }
    else{
      H = 1;
    }
    mean += H - cdf[i];
  }
  return mean;
}

template <class RealType>
std::vector<RealType> getDiscretePDF(std::vector<RealType> comp_cdf, RealType nParticles){
  std::vector<RealType> pdf(comp_cdf.size()-1);
  RealType cdf_current, cdf_prev;
  RealType sum=0;
  for (unsigned long int i=1; i < comp_cdf.size(); i++){
    cdf_current = exp(-comp_cdf[i] * nParticles);
    cdf_prev = exp(-comp_cdf[i-1] * nParticles);
    pdf[i-1] = cdf_current - cdf_prev;
    sum += pdf[i-1];
  }
  return pdf;
}

template <class RealType>
std::vector<RealType> pdf_to_comp_cdf(std::vector<RealType> pdf, RealType norm){
  std::vector<RealType> comp_cdf(pdf.size()+1);
  comp_cdf[0] = 1;
  RealType sum = 0;
  for (auto i = 0; i < pdf.size(); i++){
    sum += pdf[i];
    comp_cdf[i+1] = 1.0 - sum / norm;
  }
  return comp_cdf;
}

template <class RealType, class x_numeric>
RealType getGumbelVarianceCDF(std::vector<x_numeric> xvals, std::vector<RealType> comp_cdf, RealType nParticles){
  std::vector<RealType> discrete_pdf = getDiscretePDF(comp_cdf, nParticles);
  return calculateVarianceFromPDF(xvals, discrete_pdf);
}

template<class RealType, class x_numeric>
RealType getGumbelVariancePDF(std::vector<x_numeric> xvals, std::vector<RealType> pdf, RealType nParticles, RealType norm){
  std::vector<RealType> comp_cdf = pdf_to_comp_cdf(pdf, norm);
  return getGumbelVarianceCDF(xvals, comp_cdf, nParticles);
}
