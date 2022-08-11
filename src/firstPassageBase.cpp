#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <math.h>

#include "firstPassageBase.hpp"
#include "randomNumGenerator.hpp"
#include "stat.hpp"
#include "particleData.hpp"

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

FirstPassageBase::FirstPassageBase(const unsigned long int _maxPosition)
{
  maxPosition = _maxPosition;
  PDF.resize(maxPosition + 2);

  firstPassageCDF = 0;

  // Set first element of array to 1
  PDF[0] = 1;
  haltingFlag = false;
}

void FirstPassageBase::iterateTimeStep(std::vector<RealType> biases)
{
  std::vector<RealType> pdf_new(PDF.size());
  RealType bias;
  RealType newfirstPassageCDF = 0; // Reset to zero

  // Okay this stuff is workign as expected
  if (t < maxPosition) {
    for (unsigned int i = 0; i <= t; i++) {
      bias = biases.at(i);
      pdf_new.at(i) += PDF.at(i) * bias;
      pdf_new.at(i + 1) += PDF.at(i) * (1 - bias);
    }
    if(t == maxPosition - 1){
      newfirstPassageCDF += pdf_new.at(0) + pdf_new.at(pdf_new.size()-2);
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
          bias = biases.at(i);
          pdf_new.at(i) += PDF.at(i) * (1-bias);
          pdf_new.at(i - 1) += PDF.at(i) *  bias;
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
          bias = biases.at(i);
          pdf_new.at(i) += PDF.at(i) * bias;
          pdf_new.at(i + 1) += PDF.at(i) * (1-bias);
        }
      }
    }
    newfirstPassageCDF += pdf_new.at(0);
    if (pdf_new.back() == 0) {
      newfirstPassageCDF += pdf_new.at(pdf_new.size() - 2);
    }
    else {
      newfirstPassageCDF += pdf_new.back();
    }
  }
  if (firstPassageCDF > newfirstPassageCDF){
    throw std::runtime_error("Trying to set first passage cdf to value smaller than previous first passage cdf");
  }
  firstPassageCDF = newfirstPassageCDF;
  PDF = pdf_new;
  t += 1;
}
