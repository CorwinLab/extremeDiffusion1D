#ifndef FISRTPASSAGEDRIVER_HPP_
#define FISRTPASSAGEDRIVER_HPP_

#include <vector> 
#include "firstPassagePDF.hpp"
#include "randomNumGenerator.hpp"

class FirstPassageDriver : public RandomNumGenerator {
protected:
    std::vector<FirstPassagePDF> pdfs;
    unsigned int t;
    std::vector<unsigned int long> maxPositions;

public:
  FirstPassageDriver(const double _beta, std::vector<unsigned int long> maxPositions);
  ~FirstPassageDriver(){};

  void iterateTimeStep();
  std::vector<RealType> getBiases();

  std::vector<FirstPassagePDF> getPDFs() { return pdfs; };
  void setPDFs(std::vector<FirstPassagePDF> _pdfs) { pdfs = _pdfs; };
};

#endif /* FISRTPASSAGEDRIVER_HPP_ */