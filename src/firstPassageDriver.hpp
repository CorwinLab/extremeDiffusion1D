#ifndef FISRTPASSAGEDRIVER_HPP_
#define FISRTPASSAGEDRIVER_HPP_

#include "firstPassagePDF.hpp"
#include "randomNumGenerator.hpp"
#include <vector>

class FirstPassageDriver : public RandomNumGenerator {
protected:
  std::vector<FirstPassagePDF> pdfs;
  unsigned int t;
  std::vector<unsigned int long> maxPositions;

public:
  FirstPassageDriver(const double _beta,
                     std::vector<unsigned int long> maxPositions);
  ~FirstPassageDriver(){};

  std::vector<FirstPassagePDF> getPDFs() { return pdfs; };
  void setPDFs(std::vector<FirstPassagePDF> _pdfs) { pdfs = _pdfs; };

  void iterateTimeStep();
  std::vector<RealType> getBiases();
  std::tuple<std::vector<unsigned int long>,
             std::vector<RealType>,
             std::vector<unsigned int long>>
  evolveToCutoff(RealType nParticles, RealType cutoff);
};

#endif /* FISRTPASSAGEDRIVER_HPP_ */