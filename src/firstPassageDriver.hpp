#ifndef FISRTPASSAGEDRIVER_HPP_
#define FISRTPASSAGEDRIVER_HPP_

#include "firstPassageBase.hpp"
#include "randomNumGenerator.hpp"
#include <vector>

class FirstPassageDriver : public RandomNumGenerator {
protected:
  std::vector<FirstPassageBase> pdfs;
  unsigned int t;
  std::vector<unsigned int long> maxPositions;

public:
  FirstPassageDriver(const double _beta,
                     std::vector<unsigned int long> _maxPositions);
  ~FirstPassageDriver(){};

  std::vector<FirstPassageBase> getPDFs() { return pdfs; };
  void setPDFs(std::vector<FirstPassageBase> _pdfs) { pdfs = _pdfs; };

  unsigned int getTime() { return t; };
  void setTime(unsigned int _t) { t=_t; };

  void iterateTimeStep();
  std::vector<RealType> getBiases();
  std::tuple<std::vector<unsigned int long>,
             std::vector<RealType>,
             std::vector<unsigned int long>>
  evolveToCutoff(RealType nParticles, RealType cutoff, bool writeHeader);
};

#endif /* FISRTPASSAGEDRIVER_HPP_ */