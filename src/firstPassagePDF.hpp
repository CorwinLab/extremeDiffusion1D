#ifndef FISRTPASSAGEPDF_HPP_
#define FISRTPASSAGEPDF_HPP_

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

#include "firstPassagePDFBase.hpp"
#include "randomNumGenerator.hpp"

class FirstPassagePDFMain : public RandomNumGenerator {
private:
    std::vector<FirstPassagePDFBase> pdfs;
    std::vector<unsigned long int> maxPositions;
    std::vector<double> transitionProbabilities;
    unsigned long int t;
    RealType nParticles;

public:
    FirstPassagePDFMain(const double _beta, std::vector<unsigned long int> _maxPosition, RealType nParticles);
    ~FirstPassagePDFMain(){};

    std::vector<FirstPassagePDFBase> getPDFs() { return pdfs; };
    void setPDFs(std::vector<FirstPassagePDFBase> _pdfs) { pdfs = _pdfs; };

    unsigned long int getTime(){ return t; };
    void setTime(unsigned long int _t){ t = _t; };

    std::vector<double> getTransitionProbabilities() { return transitionProbabilities; };
    std::vector<double> generateTransitionProbabilities();

    std::vector<unsigned long int> getMaxPositions() { return maxPositions; };
    void setMaxPositions(std::vector<unsigned long int> _maxPositions){ maxPositions = _maxPositions; };

    std::vector<RealType> iteratePDF(std::vector<RealType> pdf, std::vector<double> transitionProbabilities, int transitionProbIndex);
    void iterateTimeStep();

    std::tuple<std::vector<unsigned long int>, std::vector<RealType> > evolveToCutoff(RealType cutoff, RealType nParticles);
};

#endif /* FIRSTPASSAGEPDF_HPP_ */ 