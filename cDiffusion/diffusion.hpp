#include <vector>
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <random>
#include <utility>
#include <assert.h>

#ifndef DIFFUSION_HPP_
#define DIFFUSION_HPP_


class Diffusion{
	private:
		std::vector<double> occupancy;
		double nParticles;
    // It would be nice if this could be a generic distribution as:
    // boost::random::distribution bias
		boost::random::beta_distribution<>::param_type betaParams;
    std::random_device rd;
    //boost::random::mt19937_64 gen(rd());
    boost::random::mt19937_64 gen;
    //std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> dis;
    boost::random::binomial_distribution<> binomial;
    boost::random::normal_distribution<> normal;
    boost::random::beta_distribution<> beta_dist;

		double smallCutoff;
		double largeCutoff;
    bool setProbDistFlag;

		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edges;
		unsigned long int time;

    double toNextSite(double currentSite, ???);


	public:
    Diffusion(const double _nParticles, const double _beta, const double _smallCutoff=pow(2, 31)-2,	const double _largeCutoff=1e31, const bool _probDistFlag=false, const long occupancySize);
    ~Diffusion() {};

    //void initializeOccupationAndEdges(unsigned long int size=0);
    //getters and setters
    double getNParticles();

    double getBeta();

    void setSmallCutoff(double);
    double getSmallCutoff();

    void setLargeCutoff(double);
    double getLargeCutoff();

    void setProbDistFlag(bool);

    void setOccupancy(const std::vector<double>);
    std::vector<double> getOccupancy();

    double getTime();

    std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > getEdges();

    //Functions that do things

    void iterateTimestep();

  };

// for time in pythonGeneratorWhatever(maxTime):
//    if mod(time, 10000):
//      saveData()
#endif /* DIFFUSION_HPP_ */
