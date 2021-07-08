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
		double smallCutoff;
		double largeCutoff;
		bool ProbDistFlag;

    // It would be nice if this could be a generic distribution as:
    // boost::random::distribution bias
		boost::random::beta_distribution<>::param_type betaParams;

		std::random_device rd;
		boost::random::mt19937_64 gen;

		//std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> dis;
    boost::random::binomial_distribution<> binomial;
    boost::random::normal_distribution<> normal;
    boost::random::beta_distribution<> beta_dist;

		std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > edges;
		unsigned long int time;

    double toNextSite(double currentSite, double bias);
		double generateBeta();

	public:
    Diffusion(
      const double _nParticles,
      const double _beta,
      const unsigned long int occupancySize,
      const double _smallCutoff=pow(2, 31)-2,
      const double _largeCutoff=1e31,
      const bool _probDistFlag=false);
    ~Diffusion() {};

    double getNParticles(){ return nParticles; };

    double getBeta(){ return betaParams.beta(); };

    void setSmallCutoff(double _smallCutoff){ smallCutoff = _smallCutoff; };
    double getSmallCutoff(){ return smallCutoff; };

    void setLargeCutoff(double _largeCutoff){ largeCutoff = _largeCutoff; };
    double getLargeCutoff(){ return largeCutoff; };

    void setProbDistFlag(bool _probDistFlag){ ProbDistFlag = _probDistFlag; };
    bool getProbDistFlag(){ return ProbDistFlag; };

    void setOccupancy(const std::vector<double> _occupancy){ occupancy = _occupancy; };
    std::vector<double> getOccupancy(){ return occupancy; };

    double getTime(){ return time; };

    std::pair<std::vector<unsigned long int>, std::vector<unsigned long int> > getEdges(){ return edges; };

    //Functions that do things

    void iterateTimestep();

		double NthquartileSingleSided(const double NQuart);
  };

#endif /* DIFFUSION_HPP_ */
