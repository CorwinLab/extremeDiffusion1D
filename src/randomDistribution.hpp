#ifndef RANDOMDISTRIBUTION_HPP_
#define RANDOMDISTRIBUTION_HPP_

#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/triangle_distribution.hpp>
#include <random>
#include <string>
#include <vector>

class RandomDistribution{
    protected:
        // Generator and uniform random
        boost::random::mt19937_64 gen;
        std::uniform_real_distribution<> dis;

        // Beta distribution
        boost::random::beta_distribution<>::param_type betaParams;
        std::random_device rd;
        boost::random::beta_distribution<> beta_dist;
        double getBetaDistributed();

        // Bates distribution
        double getBatesDistributed();

        // Triangular distribution
        boost::random::triangle_distribution<> triang_dist;
        double getTriangularDistributed();

        // Cutoff Uniform distribution
        std::uniform_real_distribution<> cutoff_uniform;
        double getUniformDistributed();

        // Quadratic distribution
        double alpha;
        double beta;
        double getQuadraticDistributed();

        // Delta distribution
        std::discrete_distribution<> disc_dist;
        double getDeltaDistributed();
        
        std::string distributionName;
        std::vector<double> parameters;
        
    public: 
        RandomDistribution(std::string _distributionName, std::vector<double> _parameters);
        ~RandomDistribution(){};

        std::string getDistributionName(){ return distributionName; };
        void setDistributionName(std::string _distributionName){ distributionName = _distributionName; };

        std::vector<double> getParameters() { return parameters; };
        void setParameters(std::vector<double> _parameters){ parameters = _parameters; };

        void setGeneratorSeed(const unsigned int seed) { gen.seed(seed); };
        double generateRandomVariable();
};

#endif /* RANDOMDISTRIBUTION_HPP_ */