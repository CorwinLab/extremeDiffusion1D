#include "randomDistribution.hpp"
#include <cmath>

RandomDistribution::RandomDistribution(std::string _distributionName,
                                       std::vector<double> _parameters)
    : distributionName(_distributionName), parameters(_parameters)
{
    if (distributionName != "beta" && distributionName != "bates" && distributionName != "triangular" && distributionName != "uniform" && distributionName != "quadratic"){
        throw std::runtime_error("distributionName must be either 'beta' or 'bates'");
    }
    /* Set up beta distribution */
    if (distributionName == "beta"){
        if (parameters[0] != 0  || parameters[1] != 0) {
            boost::random::beta_distribution<>::param_type params(parameters[0], parameters[1]); /* (alpha, beta) */ 
            betaParams = params;
        }
    }

    /* Set up triangular distribution */ 
    if (distributionName == "triangular"){
        boost::random::triangle_distribution<>::param_type t_params(parameters[0], parameters[1], parameters[2]); /* (a, b, c) */
        triang_dist.param(t_params);
    }

    /* Set up cutoff uniform distribution */ 
    if (distributionName == "uniform"){
        std::uniform_real_distribution<>::param_type cutoffParams(parameters[0], parameters[1]);
        cutoff_uniform.param(cutoffParams);
    }

    /* Set up quadratic distribution */
    if (distributionName == "quadratic"){
        beta = (parameters[0] + parameters[1]) / 2;
        alpha = 12 / (pow(parameters[1] - parameters[0], 3));
    }

    /* Set up random uniform distribution*/
    std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
    dis.param(unifParams);

    /* Seed generator */
    gen.seed(rd());
}

double RandomDistribution::getBetaDistributed(){
    if (parameters.size() != 2){
        throw std::runtime_error("Parameters is not correct size");
    }
    if (parameters[0] == 0.0 && parameters[1] == 0.0) {
        return round(dis(gen));
    }
    // If beta = 1 use random uniform distribution
    else if (parameters[0] == 1 && parameters[1] == 1) {
        return dis(gen);
    }
    // If beta = inf return 0.5
    else if (isinf(parameters[0]) && isinf(parameters[1])) {
        return 0.5;
    }
    else {
        double randomVal = beta_dist(gen, betaParams);
        // If returned value is NaN, there's either an overflow or underflow 
        // So return 0 or 1. 
        if (isnan(randomVal)){
            randomVal = round(dis(gen));
        }
        return randomVal;
    }
}

double RandomDistribution::getBatesDistributed(){
    double mean = 0;
    for (unsigned int i=0; i < (unsigned int)parameters[0]; i++){
        mean += dis(gen);
    }
    return mean / parameters[0];
}

double RandomDistribution::getTriangularDistributed(){
    return triang_dist(gen);
}

double RandomDistribution::getUniformDistributed(){
    return cutoff_uniform(gen);
}

double RandomDistribution::getQuadraticDistributed(){
    double val = (3 * dis(gen) / alpha - pow(beta - parameters[0], 3));
    if (val < 0){
        val = -pow(abs(val), 1./3);
    }
    else{
        val = pow(val, 1./3);
    }
    return val + beta;
}

double RandomDistribution::generateRandomVariable(){
    if (distributionName == "beta"){
        return getBetaDistributed();
    }
    else if (distributionName == "bates"){
        return getBatesDistributed();
    }
    else if (distributionName == "triangular"){
        return getTriangularDistributed();
    }
    else if (distributionName == "uniform"){
        return getUniformDistributed();
    }
    else if (distributionName == "quadratic"){
        return getQuadraticDistributed();
    }
    else {
        throw;
    }
}