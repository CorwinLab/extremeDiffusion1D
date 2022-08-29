#include "randomDistribution.hpp"

RandomDistribution::RandomDistribution(std::string _distributionName,
                                       std::vector<double> _parameters)
    : distributionName(_distributionName), parameters(_parameters)
{
    /* Set up beta distribution */
    if (distributionName == "beta"){
        if (parameters[0] != 0  || parameters[1] != 0) {
            boost::random::beta_distribution<>::param_type params(parameters[0], parameters[1]); /* (alpha, beta) */ 
            betaParams = params;
        }
    }

    /* Set up random uniform distribution*/
    std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
    dis.param(unifParams);

    /* Seed generator */
    gen.seed(rd());
}

double RandomDistribution::getBetaDistributed(){
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

double RandomDistribution::generateRandomVariable(){
    if (distributionName == "beta"){
        return getBetaDistributed();
    }
    else if (distributionName == "bates"){
        return getBatesDistributed();
    }
    else {
        throw;
    }
}