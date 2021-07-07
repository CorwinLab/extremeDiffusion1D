#include "diffusion.hpp"

//Constuctor and destructor
Diffusion::Diffusion(
  const double _nParticles,
  const double _beta,
  const double _smallCutoff=pow(2, 31)-2,
  const double _largeCutoff=1e31,
  const bool _probDistFlag=false;
  const long occupancySize)
  : nParticles(_nParticles), beta(_beta), smallCutoff(_smallCutoff), largeCutoff(_largeCutoff), probDistFlag(_probDistFlag)
{
  // create the occupancylist and the edges, etc
}
