#include <iostream>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <random>

class DiffusionND{
private:
  const gsl_rng_type * T = gsl_rng_default;
  gsl_rng * r;
  int dim;
  std::vector<std::vector<double> > occupancy;
  unsigned long int tMax;
  
public:
  DiffusionND(int _dim, unsigned long int _tMax) : dim(_dim), tMax(_tMax) {
    gsl_rng_env_setup();
    r = gsl_rng_alloc(T);
    std::random_device rd;
    gsl_rng_set(r, rd());
  };

  ~DiffusionND(){ gsl_rng_free(r); };

  std::vector<double> generateRand(){
    std::vector<double> theta(dim);
    std::vector<double> alpha(dim, 1);
    gsl_ran_dirichlet(r, theta.size(), &alpha[0], &theta[0]);
    for (auto i=0; i < theta.size(); i++){
      std::cout << theta[i] << std::endl;
    }
    return theta;
  }
};

int main(void){
  gsl_rng_env_setup(); // set gsl_rng_default seed

  // Create random number generator of type gsl_rng_default
  const gsl_rng_type * T = gsl_rng_default;
  gsl_rng * r = gsl_rng_alloc(T);

  for (int i = 0; i < 100; i++)
    {
      std::vector<double> theta(3);
      std::vector<double> alpha = { 1, 1, 1};
      int n = theta.size();
      gsl_ran_dirichlet(r, n, &alpha[0], &theta[0]);
      for (int i = 0; i < theta.size(); i++){
        std::cout << theta[i] << std::endl;
      }
      std::cout << "Finished this thing \n" << std::endl;
    }

  gsl_rng_free(r); // free memory associated with generator r

  std::cout << "moving to Class" << std::endl;
  DiffusionND d = DiffusionND(10);
  d.generateRand();

  d.generateRand();

  return 0;
}
