#include <iostream>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int main(void){
  gsl_rng_env_setup();

  const gsl_rng_type * T = gsl_rng_default;
  gsl_rng * r = gsl_rng_alloc(T);

  for (int i = 0; i < 100; i++)
    {
      std::vector<double> theta = { 1.0 };
      std::vector<double> alpha = { 1.0 };
      int n = theta.size();
      gsl_ran_dirichlet(r, n, alpha.begin(), theta.begin());
      std::cout << theta << std::endl;
    }

  gsl_rng_free(r);

  return 0;
}
