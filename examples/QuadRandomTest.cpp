#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/random.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <chrono>

int main()
{
   using std::chrono::high_resolution_clock;
   using std::chrono::duration_cast;
   using std::chrono::milliseconds;
   using namespace boost::multiprecision;
   using namespace boost::random;
   using RealType = boost::multiprecision::float128;

   //
   // Declare our random number generator type, the underlying generator
   // is the Mersenne twister mt19937 engine, and we'll generate 256 bit
   // random values, independent_bits_engine will make multiple calls
   // to the underlying engine until we have the requested number of bits:
   //
   independent_bits_engine<mt19937, std::numeric_limits<RealType>::digits, cpp_int> gen;
   uniform_real_distribution<RealType> dis(0, 1);
   beta_distribution<RealType> beta;
   beta_distribution<RealType>::param_type params(1, 1);
   //
   // Generate some values:
   //
   unsigned numofPoints = 1e6;
   auto t1 = high_resolution_clock::now();
   for(unsigned i = 0; i < numofPoints; ++i){
      RealType val = beta(gen, params);
   }
   auto t2 = high_resolution_clock::now();
   auto ms_int = duration_cast<milliseconds>(t2 - t1);
   std::cout << "Quad precision: " << ms_int.count() << "ms\n";

   // Compare to double precision random values
   beta_distribution<double> doubleBeta;
   beta_distribution<double>::param_type doubleParams(1, 1);
   mt19937_64 doubleGen;
   t1 = high_resolution_clock::now();
   for(unsigned i=0; i < numofPoints; ++i){
     double doubleVal = doubleBeta(doubleGen, doubleParams);
   }
   t2 = high_resolution_clock::now();
   ms_int = duration_cast<milliseconds>(t2-t1);
   std::cout << "Double precision: " << ms_int.count() << "ms\n";
}
