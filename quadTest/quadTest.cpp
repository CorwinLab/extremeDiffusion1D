#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <boost/multiprecision/float128.hpp>
#include <iostream>

typedef boost::multiprecision::float128 RealType;

void func(RealType* dest, unsigned char* src){
  dest = (RealType*) src;
}

void chartofloat128(std::vector<unsigned char> x){
  RealType y = RealType(*x[0]);
  func(&y, &x[0])
  std::cout << y << std::endl;
}

PYBIND11_MODULE(quadTest, m){
  m.def("chartofloat128", &chartofloat128);
}
