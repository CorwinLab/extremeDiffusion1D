#include "environment.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// Constructor
Environment::Environment(const unsigned long int tMax,
                        const unsigned long int xMax){
  biases.resize(tMax, std::vector<double> (xMax, 0));
}

void Environment::fillBiases(){
  
}

PYBIND11_MODULE(environment, m)
{
  m.doc() = "C++ for generating environments or random numbers";

  py::class_<Environment>(m, "Environment")
    .def(py::init<const unsigned long int, const unsigned long int>())
    .def("getBiases", &Environment::getBiases);
}
