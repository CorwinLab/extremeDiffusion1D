#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) diffusionND.cpp -I/c/modular-boost -lquadmath -lgsl -lgslcblas -o diffusionND.so -I"../pybind11/include"
