#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=c++11 -fPIC $(python3-config --includes) diffusion.cpp -o diffusion.so -I"../../pybind11/include"
