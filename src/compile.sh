#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) libDiffusion.cpp -I/c/modular-boost -lquadmath -o libDiffusion.so -I"../../pybind11/include"