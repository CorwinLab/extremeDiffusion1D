#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) diffusionSystems.cpp -I/c/modular-boost -lquadmath -o diffusionSystems.so -I"../../pybind11/include"
