#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) diffusion.cpp -I/c/modular-boost -lquadmath -o diffusion.so -I"../../pybind11/include"
