#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) environment.cpp -I/c/modular-boost -lquadmath -o environment.so -I"../../pybind11/include"
