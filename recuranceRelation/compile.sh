#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) recurrance.cpp -I/c/modular-boost -lquadmath -o recurrance.so -I"../../pybind11/include"
