#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=c++11 -fPIC $(python3-config --includes) quadTest.cpp -I/c/modular-boost -lquadmath -o quadTest.so -I"../../pybind11/include"
