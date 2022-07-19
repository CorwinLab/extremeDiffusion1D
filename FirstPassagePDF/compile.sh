#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) firstPassagePDF.cpp -I/c/modular-boost -lquadmath -o firstPassagePDF.so -I"../../pybind11/include"
