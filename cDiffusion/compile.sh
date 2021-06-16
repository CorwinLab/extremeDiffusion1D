#!/bin/bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) cDiffusion.cpp -o cDiffusion.so -I"../../pybind11/include"
