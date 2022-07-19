#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) continuousDiffusion.cpp -I/c/modular-boost -lquadmath -o continuousDiffusion.so -I"../../pybind11/include"
