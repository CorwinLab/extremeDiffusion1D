#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) diffusionCDF.cpp -I/c/modular-boost -lquadmath -o diffusionCDF.so -I"../../pybind11/include"
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) diffusionPDF.cpp -I/c/modular-boost -lquadmath -o diffusionPDF.so -I"../../pybind11/include"
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) firstPassagePDF.cpp -I/c/modular-boost -lquadmath -o firstPassagePDF.so -I"../../pybind11/include"