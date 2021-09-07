#!/bin/bash
c++ -O3 -Wall -std=gnu++11 -I/usr/include/ -o test.out diffusionND.cpp -L/usr/lib/ -lgsl -lgslcblas -lm
