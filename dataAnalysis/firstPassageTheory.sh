#!/bin/bash

NPARTICLES=1e24
MAXDISTANCE=55262

math -script getEnvFirstPassage.m $NPARTICLES $MAXDISTANCE
