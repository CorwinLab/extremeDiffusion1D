#!/bin/bash

NPARTICLES=1e24
MAXDISTANCE=276310
BETA=0.01

math -script getGeneralBetaVariance.m $NPARTICLES $MAXDISTANCE $BETA
math -script getGeneralBetaSamplingVariance.m $NPARTICLES $MAXDISTANCE $BETA