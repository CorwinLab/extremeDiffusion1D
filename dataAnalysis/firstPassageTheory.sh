#!/bin/bash

NPARTICLES=1e7
MAXDISTANCE=16118

math -script getEnvFirstPassage.m $NPARTICLES $MAXDISTANCE
