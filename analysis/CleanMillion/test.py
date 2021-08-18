import sys

sys.path.append("../../src")
from fileIO import loadArrayQuad

file = "/home/jacob/Desktop/corwinLabMount/CleanData/QuartilesMillion/Quartiles0.txt"

with open(file, "r") as f:
    print(f.readline())
