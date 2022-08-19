import numpy as np
import npquad 
import sys 
from pyDiffusion import FirstPassageDriver, FirstPassagePDF
maxPositions = [3, 4, 5]
beta = 1

pdf = FirstPassageDriver(beta, maxPositions)
pdf.setBetaSeed(0)
max_time = 5
for i in range(max_time):
    print(f"i:")
    pdf.iterateTimeStep()
    pdfs = pdf.getPDFs()
    for p in pdfs: 
        print(list(np.array(p.getPDF()).astype(float)), sum(p.getPDF()))
    print("\n")

N = 100
pdf = FirstPassageDriver(beta, maxPositions)
quantile, variance, positions = pdf.evolveToCutoff(100, 'test1.csv', 1, writeHeader=True)
print(quantile, variance, positions)

for d in maxPositions:
    pdf = FirstPassagePDF(beta, d, False)
    pdf.setBetaSeed(0)
    quantile, variance, Ns = pdf.evolveToCutoffMultiple([N], 1)
    print(quantile, variance, Ns)

N = 1e24
maxPosition = 1000
beta = np.inf
pdf = FirstPassageDriver(beta, [maxPosition])
pdf.setBetaSeed(0)
quantile, variance, positions = pdf.evolveToCutoff(N, 'test2.csv', 1, writeHeader=True)
print(f"Driver: \nQuantile: {quantile[0]}, Variance:{variance[0]}")

pdf = FirstPassagePDF(beta, maxPosition, False)
pdf.setBetaSeed(0)
quantile, variance, Ns = pdf.evolveToCutoffMultiple([N], 1)
print(f"Single: \nQuantile: {quantile[0]}, Variance:{variance[0]}")


maxPosition = 5
beta = np.inf 

pdf = FirstPassageDriver(beta, [maxPosition])
maxTime = 10
for _ in range(maxTime):
    pdf.iterateTimeStep()
    print(list(np.array(pdf.getPDFs()[0].getPDF()).astype(float)), sum(np.array(pdf.getPDFs()[0].getPDF())))

pdf = FirstPassagePDF(beta, maxPosition, staticEnvironment=False)
maxTime = 10
for _ in range(maxTime):
    pdf.iterateTimeStep()
    print(list(np.array(pdf.getPDF()).astype(float)), sum(pdf.getPDF()))