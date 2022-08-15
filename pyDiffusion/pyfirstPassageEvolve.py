import numpy as np
import npquad
from typing import List, Tuple
import csv

from .lDiffusionLink import libDiffusion

class FirstPassageEvolve(libDiffusion.FirstPassageEvolve):
    def __init__(self, beta: float, maxPositions: List[int], nParticles: np.quad):
        super().__init__(beta, maxPositions, nParticles)
    
    def checkParticleData(self) -> Tuple[List[int], List[np.quad], List[int]]:
        return super().checkParticleData()

    @property
    def numberOfPositions(self):
        return self.getNumberOfPositions()

    @property 
    def particleData(self):
        return self.getParticleData()
    
    @particleData.setter
    def particleData(self, _particleData):
        self.setParticleData(_particleData)

    @property
    def numberHalted(self):
        return self.getNumberHalted()
    
    @numberHalted.setter
    def numberHalted(self, _numberHalted):
        self.setNumberHalted(_numberHalted)
    
    @property 
    def nParticles(self):
        return self.getNParticles()

    @nParticles.setter
    def nParticles(self, _nParticles):
        self.setNParticles(_nParticles)

    def evolveToCutoff(self, file: str, writeHeader: bool = True, cutoff: float=1):
        f = open(file, 'a')
        writer = csv.writer(f)
        if writeHeader:
            writer.writerow(["position", "quantile", "variance"])
        

        while (self.numberHalted < self.numberOfPositions):
            self.iterateTimeStep()
            quantile, variance, position = self.checkParticleData()
            if (not quantile) and (not variance) and (not position): 
                continue
            for p, q, v in zip(position, quantile, variance):
                writer.writerow([p, q, v])
                f.flush()
        f.close()