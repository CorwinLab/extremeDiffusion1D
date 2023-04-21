from .lDiffusionLink import libDiffusion
import numpy as np
import npquad 
from typing import List
import csv

class ScatteringModel(libDiffusion.Scattering):
    def __init__(self, distName: str, params: List[float], size: int):
        super().__init__(distName, params, size)

    @property
    def pright(self):
        return np.array(self.getpright())
    
    @property
    def pleft(self):
        return np.array(self.getpleft())
    
    def evolveToTime(self, t):
        while self.getTime() < t: 
            self.iterateTimestep()
    
    def getVelocities(self, times, vs, save_files):
        files = [open(i, 'a') for i in save_files]
        writers = [csv.writer(f) for f in files]

        for i in range(len(writers)):
            writers[i].writerow(["Time", "Position", "logP", "Delta", "P"])
            files[i].flush()

        for t in times:
            self.evolveToTime(t)
            xvals = np.floor(vs * self.getTime() **(3/4)) 
            idx = np.ceil((xvals + self.getTime()) / 2).astype(int) # need to account for shifting index

            for i in range(len(xvals)): 
                prob = self.getProbAbove(idx[i])
                delta = self.getDeltaAt(idx[i])
                probAt = self.getProbAt(idx[i])

                writers[i].writerow([self.getTime(), xvals[i], np.log(prob).astype(float), delta.astype(float), probAt.astype(float)])
                files[i].flush()

        for f in files:
            f.close()