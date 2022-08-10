import numpy as np
import npquad
from typing import List

from .lDiffusionLink import libDiffusion


class FirstPassageDriver(libDiffusion.FirstPassageDriver):
    def __init__(self, beta: float, maxPositions: List[int]):
        super().__init__(beta, maxPositions)

    @property
    def time(self) -> int:
        return self.getTime()
    
    @time.setter
    def time(self, time: int):
        self.setTime(time)
    
    def iterateTimeStep(self):
        super().iterateTimeStep()
    
    def getBiases() -> List[np.quad]: 
        return super().getBiases()
    
    def getPDFs(self) -> List["FirstPassageBase"]:
        return super().getPDFs()

    def setPDFs(self, pdfs: List["FirstPassageBase"]):
        super().setPDFs(pdfs)


    def evolveToCutoff(self, nParticles: np.quad, filePath: str, cutoff: float=1, writeHeader: bool=True):
        super().evolveToCutoff(nParticles, cutoff, filePath, writeHeader)
