import numpy as np
import npquad
from typing import List, Tuple
import csv
import json
import os

from .lDiffusionLink import libDiffusion


class FirstPassageEvolve(libDiffusion.FirstPassageEvolve):
    def __init__(self, beta: float, maxPositions: List[int], nParticles: np.quad):
        super().__init__(beta, maxPositions, nParticles)
        self.id = None
        self.save_dir = "."

    def checkParticleData(self) -> Tuple[List[int], List[np.quad], List[int]]:
        return super().checkParticleData()

    def __eq__(self, other: "FirstPassageEvolve"):
        if not isinstance(other, FirstPassageEvolve):
            raise TypeError(
                f"Comparison must be between same object types, but other of type {type(other)}"
            )

        if (
            self.time == other.time
            and np.all(self.particleData == other.particleData)
            and np.all(self.getPDFs() == other.getPDFs())
            and np.all(self.maxPositions == other.maxPositions)
            and self.numberHalted == other.numberHalted
            and self.nParticles == other.nParticles
            and self.getBeta() == other.getBeta()
            and self.id == other.id 
            and self.save_dir == other.save_dir
        ):
            return True
        return False

    def saveState(self):
        # Save particleData objects
        particleDataDict = {}
        for i in range(self.numberOfPositions):
            particleData = self.particleData[i]
            position = self.maxPositions[i]
            particleDataDict.update(
                {
                    position: {
                        "quantileTime": particleData.quantileTime,
                        "variance": str(particleData.variance),
                        "quantileSet": particleData.quantileSet,
                        "varianceSet": particleData.varianceSet,
                        "cdfPrev": str(particleData.cdfPrev),
                        "runningSumSquared": str(particleData.runningSumSquared),
                        "runningSum": str(particleData.runningSum),
                    }
                }
            )
        # Save pdf object data
        pdfsDataDict = {}
        for pdf in self.getPDFs():
            pdfsDataDict.update(
                {
                    pdf.getMaxPosition(): {
                        "time": pdf.getTime(),
                        "pdf": [str(i) for i in pdf.getPDF()],
                        "cdf": str(pdf.getFirstPassageCDF()),
                    }
                }
            )

        selfAttributes = {}
        vars = {
            "maxPositions": self.maxPositions,
            "time": self.time,
            "numberHalted": self.numberHalted,
            "nParticles": str(self.nParticles),
            "beta": self.getBeta(),
            "id": self.id,
            "save_dir": self.save_dir,
        }
        total_vars = {
            "particleData": particleDataDict,
            "pdfsData": pdfsDataDict,
            "vars": vars,
        }
        save_file = os.path.join(self.save_dir, "Scalars.json")
        with open(save_file, "w") as outfile:
            json.dump(total_vars, outfile, indent=4)

    @classmethod
    def fromFile(cls, file: str) -> "FirstPassageEvolve":
        with open(file, "r") as file:
            vars = json.load(file)

        beta = vars["vars"]["beta"]
        maxPositions = vars["vars"]['maxPositions']
        nParticles = np.quad(vars["vars"]['nParticles'])
        pdf = FirstPassageEvolve(beta, maxPositions, nParticles)
        pdf.id = vars["vars"]["id"]
        pdf.save_dir = vars["vars"]["save_dir"]
        pdf.numberHalted = vars["vars"]["numberHalted"]
        pdf.time = vars["vars"]["time"]
        pdf.nParticles = np.quad(vars["vars"]["nParticles"])
        
        PDFs = []
        for key, pdfData in vars["pdfsData"].items():
            pdfBase = libDiffusion.FirstPassageBase(int(key))
            pdfBase.setTime(pdfData['time'])
            pdfBase.setPDF([np.quad(i) for i in pdfData['pdf']])
            pdfBase.setMaxPosition(int(key))
            pdfBase.setFirstPassageCDF(np.quad(pdfData['cdf']))
            PDFs.append(pdfBase)
        
        pdf.setPDFs(PDFs)

        particlesData = []
        for key, data in vars["particleData"].items():
            pData = libDiffusion.ParticleData(np.quad(vars["vars"]["nParticles"]))
            pData.quantileTime = data["quantileTime"]
            pData.variance = np.quad(data["variance"])
            pData.quantileSet = data["quantileSet"]
            pData.varianceSet = data["varianceSet"]
            pData.cdfPrev = np.quad(data["cdfPrev"])
            pData.runningSumSquared = np.quad(data["runningSumSquared"])
            pData.runningSum = np.quad(data["runningSum"])
            particlesData.append(pData)
        
        pdf.particleData = particlesData
        
        return pdf

    @property
    def maxPositions(self):
        return self.getMaxPositions()

    @maxPositions.setter
    def maxPositions(self, _maxPositions):
        self.setMaxPositions(_maxPositions)

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

    @property
    def numberOfPositions(self):
        return self.getNumberOfPositions()

    @property
    def particleData(self) -> List["ParticleData"]:
        return self.getParticleData()

    @particleData.setter
    def particleData(self, _particleData: List["ParticleData"]):
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

    def evolveToCutoff(self, file: str, writeHeader: bool = True, cutoff: float = 1):
        f = open(file, "a")
        writer = csv.writer(f)
        if writeHeader:
            writer.writerow(["position", "quantile", "variance"])

        while self.numberHalted < self.numberOfPositions:
            self.iterateTimeStep()
            quantile, variance, position = self.checkParticleData()
            if (not quantile) and (not variance) and (not position):
                continue
            for p, q, v in zip(position, quantile, variance):
                writer.writerow([p, q, v])
                f.flush()
        f.close()
