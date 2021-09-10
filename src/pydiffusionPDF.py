import numpy as np
import sys
import os
import time
import json

# Need to link to diffusionPDF library (PyBind11 code)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DiffusionPDF")
sys.path.append(path)

import diffusionPDF
import csv
import npquad
import fileIO


class DiffusionPDF(diffusionPDF.DiffusionPDF):
    """
    Helper class for C++ Diffusion object. Allows simulating random walks with
    biases drawn from a beta distribution. Includes multiple helper functions to
    return important data such as maximum distance and quantiles.

    Parameters
    ----------
    numberOfParticles : int or float
        Number of particles to include in the simulation.

    beta : int or float
        Value of the beta distribution to use. Must satisfy 0 <= beta <= 1.

    occupancySize : int
        Size of the edges and occupancy arrays to initialize to. This needs to
        be at least the size of the number of timesteps that are planned to run.

    probDistFlag : bool (true)
        Whether or not to include fractional particles or not. If True doesn't
        round the particles shifting and if False then rounds the particles so
        there is always a whole number of particles.

    id : int
        SLURM ID used to save state periodically.

    Attributes
    ----------
    time : numpy array
        Time of the system

    currentTime : int
        Current time of the system - this is just max(time)

    center : numpy array
        Center of the occupancy over time.

    minDistance : numpy array
        The distance from the left side of the occupancy over time.

    maxDistance : numpy array
        The distance from the right side of the occupancy over time. This
        is generally the one we care about.

    occupancy : numpy array
        Number of particles at each position in the system. More formally, this
        is referred to as the partition function.

    nParticles : float
        Number of particles in the system

    beta : float
        Beta value of the beta distribution

    probDistFlag : bool
        Whether or not to include fractional particles or not. If True doesn't
        round the particles shifting and if False then rounds the particles so
        there is always a whole number of particles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_saved_time = time.process_time()  # seconds
        self._save_interval = 3600 * 2  # Set to save occupancy every 2 hours.
        self.id = None  # Need to also get SLURM ID

    def __str__(self):
        return f"DiffusionPDF(N={self.getNParticles()}, beta={self.getBeta()}, size={len(self.getEdges()[0])}, time={self.getTime()})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        """
        Doesn't check if edges are the same b/c we don't really care/use those
        a whole lot anyways.
        """

        if not isinstance(other, DiffusionPDF):
            raise TypeError(
                f"Comparison must be between same object types, but other of type {type(other)}"
            )

        occ_same = np.all(self.occupancy == other.occupancy)
        time_same = self.currentTime == other.currentTime
        nParticles_same = self.nParticles == other.nParticles
        beta_same = self.beta == other.beta
        probDistFlag_same = self.probDistFlag == other.probDistFlag
        edges_same_length = len(self.getEdges()[0]) == len(other.getEdges()[0])

        if (
            occ_same
            and time_same
            and nParticles_same
            and beta_same
            and probDistFlag_same
            and edges_same_length
        ):
            return True

        return False

    @property
    def time(self):
        return np.arange(0, self.getTime() + 1)

    @property
    def currentTime(self):
        return self.getTime()

    @currentTime.setter
    def currentTime(self, time):
        self.setTime(time)

    @property
    def center(self):
        return self.time * 0.5

    @property
    def minDistance(self):
        minEdge = np.array(self.getEdges()[0][: self.currentTime + 1])
        return minEdge - self.center

    @property
    def maxDistance(self):
        maxEdge = np.array(self.getEdges()[1][: self.currentTime + 1])
        return maxEdge - self.center

    @property
    def occupancy(self):
        return np.array(self.getOccupancy(), dtype=np.quad)

    @occupancy.setter
    def occupancy(self, occupancy):
        self.setOccupancy(occupancy)

    @property
    def nParticles(self):
        return self.getNParticles()

    @property
    def beta(self):
        return self.getBeta()

    @property
    def probDistFlag(self):
        return self.getProbDistFlag()

    @probDistFlag.setter
    def probDistFlag(self, flag):
        self.setProbDistFlag(flag)

    @property
    def smallCutoff(self):
        return self.getSmallCutoff()

    @smallCutoff.setter
    def smallCutoff(self, smallCutoff):
        self.setSmallCutoff(smallCutoff)

    @property
    def largeCutoff(self):
        return self.getLargeCutoff()

    @largeCutoff.setter
    def largeCutoff(self, largeCutoff):
        self.setLargeCutoff(largeCutoff)

    @property
    def edges(self):
        return self.getEdges()

    @edges.setter
    def edges(self, edges):
        self.setEdges(edges)

    def resizeOccupancyAndEdges(self, size):
        """
        Add elements to the end of the occupancy vector.

        Parameters
        ----------
        size : int
            Number of elements to add to occupancy
        """

        super().resizeOccupancyAndEdges(size)

    def saveVariables(self):
        """
        Save the current object's variables. These should be constants so
        we'll just save them at the first timestep.
        """

        vars = {
            "nParticles": str(self.nParticles),
            "beta": self.beta,
            "probDistFlag": self.probDistFlag,
            "smallCutoff": self.smallCutoff,
            "largeCutoff": self.largeCutoff,
        }

        with open(f"Variables{self.id}.json", "w+") as file:
            json.dump(vars, file)

    def saveState(self):
        """
        Save all the relevant data of the current object.
        """

        minIdx = self.edges[0][self.currentTime]
        maxIdx = self.edges[1][self.currentTime]
        fileIO.saveArrayQuad(
            f"Occupancy{self.id}.txt", self.occupancy[minIdx : maxIdx + 1]
        )

        vars = {
            "time": self.currentTime,
            "minEdges": self.edges[0][: self.currentTime + 1],
            "maxEdges": self.edges[1][: self.currentTime + 1],
            "minIdx": minIdx,
            "maxIdx": maxIdx,
        }

        with open(f"Edges{self.id}.json", "w+") as file:
            json.dump(vars, file)

    @classmethod
    def fromFiles(cls, variables_file, occupancy_file, edges_file):
        """
        Create a DiffusionPDF class from variables saved with saveVariables()
        and saveState().

        Parameters
        ----------
        variables_file : str
            File that contains the parameters, nParticles, beta, probDistFlag,
            smallCutoff, and largeCutoff.

        occupancy_file : str
            Occupancy at the current state

        edges_file : str
            File that contains the parameters minEdge, maxEdge and time.

        Returns
        -------
        d : DiffusionPDF
            Diffusion object ready to pick up the simulation.
        """

        with open(variables_file, "r") as file:
            vars = json.load(file)

        with open(edges_file, "r") as file:
            edges = json.load(file)

        occupancySize = edges["maxIdx"] - edges["minIdx"] + 1
        occupancy = fileIO.loadArrayQuad(occupancy_file, shape=occupancySize)

        d = DiffusionPDF(
            np.quad(vars["nParticles"]),
            vars["beta"],
            occupancySize,
            vars["probDistFlag"],
        )

        occupancy = np.concatenate(
            [np.zeros(edges["minIdx"], dtype=np.quad), occupancy]
        )
        d.occupancy = occupancy
        d.smallCutoff = vars["smallCutoff"]
        d.largeCutoff = vars["largeCutoff"]
        d.edges = (edges["minEdges"], edges["maxEdges"])
        d.currentTime = edges["time"]

        return d

    def setBetaSeed(self, seed):
        """
        Set the random generator seed for the beta distribution.

        Parameters
        ----------
        seed : int
            Seed for random beta distribution generator
        """

        self.setBetaSeed(seed)

    def iterateTimestep(self):
        """
        Move the occupancy forward one timestep drawing biases from the beta
        distribution.
        """

        if self.currentTime == 0:
            self.saveVariables()

        # Save the occupancy periodically so we can start it up later.
        if (time.process_time() - self._last_saved_time) > self._save_interval:
            self.saveState()
            self._last_saved_time = time.process_time()

        # Need to throw error if trying to go past the edges
        if self.currentTime + 2 > len(self.edges[0]):
            raise RuntimeError("Cannot iterate past the size of the edges")

        super().iterateTimestep()

    def findQuantile(self, quantile):
        """
        Get the rightmost Nth quantile of the occupancy.

        Parameters
        ----------
        quantile : float
            Nth quantile to find. Must satisfy 1 < NQuart.

        Returns
        -------
        float
            Distance from the center of the Nth quantile position.

        Examples
        --------
        >>> d = Diffusion(1, beta=np.inf, occupancySize=5)
        >>> d.evolveToTime(5)
        >>> print(d.occupancy)
        [0.03125 0.15625 0.3125 0.3125 0.15625 0.03125]
        >>> print(d.findQuantile(100))
        """

        assert quantile > 1, "Quantile must be > 1, but quantile: {quantile}"

        return np.array(super().findQuantile(quantile))

    def findQuantiles(self, quantiles):
        """
        Get the rightmost Nth quantile of the occupancy for multiple quantiles.
        Will be faster than writing a for loop over the findQuantile for most
        cases (large nParticles and small Nth quantiles).

        Parameters
        ----------
        quantiles : list
            Nth quantiles to find. Must satisfy quantiles > 1.

        Returns
        -------
        numpy array
            Distance from the center of the Nth quantile position.

        Note
        ----
        Expects the incoming quantiles array to be in ascending order. The algorithm
        will sort the quantiles in ascending order and then return the Nquarts in
        ascending order.

        Looks like this is much faster than using list comprehension with
        NthquantileSingleSided.

        Examples
        --------
        >>> d = Diffusion(1, beta=np.inf, occupancySize=5)
        >>> d.evolveToTime(5)
        >>> print(d.occupancy)
        [0.03125 0.15625 0.3125 0.3125 0.15625 0.03125]
        >>> print(diff.findQuantiles([100, 10]) # Quantiles should be in ascending order!
        [2.5, 1.5]
        """

        assert np.all(np.array(quantiles) > 1), "All quantiles must be > 1."

        return np.array(super().findQuantiles(quantiles))

    def pGreaterThanX(self, idx):
        """
        Get the probability of a particle being greater than index x.

        Parameters
        ----------
        idx : int
            Index to find the number of particles in the occupancy that are greater
            than the index position.
        """

        return super().pGreaterThanX(idx)

    def calcVsAndPb(self, num):
        """
        Calculate the velocity and probability being greater than v*t at the
        current time for a given number of points. Accrues velocities by moving
        from greatest filled index inward.

        Parameters
        ----------
        num : int
            Number of velocities to calculate

        Returns
        -------
        tuple(numpy array, numpy array)
            Velocities and logged probabilities or ln(Pb(vt, t)) as a tuple (v, ln(Pb))
        """

        return super().calcVsAndPb(num)

    def VsAndPv(self, minv=0.0):
        """
        Calculate velocities and ln(Pb(vt, t)) until minimum velocity is reached.

        Parameters
        ----------
        minv : float (0.0)
            Minimum velocity to calculate to

        Returns
        -------
        tuple(numpy array, numpy array)
            Velocities and logged probabilities or ln(Pb(vt, t)) as a tuple (v, ln(Pb))
        """

        return super().VsAndPb(minv)

    def evolveTimeSteps(self, iterations):
        """
        Evolves the system forward a specified number of timesteps.

        Parameters
        ----------
        iterations : int
            Number of timesteps to iterate forward
        """

        for _ in range(iterations):
            self.iterateTimestep()

    def evolveToTime(self, time):
        """
        Evolve the system to a specified time. If the input time is greater than
        the system's current time it won't actually do anything.

        Parameters
        ----------
        time : int
            System time to evolve the system forward to
        """

        while self.getTime() < time:
            self.iterateTimestep()

    def evolveAndSaveQuantiles(self, time, quantiles, file, append=False):
        """
        Incrementally evolves the system forward to the specified times and saves
        the specified quantiles after each increment.

        Parameters
        ----------
        time : list or numpy array
            Times to evolve the system to and save the quantiles at

        quantiles : list or numpy array
            Quantiles to save at each time. These should all be > 1.

        file : string
            Filename (or path) to save the time and quantiles

        append : bool
            Whether or not to append to the input file. This is generally used
            to continue evolving an occupancy after its been saved.

        Examples
        --------
        >>> diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
        >>> diff.evolveAndSaveQuantiles(time=[1, 2, 3, 4, 5], quantiles=[100, 10], file="Data.txt")
        >>> with open("Data.txt", "r") as f:
                print(f.read())
        time,MaxEdge,100,10
        1,1,0.5,0.5
        2,2,1.0,1.0
        3,3,1.5,1.5
        4,4,2.0,1.0
        5,5,2.5,1.5

        Notes
        -----
        Looks like this is a bit faster than the evolveAndSave method which
        stores everything to a numpy array and then saves it.
        """

        if append:
            f = open(file, "a")
        else:
            f = open(file, "w")

        writer = csv.writer(f)

        # Sort quantiles in descending order
        quantiles = np.sort(quantiles)[::-1]

        if not append:
            header = ["time", "MaxEdge"] + list(quantiles)
            writer.writerow(header)

        for t in time:
            self.evolveToTime(t)

            NthQuantiles = self.findQuantiles(quantiles)
            maxEdge = self.maxDistance[-1]
            # need to unpack NthQuantiles since it's returned as a np array
            row = [self.getTime(), maxEdge, *NthQuantiles]
            writer.writerow(row)
        f.close()

    def evolveAndSaveV(self, time, vs, file):
        """
        Incrementally evolves the system forward to the specified times and saves
        the number of particles greater than position v * time after each increment.
        This is to evaluate Pb(vt, t) at the specified times where Pb(x, t) is the
        probability of a particle being greater than x at time t. Need to divide
        by nParticles to get the probability.

        Parameters
        ----------
        time : list or numpy array
            Times to evolve the system to and save the quantiles at

        vs : list or numpy array
            Velocities to save at each time. These should all satisfy 0 <= v <= 1.
            Note that v=1 corresponds to the probability of a particle being greater
            than x=t and v=0 corresponds to the probability of a particle being
            greater than x=t/2 or the center.

        file : string
            Filename (or path) to save the time and probabilities

        Examples
        --------
        >>> N = 1e300
        >>> num_of_steps = 100_000
        >>> d = Diffusion(N, beta=beta, occupancySize=num_of_steps, smallCutoff=0, largeCutoff=0, probDistFlag=True)
        >>> save_times = np.geomspace(1, num_of_steps, 1000, dtype=np.int64)
        >>> save_times = np.unique(save_times)
        >>> vs = np.geomspace(1e-7, 1, 50)
        >>> save_file = 'Data.txt'
        >>> d.evolveAndSaveV(save_times, vs, save_file)

        Notes
        -----
        Looks like this is a bit faster than the evolveAndSave method which
        stores everything to a numpy array and then saves it.

        To get the index we use the formula:
                        idx = round((v * t + t) / 2)
        which satisfies the constraints mentioned in the velocity definition.
        """

        f = open(file, "w")
        writer = csv.writer(f)
        header = ["time"] + [str(v) for v in vs]
        writer.writerow(header)
        for t in time:
            self.evolveToTime(t)
            idx = (self.getTime() * vs + self.getTime()) / 2
            idx = np.round(idx).astype(np.int64)
            pos = [self.pGreaterThanX(i) for i in idx]
            row = [self.getTime()] + pos
            writer.writerow(row)
        f.close()

    def evolveAndSave(self, time, quantiles, file):
        """
        Incrementally evolves the system forward to the specified times and saves
        the specified quantiles a2fter each increment. The data is stored as a
        numpy array which may make it slower than the evolveAndSaveQuantile method.

        Parameters
        ----------
        time : list or numpy array
            Times to evolve the system to and save the quantiles at

        quantiles : list or numpy array
            Quantiles to save at each time. These should all be < 1.

        file : string
            Filename (or path) to save the time and quantiles

        Notes
        -----
        Looks like this is a bit slower than the evolveAndSaveQuantile method which
        stores writes to the file incrementally.

        Also shouldn't be used b/c I don't think it will be compatible with np.quad.
        """

        save_array = np.zeros(shape=(len(time), len(quantiles) + 2))
        for row_num, t in enumerate(time):
            self.evolveToTime(t)

            quantiles = np.array(quantiles)
            quantiles.sort()
            NthQuantile = self.findQuantiles(quantiles)

            maxEdge = self.maxDistance
            row = [self.getTime(), maxEdge] + NthQuantile
            save_array[row_num, :] = row
        np.savetxt(file, save_array)

    def ProbBiggerX(self, vs, timesteps):
        """
        Troubleshooting function to make sure that pGreaterThanX function
        works properly.

        Parameters
        ----------
        vs : list or numpy array
            Velocities to print out

        timesteps : int
            How many timesteps to iterate forward (generally good to keep this small
            so you can actually add up the occupancy)

        Examples
        --------
        # Note the output will change each time this is run since the biases are random
        >>> N = 1e300
        >>> d = Diffusion(N, beta=1, occupancySize=10, smallCutoff=0, largeCutoff=0, probDistFlag=True)
        >>> d.ProbBiggerX(np.array([0.5, 1]), 1)
        Bigger than Index: [3.058954085425106e+299, 3.058954085425106e+299]
        Indices:  [1 1]
        Occupancy: [6.94104591e+299 3.05895409e+299]
        Prob:  [0.30589541 0.30589541]
        """

        for _ in range(timesteps):
            self.iterateTimestep()

        # It looks like this produces the proper indeces we are looking for!
        idx = (self.getTime() * vs + self.getTime()) / 2
        idx = np.round(idx).astype(np.int64)

        nonzeros = np.nonzero(self.getOccupancy())[0]
        Ns = [self.pGreaterThanX(i) for i in idx]
        print("Bigger than Index:", Ns)
        print("Indices: ", idx)
        print("Occupancy:", np.array(self.getOccupancy())[nonzeros])
        print("Prob: ", np.array(Ns) / self.getNParticles())
