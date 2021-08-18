import numpy as np
import npquad
import glob
from fileIO import loadArrayQuad
import theory as th
import os
import matplotlib
from quadMath import prettifyQuad

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class Database:
    def __init__(self, files, delimiter=",", skiprows=1):
        """
        Create a Database with the selected files.

        Parameters
        ----------
        files : list
            Files to create database from
        """
        self.files = files

        # Need to first parse the shape of the arrays we'll be loading
        temp = np.loadtxt(files[0], delimiter=delimiter, skiprows=skiprows)
        self.shape = temp.shape

        # Load the first array so we can get simple stuff like len of the files
        # and max/min times
        self._example_file = loadArrayQuad(
            files[0], self.shape, delimiter=delimiter, skiprows=skiprows
        )

        # Set some easy properties
        self.delimiter = delimiter
        self.skiprows = skiprows

    def __len__(self):
        return len(self.files)

    @property
    def time(self):
        return self._example_file[:, 0].astype(np.float64)

    @classmethod
    def fromDir(cls, directory):
        """
        Create a Database class from a directory

        Parameters
        ----------
        directory : str
            Path to files
        """
        directory = os.path.abspath(directory)
        files = glob.glob(os.path.join(director, "*.txt"))

        return cls(files)

    def loadMean(self, file, quad=False):
        """
        Load the mean of the dataset from a file.

        Parameters
        ----------
        file : str
            File path of the mean to load
        """
        if quad:
            self.mean = loadArrayQuad(file)
        else:
            self.mean = np.loadtxt(file)

    def loadVar(self, file, quad=False):
        """
        Load the variance of the dataset from a file.

        Parameters
        ----------
        file : str
            File path to the variance to load
        """
        if quad:
            self.var = loadArrayQuad(file)
        else:
            self.var = np.loadtxt(file)


class QuartileDatabase(Database):
    def __init__(self, files, readNs=True, delimiter=",", skiprows=1):
        """
        Create a Database with the selected files.

        Parameters
        ----------
        files : list
            list of files to create database from
        """
        super().__init__(files, delimiter, skiprows)

        # Set some easy properties
        if readNs:
            self.Ns = self.getNs()

    def calculateMeanVar(self, verbose=False):
        """
        Calculate the mean of the selected data along the columns or rows.
        Assumes that the first column is the time.

        Parameters
        ----------
        verbose : bool
            Whether to print the file names out when each is finished or not.
            Really only used if you want to see progress over time.
        """

        squared_sum = None
        mean_sum = None

        for f in self.files:
            data = loadArrayQuad(
                f, self.shape, delimiter=self.delimiter, skiprows=self.skiprows
            )
            time = data[:, 0]
            maxEdge = (
                2 * data[:, 1]
            )  # second column is maximum edge which we don't really care about for probDist=True
            data = 2 * data[:, 2:]

            if squared_sum is None:
                squared_sum = data ** 2
            else:
                squared_sum += data ** 2

            if mean_sum is None:
                mean_sum = data
            else:
                mean_sum += data

            if verbose:
                print(f)

        self.mean = mean_sum.astype(np.float64) / len(self)
        self.var = squared_sum.astype(np.float64) / len(self) - self.mean ** 2

    def getNs(self):
        """
        Returns the measured N quartile values from the file.

        Returns
        -------
        Ns : list
            The 1/Nth quartiles recorded as quads
        """

        with open(self.files[0]) as f:
            Ns = f.readline().split(",")[2:]
            Ns = [np.quad(N) for N in Ns]

        return Ns

    def setNs(self, Ns):
        """
        Set the measured N quartile values.

        Parameters
        ----------
        Ns : list (of quads)
            The 1/Nth quartiles in the database
        """

        self.Ns = Ns

    def plotMeans(self, save_dir=".", xscale=True, verbose=False):
        """
        Plot the mean 1/Nth quantiles for all N's in the database.

        Parameters
        ----------
        save_dir : str
            Directory to save plots to.

        xscale : str
            Whether or not to scale the x-axis by logN or not.
        """

        for i, N in enumerate(self.Ns):
            Nstr = prettifyQuad(N)
            if np.isinf(N):
                continue
            if verbose:
                print(Nstr)

            theory = th.theoreticalNthQuart(N, self.time)
            fig, ax = plt.subplots()
            ax.set_ylabel("Mean Nth Quartile")
            ax.set_title(f"N={Nstr}")

            if xscale:
                time = self.time / np.log(N).astype(np.float64)
                ax.set_xlabel("Time / ln(N)")
            else:
                time = self.time
                ax.set_xlabel("Time")

            ax.plot(
                self.time / np.log(N).astype(np.float64),
                self.mean[:, i],
                label="Mean",
            )
            ax.plot(
                self.time / np.log(N).astype(np.float64), theory, label=th.NthQuartStr
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), f"Mean{Nstr}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotVars(self, save_dir=".", verbose=False):
        """
        Plot the variance of the 1/Nth quartiles for all N's in the database.

        Parameters
        ----------
        save_dir : str
            Directory to save plots to.
        """

        for i, N in enumerate(self.Ns):
            Nstr = prettifyQuad(N)

            if np.isinf(N):
                continue
            if verbose:
                print(Nstr)

            theory = th.theoreticalNthQuartVar(N, self.time)
            logTheory = th.theoreticalNthQuartVarLargeTimes(N, self.time)

            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("Variance of Nth Quartile")
            ax.set_title(f"N={Nstr}")
            ax.plot(
                self.time / np.log(N).astype(np.float64),
                self.var[:, i],
                label="Variance",
            )
            ax.plot(
                self.time / np.log(N).astype(np.float64),
                theory,
                label=th.NthQuartVarStr,
            )
            ax.plot(
                self.time / np.log(N).astype(np.float64),
                logTheory,
                label=th.NthQuartVarStrLargeTimes,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=12)
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), f"Var{Nstr}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotMeansEvolve(self, save_dir=".", legend=True):
        """
        Plot all the means together on the same plot.
        """

        fig, ax = plt.subplots()
        ax.set_xlabel("Time / lnN")
        ax.set_ylabel("Mean Nth Quartile")

        cm = plt.get_cmap("gist_heat")
        colors = [cm(1.0 * i / len(self.Ns) / 1.5) for i in range(len(self.Ns))]

        for i, N in enumerate(self.Ns):
            Nstr = prettifyQuad(N)
            if np.isinf(N):
                continue
            ax.plot(
                self.time / np.log(N).astype(np.float64),
                self.mean[:, i],
                c=colors[i],
                label=f"N={Nstr}",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        if legend:
            ax.legend()
        fig.savefig(
            os.path.join(os.path.abspath(save_dir), f"Means.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

    def plotVarsEvolve(self, save_dir=".", legend=True, yscale=True, theory=True):
        """
        Plot all the variances together on the same plot.
        """

        fig, ax = plt.subplots()
        ax.set_xlabel("Time / lnN")
        ax.set_ylabel("Variance of Nth Quartile / lnN^(2/3)")

        cm = plt.get_cmap("gist_heat")
        colors = [cm(1.0 * i / len(self.Ns) / 1.5) for i in range(len(self.Ns))]

        for i, N in enumerate(self.Ns):
            if np.isinf(N):
                continue
            Nstr = prettifyQuad(N)
            ax.plot(
                self.time / np.log(N).astype(np.float64),
                self.var[:, i] / np.log(N).astype(float) ** (2 / 3),
                c=colors[i],
                label=None,
            )

        theory = th.theoreticalNthQuartVar(N, self.time)
        theory_Large = th.theoreticalNthQuartVarLargeTimes(N, self.time)
        ax.plot(
            self.time / np.log(N).astype(np.float64),
            theory / np.log(N).astype(float) ** (2 / 3),
            c="b",
            label=th.NthQuartVarStr,
        )
        ax.plot(
            self.time / np.log(N).astype(np.float64),
            theory_Large / np.log(N).astype(float) ** (2 / 3),
            c="g",
            label=th.NthQuartVarStrLargeTimes,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        if legend:
            ax.legend()
        fig.savefig(
            os.path.join(os.path.abspath(save_dir), f"Vars.png"), bbox_inches="tight"
        )
        plt.close(fig)


class VelocityDatabase(Database):
    def __init__(self, files):
        """
        Initialize a Database with the selected files.

        Parameters
        ----------
        files : str
            Files to create database from
        """

        super().__init__(files)

        self.vs = self.getVs()

    def calculateMeanVar(self, verbose=False):
        """
        Calculate the mean and variance of the dataset.
        """

        squared_sum = None
        mean_sum = None

        for f in self.files:
            data = loadArrayQuad(f, self.shape, delimiter=",", skiprows=1)
            time = data[:, 0]
            data = data[:, 1:]
            data = np.log(data)

            if squared_sum is None:
                squared_sum = data ** 2
            else:
                squared_sum += data ** 2

            if mean_sum is None:
                mean_sum = data
            else:
                mean_sum += data

            if verbose:
                print(f)

        self.mean = mean_sum.astype(np.float64) / len(self)
        self.var = squared_sum.astype(np.float64) / len(self) - self.mean ** 2

    def getVs(self):
        """
        Return the measured velocities from the files.

        Returns
        -------
        vs : list
            Velocities from the files
        """

        with open(self.files[0]) as f:
            vs = g.readline().split(",")[1:]
            vs = [float(i) for i in vs]
        return vs

    def plotMean(self, save_dir="."):
        """
        Plot the mean of each velocity as a function of time and compare to the
        theory.
        """

        for i, v in enumerate(self.vs):
            theory = th.theoreticalPbMean(v, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("ln(Pb(vt, t))")
            ax.plot(self.time, self.mean[:, i], label="Data", c="r")
            ax.plot(self.time, theory, label="Theory")
            ax.set_title(f"v={v}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), "Mean{v}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotVar(self, save_dir="."):
        """
        Plot the variance of each velocity as a function of time and compare to
        the theory.
        """

        for i, v in enumerate(self.vs):
            theory = th.theoreticalPbVar(v, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("Var(ln(Pb(vt, t)))")
            ax.plot(self.time, self.var[:, i], label="Data", c="r")
            ax.plot(self.time, theory, label="Theory")
            ax.set_title(f"v={v}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), "Var{v}.png"),
                bbox_inches="tight",
            )
