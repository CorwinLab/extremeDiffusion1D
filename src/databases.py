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
from TracyWidom import TracyWidom


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

    @property
    def center(self):
        return self.time * 0.5

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
    def __init__(
        self, files, readQuantiles=True, delimiter=",", skiprows=1, nParticles=None
    ):
        """
        Create a Database with the selected files.

        Parameters
        ----------
        files : list
            list of files to create database from
        """
        super().__init__(files, delimiter, skiprows)

        # Set some easy properties
        if readQuantiles:
            self.quantiles = self.getQuantiles()

        self.nParticles = nParticles

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

        rows = self.shape[0]
        cols = self.shape[1] - 2  # exclude the time and max columns
        shape = (rows, cols)
        squared_sum = np.zeros(shape, dtype=np.quad)
        mean_sum = np.zeros(shape, dtype=np.quad)

        maxEdge_mean_sum = np.zeros(rows, dtype=np.quad)
        maxEdge_squared_sum = np.zeros(rows, dtype=np.quad)

        for f in self.files:
            data = loadArrayQuad(
                f, self.shape, delimiter=self.delimiter, skiprows=self.skiprows
            )
            time = data[:, 0]
            # second column is maximum edge which we don't really care about
            # for probDist=True
            maxEdge = 2 * (data[:, 1] - self.center)
            data = 2 * data[:, 2:]

            squared_sum += data ** 2
            mean_sum += data

            maxEdge_mean_sum += maxEdge
            maxEdge_squared_sum += maxEdge ** 2

            if verbose:
                print(f)

        mean_sum = mean_sum.astype(np.float64)
        squared_sum = squared_sum.astype(np.float64)
        maxEdge_mean_sum = maxEdge_mean_sum.astype(np.float64)
        maxEdge_squared_sum = maxEdge_squared_sum.astype(np.float64)

        self.mean = mean_sum / len(self)
        self.var = squared_sum / len(self) - self.mean ** 2

        self.maxMean = maxEdge_mean_sum / len(self)
        self.maxVar = maxEdge_squared_sum / len(self) - self.maxMean ** 2

    def getQuantiles(self):
        """
        Returns the measured N quartile values from the file.

        Returns
        -------
        quantiles : list
            The 1/Nth quantiles recorded as quads
        """

        with open(self.files[0]) as f:
            quantiles = f.readline().split(",")[2:]
            quantiles = [np.quad(N) for N in quantiles]

        return quantiles

    def setNs(self, quantiles):
        """
        Set the measured N quartile values.

        Parameters
        ----------
        Ns : list (of quads)
            The 1/Nth quantiles in the database
        """

        self.quantiles = quantiles

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

        for i, quant in enumerate(self.quantiles):
            Nstr = prettifyQuad(quant)
            if np.isinf(quant):
                continue
            if verbose:
                print(Nstr)

            theory = th.theoreticalNthQuart(quant, self.time)
            fig, ax = plt.subplots()
            ax.set_ylabel("Mean Nth Quartile")
            ax.set_title(f"N={Nstr}")

            if xscale:
                time = self.time / np.log(quant).astype(np.float64)
                ax.set_xlabel("Time / ln(N)")
            else:
                time = self.time
                ax.set_xlabel("Time")

            ax.plot(
                self.time / np.log(quant).astype(np.float64),
                self.mean[:, i],
                label="Mean",
            )
            ax.plot(
                self.time / np.log(quant).astype(np.float64),
                theory,
                label=th.NthQuartStr,
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
        Plot the variance of the 1/Nth quantiles for all N's in the database.

        Parameters
        ----------
        save_dir : str
            Directory to save plots to.
        """

        for i, quant in enumerate(self.quantiles):
            Nstr = prettifyQuad(quant)

            if np.isinf(quant):
                continue
            if verbose:
                print(Nstr)

            theory = th.theoreticalNthQuartVar(quant, self.time)
            logTheory = th.theoreticalNthQuartVarLargeTimes(quant, self.time)

            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("Variance of Nth Quartile")
            ax.set_title(f"N={Nstr}")
            ax.plot(
                self.time / np.log(quant).astype(np.float64),
                self.var[:, i],
                label="Variance",
            )
            ax.plot(
                self.time / np.log(quant).astype(np.float64),
                theory,
                label=th.NthQuartVarStr,
            )
            ax.plot(
                self.time / np.log(quant).astype(np.float64),
                logTheory,
                label=th.NthQuartVarStrLargeTimes,
            )
            ax.plot(
                self.time / np.log(quant).astype(np.float64),
                self.time / np.log(quant).astype(np.float64),
                label="Linear",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=12)
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), f"Var{Nstr}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotMaxMean(self, save_dir=".", xscale=True):
        """
        Plot the mean maximum particle position. Need to have set nPartilces for
        this to work properly.
        """

        Nstr = prettifyQuad(self.nParticles)

        theory = th.theoreticalNthQuart(self.nParticles, self.time)
        fig, ax = plt.subplots()
        ax.set_ylabel("Mean Maximum Particle Position")
        ax.set_title(f"N={Nstr}")

        if xscale:
            time = self.time / np.log(self.nParticles).astype(np.float64)
            ax.set_xlabel("Time / ln(N)")
        else:
            time = self.time
            ax.set_xlabel("Time")

        ax.plot(
            self.time / np.log(self.nParticles).astype(np.float64),
            self.maxMean,
            label="Mean",
        )
        ax.plot(
            self.time / np.log(self.nParticles).astype(np.float64),
            theory,
            label=th.NthQuartStr,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        fig.savefig(
            os.path.join(os.path.abspath(save_dir), f"MaxMean{Nstr}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

    def plotMaxVar(self, save_dir=".", xscale=True):
        """
        Plot the variance of the maximum particle position. Need to have set nParticles
        for this to work.
        """

        Nstr = prettifyQuad(self.nParticles)

        theory = th.theoreticalNthQuartVar(self.nParticles, self.time)
        logTheory = th.theoreticalNthQuartVarLargeTimes(self.nParticles, self.time)

        fig, ax = plt.subplots()
        ax.set_xlabel("Time / lnN")
        ax.set_ylabel("Variance of Maximum Particle")
        ax.set_title(f"N={Nstr}")
        ax.plot(
            self.time / np.log(self.nParticles).astype(np.float64),
            self.maxVar,
            label="Variance",
        )
        ax.plot(
            self.time / np.log(self.nParticles).astype(np.float64),
            theory,
            label=th.NthQuartVarStr,
        )
        ax.plot(
            self.time / np.log(self.nParticles).astype(np.float64),
            logTheory,
            label=th.NthQuartVarStrLargeTimes,
        )
        ax.plot(
            self.time / np.log(self.nParticles).astype(np.float64),
            self.time / np.log(self.nParticles).astype(np.float64),
            label="Linear",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=12)
        fig.savefig(
            os.path.join(os.path.abspath(save_dir), f"MaxVar{Nstr}.png"),
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
        colors = [
            cm(1.0 * i / len(self.quantiles) / 1.5) for i in range(len(self.quantiles))
        ]

        for i, quant in enumerate(self.quantiles):
            Nstr = prettifyQuad(quant)
            if np.isinf(quant):
                continue
            ax.plot(
                self.time / np.log(quant).astype(np.float64),
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
        colors = [
            cm(1.0 * i / len(self.quantiles) / 1.5) for i in range(len(self.quantiles))
        ]

        for i, quant in enumerate(self.quantiles):
            if np.isinf(quant):
                continue
            Nstr = prettifyQuad(quant)
            ax.plot(
                self.time / np.log(quant).astype(np.float64),
                self.var[:, i] / np.log(quant).astype(float) ** (2 / 3),
                c=colors[i],
                label=None,
            )

        theory = th.theoreticalNthQuartVar(quant, self.time)
        theory_Large = th.theoreticalNthQuartVarLargeTimes(quant, self.time)
        ax.plot(
            self.time / np.log(quant).astype(np.float64),
            theory / np.log(quant).astype(float) ** (2 / 3),
            c="b",
            label=th.NthQuartVarStr,
        )
        ax.plot(
            self.time / np.log(quant).astype(np.float64),
            theory_Large / np.log(quant).astype(float) ** (2 / 3),
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

    def plotVarKPZ(self, save_dir="."):
        """
        Plot the variance versus the predicted KPZ behavior.
        """

        for i, quant in enumerate(self.quantiles):
            Nstr = prettifyQuad(quant)
            fig, ax = plt.subplots()
            logN = np.log(quant).astype(np.float64)
            xaxis = logN ** 2 / self.time
            yaxis = logN / self.time * self.var[:, i]
            ax.plot(xaxis, yaxis)
            ax.set_xlabel("lnN^2 / t")
            ax.set_ylabel("lnN / t * Var(Qb(N,t))")
            ax.set_title(f"N={Nstr}")
            ax.set_xscale("log")
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), f"KPZVar{Nstr}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotGumbalDistOverTime(self, quantile):
        """
        Look for the differences in 10*quantile - 0.1*quantile and see if they
        look like the Gumbal distribution.
        """

        diff = self.getGumbalDiff(quantile)

        fig, ax = plt.subplots()
        ax.plot(self.time, diff, label="Date")
        ax.plot(self.time, np.sqrt(self.time), label="t^(1/2)")
        ax.set_title(f"N={quantile}")
        ax.set_ylabel("10*N - 0.1*N")
        ax.set_xlabel("Time")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.savefig("GumbelNumber.png")

    def getGumbalDiff(self, quantile):
        """
        Return the difference for 10*quantile - 0.1*quantile to see if they look
        like the Gumbal distribution.

        Parameters
        ----------
        quantile : numpy quad or float
            Quantile to get difference of

        Returns
        -------
        diff : numpy array
            Difference between 10*quantile - 0.1*quantile
        """

        idx = self.quantiles.index(quantile)

        # Make sure the quantiles differ by a factor of 10. The quantiles go
        # largest to smallest. So idx-1 is largest and idx+1 is smallest.
        assert self.quantiles[idx - 1] / self.quantiles[idx] == 10
        assert self.quantiles[idx] / self.quantiles[idx + 1] == 10
        assert self.quantiles[idx - 1] / self.quantiles[idx + 1] == 100

        diff = self.mean[:, idx - 1] - self.mean[:, idx + 1]
        return diff

    def plotGumbalDist(self, quantile, verbose=False):
        """
        Make a histogram of the difference between the 10*quantile and 0.1*quantile
        quantiles at the maximum time.
        """

        idx = self.quantiles.index(quantile)

        # Make sure the quantiles differ by a factor of 10. The quantiles go
        # largest to smallest. So idx-1 is largest and idx+1 is smallest.
        assert self.quantiles[idx - 1] / self.quantiles[idx] == 10
        assert self.quantiles[idx] / self.quantiles[idx + 1] == 10
        assert self.quantiles[idx - 1] / self.quantiles[idx + 1] == 100

        diff = []
        for f in self.files:
            final_time = loadArrayQuad(
                f, skiprows=len(self._example_file) - 1, shape=len(self.quantiles) + 2
            )
            final_time = final_time[2:]
            diff.append((final_time[idx - 1] - final_time[idx + 1]).astype(np.float64))
            if verbose:
                print(f)

        fig, ax = plt.subplots()
        ax.hist(diff, bins=100)
        fig.savefig("GumbelHist.png")


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

        self.velocities = self.getVelocities()

    def calculateMeanVar(self, verbose=False):
        """
        Calculate the mean and variance of the dataset.
        """

        rows = self.shape[0]
        cols = self.shape[1] - 1  # exclude the time column
        shape = (rows, cols)
        squared_sum = np.zeros(shape, dtype=np.quad)
        mean_sum = np.zeros(shape, dtype=np.quad)

        for f in self.files:
            data = loadArrayQuad(f, self.shape, delimiter=",", skiprows=1)
            time = data[:, 0]
            data = data[:, 1:]
            data = np.log(data)

            squared_sum += data ** 2
            mean_sum += data

            if verbose:
                print(f)

        mean_sum = mean_sum.astype(np.float64)
        squared_sum = squared_sum.astype(np.float64)

        self.mean = mean_sum / len(self)
        self.var = squared_sum / len(self) - self.mean ** 2

    def getVelocities(self):
        """
        Return the measured velocities from the files.

        Returns
        -------
        velocities : list
            Velocities from the files
        """

        with open(self.files[0]) as f:
            velocities = f.readline().split(",")[1:]
            velocities = [float(i) for i in velocities]
        return velocities

    def plotMeans(self, save_dir="."):
        """
        Plot the mean of each velocity as a function of time and compare to the
        theory.
        """

        for i, v in enumerate(self.velocities):
            theory = th.theoreticalPbMean(v, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("|ln(Pb(vt, t))|")
            ax.plot(self.time, abs(self.mean[:, i]), label="Data", c="r")
            ax.plot(self.time, abs(theory), label=th.PbMeanStr, ls="--")
            ax.set_title(f"v={v}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), f"Mean{v}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotVars(self, save_dir="."):
        """
        Plot the variance of each velocity as a function of time and compare to
        the theory.
        """

        for i, v in enumerate(self.velocities):
            theory = th.theoreticalPbVar(v, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("Var(ln(Pb(vt, t)))")
            ax.plot(self.time, self.var[:, i], label="Data", c="r")
            ax.plot(self.time, theory, label=th.PbVarStr, ls="--")
            ax.set_title(f"v={v}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(
                os.path.join(os.path.abspath(save_dir), f"Var{v}.png"),
                bbox_inches="tight",
            )

    def plotDistribution(
        self, save_dir=".", save_file="FinalTime.txt", load_file=None, verbose=False
    ):
        """
        Plot the Tracy Widom distribution at the maximum time.
        """

        if load_file is not None:
            total_data = np.loadtxt(load_file)

        else:
            total_data = np.empty((len(self.files), self.shape[1] - 1))
            for row, f in enumerate(self.files):
                data = loadArrayQuad(f, self.shape, delimiter=",", skiprows=1)
                data = data[-1, 1:]
                data = np.log(data).astype(np.float64)
                total_data[row] = data
                if verbose:
                    print(f)

            np.savetxt(save_file, total_data)

        tw = TracyWidom(beta=2)
        for i, v in enumerate(self.velocities):
            data = total_data[:, i]
            data = data[~np.isinf(data)]
            data = data[~np.isnan(data)]
            print(len(data))

            I = 1 - np.sqrt(1 - v ** 2)
            sigma = ((2 * I ** 2) / (1 - I)) ** (1 / 3)
            scale = self.time[-1] ** (1 / 3) * sigma
            offset = I * self.time[-1]

            fig, ax = plt.subplots()
            ax.set_xlabel("(ln(Pb(vt, t)) + I(v)t) / t^(1/3) * sigma")
            ax.set_ylabel("Probability Density")
            ax.set_title(f"v={v}")
            ax.hist((data + offset) / scale, density=True, bins=100)

            x = np.linspace(-5, 5, 100)
            pdf = tw.pdf(x)
            ax.plot(x, pdf, label="TW Distribution")
            ax.set_yscale("log")
            ax.legend()

            fig.savefig(
                os.path.join(save_dir, f"Histogram{v}.png"), bbox_inches="tight"
            )
            plt.close(fig)

    def plotAllVars(self, save_dir="."):
        """
        Plot all the velocities on the same plot.
        """
        cm = plt.get_cmap("gist_heat")
        colors = [
            cm(1.0 * i / len(self.velocities) / 1.5)
            for i in range(len(self.velocities))
        ]
        fig, ax = plt.subplots()
        for i, v in enumerate(self.velocities):
            I = 1 - np.sqrt(1 - v ** 2)
            sigma = ((2 * I ** 2) / (1 - I)) ** (1 / 3)
            var = self.var[:, i]
            ax.plot(
                self.time,
                var / self.time ** (2 / 3) / sigma ** 2 / 0.813,
                label="{0:.1f}".format(v),
                c=colors[i],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Variance / (t^2/3 * sigma^2 * 0.813)")
        ax.legend()
        fig.savefig(os.path.join(save_dir, f"Vars.png"), bbox_inches="tight")

    def plotResidual(self, save_dir="."):
        """
        Plot the residuals of the variance. And try to do a curve fit - this isn't
        working at all.
        """
        cm = plt.get_cmap("gist_heat")
        colors = [
            cm(1.0 * i / len(self.velocities) / 1.5)
            for i in range(len(self.velocities))
        ]
        for i, v in enumerate(self.velocities):
            fig, ax = plt.subplots()
            theory = th.theoreticalPbVar(v, self.time)
            residual = abs(self.var[:, i] - theory) / theory * 100
            idx = np.where(~np.isnan(residual))[0]
            time = self.time[idx]
            residual = residual[idx]
            ax.plot(time[100:], residual[100:], label="{0:.1f}".format(v), c="k")
            conf = np.polyfit(np.log(time[100:]), np.log(residual[100:]), 1)
            # ax.plot(time[100:], conf[0] * time[100:] + conf[1])
            ax.set_xscale("log")
            ax.set_ylabel("Variance Percent Change")
            ax.legend()
            fig.savefig(os.path.join(save_dir, f"Residual{v}.png"), bbox_inches="tight")


class CDFQuartileDatabase(QuartileDatabase):
    """
    Making this b/c the QuartileDatabase needs to factor everything by multiply
    the data by 2 to get it to match the theoretically predicted values.
    """

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

        rows = self.shape[0]
        cols = self.shape[1] - 1  # exclude the time and max columns
        shape = (rows, cols)
        squared_sum = np.zeros(shape, dtype=np.quad)
        mean_sum = np.zeros(shape, dtype=np.quad)

        for f in self.files:
            data = loadArrayQuad(
                f, self.shape, delimiter=self.delimiter, skiprows=self.skiprows
            )
            time = data[:, 0]
            # second column is maximum edge which we don't really care about
            # for probDist=True
            data = data[:, 1:]

            squared_sum += data ** 2
            mean_sum += data

            if verbose:
                print(f)

        mean_sum = mean_sum.astype(np.float64)
        squared_sum = squared_sum.astype(np.float64)

        self.mean = mean_sum / len(self)
        self.var = squared_sum / len(self) - self.mean ** 2
