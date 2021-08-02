import numpy as np
import npquad
import glob
import pydiffusion as diff
import os

class Database:

    def __init__(self, files):
        '''
        Create a Database with the selected files.

        Parameters
        ----------
        files : list
            Files to create database from
        '''
        self.files = files

        # Need to first parse the shape of the arrays we'll be loading
        temp = np.loadtxt(files[0], delimiter=",", skiprows=1)
        self.shape = temp.shape

        # Load the first array so we can get simple stuff like len of the files
        # and max/min times
        self._example_file = diff.loadArrayQuad(files[0], self.shape, delimiter=",", skiprows=1)

        # Set some easy properties
        self.Ns = self.getNs()

    def __len__(self):
        return len(self.files)

    @property
    def time(self):
        return self._example_file[:, 0]

    @classmethod
    def fromDir(cls, directory):
        '''
        Create a Database class from a directory

        Parameters
        ----------
        directory : str
            Path to files
        '''
        directory = os.path.abspath(directory)
        files = glob.glob(os.path.join(director, '*.txt'))

        return cls(files)

    def loadMean(self, file, quad=False):
        '''
        Load the mean of the dataset from a file.

        Parameters
        ----------
        file : str
            File path of the mean to load
        '''
        if quad:
            self.mean = diff.loadArrayQuad(file)
        else:
            self.mean = np.loadtxt(file)

    def loadVar(self, file, quad=False):
        '''
        Load the variance of the dataset from a file.

        Parameters
        ----------
        file : str
            File path to the variance to load
        '''
        if quad:
            self.mean = diff.loadArrayQuad(file)
        else:
            self.mean = np.loadtxt(file)


class QuartileDatabase(Database):

    def __init__(self, files):
        '''
        Create a Database with the selected files.

        Parameters
        ----------
        files : list
            list of files to create database from
        '''
        super().__init__(files)

        # Set some easy properties
        self.Ns = self.getNs()

    def calculateMeanVar(self):
        '''
        Calculate the mean of the selected data along the columns or rows.
        Assumes that the first column is the time.
        '''

        squared_sum = None
        mean_sum = None

        for f in self.files:
            data = diff.loadArrayQuad(f, self.shape, delimiter=",", skiprows=1)
            time = data[:, 0]
            data = 2 * data[:, 1:]

            if squared_sum is None:
                squared_sum = data ** 2
            else:
                squared_sum += data ** 2

            if mean_sum is None:
                mean_sum = data
            else:
                mean_sum += data

        self.mean = mean_sum.astype(np.float64) / len(self)
        self.var = squared_sum.astype(np.float64) / len(self) - self.mean ** 2

    def getNs(self):
        '''
        Returns the measured N quartile values from the file.

        Returns
        -------
        Ns : list
            The 1/Nth quartiles recorded as quads
        '''

        with open(self.files[0]) as f:
            Ns = g.readline().split(",")[2:]
            Ns = [np.quad(N) for N in Ns]

        return Ns

    def plotMeans(self, save_dir='.'):
        '''
        Plot the mean 1/Nth quantiles for all N's in the database.

        Parameters
        ----------
        save_dir : str
            Directory to save plots to.
        '''

        for i, N in enumerate(self.Ns):
            theory = diff.theoreticalNthQuart(N, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("Mean Nth Quartile")
            ax.set_title("N={N}")
            ax.plot(self.time, self.mean, label='Mean')
            ax.plot(self.time, theory, label='Theory')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(os.path.join(os.path.abspath(save_dir), f"Mean{N}.png"),
                        bbox_inches='tight')
            plt.close(fig)

    def plotVars(self, save_dir='.'):
        '''
        Plot the variance of the 1/Nth quartiles for all N's in the database.

        Parameters
        ----------
        save_dir : str
            Directory to save plots to.
        '''

        for i, N in enumerate(self.Ns):
            theory = diff.theoreticalNthQuartVar(N, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("Variance of Nth Quartile")
            self.title("N={N}")
            ax.plot(self.time, self.var[:, i], label='Variance')
            ax.plot(self.time, theory, label='Theory')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(os.path.join(os.path.abspath(save_dir), f"Var{N}.png"),
                        bbox_inches='tight')
            plt.close(fig)

class VelocityDatabase(Database):

    def __init__(self, files):
        '''
        Initialize a Database with the selected files.

        Parameters
        ----------
        files : str
            Files to create database from
        '''

        super().__init__(files)

        self.vs = self.getVs()

    def calculateMeanVar(self):
        '''
        Calculate the mean and variance of the dataset.
        '''

        squared_sum = None
        mean_sum = None

        for f in self.files:
            data = diff.loadArrayQuad(f, self.shape, delimiter=",", skiprows=1)
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

        self.mean = mean_sum.astype(np.float64) / len(self)
        self.var = squared_sum.astype(np.float64) / len(self) - self.mean ** 2

    def getVs(self):
        '''
        Return the measured velocities from the files.

        Returns
        -------
        vs : list
            Velocities from the files
        '''

        with open(self.files[0]) as f:
            vs = g.readline().split(",")[1:]
            vs = [float(i) for i in vs]
        return vs

    def plotMean(self, save_dir='.'):
        """
        Plot the mean of each velocity as a function of time and compare to the
        theory.
        """

        for i, v in enumerate(self.vs):
            theory = diff.theoreticalPbMean(v, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("ln(Pb(vt, t))")
            ax.plot(self.time, self.mean[:, i], label="Data", c='r')
            ax.plot(self.time, theory, label='Theory')
            ax.set_title(f"v={v})
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(os.path.join(os.path.abspath(save_dir), "Mean{v}.png"),
                        bbox_inches='tight')
            plt.close(fig)

    def plotVar(self, save_dir='.'):
        """
        Plot the variance of each velocity as a function of time and compare to
        the theory.
        """

        for i, v in enumerate(self.vs):
            theory = diff.theoreticalPbVar(v, self.time)
            fig, ax = plt.subplots()
            ax.set_xlabel("Time")
            ax.set_ylabel("Var(ln(Pb(vt, t)))")
            ax.plot(self.time, self.var[:, i], label="Data", c='r')
            ax.plot(self.time, theory, label='Theory')
            ax.set_title(f"v={v})
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(os.path.join(os.path.abspath(save_dir), "Var{v}.png"),
                        bbox_inches='tight')
