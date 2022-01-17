import json
import os
import quadMath
import numpy as np
import npquad
import glob
import fileIO
import pandas as pd

def getQuantilesGumbelFile(file):
    '''
    Get the quantiles for a CDF file.
    '''
    quantiles = []
    with open(file) as f:
        file_quantiles = f.readline().split(",")[1:] # need to get rid of time column
        for q in file_quantiles:
            if 'var' in q:
                continue
            else:
                q = quadMath.prettifyQuad(np.quad(q))
                q_exp = q.split('e')[-1]
                quantiles.append(q_exp)
    return quantiles

def calculateMeanVar(files, skiprows=1, delimiter=',', verbose=False, nFiles=-1, maxTime=None):
    '''
    Calculate mean and variance of arrays in files.
    '''
    squared_sum = None
    sum = None
    return_time = None
    number_of_files = 0
    for f in files[:nFiles]:
        data = fileIO.loadArrayQuad(f, delimiter=delimiter, skiprows=skiprows)
        time = data[:, 0].astype(np.float64)
        data = data[:, 1:]

        if maxTime is not None:
            if max(time) < maxTime:
                continue
            else:
                time = time[time <= maxTime]
                return_time = time
                number_of_files += 1
        else:
            maxTime = max(time)
            number_of_files += 1
        maxIdx = len(time)
        data = data[:maxIdx, :]

        if squared_sum is None:
            squared_sum = np.zeros(data.shape, dtype=np.quad)
        if sum is None:
            sum = np.zeros(data.shape, dtype=np.quad)

        squared_sum += data ** 2
        sum += data

        if verbose:
            print(f)

    if return_time is None:
        return_time = time
    mean = (sum / number_of_files).astype(np.float64)
    var = (squared_sum / number_of_files).astype(np.float64) - mean ** 2

    return return_time, mean, var, maxTime, number_of_files

class Database:
    def __init__(self):
        self.dirs = {}

    def __len__(self):
        return len(self.dirs.keys())

    def add_directory(self, directory, dir_type):
        '''
        Load a directory into the database.
        '''
        vars_file = os.path.join(directory, 'variables.json')
        with open(vars_file, "r") as file:
            try:
                vars = json.load(file)
            except Exception as e:
                print(f"Exception caught in dir {directory} trying to read variables.json file. Trying to load scalars.json file")
                scalars_file = os.path.join(directory, 'Scalars0.json')
                with open(scalars_file, "r") as file:
                    vars = json.load(file)

        self.dirs[directory] = vars
        if 'N_exp' not in self.dirs[directory].keys():
            quantile_file = os.path.join(directory, 'Quartiles0.txt')
            quantiles = getQuantilesGumbelFile(quantile_file)
            self.dirs[directory]['N_exp'] = quantiles

        self.dirs[directory]['type'] = dir_type
        mean_file = os.path.join(directory, 'Mean.txt')
        var_file = os.path.join(directory, 'Var.txt')

        if os.path.exists(mean_file) and os.path.exists(var_file):
            self.dirs[directory]['mean'] = mean_file
            self.dirs[directory]['var'] = var_file

    @classmethod
    def fromDirs(cls, dirs, dir_types):
        '''
        Initialize a Database object from a list of directories.
        '''

        db = cls()
        for dir_type, dir in zip(dir_types, dirs):
            db.add_directory(dir, dir_type)
        return db

    def getBetas(self, beta):
        '''
        Get all datasets with specified beta value.
        '''
        beta_dirs = []
        dir_types = []
        for d in self.dirs.keys():
            beta_dir = float(self.dirs[d]['beta'])
            if beta == beta_dir:
                beta_dirs.append(d)
                dir_types.append(self.dirs[d]['type'])

        if not beta_dirs:
            raise ValueError(f"beta=={beta} not in database")

        return self.fromDirs(beta_dirs, dir_types)

    def getN(self, N):
        '''
        Get all datasets with specified number of particles.
        '''
        N_dirs = []
        dir_types = []
        for d in self.dirs.keys():
            N_exp = self.dirs[d]['N_exp']
            if isinstance(N_exp, list):
                N_exp = [int(i) for i in N_exp]
                if N in N_exp:
                    N_dirs.append(d)
                    dir_types.append(self.dirs[d]['type'])
            else:
                if N == int(N_exp):
                    N_dirs.append(d)
                    dir_types.append(self.dirs[d]['type'])

        if not N_dirs:
            raise ValueError(f"N={N} not in database")

        return self.fromDirs(N_dirs, dir_types)

    def calculateMeanVar(self, directory, verbose=False, nFiles=-1, maxTime=None):
        '''
        Calculate the mean and variance over a directory.
        '''
        assert directory in self.dirs.keys()

        search_path = os.path.join(directory, 'Q*.txt')
        files = glob.glob(search_path)
        time, mean, var, maxTime, number_of_files = calculateMeanVar(files, verbose=verbose, nFiles=nFiles, maxTime=maxTime)
        time = time.reshape((mean.shape[0], 1))

        mean = np.hstack([time, mean])
        var = np.hstack([time, var])

        with open(files[0]) as f:
            header = f.readline().replace('\n', '')

        mean_file = os.path.join(directory, 'Mean.txt')
        var_file = os.path.join(directory, 'Var.txt')
        np.savetxt(mean_file, mean, header=header, comments='', delimiter=',')
        np.savetxt(var_file, var, header=header, comments='', delimiter=',')

        self.dirs[directory]['mean'] = mean_file
        self.dirs[directory]['var'] = var_file
        self.dirs[directory]['number_of_systems'] = number_of_files
        self.dirs[directory]['maxTime'] = maxTime

        analysis_file = os.path.join(directory, 'analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(self.dirs[directory], f)

    def getMeanVarN(self, N, delimiter=','):
        db = self.getN(N)
        cdf_df = pd.DataFrame()
        max_df = pd.DataFrame()
        for d in db.dirs.keys():
            mean_df = pd.read_csv(db.dirs[d]['mean'], sep=delimiter)
            var_df = pd.read_csv(db.dirs[d]['var'], sep=delimiter)

            if db.dirs[d]['type'] == 'Gumbel':
                cdf_df['time'] = mean_df['time']
                for column in mean_df.columns[1:]:
                    if 'var' in column:
                        N_column = np.quad(column.replace("var", ''))
                        exp = quadMath.prettifyQuad(N_column).split("e")[-1]
                        if int(exp) == N:
                            cdf_df['Gumbel Mean Variance'] = mean_df[column]
                    else:
                        N_column = np.quad(column)
                        exp = quadMath.prettifyQuad(N_column).split("e")[-1]
                        if int(exp) == N:
                            cdf_df['Mean Quantile'] = mean_df[column]

                for column in var_df.columns[1:]:
                    if 'var' in column:
                        continue
                    else:
                        N_column = np.quad(column)
                        exp = quadMath.prettifyQuad(N_column).split("e")[-1]
                        if int(exp) == N:
                            cdf_df['Var Quantile'] = var_df[column]

            if db.dirs[d]['type'] == 'Max':
                mean_df = pd.read_csv(db.dirs[d]['mean'], sep=delimiter)
                var_df = pd.read_csv(db.dirs[d]['var'], sep=delimiter)
                max_df['time'] = mean_df['time']
                max_df['Mean Max'] = mean_df['MaxEdge']
                max_df['Var Max'] = var_df['MaxEdge']

        return cdf_df, max_df


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import theory
    from matplotlib.colors import LinearSegmentedColormap

    def getLessThanT(time, mean):
        greater = mean >= time-1
        nonzero = np.nonzero(greater)[0][-1]
        return time[nonzero]

    db = Database()
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxBetaSweep2/1/', dir_type='Max')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxBetaSweep2/5/', dir_type='Max')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/SweepVariance/', dir_type='Gumbel')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart100/', dir_type='Max')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart300/', dir_type='Max')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPartSmall/2/', dir_type='Max')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPartSmall/6/', dir_type='Max')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPartSmall/20/', dir_type='Max')
    db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/CDFSmallBeta1/', dir_type='Gumbel')

    #db.calculateMeanVar('/home/jacob/Desktop/corwinLabMount/CleanData/CDFSmallBeta1/', verbose=True)
    db = db.getBetas(1)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t / \ln(N)$")
    ax.set_ylabel(r"$\sigma^{2}_{Q} / \ln(N)^{2/3}$")

    quantiles = [2, 6, 20, 100, 300]
    cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
    colors = [
        cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
    ]
    ypower = 2/3
    for i, N in enumerate(quantiles):
        cdf_df, max_df = db.getMeanVarN(N)

        Nquad = np.quad(f"1e{N}")
        logN = np.log(Nquad).astype(float)
        crossover=logN**2
        predicted = theory.quantileVar(Nquad, cdf_df['time'].values, crossover=crossover, width=crossover/10)
        ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**ypower, alpha=0.5, c=colors[i])
        ax.plot(cdf_df['time'] / logN, predicted / logN**ypower, ls='--', c=colors[i])

    ax.set_xlim([0.5, 5*10**3])
    fig.savefig("QuantileVar.png")

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t / \ln(N)$")
    ax.set_ylabel(r"$\bar{X}_Q(N, t)$")

    fig2, ax2 = plt.subplots()
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$t / \ln(N)$")
    ax2.set_ylabel(r"$Residual$")

    quantiles = [2, 6, 20, 100, 300]
    cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
    colors = [
        cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
    ]
    ypower = 2/3
    for i, N in enumerate(quantiles):
        cdf_df, max_df = db.getMeanVarN(N)
        Nquad = np.quad(f"1e{N}")
        logN = np.log(Nquad).astype(float)
        predicted = theory.quantileMean(Nquad, (cdf_df['time']).values)
        ax.plot((cdf_df['time']) / logN, cdf_df['Mean Quantile'] - 2, alpha=0.5, c=colors[i])
        ax.plot((cdf_df['time']) / logN, predicted, ls='--', c=colors[i])
        ax2.plot(cdf_df['time'] / logN, predicted - cdf_df['Mean Quantile'], c=colors[i])

    ax.set_xlim([10**-3, 3*10**3])
    ax.set_ylim([1, 7*10**4])
    fig.savefig("QuantileMean.png")
    ax2.set_xlim([1, 10**4])
    ax2.set_ylim([10**-2, 2 * 10**2])
    fig2.savefig("ResidualMean.png")

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t / \ln(N)$")
    ax.set_ylabel(r"$\sigma^{2}_{max} (N, t) / \ln(N)^{2/3}$")
    quantiles = [2, 6, 20, 100, 300]
    cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
    colors = [
        cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
    ]

    for i, quantile in enumerate(quantiles):
        N = np.quad(f"1e{quantile}")
        logN = np.log(N).astype(float)
        cdf_df, max_df = db.getMeanVarN(quantile)

        if quantile != 100:
            max_df['Mean Max'] = max_df['Mean Max'] * 2
        else:
            max_df['Mean Max'] = (max_df['Mean Max'] - (max_df['time'] / 2))
        max_df['Var Max'] = max_df['Var Max'] * 4

        var_theory = theory.quantileVar(N, max_df['time'].values)

        ax.plot(max_df['time'] / logN, (var_theory)/ logN**(2/3) + np.pi**2 / 12 * max_df['time'] / logN / logN**(2/3) , '--', c=colors[i])
        ax.plot(max_df['time'] / logN, max_df['Var Max'] / logN**(2/3), label=quantile, c=colors[i], alpha=0.5)

    end_coord = (200, 0.5)
    start_coord = (100, 2*10**2)
    dx = np.log(start_coord[0]) - np.log(end_coord[0])
    dy = np.log(start_coord[1]) - np.log(end_coord[1])
    theta = np.rad2deg(np.arctan2(dy, dx))
    ax.annotate("", xy=end_coord, xytext=start_coord,
            arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=20, headwidth=50, headlength=30, alpha=0.5), zorder=0)
    ax.annotate(r"$N=10^{2}$", xy=(50, 3 *10**2), c=colors[0], rotation=-(90-theta), rotation_mode='anchor')

    ax.set_xlim([0.3, 5*10**3])
    ax.set_ylim([10**-4, 10**3])
    fig.savefig("Var.png")

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t / \ln(N)$")
    ax.set_ylabel(r"$\overline{X_{max}}(N, t)$")

    ax2 = fig.add_axes([0.2, 0.57, 0.25, 0.25])
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\ln(N)$", fontsize=8, labelpad=0)
    ax2.set_ylabel(r"$\tau$", fontsize=8, labelpad=0)
    ax2.tick_params(axis='both', which='major', labelsize=6)

    quantiles = [2, 6, 20, 100, 300]
    cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
    colors = [
        cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
    ]
    logNs = []
    t_less_than = []

    for i, quantile in enumerate(quantiles):
        N = np.quad(f"1e{quantile}")
        logN = np.log(N).astype(float)
        cdf_df, max_df = db.getMeanVarN(quantile)

        if quantile != 100:
            max_df['Mean Max'] = max_df['Mean Max'] * 2
        else:
            max_df['Mean Max'] = 2*(max_df['Mean Max'] - (max_df['time'] / 2))

        max_df['Var Max'] = max_df['Var Max'] * 4

        var_theory = theory.quantileMean(N, max_df['time'].values)

        ax.plot(max_df['time'] / logN, max_df['Mean Max'], label=quantile, c=colors[i], alpha=0.8)
        ax.plot(max_df['time'] / logN, var_theory, '--', c=colors[i])
        ax.plot(max_df['time'] / logN, np.sqrt(2 * max_df['time'] * logN), c=colors[i], alpha=0.8, ls='dotted')

        t_less_than.append(getLessThanT(max_df['time'].values, max_df['Mean Max'].values))
        logNs.append(logN)

    start_coord = (200, 30)
    end_coord = (8, 2*10**4)
    dx = np.log(start_coord[0]) - np.log(end_coord[0])
    dy = np.log(start_coord[1]) - np.log(end_coord[1])
    theta = np.rad2deg(np.arctan2(dy, dx))
    ax.annotate("", xy=end_coord, xytext=start_coord,
            arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=50, headwidth=100, headlength=60, alpha=0.5), zorder=0)
    ax.annotate(r"$N=10^{2}$", xy=(start_coord[0] - 90, start_coord[1] - 17), c=colors[0], rotation=90-abs(theta), rotation_mode='anchor')
    ax.annotate(r"$N=10^{300}$", xy=(3, 1.5*10**4), c=colors[-1], rotation=90-abs(theta), rotation_mode='anchor')
    ax2.scatter(logNs, t_less_than, c=colors)
    ax2.plot(logNs, logNs, c='k', ls='--')
    ax.set_xlim([10**-3, 5*10**3])
    ax.set_ylim([1, 10**5])
    fig.savefig("Mean.png")
