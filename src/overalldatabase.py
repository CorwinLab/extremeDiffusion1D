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

def calculateMeanVarHelper(files, skiprows=1, delimiter=',', verbose=False, maxTime=None):
    '''
    Calculate mean and variance of arrays in files.
    '''
    squared_sum = None
    sum = None
    return_time = None
    number_of_files = 0
    for f in files:
        try:
            data = np.loadtxt(f, delimiter=delimiter, skiprows=skiprows)
        except StopIteration:
            continue
        #data = fileIO.loadArrayQuad(f, delimiter=delimiter, skiprows=skiprows)
        df = pd.DataFrame(data.astype(float))
        df = df.drop_duplicates(subset=[0], keep='last')

        data = df.to_numpy()
        time = data[:, 0].astype(np.float64)
        data = data[:, 1:]

        ''' This implementation didn't work b/c flip is kind of weird with a 2D array
        # We need to remove the artifacts of stopping and then restarting at
        # an earlier time.  We want to retain the *last* valid element
        time, index = np.unique(np.flip(rawtime), return_index=True)
        data = np.flip(data)[index]
        data = np.flip(data)
        '''
        if maxTime is not None:
            if max(time) < maxTime:
                continue
            elif max(time) == maxTime:
                time = time[time <= maxTime]
                if return_time is not None:
                    if len(time) < len(return_time):
                        print("Missing times:", np.setdiff1d(return_time, time))
                        continue
                return_time = time
        else:
            maxTime = max(time)

        maxIdx = len(time)
        data = data[:maxIdx, :]

        if squared_sum is None:
            squared_sum = np.zeros(data.shape, dtype=np.quad)
        if sum is None:
            sum = np.zeros(data.shape, dtype=np.quad)

        squared_sum += data ** 2
        sum += data
        number_of_files += 1

        if verbose:
            print(f, time.shape)


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

    def add_directory(self, directory, dir_type, var_file='variables.json'):
        '''
        Load a directory into the database.
        '''
        vars_file = os.path.join(directory, var_file)
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

    def N(self, dir_type):
        '''
        Get the quantiles for a specific datatype
        '''
        Ns = []
        for d in self.dirs.keys():
            if self.dirs[d]['type'] == 'Max':
                Ns.append(int(self.dirs[d]['N_exp']))
        return Ns

    def betas(self):
        '''
        Get the beta values for a specific datatype
        '''
        betas = []
        for d in self.dirs.keys():
                betas.append(float(self.dirs[d]['beta']))
        return betas

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

    def calculateMeanVar(self, directories, verbose=False, maxTime=None):
        '''
        Calculate the mean and variance over a directory.
        '''
        files = []
        if isinstance(directories, list):
            for d in directories:
                assert d in self.dirs.keys()

            for d in directories:
                search_path = os.path.join(d, 'Q*.txt')
                files += glob.glob(search_path)

        elif isinstance(directories, str):
            assert directories in self.dirs.keys()
            search_path = os.path.join(directories, 'Q*.txt')
            files += glob.glob(search_path)

        time, mean, var, maxTime, number_of_files = calculateMeanVarHelper(files, verbose=verbose, maxTime=maxTime)
        time = time.reshape((mean.shape[0], 1))

        mean = np.hstack([time, mean])
        var = np.hstack([time, var])

        if isinstance(directories, list):
            for directory in directories:
                with open(os.path.join(directory, 'Quartiles0.txt')) as f:
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

        elif isinstance(directories, str):
            with open(os.path.join(directories, 'Quartiles0.txt')) as f:
                header = f.readline().replace('\n', '')

            mean_file = os.path.join(directories, 'Mean.txt')
            var_file = os.path.join(directories, 'Var.txt')
            np.savetxt(mean_file, mean, header=header, comments='', delimiter=',')
            np.savetxt(var_file, var, header=header, comments='', delimiter=',')

            self.dirs[directories]['mean'] = mean_file
            self.dirs[directories]['var'] = var_file
            self.dirs[directories]['number_of_systems'] = number_of_files
            self.dirs[directories]['maxTime'] = maxTime

            analysis_file = os.path.join(directories, 'analysis.json')
            with open(analysis_file, 'w') as f:
                json.dump(self.dirs[directories], f)

        print('Done Calculating Mean')

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
