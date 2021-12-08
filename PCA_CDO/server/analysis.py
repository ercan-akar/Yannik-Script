import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA
import matplotlib.pyplot as plt
from sklearn import datasets,preprocessing
import scipy
import h5py
from sklearn.preprocessing import StandardScaler
import pandas as pd
import t2contrib
import scipy.interpolate


class PLSModel():
    """
    A PLSModel of a batch process. Defined through a dataset of good reference batches.
    """
    def __init__(self, data, id_column, time_column, x_columns, n_components=3):
        """
            Args:
                data: batch data, pandas.DataFrame with columns for X and Y variables and batch id
        """
        # remove rows that miss time
        self.data = data.dropna(subset=['Time']).reset_index()
        self.id_column = id_column
        self.time_column = time_column
        self.x_columns = x_columns
        self.n_components = n_components

        self.ids = data[id_column].unique()
        self.time_grid = self.get_time_grid()
        
        self.n_batches = len(self.ids)
        self.n_time = len(self.time_grid)
        self.n_variables = len(x_columns)

        self.Y = np.repeat(self.time_grid.reshape(1, -1), self.n_batches, axis=0)
        self.X = self.map_to_grid(self.data, self.time_grid)

        self.fill_missing(self.X, self.time_grid, self.data)

        self.mean = np.nanmean(self.X, axis=0)
        self.std = np.nanstd(self.X, axis=0)

        self.pls = _train_pls(
            self.X.reshape(self.n_batches * self.n_time, self.n_variables), 
            self.Y.reshape(self.n_batches * self.n_time),
            self.n_components)
        


        self.P = self.pls.x_loadings_
        self.W = self.pls.x_rotations_

        self.scores = self.pls.x_scores_
        self.scores_mean = np.mean(self.pls.x_scores_.reshape(self.n_batches, self.n_time, -1), axis=0)
        self.scores_std = np.std(self.pls.x_scores_.reshape(self.n_batches, self.n_time, -1), axis=0)
        self.scores_covariance = 1/(self.scores.shape[0]-1) * (self.scores.T @ self.scores)
        self.scores_eigenvalues = np.diagonal(self.scores_covariance)

        dmodx, dmodx_contrib = self.get_dModX_pls(self.X.reshape(self.n_batches * self.n_time, self.n_variables))
        self.dmodx = dmodx.reshape(self.n_batches, self.n_time)
        self.dmodx_contrib = dmodx_contrib.reshape(self.n_batches, self.n_time, self.n_variables)
        self.dmodx_mean = np.mean(self.dmodx, axis=0)
        self.dmodx_std = np.std(self.dmodx, axis=0)
        self.dmodx_contrib_mean = np.mean(self.dmodx_contrib, axis=0)

        self.t2 = self.get_t2_pls(self.X.reshape(self.n_batches * self.n_time, self.n_variables)).reshape(self.n_batches, self.n_time)
        # self.t2_contrib = t2_contrib.reshape(self.n_batches, self.n_time, self.n_variables)
        self.t2_mean = np.mean(self.dmodx, axis=0)
        self.t2_std = np.std(self.dmodx, axis=0)
        # self.t2_contrib_mean = np.mean(self.t2_contrib, axis=0)

    def get_time_grid(self, alpha=0.1):
        """Calculates a equally spaced time grid from batchwise time values"""
        times = np.unique(self.data[self.time_column])
        # remove potentially occuring NaN values
        times = times[~np.isnan(times)]
        # count the number of timepoints with min distance
        # if two time points are spaced less then alpha * average_spacing they are counted as one
        mean_spacing = np.mean(times[1:] - times[0:-1])
        count = 1
        last = times[0]
        for t in times[1:]:
            diff = t - last
            if diff >= alpha * mean_spacing:
                count += 1
                last = t
        
        return np.linspace(times[0], times[-1], count)

    def map_to_grid(self, data, grid):
        """Maps measurement wise batch data to a time grid by interpolating."""
        time_df = pd.DataFrame({self.time_column: grid})
        batches = [] 
        for id in self.ids:
            batch_data = data[data[self.id_column]==id]
            # align on time grid and fill in nan rows for times that dont match
            batch_data_aligned = _join_on_float(time_df, batch_data, self.time_column, 'left', precision=3)[self.x_columns].to_numpy(dtype=np.float64)
           

            batches.append(batch_data_aligned)
        
        return np.stack(batches)

    def fill_missing(self, data_aligned, grid, data, interpolation='cubic'):
        """Fills missing values in the grid aligned data array based on the raw unaligend data.
        Args:
            data_aligned: the grid aligned data array
            grid: the time grid the data is aligned to
            data: the original unaligned data as DataFrame
            interpolation: the interpolation method to use
        """
        mean_batch = np.nanmean(data_aligned, axis=0)

        for id_idx, id in enumerate(self.ids):
            batch_data = data[data[self.id_column]==id]

            # interpolate missing values for each variable
            for c_idx, c in enumerate(self.x_columns):
                nans = np.isnan(data_aligned[id_idx, :, c_idx])

                if nans.any():
                    gaps = np.where(nans)[0]
                    
                    # extrapolate start and end with mean
                    not_gaps = np.where(~nans)[0]
                    data_aligned[id_idx, 0: not_gaps[0], c_idx] = mean_batch[0: not_gaps[0], c_idx]
                    data_aligned[id_idx, not_gaps[-1]+1:, c_idx] = mean_batch[not_gaps[-1]+1:, c_idx]
                    
                    # remove extrapolated gaps before interpolating
                    gaps = np.array([x for x in gaps if x >= not_gaps[0] and x <= not_gaps[-1]])
                    
                    valid_variable_data = batch_data[[self.time_column, c]].dropna()
                    
                    interp = scipy.interpolate.interp1d(
                        valid_variable_data[self.time_column].to_numpy(),
                        valid_variable_data[c].to_numpy(),
                        kind=interpolation)
                    
                    data_aligned[id_idx, gaps, c_idx] = interp(grid[gaps])

    def project(self, X):
        is_batch = True
        
        if len(X.shape) < 2:
            is_batch = False
            X = X.reshape(1, -1)
        
        X_projected = self.pls.transform(X)
        
        if is_batch:
            return X_projected
        else:
            return X_projected[0]

    def get_t2_pls(self, X):
        n = X.shape[0]
        is_batch = True
        
        if len(X.shape) < 2:
            is_batch = False
            n = 1
            X = X.reshape(1, -1)
        
        t2 = np.zeros(n)

        scores = self.pls.transform(X)

        # best reproduction from rolf:
        # t2[j] = np.sum(X_scores[j,:]/eigenvalues*X_scores[j,:]) 
        for i in range(n):
            t2[i] = np.sum(scores[i, :] / self.scores_eigenvalues * scores[i, :])

        if is_batch:
            return t2
        else:
            return t2[0]

    def get_t2_contrib_pls(self, X, t, components=[0], variant="P"):
        is_batch = True
        
        if len(X.shape) < 2:
            is_batch = False
            X = X.reshape(1, -1)
            t = np.array([t])
        else:
            assert X.shape[0] == t.shape[0]

        contributions = getattr(t2contrib, variant)(self, X, t, components)
                
        if is_batch:
            return contributions
        else:
            return contributions[0]

    def get_dModX_pls(self, X):

        is_batch = True
        
        if len(X.shape) < 2:
            is_batch = False
            X = X.reshape(1, -1)
        
        # scores = self.pls.transform(X)
        # n_train = self.n_batches * self.n_time            

        X_scaled = (X - self.pls.x_mean_) / self.pls.x_std_
        P = self.pls.x_loadings_
        R = self.pls.x_rotations_
        
        contributions = X_scaled - X_scaled @ (P @ R.T)

        DModX = np.linalg.norm(contributions, axis = 1)#/np.sqrt((I-A)/A)
        
        if is_batch:
            return DModX, contributions
        else:
            return DModX[0], contributions[0]


def _train_pls(X, Y, ncomp=3):
    """Trains and returns a PLSRegressor on the given data."""
    assert(X.shape[0] == Y.shape[0])
    if len(X.shape) == 3: # unfold batch dim
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        Y = Y.reshape(Y.shape[0] * Y.shape[1], -1)

    # use PLS, data is already scaled
    pls = PLSRegression(n_components=ncomp, scale = True, copy=True)

    pls.fit(X = X, Y = Y)
    # pls.x_loadings_[:,-1] *= -1
    # pls.x_rotations_[:, -1] *= -1
    # pls.x_scores_[:, -1] *= -1
    # pls.x_weights_[:, -1] *= -1
    return pls


def _train_pca(X, ncomp=3):
    if len(X.shape) == 3: # unfold batch dim
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=ncomp, svd_solver='full')
    pca.fit(X_scaled)

    return pca, scaler

def _join_on_float(left, right, on, how, precision=4):
    """Join outer two DataFrames with approximate comparison operator.
    Args:
        df1: DataFrame to join into
        df2: other
        on: column name to join on
        how: join method, 'left', 'right' or 'outer'
        precision: decimal points to compare
    """
    if how == 'left':
        return _join_left_on_float(left, right, on, precision)
    elif how == 'outer':
        return _join_outer_on_float(left, right, on, precision)
    elif how == 'right':
        return _join_left_on_float(right, left, on, precision)
    else:
        raise ValueError("Unsuppeorted join method")

def _join_left_on_float(left, right, on, precision=4):
    e = 0.1 ** (precision+1)
    df_join = pd.DataFrame(index=left.index, columns=set(list(left.columns)+list(right.columns)))

    left_len = len(left.index)
    right_len = len(right.index)
    left_idx, right_idx = 0, 0

    # sort both along join column
    left = left.sort_values(by=[on])
    right = right.sort_values(by=[on])

    while left_idx < left_len:
        if (right_idx >= right_len):
            # just append whats left in left
            df_join.loc[left_idx:, left.columns] = left.iloc[left_idx:]
            break

        # append left, right or joinded
        left_val = left[on].iloc[left_idx].round(precision)
        right_val = right[on].iloc[right_idx].round(precision)
        
        if abs(left_val - right_val) <= e:
            # equal fill with merged values
            df_join.loc[left_idx, left.columns] = left.iloc[left_idx]
            df_join.loc[left_idx, right.columns] = right.iloc[right_idx]
            left_idx += 1
            right_idx += 1
        
        elif left_val < right_val:
            # not equal, take left, leave rest NaN
            df_join.loc[left_idx, left.columns] = left.iloc[left_idx]
            left_idx += 1

        elif left_val > right_val:
            # not equal, ignore right
            right_idx += 1
        
        else:
            print("Invalid comparison state during float join")

    return df_join


def _join_outer_on_float(left, right, on, precision=4):
    e = 0.1 ** (precision+1)
    df_join = pd.DataFrame(columns=set(list(left.columns)+list(right.columns)))

    left_len = len(left.index)
    right_len = len(right.index)
    left_idx, right_idx = 0, 0

    # sort both along join column
    left = left.sort_values(by=[on])
    right = right.sort_values(by=[on])

    while left_idx < left_len or right_idx < right_len:
        if (left_idx >= left_len):
            # just append whats left in right
            df_join = df_join.append(right.iloc[right_idx:], ignore_index=True)
            break

        elif (right_idx >= right_len):
            # just append whats left in left
            df_join = df_join.append(left.iloc[left_idx:], ignore_index=True)
            break

        else:
            # append left, right or joinded
            left_val = left.loc[left_idx, on].round(precision)
            right_val = right.loc[right_idx, on].round(precision)
            
            # next row
            df_row = pd.DataFrame(index=[0], columns=df_join.columns)
            
            if left_val - right_val <= e:
                # equal fill with merged values
                df_row.loc[0, left.columns] = left.iloc[left_idx]
                df_row.loc[0, right.columns] = right.iloc[right_idx]
                left_idx += 1
                right_idx += 1
            
            elif left_val < right_val:
                # not equal, take left, leave rest NaN
                df_row.loc[0, left.columns] = left.iloc[left_idx]
                left_idx += 1

            elif left_val > right_val:
                # not equal, take right, leave rest NaN
                df_row.loc[0, right.columns] = right.iloc[right_idx]
                right_idx += 1

            else:
                print("Invalid comparison state during float join")

            # append the next row
            df_join = df_join.append(df_row, ignore_index=True)

    return df_join

            


