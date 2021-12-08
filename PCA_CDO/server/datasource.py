import pandas as pd
import numpy as np


class XLSYeast():
    """Hardcoded data source for yeast data from excel sheet.
        Raw batch data aswell as starting conditions and quality.
    """

    def __init__(self, xls_file, good_batch_ids):
        self.X_columns = [3,4,5,6,7,8,9]
        self.Y_columns = [2]
        self.n_variables = 8
        self.n_time = 83
        
        self.yeast_data = pd.read_excel(xls_file)
        self.qual_data = pd.read_excel(xls_file,2)

        self.variable_names = np.array(self.yeast_data.columns, dtype = np.str)[self.X_columns]
        
        self.good_batch_ids = good_batch_ids

        _, var_data, _ = self.training_data()
        self.mean = np.mean(var_data, axis=0)
        self.std = np.std(var_data, axis=0)


    def variable_records(self, batch_id, t=None):
        """Returns measurements of this variable for all timesteps and all batches.
            [{'batch1ID': value, 'batch2ID': value, ...}, ...]

            Args:
                batch_id: id of the batch to get variable plots of
                t: current time
            Rerturns: dict with records for each variable or list of records
        """
        batch_ids, var_data = self.batch_data_numpy()
        var_data = var_data[np.where(batch_ids==batch_id)][0]
        
        time = self.get_time()
        if t is None:
            t = self.n_time
        
        return {
            var_name: [
                {  
                    **({ 
                        'value': var_data[j, k]
                    } if j <= t else {}),
                    **{
                        't': time[j],
                        'idx': j,
                        'mean': self.mean[j, k],
                        'std+3': self.mean[j, k] + 3 * self.std[j, k],
                        'std-3': self.mean[j, k] - 3 * self.std[j, k]
                    }
                }
                for j in range(self.n_time)
            ]
            for k, var_name in enumerate(self.variable_names)
        }


    def transformed_variable_records(self, model, batch_id, t=None):
        """Takes in a model with and applies model.transform on all variables before returning them as records.
            Args:
                batch_id: id of the batch to get scores of
        """
        scores = model.x_scores_.reshape(len(self.good_batch_ids), self.n_time, -1)
        mean_scores = np.mean(scores, axis=0)
        std_scores = np.std(scores, axis=0)

        batch_ids, var_data = self.batch_data_numpy() 
        var_data = var_data[np.where(batch_ids==batch_id)][0]
        scores = model.transform(var_data)
        D = scores.shape[-1]

        time = self.get_time()
        if t is None:
            t = self.n_time

        return {
            't'+str(k): [
                {
                    **({ 
                        'value': scores[j, k]
                    } if j <= t else {}),
                    **{
                        'idx': j,
                        't': time[j],
                        'mean': mean_scores[j, k],
                        'std+3': mean_scores[j, k] + 3 * std_scores[j, k],
                        'std-3': mean_scores[j, k] - 3 * std_scores[j, k]
                    }
                }
                for j in range(self.n_time)
            ]
            for k in range(D)
        }


    def get_time(self):
        return self.yeast_data.iloc[:, [2]].to_numpy().reshape(-1)


    def batch_data_numpy(self, Y=False, flat=False):
        """Return the batches in a 3D numpy array.
            returns: batchIDs, batchData as numpy arrays
        """
        ids = self.yeast_data['BatchID'].unique()
        X = np.zeros((len(ids), self.n_time, self.n_variables if Y else len(self.X_columns)))
        columns = self.X_columns
        if Y:
            columns += self.Y_columns
        for idx, bID in enumerate(ids):
            X[idx] = self.yeast_data[self.yeast_data['BatchID']==bID].to_numpy()[:self.n_time, columns]
        return ids, X


    def training_data(self, flat=False):
        """Using information on quality and good batch ids, this function selects a training setof batches
            returns: batchIDs, batchDataX, batchDataY as numpy arrays
        """
        X = np.zeros((len(self.good_batch_ids), self.n_time, len(self.X_columns))) 
        Y = np.zeros((len(self.good_batch_ids), self.n_time, len(self.Y_columns)))
        for idx, bID in enumerate(self.good_batch_ids):
            batch = self.yeast_data[self.yeast_data['BatchID']==bID].to_numpy()
            X[idx] = batch[:self.n_time, self.X_columns]
            Y[idx] = batch[:self.n_time, self.Y_columns]
        return self.good_batch_ids, X, Y
