import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler
from analysis import PLSModel
import t2contrib

def from_config(config):
    """Creates a new Equipment from a config dict."""
    required_fields = [
        'name',
        'model_data',
        'variables',
        'time_column',
        'id_column'
    ]
    for field_name in required_fields:
        if not field_name in config:
            raise ValueError("field {} missing in equipement config".format(field_name))

    if not os.path.exists(config['model_data']):
        raise FileNotFoundError("model data path {} not found".format(config['model_data'])) 
    model_data = pd.read_csv(config['model_data'])

    if len(config['variables']) == 0:
        raise ValueError("Variables array should be non empty")

    for variable in config['variables']:
        if not 'tag' in variable:
            raise ValueError("Variables require a tag field")
        if not 'name' in variable:
            raise ValueError("Variables require a name")

    return Equipment(
        config['name'],
        model_data,
        config['variables'],
        config['time_column'],
        config['id_column']
    )

class Equipment():
    """ A monitored equipment in a production pipeline.
    A monitored equipment has measurements of multiple variables for the current
    batch. It also has a PLS self.model.on how these measurements should behave under
    normal conditions.
    Multiple equipments are part of a processing step.
    Equipment names have to be unique among a processing step.

    Args:
        name: the name of the equipment. Should be unique among the processing step
        model_data: a DataFrame containing the measurements of multiple good batches
        variables: the names of the columns in the training data that contain x variable observations and the tag of the corresponding sensor in the machine
        y_columns: the names of the columns in the training data that contain y variable observations
        id_column: the name of the column that contains the batch id of that sample
    """
    def __init__(self, name, model_data, variables, time_column='time', id_column='batchid'):

        self.name = name
        self.variables = variables
        self.x_columns = [v['name'] for v in variables]
        self.status = 0
        self.time_column = time_column
        self.id_column = id_column

        self.model = PLSModel(
            model_data,
            self.id_column,
            self.time_column,
            self.x_columns)

        self._t = 0 # index into model.time_grid that is closest to self.time[-1]  
        self.batch_start_time = 0
        self.time = np.array([]) # batch rel. time
        self._X = np.array([]) # measurements
        self.n_time = len(self.model.time_grid) # expected number of measurements

        # sets batch_start time to 0 
        self.new_batch(0)

    def get_data(self):
        return {
            'name': self.name,
            'status': self.status,
            'variables': self.variables
        }

    def get_plot_data(self):
        return {
            'name': 'Ruehrmischer',
            'status': 'Running',
            'process_step': 'Mischen',
            'time_stamps': self.model.time_grid,
            'n_time_steps': self.n_time,
            '_batch_id': 'Ja',
            'variables': {
                name: self.merge_line_plots(
                    self.plot_lines(self.model.time_grid, {
                        'mean': self.model.mean[:, i],
                        'std+3': self.model.mean[:, i] + 3 * self.model.std[:, i],
                        'std-3': self.model.mean[:, i] - 3 * self.model.std[:, i]
                    }),
                    self.plot_lines(self.time, {'value': self._X[:, i]}, index_key='idx')
                )

                for i, name in enumerate(self.x_columns)
            },
            'scores': {
                's'+str(i): self.merge_line_plots(
                    self.plot_lines(self.model.time_grid, {
                        'mean': self.model.scores_mean[:, i],
                        'std+3': self.model.scores_mean[:, i] + 3 * self.model.scores_std[:, i],
                        'std-3': self.model.scores_mean[:, i] - 3 * self.model.scores_std[:, i]
                    }),
                    self.plot_lines(self.time, {
                        'value': self._scores[:, i]
                    }, index_key='idx')
                )
                for i in range(self.model.n_components)
            },
            't2': self.merge_line_plots(
                self.plot_lines(self.model.time_grid, {
                    'Crit95': np.ones_like(self.model.time_grid) * 10.7,
                    'Crit99': np.ones_like(self.model.time_grid) * 17.3
                }),
                self.plot_lines(self.time, {
                    'value': self._t2
                }, index_key='idx')
            ),
            't2_contrib': {
                variant: self.plot_bars(value, self.x_columns)
                for variant, value in self._t2_contrib.items()
            },
            't2_contrib_composed': {
                variant: self.plot_lines(self.time, {'value': value}, index_key='idx')
                for variant, value in self._t2_contrib_composed.items()
            },
            'dmodx': self.merge_line_plots(
                self.plot_lines(self.model.time_grid, {
                    'std+3': self.model.dmodx_mean + self.model.dmodx_std * 3
                }),
                self.plot_lines(self.time, {
                    'value': self._dmodx
                }, index_key='idx')
            ),
            'dmodx_contrib': self.plot_bars(self._dmodx_contrib, self.x_columns)
        }

    def _process_recent_measurement(self):
        """Updates internals for most recent measurement self._X[-1]"""
        x = self._X[-1:, :]
        s = self.model.project(x)
        dmodx, dmodx_contrib = self.model.get_dModX_pls(x)
        t2 = self.model.get_t2_pls(x)
        
        self._scores = np.concatenate((self._scores, s)) if self._scores.shape[0] > 0 else s
        self._dmodx = np.concatenate((self._dmodx, dmodx)) if self._dmodx.shape[0] > 0 else dmodx
        self._dmodx_contrib = np.concatenate((self._dmodx_contrib, dmodx_contrib)) if self._dmodx_contrib.shape[0] > 0 else dmodx_contrib
        self._t2 = np.concatenate((self._t2, t2)) if self._t2.shape[0] > 0 else t2
       
        for variant in [v for v, f in t2contrib.__dict__.items() if callable(f)]:
            t2_contrib = self.model.get_t2_contrib_pls(x, self.time[-1:], variant=variant)
            composed = np.sum(t2_contrib, axis=1)
            self._t2_contrib[variant] = np.concatenate((self._t2_contrib[variant], t2_contrib)) if self._t2_contrib[variant].shape[0] > 0 else t2_contrib
            self._t2_contrib_composed[variant] = np.concatenate((self._t2_contrib_composed[variant], composed)) if self._t2_contrib_composed[variant].shape[0] > 0 else composed

        self._update_status()

    def _update_status(self):
        """The status can only get worse inside a batch. It is resetted druing batch restart
        """
        t2_95_limit = 10
        t2_99_limit = 17

        if self.status == 0 and self._t2[-1] > t2_95_limit:
            self.status = 1
        
        if self.status < 2 and self._t2[-1] > t2_99_limit:
            self.status = 2

        if self.status < 1:
            for i, x in enumerate(self._X[-1]):
                if x > self.model.mean[self._t][i] + 3*self.model.std[self._t][i]:
                    self.status = 1
                    break

            for i, x in enumerate(self._scores[-1]):
                if x > self.model.scores_mean[self._t][i] + 3*self.model.scores_std[self._t][i]:
                    self.status = 1
                    break

    def add_measurement(self, m):
        """Adds a new measurement value to the currently running batch
        """
        batch_relative_time = m[self.time_column] - self.batch_start_time

        if not isinstance(m, dict):
            raise ValueError("Expects measurement of format: {time, v1, v2...}")
        if not self.time_column in m:
            raise ValueError("Measurement is missing time")

        if batch_relative_time >= self.model.time_grid[-1]:
            print("Time overflow, new batch should habe been started by now.")
            self.status = 5 # set overtime status
        else:
            X = np.empty((1, len(self.x_columns)))
            for k, v in m.items():
                try:
                    idx = self.x_columns.index(k)
                    X[0, idx] = v
                except (ValueError, IndexError):
                    if k != self.time_column:
                        print("Warning measurement {} contained unknown variable {}".format(m, k))
            
            self.time = np.append(self.time, batch_relative_time)
            self._X = np.concatenate((self._X, X)) if self._X.shape[0] > 0 else X
            self._t = np.argmin(np.abs(self.model.time_grid - batch_relative_time))

            self._process_recent_measurement()

    def new_batch(self, start_time):
        """Clears the internal states for the current batch. May save to hist.
        """
        print("New batch")
        if len(self.time) > 0:
            self.batch_start_time = self.time[-1]
        else:
            self.batch_start_time = 0
        self.status = 0
        self._t = 0
        self.time = np.array([])
        self._batch_id = ''
        self._X = np.array([])
        self._scores = np.array([])
        self._t2 = np.array([])
        self._t2_contrib = {
            v: np.array([]) for v in [v for v, f in t2contrib.__dict__.items() if callable(f)]
        }
        self._t2_contrib_composed = {
            v: np.array([]) for v in [v for v, f in t2contrib.__dict__.items() if callable(f)]
        }
        self._dmodx = np.array([])
        self._dmodx_contrib = np.array([])

    def plot_bars(self, bars, categories):
        """Creates a barchart for each vector in last dimension of 2D array bars with 
        labels from categories.
        """
        return [
            [
                {
                    'name': name,
                    'value': bars[i][idx]
                }
                for idx, name in enumerate(categories)
            ]
            for i in range(len(bars))
        ]

    def plot_lines(self, time_grid, series={}, lengths={}, index_key=None):
        """Args:
            index: whether to append the key 'index'=idx to each datapoint
        """
        plot_data = []

        for i in range(len(time_grid)):
            point = {}
            if index_key is not None:
                point[index_key] = i
            point['t'] = time_grid[i]
            for k, v in series.items():
                if i < lengths.get(k, 1000000) and i < len(v):
                    point[k] = v[i]
            plot_data.append(point)
        
        return plot_data

    def merge_line_plots(self, *line_plots, key='t', precision=1e-4):
        """Creates a common grid and merges values that have a common
        key (up to precision if key is float).
        ATTENTION: value merging does only work if all the lines have 
        differen datapoint keys.
        DISCLAIMER: VERY HACKY CODE
        """
        out = []
        idxs = [0 for _ in line_plots]
        lens = [len(s) for s in line_plots]

        while True in [i < l for i, l in zip(idxs, lens)]:
            key_vals = [
                s[i][key] if i < l else 9999999 
                for s, i, l in zip(line_plots, idxs, lens)
            ]
            min_val = min(key_vals)
            
            point = {}
            for i, key_val in enumerate(key_vals):
                if key_val - min_val < precision:
                    point = {**point, **line_plots[i][idxs[i]]}
                    idxs[i] += 1
            
            point[key] = min_val
            out.append(point)

        return out


