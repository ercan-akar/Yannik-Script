import pandas as pd
import numpy as np
from datasource import XLSYeast
import time
import requests
import os
import pyodbc
import threading

### manually load example batch from yeast dataset
data_source = XLSYeast(
    os.path.join('mockdata', 'bakersyeast.xls'),
    good_batch_ids=['Ba', 'Ca', 'Ia', 'Ma', 'Na', 'Qa', 'Ra', 'Ta', 'Va', 'Xa', 'Za', 'ab', 'bb', 'cb', 'db', 'eb', 'fb', 'gb', 'hb', 'ib']
) # to be replaced with an actual dataprovider with access to a database

df = data_source.yeast_data

batch_ids, batch_data = data_source.batch_data_numpy()
batches = [
    batch_data[np.where(batch_ids=='Ja')][0],
    batch_data[np.where(batch_ids=='Aa')][0],
    batch_data[np.where(batch_ids=='Ga')][0],
    batch_data[np.where(batch_ids=='Pa')][0],
    batch_data[np.where(batch_ids=='Ma')][0]
]
time_grid = data_source.get_time()
x_columns = data_source.variable_names

def query_data(receiver_url):
    """This is a concurrent process that periodically pushes new data to the server."""
    plant = requests.get(url = receiver_url+"/plant").json()    

    t = 0
    t_end = 82
    while True:
        if t <= t_end:
            k = 0
            for step in plant['steps']:
                for equipment in step['equipments']:
                    url = "{}/equipment/{}/{}".format(receiver_url, step['name'], equipment['name'])
                    m = {'Time': time_grid[t]}
                    m = {**m, **{x: v for x, v in zip(x_columns, batches[k][t])}}
                    requests.post(url, json=m)
                    k += 1 
            t += 1
        else:
            t = 0
        time.sleep(2)