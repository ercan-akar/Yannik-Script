from flask import Flask, send_from_directory, request
import numpy as np
import pandas as pd
import json
import os
import time
from sklearn.preprocessing import StandardScaler
import argparse
from threading import Thread
from argparse import ArgumentParser

from query import query_data
from datasource import XLSYeast
from analysis import PLSModel
import plant
from util import make_serializable


# Command line arguments
parser = ArgumentParser('plant-monitor-server')
parser.add_argument('config', type=str, default='config.json', help='Path to configuration file')
args = parser.parse_args()


# Load configuration file
with open(args.config) as f:
    plant_config = json.load(f)

plant = plant.from_config(plant_config)


# Define flask app
app = Flask(__name__)

@app.route('/equipment/<step>/<name>', methods=['GET', 'POST'])
def equipment(step, name):
    if request.method == 'GET':
        return get_equipment(step, name)
    elif request.method == 'POST':
        return post_equipment(step, name, request.json)

def get_equipment(step, name):
    equipment = plant.get_step_by_name(step).get_equipment_by_name(name)
    
    data = equipment.get_plot_data()
    serializable = make_serializable(data)
    return json.dumps(serializable)

def post_equipment(step, name, data):
    equipment = plant.get_step_by_name(step).get_equipment_by_name(name)

    missing_field_error = "data missing field '{}'"

    # check that data fits the equipment
    if not equipment.time_column in data:
        return missing_field_error.format(equipment.time_column), 400
    for x in equipment.x_columns:
        if not x in data:
            return missing_field_error.format(x), 400

    equipment.add_measurement(data)

    return "", 200

@app.route('/reset', methods=['POST'])
def post_reset():
    plant.reset()

@app.route('/plant')
def get_plant():
    return json.dumps(plant.get_data())

if __name__=='__main__':
    print("Starting flask task")
    t_flask = Thread(target=app.run)
    t_flask.start()

# Start the data query task that feed the app with live data
print("Starting query task")
t_query = Thread(target=query_data, args=('http://localhost:5000',))
t_query.start()
