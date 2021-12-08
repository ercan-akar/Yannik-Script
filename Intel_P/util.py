import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

import os
import codecs
import json
import pickle
import base64
import pandas as pd
import re

def encode_to_text(data):
    return json.dumps(codecs.encode(pickle.dumps(data), 'base64').decode())

def decode_from_text(text):
    return pickle.loads(codecs.decode(json.loads(text).encode(), 'base64'))

def get_trigger():
    return dash.callback_context.triggered[0]['prop_id'].split('.')[0]

def create_separator(colSpan):
    return html.Td(html.Hr(), colSpan = colSpan)

def read_uploaded_excel(data):
    _, data = data.split(',')
    data = base64.b64decode(data)
    dfs = pd.read_excel(data, sheet_name = None)
    return dfs

def create_excel_upload_callback(app, btn_elem, sheet_elem, storage_elem, desc_keys):
    @app.callback(
        [Output(sheet_elem, 'options'),
         Output(sheet_elem, 'value'),
         Output(storage_elem, 'children')],
        [Input(btn_elem, 'contents'),
         Input('load-trigger', 'value')],
        State('project-description', 'children')
    )
    def excel_upload_callback(contents, lt, desc):
        # print(get_trigger())
        if get_trigger() == btn_elem:
            dfs = read_uploaded_excel(contents)
            options = [{'label': name, 'value': name} for name in dfs.keys()]
            value = None if len(options) != 1 else options[0]['value']
            storage = encode_to_text(dfs)
            return [options, value, storage]
        if get_trigger() == 'load-trigger':
            desc = json.loads(desc)
            print(desc)
            inner_desc = desc
            for key in desc_keys:
                if key in inner_desc: # filter out sp700a/sa02 part
                    inner_desc = inner_desc[key]
                else:
                    return [[], None, json.dumps({})]
            pkl_file = inner_desc['path']
            sheet = inner_desc['sheet']

            print(pkl_file)
            print(sheet)

            if sheet is None or pkl_file == None:
                return [[], None, json.dumps({})]

            full_path = os.path.join(get_program_folder(), desc['folder'], pkl_file)

            dfs = pickle.load(open(full_path, 'rb'))
            options = [{'label': name, 'value': name} for name in dfs.keys()]

            print(sheet)
            print(options)
            return [options, sheet, encode_to_text(dfs)]

        return [[], None, json.dumps({})]

def get_program_folder():
    appdata = os.getenv('APPDATA')
    our_folder = os.path.join(appdata, 'MerckDataScienceModellingToolkit')
    return our_folder

def desc_to_directory(desc):
    #base = get_program_folder()
    #name = desc['name']
    #sanitized_name = re.sub('[^a-zA-Z0-9\s]', '', name)
    #uuid = desc['uuid']
    #return os.path.join(base, '{}.{}'.format(sanitized_name, uuid_new))
    base = get_program_folder()
    folder = desc['folder']
    return os.path.join(base, folder)
