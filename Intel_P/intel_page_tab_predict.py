import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import json
import os
import base64
import pandas as pd
import util

import intel_page_tab_model
import modelling_part_intel as modelling
import prediction_intel as prediction

import pickle

def get_prefix():
    return 'predict'

def create_layout():
    predict_table = html.Table([
        html.Tr([
            html.Td(html.Div('Select Data File'), style = {'width': '6cm'}),
            html.Td(dcc.Upload('Choose File', id = 'sp700s-dataset-predict-btn', className = 'chooser-button'), style = {'width': '5cm'})
        ]),
        html.Tr([
            html.Td('Sheet Name'),
            html.Td(dcc.Dropdown(options = [], value = None, id = 'sp700s-predict-dataset-sheet'))
        ]),
        util.create_separator(2),
        html.Tr([
            html.Td('Include Outliers'),
            html.Td(dcc.RadioItems(options = [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value = 1, id = 'sp700s-predict-include-outliers'))
        ]),
        html.Tr([
            html.Td('Select Batches'),
            html.Td(html.Table([
                html.Tr([
                    html.Td(html.Button('Select all', id = 'sp700s-predict-batches-sel-all')),
                    html.Td(html.Button('De-select all', id = 'sp700s-predict-batches-sel-none'))
                ]),
                html.Tr(html.Td(dcc.Checklist(options = [], value = [], id = 'sp700s-predict-batches', labelStyle = {'display': 'block'}), colSpan = 2))
            ]))
        ]),
        html.Tr(html.Td(html.Button('Predict', id = 'sp700s-predict-evaluate-btn', style = {'width': '100%'}), colSpan = 2))
    ])

    return predict_table

def create_storage():
    return html.Div([
        html.Div(json.dumps({}), id = 'sp700s-predict-data-storage'),
        dcc.Input(type = 'text', id = 'sp700s-prediction-created', value = ''),
    ])

def create_callbacks(app):
    util.create_excel_upload_callback(app, 'sp700s-dataset-predict-btn', 'sp700s-predict-dataset-sheet', 'sp700s-predict-data-storage', ['sp700s-attributes', 'prediction', 'dataset'])

    @app.callback(
        Output('sp700s-predict-evaluate-btn', 'disabled'),
        [Input('sp700s-predict-dataset-sheet', 'value'),
         Input('sp700s-predict-batches', 'value')]
    )
    def disable_predict_btn(sheet, batches):
        return sheet == None or batches == None or len(batches) == 0

    @app.callback(
        Output('sp700s-predict-batches', 'options'),
        Input('sp700s-predict-dataset-sheet', 'value'),
        [State('sp700s-predict-data-storage', 'children'),
         State('sp700s-model-product-code', 'value'),
         State('sp700s-predict-batches', 'options')]
    )
    def fill_batch_list(sheet, dataset, code, current_options):
        if sheet is not None:
            if util.get_trigger() == 'sp700s-predict-dataset-sheet':
                dataset = util.decode_from_text(dataset)[sheet]
                dataset = modelling.prepare_main_dataset(dataset)

                intel_num = modelling.product_mapping[code]['Intel_num']

                dataset = dataset.dropna(subset = [intel_num])
                batches = dataset.loc[:, 'Lbl_6'].unique().tolist()
                options = [{'label': batch, 'value': batch} for batch in batches]
                return options
        return []

    @app.callback(
        Output('sp700s-predict-include-outliers', 'value'),
        Input('load-trigger', 'value'),
        State('project-description', 'children')
    )
    def load_outliers(load_trigger, desc):
        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sp700s':
                return json.loads(desc)['sp700s-attributes']['prediction']['include_outliers']
        return 1

    @app.callback(
        Output('sp700s-predict-batches', 'value'),
        [Input('sp700s-predict-dataset-sheet', 'value'),
         Input('sp700s-predict-batches-sel-all', 'n_clicks'),
         Input('sp700s-predict-batches-sel-none', 'n_clicks'),
         Input('load-trigger', 'value')],
        [State('sp700s-predict-batches', 'options'),
         State('project-description', 'children')]
    )
    def batch_selection(sheet, clicks1, clicks2, load_trigger, current_options, desc):
        if util.get_trigger() == 'sp700s-predict-dataset-sheet':
            return []
        if util.get_trigger() == 'sp700s-predict-batches-sel-all':
            return [elem['value'] for elem in current_options]
        if util.get_trigger() == 'sp700s-predict-batches-sel-none':
            return []
        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sp700s':
                return json.loads(desc)['sp700s-attributes']['prediction']['batches']
        return []

    @app.callback(
        Output('sp700s-prediction-created', 'value'),
        [Input('sp700s-predict-evaluate-btn', 'n_clicks'),
         Input('sp700s-model-created', 'value'),
         Input('load-trigger', 'value')],
        [State('sp700s-predict-data-storage', 'children'),
         State('sp700s-predict-dataset-sheet', 'value'),
         State('sp700s-predict-batches', 'value'),
         State('sp700s-predict-include-outliers', 'value'),
         State('sp700s-prediction-created', 'value'),
         State('project-description', 'children')]
    )
    def evaluate(clicks, model_created, load_trigger, data, sheet, batches, outliers, created, desc):
        if util.get_trigger() == 'sp700s-predict-evaluate-btn':
            path = util.desc_to_directory(json.loads(desc))
            all_data = util.decode_from_text(data)
            data = all_data[sheet]
            intel_num = modelling.product_mapping[json.loads(desc)['sp700s-attributes']['modelling']['product_code']]['Intel_num']
            data = prediction.prepare_dataset(data, batches)
            outliers = 0 if outliers == 'yes' else 1
            prediction.predict(data, intel_num, path, outliers)

            # save input data to disc for possibly loading it later
            desc = json.loads(desc)
            data_path = os.path.join(util.get_program_folder(), desc['folder'], desc['sp700s-attributes']['prediction']['dataset']['path'])
            pickle.dump(all_data, open(data_path, 'wb'))

            created = model_created
            return created
        if util.get_trigger() == 'sp700s-model-created':
            if created != model_created:
                created = ''
            return created

        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sp700s':
                created = json.loads(desc)['sp700s-attributes']['prediction']['id']
                return created

        return dash.no_update
