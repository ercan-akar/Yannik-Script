import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import json
import os
import base64
import pandas as pd
import util
import uuid
import pickle

import modelling_part_intel as modelling

def create_layout():
    product_codes = [{'label': name, 'value': name} for name in modelling.product_mapping.keys()]

    qc_modes = [
        {
            'label': 'All QC',
            'value': 'ALL_QC'
        },
        {
            'label': 'PP RM',
            'value': 'PP_RM'
        },
        {
            'label': 'All QC until bulk',
            'value': 'ALL_QC_until_bulk'
        },
        {
            'label': 'PP RM until bulk',
            'value': 'PP_RM_until_bulk'
        }
    ]

    model_table = html.Table([
        html.Tr([
            html.Td(html.Div('Select Data File'), style = {'width': '6cm'}),
            html.Td(dcc.Upload('Choose File', id = 'sp700s-model-dataset-btn', className = 'chooser-button'), style = {'width': '5cm'})
        ]),
        html.Tr([
            html.Td('Sheet Name'),
            html.Td(dcc.Dropdown(options = [], value = None, id = 'sp700s-model-dataset-sheet'))
        ]),
        html.Tr(html.Td(html.Hr(), colSpan = 2)),
        html.Tr([
            html.Td(html.Div('Select Exclusion File')),
            html.Td(dcc.Upload('Choose File', id = 'sp700s-model-exclude-btn', className = 'chooser-button'))
        ]),
        html.Tr([
            html.Td('Sheet Name'),
            html.Td(dcc.Dropdown(options = [], value = None, id = 'sp700s-model-exclude-sheet'))
        ]),
        html.Tr(html.Td(html.Hr(), colSpan = 2)),
        html.Tr([
            html.Td('Product Code'),
            html.Td(dcc.Dropdown(options = product_codes, value = None, id = 'sp700s-model-product-code'))
        ]),
        html.Tr([
            html.Td('Mode'),
            html.Td(dcc.Dropdown(options = qc_modes, value = 'ALL_QC', id = 'sp700s-qc-mode'))
        ], style = {'display': 'none'}),
        html.Tr(html.Td(html.Hr(), colSpan = 2)),
        html.Tr(html.Td(html.Button('Create Models', id = 'sp700s-model-evaluate-btn', style = {'width': '100%'}), colSpan = 2))
    ])

    return model_table

def create_storage():
    return html.Div([
        html.Div(json.dumps({}), id = 'sp700s-model-dataset-storage'),
        html.Div(json.dumps({}), id = 'sp700s-model-exclude-storage'),
        dcc.Input(type = 'text', id = 'sp700s-model-created', value = ''),
    ])

def create_callbacks(app):
    util.create_excel_upload_callback(app, 'sp700s-model-dataset-btn', 'sp700s-model-dataset-sheet', 'sp700s-model-dataset-storage', ['sp700s-attributes', 'modelling', 'dataset'])
    util.create_excel_upload_callback(app, 'sp700s-model-exclude-btn', 'sp700s-model-exclude-sheet', 'sp700s-model-exclude-storage', ['sp700s-attributes', 'modelling', 'exclude'])

    @app.callback(
        Output('sp700s-model-evaluate-btn', 'disabled'),
        [Input('sp700s-model-dataset-sheet', 'value'),
         Input('sp700s-model-exclude-sheet', 'value'),
         Input('sp700s-model-product-code', 'value'),
         Input('sp700s-qc-mode', 'value')]
    )
    def disable_model_run_btn(value1, value2, value3, value4):
        return value1 == None or value2 == None or value3 == None or value4 == None

    @app.callback(
        Output('sp700s-model-product-code', 'value'),
        Input('load-trigger', 'value'),
        State('project-description', 'children')
    )
    def load_code(value, desc):
        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sp700s':
                desc = json.loads(desc)
                return desc['sp700s-attributes']['modelling']['product_code']
        return None

    @app.callback(
        Output('sp700s-model-created', 'value'),
        [Input('sp700s-model-evaluate-btn', 'n_clicks'),
         Input('load-trigger', 'value')],
        [State('sp700s-model-dataset-storage', 'children'),
         State('sp700s-model-dataset-sheet', 'value'),
         State('sp700s-model-exclude-storage', 'children'),
         State('sp700s-model-exclude-sheet', 'value'),
         State('sp700s-model-product-code', 'value'),
         State('sp700s-model-created', 'value'),
         State('project-description', 'children')]
    )
    def build_models(clicks, load_trigger, dataset, sheet, exclude, exclude_sheet, product, created, desc):
        if util.get_trigger() == 'sp700s-model-evaluate-btn':
            directory = util.desc_to_directory(json.loads(desc))
            all_data = util.decode_from_text(dataset)
            all_exclude = util.decode_from_text(exclude)
            dataset = all_data[sheet]
            exclude = all_exclude[exclude_sheet]

            dataset = modelling.prepare_main_dataset(dataset)
            exclude = modelling.prepare_exclude_dataset(exclude)

            cleaned, column = modelling.clean_dataset(product, dataset, exclude)
            modelling.build_models(cleaned, column, directory)

            # save input data to disc for possibly loading it later
            desc = json.loads(desc)
            data_path = os.path.join(util.get_program_folder(), desc['folder'], desc['sp700s-attributes']['modelling']['dataset']['path'])
            exclude_path = os.path.join(util.get_program_folder(), desc['folder'], desc['sp700s-attributes']['modelling']['exclude']['path'])
            pickle.dump(all_data, open(data_path, 'wb'))
            pickle.dump(all_exclude, open(exclude_path, 'wb'))

            created = '{}'.format(uuid.uuid4())
            return created
        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sp700s':
                created = json.loads(desc)['sp700s-attributes']['modelling']['id']
                return created

        return dash.no_update
