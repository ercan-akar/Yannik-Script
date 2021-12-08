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
import feature_importance

def create_layout():
    feature_table = html.Table([
        html.Tr(html.Td(html.Button('Calculate feature importance', id = 'sp700s-feature-evaluate-btn', style = {'width': '100%'}), colSpan = 2))
    ])

    return feature_table

def create_storage():
    return html.Div([
        dcc.Input(type = 'text', value = '', id = 'sp700s-feature-created')
    ])

def create_callbacks(app):
    @app.callback(
        Output('sp700s-feature-created', 'value'),
        [Input('sp700s-feature-evaluate-btn', 'n_clicks'),
         Input('sp700s-model-created', 'value'),
         Input('load-trigger', 'value')],
        [State('sp700s-feature-created', 'value'),
         State('project-description', 'children')]
    )
    def evaluate(clicks, model_created, load_trigger, created, desc):
        if util.get_trigger() == 'sp700s-feature-evaluate-btn':
            path = util.desc_to_directory(json.loads(desc))

            feature_importance.compute_feature_importance(path)

            created = model_created
            return created
        if util.get_trigger() == 'sp700s-model-created':
            if created != model_created:
                created = ''
            return created

        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sp700s':
                created = json.loads(desc)['sp700s-attributes']['feature_importance']['id']
                return created

        return dash.no_update
