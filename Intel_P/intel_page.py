import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import json
import os
import base64
import pandas as pd
import util

import intel_page_tab_model
import intel_page_tab_predict
import intel_page_tab_feature_importance
import modelling_part_intel as modelling

def create_layout():
    return html.Table([
        html.Tr(html.Td(html.H5('SP700s Project', style = {'text-align': 'center'}), colSpan = 3)),
        html.Tr(html.Td(html.Hr(), colSpan = 3)),
        html.Tr([
            html.Td(html.Button('Modelling', id='sp700s-tab-model', className = 'tab-button'), style = {'width': '33%'}),
            html.Td(html.Button('Prediction', id='sp700s-tab-predict', className = 'tab-button'), style = {'width': '33%'}),
            html.Td(html.Button('Feature Importance', id='sp700s-tab-feature', className = 'tab-button'), style = {'width': '33%'})
        ]),
        html.Tr(html.Td(intel_page_tab_model.create_layout(), colSpan = 3), id = 'sp700s-page-modelling'),
        html.Tr(html.Td(intel_page_tab_predict.create_layout(), colSpan = 3), id = 'sp700s-page-predict'),
        html.Tr(html.Td(intel_page_tab_feature_importance.create_layout(), colSpan = 3), id = 'sp700s-page-feature')

    ], id = 'sp700s-page')

def create_storage():
    return html.Div([
        intel_page_tab_model.create_storage(),
        intel_page_tab_predict.create_storage(),
        intel_page_tab_feature_importance.create_storage(),
    ])

def create_callbacks(app):
    intel_page_tab_model.create_callbacks(app)
    intel_page_tab_predict.create_callbacks(app)
    intel_page_tab_feature_importance.create_callbacks(app),
    @app.callback(
        Output('sp700s-page', 'style'),
        [Input('project-sp700s-btn', 'n_clicks'),
         Input('project-description', 'children')]
    )
    def show_sp700s_project(clicks1, desc):
        trigger = util.get_trigger()
        if trigger == 'project-sp700s-btn':
            return {}
        if trigger == 'project-description':
            desc = json.loads(desc)
            if 'type' in desc and desc['type'] == 'sp700s':
                return {}
        return {'display': 'none'}

    @app.callback(
        [Output('sp700s-page-modelling', 'style'),
         Output('sp700s-page-predict', 'style'),
         Output('sp700s-page-feature', 'style')],
        [Input('sp700s-tab-model', 'n_clicks'),
         Input('sp700s-tab-predict', 'n_clicks'),
         Input('sp700s-tab-feature', 'n_clicks')]
    )
    def change_tab(clicks1, clicks2, clicks3):
        trigger = util.get_trigger()
        hidden = {'display': 'none'}
        visibilities = [hidden, hidden, hidden]
        visible_idx = 0
        if trigger == 'sp700s-tab-model':
            visible_idx = 0
        if trigger == 'sp700s-tab-predict':
            visible_idx = 1
        if trigger == 'sp700s-tab-feature':
            visible_idx = 2
        visibilities[visible_idx] = {}
        return visibilities

    @app.callback(
        [Output('sp700s-tab-model', 'disabled'),
         Output('sp700s-tab-predict', 'disabled'),
         Output('sp700s-tab-feature', 'disabled')],
        Input('sp700s-model-created', 'value')
    )
    def disable_tabs(model_uuid):
        if util.get_trigger() == 'sp700s-model-created':
            return [False, model_uuid == '', model_uuid == '']
        return [False, True, True]

    @app.callback(
        [Output('sp700s-tab-model', 'className'),
         Output('sp700s-tab-predict', 'className'),
         Output('sp700s-tab-feature', 'className')],
        [Input('sp700s-tab-model', 'n_clicks'),
         Input('sp700s-tab-predict', 'n_clicks'),
         Input('sp700s-tab-feature', 'n_clicks')]
    )
    def activate_tab_btn(clicks1, clicks2, clicks3):
        trigger = util.get_trigger()
        classes = ['tab-button', 'tab-button', 'tab-button']
        selected = 'active-tab-button'
        idx = 0
        if trigger == 'sp700s-tab-model':
            idx = 0
        if trigger == 'sp700s-tab-predict':
            idx = 1
        if trigger == 'sp700s-tab-feature':
            idx = 2
        classes[idx] = selected
        return classes
