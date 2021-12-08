import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import util
import json
import modelling_part_tsmc as modelling

import tsmc_page_tab_model
import tsmc_page_tab_predict

def create_layout():
    return html.Table([
        html.Tr(html.Td(html.H5('Sa-02 Project', style = {'text-align': 'center'}), colSpan = 3)),
        util.create_separator(3),
        html.Tr([
            html.Td(html.Button('Modelling', id='sa02-tab-model', className = 'tab-button'), style = {'width': '33%'}),
            html.Td(html.Button('Prediction', id='sa02-tab-predict', className = 'tab-button'), style = {'width': '33%'}),
            html.Td(html.Button('Feature Importance', id='sa02-tab-permutation', className = 'tab-button'), style = {'width': '33%'})
        ]),
        html.Tr(html.Td(tsmc_page_tab_model.create_layout(), colSpan = 3), id = 'sa02-page-modelling'),
        html.Tr(html.Td(tsmc_page_tab_predict.create_layout(), colSpan = 3), id = 'sa02-page-predict'),
    ], id = 'sa02-page')

def create_storage():
    return html.Div([
        tsmc_page_tab_model.create_storage(),
        tsmc_page_tab_predict.create_storage()
    ])

def create_callbacks(app):
    tsmc_page_tab_model.create_callbacks(app)
    tsmc_page_tab_predict.create_callbacks(app)
    @app.callback(
        Output('sa02-page', 'style'),
        [Input('project-tsmc-btn', 'n_clicks'),
         Input('project-description', 'children')]
    )
    def show_tsmc_page(clicks, desc):
        if util.get_trigger() == 'project-tsmc-btn':
            return {}
        if util.get_trigger() == 'project-description':
            desc = json.loads(desc)
            if 'type' in desc and desc['type'] == 'sa02':
                return {}
        return {'display': 'none'}

    @app.callback(
        [Output('sa02-page-modelling', 'style'),
         Output('sa02-page-predict', 'style')],
        [Input('sa02-tab-model', 'n_clicks'),
         Input('sa02-tab-predict', 'n_clicks'),
         Input('sa02-tab-permutation', 'n_clicks')]
    )
    def change_tab(clicks1, clicks2, clicks3):
        trigger = util.get_trigger()
        hidden = {'display': 'none'}
        visibilities = [hidden, hidden]
        visible_idx = 0
        if trigger == 'sa02-tab-model':
            visible_idx = 0
        if trigger == 'sa02-tab-predict':
            visible_idx = 1
        if trigger == 'sa02-tab-permutation':
            visible_idx = 2
        visibilities[visible_idx] = {}
        return visibilities

    @app.callback(
        [Output('sa02-tab-model', 'className'),
         Output('sa02-tab-predict', 'className'),
         Output('sa02-tab-permutation', 'className')],
        [Input('sa02-tab-model', 'n_clicks'),
         Input('sa02-tab-predict', 'n_clicks'),
         Input('sa02-tab-permutation', 'n_clicks')]
    )
    def activate_tab_btn(clicks1, clicks2, clicks3):
        trigger = util.get_trigger()
        classes = ['tab-button', 'tab-button', 'tab-button']
        selected = 'active-tab-button'
        idx = 0
        if trigger == 'sa02-tab-model':
            idx = 0
        if trigger == 'sa02-tab-predict':
            idx = 1
        if trigger == 'sa02-tab-permutation':
            idx = 2
        classes[idx] = selected
        return classes
