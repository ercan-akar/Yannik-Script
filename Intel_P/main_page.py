import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import os
import json
import uuid
import util
import re
import time

def create_layout():
    project_selection_table = html.Table([
        html.Tr(html.Td('Load project')),
        html.Tr(html.Td(dcc.RadioItems(options = [], value = None, id = 'existing-projects', labelStyle = {'display': 'block'}))),
        html.Tr(html.Td(html.Button('Load Selected', id = 'project-load-btn'))),
        html.Tr(html.Td('or create new project with name:')),
        html.Tr(html.Td(dcc.Input(type = 'text', placeholder = 'Project Name', id = 'project-name'))),
        html.Tr(html.Td(html.Button('SP700s Project', className = 'project-button', id = 'project-sp700s-btn'))),
        html.Tr(html.Td(html.Button('Sa-02 Project', className = 'project-button', id = 'project-tsmc-btn'))),

        ## dummy, remove later
        html.Tr([html.Td(html.Button(id = 'sa02-feature-evaluate-btn'))], style = {'display': 'none'})

    ], id = 'project-selection-screen')

    return project_selection_table

def create_storage():
    return html.Div([
        html.Div(json.dumps({}), id='project-description'),
        dcc.RadioItems(options = [{'label': 'loaded', 'value': 'loaded'}], value = None, id='load-trigger'), # well this is interesting. apparently, I cannot put an event on the children of an element. but I can put an event on anything that is supposed to be an input.
        dcc.RadioItems(options = [{'label': 'load-desired', 'value': 'load-desired'}], value = None, id = 'load-desired'),

    ])

def create_callbacks(app):
    @app.callback(
        Output('existing-projects', 'options'),
        Input('project-load-btn', 'children') # this is just to have an input. we won't use it
    )
    def list_projects(clicks):
        projects = []
        for folder in os.listdir(util.get_program_folder()):
            project_description = os.path.join(util.get_program_folder(), folder, '.project_desc.json')
            if os.path.exists(project_description):
                desc = json.load(open(project_description))
                if not desc['removed']:
                    projects.append({'label': desc['type'] + ' | ' + desc['name'], 'value': project_description})

        return projects

    @app.callback(
        Output('load-trigger', 'value'),
        Input('project-load-btn', 'n_clicks')
    )
    def send_trigger(clicks):
        if util.get_trigger() == 'project-load-btn':
            # this is stupid, like, really stupid. But currently I cannot see any other way
            # I cannot fire the trigger with the description update, because that results in circular dependencies
            # so I have to fire it simultaneously with the loading description update.
            # to 'ensure' that the description is already loaded, I just have to delay this one. It's bad design, but I am limited by dash (or my imagination) here
            time.sleep(5)
            return 'loaded'
        return dash.no_update

    @app.callback(
        Output('project-selection-screen', 'style'),
        [Input('project-sp700s-btn', 'n_clicks'),
         Input('project-tsmc-btn', 'n_clicks'),
         Input('project-load-btn', 'n_clicks'),]
    )
    def hide_project_page(clicks1, clicks2, clicks3):
        trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'project-sp700s-btn' or trigger == 'project-tsmc-btn' or trigger == 'project-load-btn':
            return {'display': 'none'}
        return {}

    @app.callback(
        Output('project-load-btn', 'disabled'),
        Input('existing-projects', 'value')
    )
    def disable_load(value):
        if util.get_trigger() == 'existing-projects':
            return value == None
        return True

    @app.callback(
        [Output('project-tsmc-btn', 'disabled'),
         Output('project-sp700s-btn', 'disabled')],
        Input('project-name', 'value')
    )
    def disable_create_btns(value):
        if util.get_trigger() == 'project-name':
            return [len(value) == 0,len(value) == 0]
        return [True, True]

    @app.callback(
        [Output('project-description', 'children'),
         Output('load-desired', 'value')],
        # loading or creating a project
        [Input('project-load-btn', 'n_clicks'),
         Input('project-sp700s-btn', 'n_clicks'),
         Input('project-tsmc-btn', 'n_clicks'),
         # sp700s
         Input('sp700s-model-created', 'value'),
         Input('sp700s-prediction-created', 'value'),
         Input('sp700s-feature-evaluate-btn', 'n_clicks'),
         # sa02
         Input('sa02-model-created', 'value'),
         Input('sa02-prediction-created', 'value'),
         Input('sa02-feature-evaluate-btn', 'n_clicks')],
        # configurations, sp700s first
        [State('sp700s-model-dataset-sheet', 'value'),
         State('sp700s-model-exclude-sheet', 'value'),
         State('sp700s-model-product-code', 'value'),
         State('sp700s-predict-dataset-sheet', 'value'),
         State('sp700s-predict-batches', 'value'),
         State('sp700s-predict-include-outliers', 'value'),
         # configuration sa02
         State('sa02-model-dataset-sheet', 'value'),
         State('sa02-predict-dataset-sheet', 'value'),
         State('sa02-predict-batches', 'value'),
         State('existing-projects', 'value'),
         State('project-name', 'value'),
         State('project-description', 'children')]
    )
    def update_desc(clicks1, clicks2, clicks3, sp700s_model_id, sp700s_prediction_id, clicks6, sa02_model_id, sa02_prediction_id, clicks9, sp700s_model_data_sheet, sp700s_model_exclude_sheet, sp700s_model_product_code, sp700s_predict_data_sheet, sp700s_predict_batches, sp700s_predict_outliers, sa02_model_data_sheet, sa02_predict_data_sheet, sa02_predict_batches, sel_path, name, current_desc):
        trigger = util.get_trigger()

        # by deault, don't trigger
        load_trigger = dash.no_update


        current_desc = json.loads(current_desc)
        # load the project from the given uuid
        if trigger == 'project-load-btn':
            current_desc = json.load(open(sel_path))
            load_trigger = 'loaded' # the value itself doesn't matter, we just need something to trigger other callbacks

        # create a new sp700s project
        if trigger == 'project-sp700s-btn':
            uuid_new = uuid.uuid4()
            sanitized_name = re.sub('[^a-zA-Z0-9\s]', '', name)
            project_path = os.path.join(util.get_program_folder(), '{}.{}'.format(sanitized_name, uuid_new))
            os.mkdir(project_path)
            desc = {
                'uuid': '{}'.format(uuid_new),
                'name': name,
                'folder': '{}.{}'.format(sanitized_name, uuid_new),
                'removed': False,
                'type': 'sp700s',
                'sp700s-attributes': {
                    'modelling': {
                        'dataset': {
                            'path': os.path.join('modelling', 'data', 'dataset.pkl'),
                            'sheet': None
                        },
                        'exclude': {
                            'path': os.path.join('modelling', 'data', 'exclude.pkl'),
                            'sheet': None
                        },
                        'product_code': None,
                        'id': ''
                    },
                    'prediction': {
                        'dataset': {
                            'path': os.path.join('predictions', 'data', 'dataset.pkl'),
                            'sheet': None
                        },
                        'include_outliers': 1,
                        'batches': [],
                        'id': ''
                    },
                    'feature_importance': {
                        'id': ''
                    }
                }
            }
            current_desc = desc

        # same but different
        if trigger == 'project-tsmc-btn':
            uuid_new = uuid.uuid4()
            sanitized_name = re.sub('[^a-zA-Z0-9\s]', '', name)
            project_path = os.path.join(util.get_program_folder(), '{}.{}'.format(sanitized_name, uuid_new))
            os.mkdir(project_path)
            desc = {
                'uuid': '{}'.format(uuid_new),
                'name': name,
                'folder': '{}.{}'.format(sanitized_name, uuid_new),
                'removed': False,
                'type': 'sa02',
                'sa02-attributes': {
                    'modelling': {
                        'dataset': {
                            'path': os.path.join('modelling', 'data', 'dataset.pkl'),
                            'sheet': None
                        },
                        'id': ''
                    },
                    'prediction': {
                        'dataset': {
                            'path': os.path.join('predictions', 'data', 'dataset.pkl'),
                            'sheet': None
                        },
                        'batches': [],
                        'id': ''
                    },
                    'feature_importance': {
                        'id': ''
                    }
                }
            }

            current_desc = desc


        if trigger == 'sp700s-model-created':
            current_desc['sp700s-attributes']['modelling']['dataset']['sheet'] = sp700s_model_data_sheet
            current_desc['sp700s-attributes']['modelling']['exclude']['sheet'] = sp700s_model_exclude_sheet
            current_desc['sp700s-attributes']['modelling']['product_code'] = sp700s_model_product_code
            current_desc['sp700s-attributes']['modelling']['id'] = sp700s_model_id


        if trigger == 'sp700s-prediction-created':
            current_desc['sp700s-attributes']['prediction']['dataset']['sheet'] = sp700s_predict_data_sheet
            current_desc['sp700s-attributes']['prediction']['batches'] = sp700s_predict_batches
            current_desc['sp700s-attributes']['prediction']['outliers'] = sp700s_predict_outliers
            current_desc['sp700s-attributes']['prediction']['id'] = sp700s_prediction_id

        # ignore for now
        if trigger == 'sp700s-feature-created':
            pass

        if trigger == 'sa02-model-created':
            current_desc['sa02-attributes']['modelling']['dataset']['sheet'] = sa02_model_data_sheet
            current_desc['sa02-attributes']['modelling']['id'] = sa02_model_id

        if trigger == 'sa02-prediction-created':
            current_desc['sa02-attributes']['prediction']['dataset']['sheet'] = sa02_predict_data_sheet
            current_desc['sa02-attributes']['prediction']['batches'] = sa02_predict_batches
            current_desc['sa02-attributes']['prediction']['id'] = sa02_prediction_id

        # ignore for now
        if trigger == 'sa02-feature-created':
            pass

        if 'folder' in current_desc and os.path.exists(os.path.join(util.get_program_folder(), current_desc['folder'])):
            json_file = os.path.join(util.get_program_folder(), current_desc['folder'], '.project_desc.json')
            json.dump(current_desc, open(json_file, 'w'))

        return [json.dumps(current_desc), load_trigger]

    #@app.callback(
        #Output('project-tsmc-btn', 'children'),
        #Input('load-trigger', 'value'),
        #State('project-description', 'children')
    #)
    #def load_triggered(l, desc):
        #print('load triggered')
        #if util.get_trigger() == 'load-trigger':
            #print('loading triggered')
            #print(json.loads(desc))
        #return 'Sa02'
