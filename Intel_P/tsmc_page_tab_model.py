import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import util
import json
import modelling_part_tsmc as modelling
import uuid

def create_layout():

    model_table = html.Table([
        html.Tr([
            html.Td('Data File', style = {'width': '6cm'}),
            html.Td(dcc.Upload('Select Data File', className='chooser-button', style = {'width': '5cm'}, id = 'sa02-model-dataset-btn'))
        ]),
        html.Tr([
            html.Td('Sheet Name'),
            html.Td(dcc.Dropdown(options = [], value = None, id = 'sa02-model-dataset-sheet'))
        ]),
        util.create_separator(2),
        #html.Tr([
            #html.Td('Start X column'),
            #html.Td(dcc.Input(value = 5, type = 'number', id = 'sa02-x-start'))
        #]),
        html.Tr([
            html.Td('Label'),
            html.Td(dcc.Input(value = 'co_sel_ecd', type = 'text', id = 'sa02-model-label'))
        ]),
        util.create_separator(2),
        html.Tr(
            html.Td(html.Button('Create Models', id = 'sa02-model-evaluate-btn', style = {'width': '100%'}), colSpan = 2)
        )
    ])

    return model_table

def create_storage():
    return html.Div([
        html.Div(json.dumps({}), id='sa02-model-dataset-storage'),
        dcc.Input(id = 'sa02-model-created', type = 'text', value = ''),
    ])

def create_callbacks(app):
    util.create_excel_upload_callback(app, 'sa02-model-dataset-btn', 'sa02-model-dataset-sheet', 'sa02-model-dataset-storage', ['sa02-attributes', 'modelling', 'dataset'])

    @app.callback(
        Output('sa02-model-evaluate-btn', 'disabled'),
        [Input('sa02-model-dataset-sheet', 'value'),
         #Input('sa02-x-start', 'value'),
         Input('sa02-model-label', 'value')]
    )
    def disable_evluate_btn(sheet, label):
        trigger = util.get_trigger()
        if trigger == 'sa02-model-dataset-sheet' or trigger == 'sa02-x-start' or trigger == 'sa02-label':
            return sheet == None or len(label) == 0
        return True

    @app.callback(
        Output('sa02-model-created', 'value'),
        [Input('sa02-model-evaluate-btn', 'n_clicks'),
         Input('load-trigger', 'value')],
        [State('sa02-model-dataset-storage', 'children'),
         State('sa02-model-dataset-sheet', 'value'),
         #State('sa02-x-start', 'value'),
         State('sa02-model-label', 'value'),
         State('sa02-model-created', 'value'),
         State('project-description', 'children')]
    )
    def create_models(clicks, load_trigger, dataset, sheet, label, created, desc):
        if util.get_trigger() == 'sa02-model-evaluate-btn':
            path = util.desc_to_directory(json.loads(desc))
            all_data = util.decode_from_text(dataset)
            dataset = all_data[sheet]
            modelling.build_models(dataset, label, path)

            # save input data to disc for possibly loading it later
            desc = json.loads(desc)
            data_path = os.path.join(util.get_program_folder(), desc['folder'], desc['sa02-attributes']['modelling']['dataset']['path'])
            pickle.dump(all_data, open(data_path, 'wb'))

            created = '{}'.format(uuid.uuid4())
            return created

        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sa02':
                created = json.loads(desc)['sa02-attributes']['modelling']['id']
                return created

        return dash.no_update
