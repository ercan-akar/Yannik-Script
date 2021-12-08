import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import util
import json
import modelling_part_tsmc as modelling

def create_layout():
    predict_table = html.Table([
        html.Tr([
            html.Td('Data File', style = {'width': '6cm'}),
            html.Td(dcc.Upload('Select Data File', className='chooser-button', style = {'width': '5cm'}, id = 'sa02-predict-dataset-btn'))
        ]),
        html.Tr([
            html.Td('Sheet Name'),
            html.Td(dcc.Dropdown(options = [], value = None, id = 'sa02-predict-dataset-sheet'))
        ]),
        util.create_separator(2),
        html.Tr([
            html.Td('Select Batches'),
            html.Td(html.Table([
                html.Tr([
                    html.Td(html.Button('Select all', id = 'sa02-predict-batches-sel-all')),
                    html.Td(html.Button('De-select all', id = 'sa02-predict-batches-sel-none'))
                ]),
                html.Tr(html.Td(dcc.Checklist(options = [], value = [], id = 'sa02-predict-batches', labelStyle = {'display': 'block'}), colSpan = 2))
            ]))
        ]),
        util.create_separator(2),
        html.Tr(
            html.Td(html.Button('Predict', id = 'sa02-predict-evaluate-btn', style = {'width': '100%'}), colSpan = 2)
        )
    ])

    return predict_table

def create_storage():
    return html.Div([
        html.Div(json.dumps({}), id='sa02-predict-data-storage'),
        dcc.Input(id = 'sa02-prediction-created', value = '', type = 'text'),
    ])

def create_callbacks(app):
    util.create_excel_upload_callback(app, 'sa02-predict-dataset-btn', 'sa02-predict-dataset-sheet', 'sa02-predict-data-storage', ['sa02-attributes', 'prediction', 'dataset'])

    @app.callback(
        Output('sa02-predict-evaluate-btn', 'disabled'),
        [Input('sa02-predict-dataset-sheet', 'value'),
         Input('sa02-predict-batches', 'value')]
    )
    def disable_predict_btn(sheet, batches):
        return sheet == None or batches == None or len(batches) == 0

    @app.callback(
        Output('sa02-predict-batches', 'options'),
        Input('sa02-predict-dataset-sheet', 'value'),
        [State('sa02-predict-data-storage', 'children'),
         State('sa02-predict-batches', 'options')]
    )
    def fill_batch_list(sheet, dataset, current_options):
        if sheet is not None:
            if util.get_trigger() == 'sa02-predict-dataset-sheet':
                dataset = util.decode_from_text(dataset)[sheet]
                batches = dataset.loc[:, 'Parameter'].unique().tolist()
                options = [{'label': batch, 'value': batch} for batch in batches]
                return options
        return []

    @app.callback(
        Output('sa02-predict-batches', 'value'),
        [Input('sa02-predict-dataset-sheet', 'value'),
         Input('sa02-predict-batches-sel-all', 'n_clicks'),
         Input('sa02-predict-batches-sel-none', 'n_clicks'),
         Input('load-trigger', 'value')],
        [State('sa02-predict-batches', 'options'),
         State('project-description', 'children')]
    )
    def batch_selection(sheet, clicks1, clicks2, load_trigger, current_options, desc):
        if util.get_trigger() == 'sa02-predict-dataset-sheet':
            return []
        if util.get_trigger() == 'sa02-predict-batches-sel-all':
            return [elem['value'] for elem in current_options]
        if util.get_trigger() == 'sa02-predict-batches-sel-none':
            return []
        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sa02':
                return json.loads(desc)['sp700s-attributes']['prediction']['batches']
        return []

    @app.callback(
        Output('sa02-prediction-created', 'value'),
        [Input('sa02-predict-evaluate-btn', 'n_clicks'),
         Input('sa02-model-created', 'value'),
         Input('load-trigger', 'value')],
        [State('sa02-predict-data-storage', 'children'),
         State('sa02-predict-dataset-sheet', 'value'),
         State('sa02-predict-batches', 'value'),
         State('sa02-prediction-created', 'value'),
         State('project-description', 'children')]
    )
    def evaluate(clicks, model_created, load_trigger, data, sheet, batches, created, desc):
        if util.get_trigger() == 'sa02-predict-evaluate-btn':
            path = util.desc_to_directory(json.loads(desc))
            all_data = util.decode_from_text(data)
            data = all_data[sheet]

            ## run the prediction here

            ##

            # save input data to disc for possibly loading it later
            desc = json.loads(desc)
            data_path = os.path.join(util.get_program_folder(), desc['folder'], desc['sa02-attributes']['prediction']['dataset']['path'])
            pickle.dump(all_data, open(data_path, 'wb'))

            created = model_created
            return created

        if util.get_trigger() == 'sa02-model-created':
            if created != model_created:
                created = ''
            return created

        if util.get_trigger() == 'load-trigger':
            if json.loads(desc)['type'] == 'sa02':
                created = json.loads(desc)['sa02-attributes']['prediction']['id']
                return created

        return dash.no_update
