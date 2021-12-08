import dash
from dash import html, dcc
import os
import pages
import util

if not os.path.exists(util.get_program_folder()):
    os.mkdir(util.get_program_folder())

app = dash.Dash('Project')

app.layout = html.Div([
    html.H4('Project (Dev)', style = {'margin-bottom': '2cm', 'text-align': 'center'}),
    pages.create_layout(),
    pages.create_storage()
])

pages.create_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
