import main_page
import intel_page
import tsmc_page

from dash import html

def create_layout():
    return html.Div([
        main_page.create_layout(),
        intel_page.create_layout(),
        tsmc_page.create_layout(),
    ], className = 'content-element')

def create_callbacks(app):
    main_page.create_callbacks(app)
    intel_page.create_callbacks(app)
    tsmc_page.create_callbacks(app)

def create_storage():
    return html.Div([
        main_page.create_storage(),
        intel_page.create_storage(),
        tsmc_page.create_storage()
    ], style = {'display': 'none'})
