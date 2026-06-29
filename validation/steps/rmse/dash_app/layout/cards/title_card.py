import dash_bootstrap_components as dbc
from dash import html

def get_title_card(recording_name: str, color_of_cards: str):
    return dbc.Card([
        html.Div([
            html.H2(recording_name, className="text-primary", style={"textAlign": "center"})
        ], style={"backgroundColor": color_of_cards})
    ])