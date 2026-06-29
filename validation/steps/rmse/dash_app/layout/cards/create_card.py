import dash_bootstrap_components as dbc
from dash import html

def create_card(title, content, color_of_cards):
    return dbc.Card([
        dbc.CardHeader(
            html.H2(title, className="text-primary"),
            className="text-primary"
        ),
        dbc.CardBody(content, style={"backgroundColor": color_of_cards}),
    ], className="mb-4 mt-4")