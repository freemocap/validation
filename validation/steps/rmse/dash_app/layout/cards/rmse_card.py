from dash import html
from .create_card import create_card

def get_rmse_card(gauges):
    content = [
        html.Div(
            id='overall-rmse-gauge',
            children=[gauges[0]],
            style={'display': 'flex', 'justify-content': 'center'}
        ),
        html.Div(
            id='axis-rmse-gauges',
            children=gauges[1:],
            style={'display': 'flex', 'justify-content': 'space-between'}
        )
    ]
    return create_card("Root Mean Squared Error (RMSE) (mm)", content, None)