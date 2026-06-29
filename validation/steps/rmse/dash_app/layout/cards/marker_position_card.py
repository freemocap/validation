
from dash import html
from .create_card import create_card

def get_marker_position_card(color_of_cards):
    content = [
        html.H3(
            id='selected-marker-trajectory',
            children="Select a marker",
            className="text-info"
        ),
        html.Div(id='trajectory-plots')
    ]
    return create_card("Marker Trajectory", content, color_of_cards)