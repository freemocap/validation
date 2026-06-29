
from dash import html
from .create_card import create_card

def get_marker_velocity_card(color_of_cards):
    content = [
        html.H3(
            id='selected-marker-velocity',
            children="Select a marker",
            className="text-info"
        ),
        html.Div(id='velocity-plots')
    ]
    return create_card("Marker Velocity", content, color_of_cards)