
from dash import html
from .create_card import create_card

def get_error_shading_plot_card(color_of_cards):
    content = [
        html.H3(
            id='selected-marker-shading-error',
            children="Select a marker",
            className="text-info"
        ),
        html.Div(id='error-shading-plots')
    ]
    return create_card("Marker Trajectory with Absolute Error Shading", content, color_of_cards)