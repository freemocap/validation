from dash import html
from .create_card import create_card

def get_absolute_error_plots_card(color_of_cards):
    content = [
        html.H3(
            id='selected-marker-absolute-error',
            children="Select a marker for error analysis",
            className="text-info"
        ),
        html.Div(id='error-plots')
    ]
    return create_card("Absolute Error Per Frame", content, color_of_cards)