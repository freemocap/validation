from .create_card import create_card
from dash import dcc

def get_scatter_plot_card(marker_figure, color_of_cards):
    content = [dcc.Graph(id='main-graph', figure=marker_figure)]
    return create_card("3D Scatter Plot", content, color_of_cards)