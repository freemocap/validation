from .create_card import create_card
from dash import dcc

def get_joint_rmse_plot_card(joint_rmse_figure, title, color_of_cards):
    content = [dcc.Graph(id=f'joint-rmse-figure-{title}', figure=joint_rmse_figure)]
    return create_card(f"Joint RMSE - {title}", content, color_of_cards)