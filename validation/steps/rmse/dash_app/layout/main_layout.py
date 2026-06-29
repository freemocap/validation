import dash_bootstrap_components as dbc
from dash import dcc
from .cards.marker_position_card import get_marker_position_card
from .cards.marker_velocity_card import get_marker_velocity_card
from .cards.marker_buttons_card import get_marker_buttons_card
from .cards.scatter_plot_card import get_scatter_plot_card
from .cards.rmse_card import get_rmse_card
from .cards.joint_rmse_plot_card import get_joint_rmse_plot_card
from .cards.absolute_error_plot_card import get_absolute_error_plots_card
from .cards.error_shading_plot_card import get_error_shading_plot_card
from .cards.info_card import get_info_card
from .cards.title_card import get_title_card

def get_layout(marker_figure, joint_rmse_figure, list_of_marker_buttons, indicators, color_of_cards, recording_name = None):
    sidebar = dbc.Col(
        [
            get_marker_buttons_card(list_of_marker_buttons, color_of_cards)
        ],
        width={"size": 1, "order": "first"},
        id="sidebar",
    )

    main_content_items = []
    
    if recording_name is not None:
        main_content_items.append(dbc.Row(get_title_card(recording_name, color_of_cards)))


    main_content_items.extend([
        dbc.Row([
            dbc.Col(get_scatter_plot_card(marker_figure, color_of_cards), width={"size": 6, "offset": 3})
        ]),
        dbc.Row(get_rmse_card(indicators)),
        dbc.Row([
            get_joint_rmse_plot_card(joint_rmse_figure['x_error'], 'X Dimension', color_of_cards),
            get_joint_rmse_plot_card(joint_rmse_figure['y_error'], 'Y Dimension', color_of_cards),
            get_joint_rmse_plot_card(joint_rmse_figure['z_error'], 'Z Dimension', color_of_cards),
        ]),
        dbc.Row([get_marker_position_card(color_of_cards)]),
        dbc.Row([get_marker_velocity_card(color_of_cards)]),
        dbc.Row([get_absolute_error_plots_card(color_of_cards)]),
        dbc.Row([get_error_shading_plot_card(color_of_cards)]),
    ])

    main_content = dbc.Col(main_content_items, width={"size": 10, "order": "last"})


    info_section = dbc.Col(
        [
            get_info_card(color_of_cards), 
        ],
        width={"size": 1, "order": "last"},
        id="info-section",
    )

    return dbc.Container([
        dcc.Store(id='store-selected-marker'),  # Add the dcc.Store component here to store the marker name
        dbc.Row([sidebar, main_content, info_section]),
    ], fluid=True)

