import dash_bootstrap_components as dbc
from dash import html


def get_marker_buttons_card(marker_list, color_of_cards):
    return dbc.Card([
        html.Div([
            html.H2("Marker List", className="text-primary"),
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id='marker-list',
                        children=marker_list,
                        style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}
                    )
                ])
            ])
        ], style={
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'height': '100vh',
            'overflow-y': 'auto',  # Enables vertical scrolling if the content overflows
            'z-index': 100,  # To keep it above other elements
            'backgroundColor': color_of_cards
        }),
    ], className="mb-4 mt-4")
