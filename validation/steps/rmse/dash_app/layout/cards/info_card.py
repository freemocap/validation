import dash_bootstrap_components as dbc
from dash import html, dcc

def get_info_card(color_of_cards):
    position_card_content = dbc.Card([
        dbc.CardHeader("Selected Marker Trajectory:"),
        dbc.CardBody(
            [
                dbc.Row([dbc.Label(id='selected-marker-info-postion-card', className = 'text-info', style={'font-weight': 'bold'})]),
                dbc.Row([
                    dbc.Label("X RMSE (mm):", className="card-text", style={'color':'darkred','font-weight': 'bold'}),
                    dbc.Label(id='info-x-position-rmse', className="card-text"),
                ]),
                dbc.Row([
                    dbc.Label("Y RMSE (mm):", className="card-text", style={'color':'darkgreen','font-weight': 'bold'}),
                    dbc.Label(id='info-y-position-rmse', className="card-text"),
                ]),
                dbc.Row([
                    dbc.Label("Z RMSE (mm):", className="card-text", style={'color':'darkblue','font-weight': 'bold'}),
                    dbc.Label(id='info-z-position-rmse', className="card-text"),
                ])
            ]
        ),
    ], className="mb-4 mt-4")

    velocity_card_content = dbc.Card([
        dbc.CardHeader("Selected Marker Trajectory:"),
        dbc.CardBody(
            [
                dbc.Row([dbc.Label(id='selected-marker-info-velocity-card', className = 'text-info', style={'font-weight': 'bold'})]),
                dbc.Row([
                    dbc.Label("X RMSE (mm):", className="card-text", style={'color':'darkred','font-weight': 'bold'}),
                    dbc.Label(id='info-x-velocity-rmse', className="card-text"),
                ]),
                dbc.Row([
                    dbc.Label("Y RMSE (mm):", className="card-text", style={'color':'darkgreen','font-weight': 'bold'}),
                    dbc.Label(id='info-y-velocity-rmse', className="card-text"),
                ]),
                dbc.Row([
                    dbc.Label("Z RMSE (mm):", className="card-text", style={'color':'darkblue','font-weight': 'bold'}),
                    dbc.Label(id='info-z-velocity-rmse', className="card-text"),
                ])
            ]
        ),
    ], className="mb-4 mt-4")

    

    save_button = dbc.Button("Save Report", id="save-report-btn", color="primary", className="mb-2")
    download_component = dcc.Download(id="download-report")

    
    return html.Div([
        position_card_content,
        velocity_card_content,
        html.Div(save_button, style={'textAlign': 'center'}),  # Button below the card
        download_component
    ], style={
        'position': 'fixed',
        'top': '0',
        'right': '0',
        'height': '100vh',
        'overflow-y': 'auto',  
        'z-index': 1000,  
        'backgroundColor': color_of_cards
    })

