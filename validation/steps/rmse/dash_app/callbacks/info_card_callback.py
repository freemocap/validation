from dash import Output, Input, Dash
from validation.steps.rmse.dash_app.ui_components.dashboard import update_joint_marker_card

def register_info_card_callback(app: Dash, rmse_dataframe, velocity_rmse_dataframe):
    @app.callback(
        [Output('info-x-position-rmse', 'children'),
        Output('info-y-position-rmse', 'children'),
        Output('info-z-position-rmse', 'children'),
        Output('info-x-velocity-rmse', 'children'),
        Output('info-y-velocity-rmse', 'children'),
        Output('info-z-velocity-rmse', 'children')],
        [Input('store-selected-marker', 'data')]
    )
    def update_info_card(stored_data):
        marker = stored_data['marker'] if stored_data else None
        x_rmse, y_rmse, z_rmse = update_joint_marker_card(marker, rmse_dataframe)
        x_velocity_rmse, y_velocity_rmse, z_velocity_rmse = update_joint_marker_card(marker, velocity_rmse_dataframe)
        return x_rmse, y_rmse, z_rmse, x_velocity_rmse, y_velocity_rmse,z_velocity_rmse
    
    # @app.callback(
    #     [Output('info-x-velocity-rmse', 'children'),
    #     Output('info-y-velocity-rmse', 'children'),
    #     Output('info-z-velocity-rmse', 'children')],
    #     [Input('store-selected-marker', 'data')]
    # )
    # def update_info_velocity_card(stored_data):
    #     marker = stored_data['marker'] if stored_data else None
    #     x_rmse, y_rmse, z_rmse = update_joint_marker_card(marker, velocity_rmse_dataframe)
    #     return x_rmse, y_rmse, z_rmse