from dash import Dash

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

from validation.steps.rmse.dash_app.data_utils.load_data import combine_freemocap_and_qualisys_into_dataframe
from validation.steps.rmse.dash_app.data_utils.file_manager import FileManager

from validation.steps.rmse.dash_app.ui_components.dashboard import prepare_dashboard_elements

from validation.steps.rmse.dash_app.layout.main_layout import get_layout

from validation.steps.rmse.dash_app.callbacks.marker_name_callbacks import register_marker_name_callbacks
from validation.steps.rmse.dash_app.callbacks.selected_marker_callback import register_selected_marker_callback
from validation.steps.rmse.dash_app.callbacks.info_card_callback import register_info_card_callback
from validation.steps.rmse.dash_app.callbacks.plot_update_callback import register_plot_update_callback
from validation.steps.rmse.dash_app.callbacks.marker_button_color_callback import register_marker_button_color_callback
from validation.steps.rmse.dash_app.callbacks.report_download_callback import register_report_download_callback


COLOR_OF_CARDS = '#F3F5F7'
FRAME_SKIP_INTERVAL = 5

from validation.steps.rmse.core.calculate_rmse import RMSEResults

def run_dash_app(data_and_error:RMSEResults, recording_name:str):
    # Initialize Dash App
    app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
    register_selected_marker_callback(app) #register a callback to find the selected marker and stored it
    register_marker_name_callbacks(app) #register a callback to update the marker name wherever it is listed in the app
    register_info_card_callback(app, data_and_error.position_rmse, data_and_error.velocity_rmse)
    register_plot_update_callback(app, data_and_error, COLOR_OF_CARDS)
    register_marker_button_color_callback(app)
    register_report_download_callback(app, data_and_error.position_joint_df, 
                                      data_and_error.position_rmse, 
                                      data_and_error.velocity_joint_df, 
                                      data_and_error.velocity_rmse, 
                                      recording_name=recording_name)

    load_figure_template('LUX')

    # Create Figures and Components
    scatter_3d_figure, indicators, marker_buttons_list, joint_rmse_plot = prepare_dashboard_elements(
        data_and_error, FRAME_SKIP_INTERVAL, COLOR_OF_CARDS)

    app.layout = get_layout(recording_name=recording_name, marker_figure=scatter_3d_figure,
                            joint_rmse_figure=joint_rmse_plot,
                            list_of_marker_buttons=marker_buttons_list,
                            indicators=indicators,
                            color_of_cards=COLOR_OF_CARDS)

    app.run(debug=False)
