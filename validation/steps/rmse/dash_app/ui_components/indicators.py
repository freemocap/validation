
from validation.steps.rmse.dash_app.data_utils.extract_rmse_values import extract_overall_rmse_values_from_dataframe
from validation.steps.rmse.dash_app.plotting.indicator_plot import create_indicator
from dash import dcc

def create_indicators_ui(rmse_error_dataframe):
    """Create the indicators for the Dash app layout."""
    rmse_values = extract_overall_rmse_values_from_dataframe(rmse_error_dataframe)
    indicators = create_indicators(rmse_values)
    return indicators


def create_indicators(rmse_values):
    return [
        dcc.Graph(figure=create_indicator(rmse_values['total'], "Total RMSE (mm)"), style={'width': '100%'}),
        dcc.Graph(figure=create_indicator(rmse_values['x'], "X RMSE (mm)", color_of_text='red'), style={'width': '33%'}),
        dcc.Graph(figure=create_indicator(rmse_values['y'], "Y RMSE (mm)", color_of_text='green'), style={'width': '33%'}),
        dcc.Graph(figure=create_indicator(rmse_values['z'], "Z RMSE (mm)", color_of_text='blue'), style={'width': '33%'}),
    ]
