from dash import Output, Input, State, ALL
from validation.steps.rmse.dash_app.ui_components.dashboard import update_marker_buttons  

def register_marker_button_color_callback(app):
    @app.callback(
        [Output({'type': 'marker-button', 'index': ALL}, 'className')],
        [Input('store-selected-marker', 'data')],
        [State({'type': 'marker-button', 'index': ALL}, 'id')]
    )
    def change_button_colors(stored_data, button_ids):
        marker = stored_data['marker'] if stored_data else None

        # Update the class names of the marker buttons based on the selected marker
        updated_classnames = update_marker_buttons(marker, button_ids)

        # Return the updated information for the outputs
        return [updated_classnames]