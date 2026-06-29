import dash
from dash import Output, Input, State, Dash, ALL
import json

def register_selected_marker_callback(app: Dash):
    """
    Register a callback to update the currently selected marker in the Dash app.

    Args:
        app (Dash): The Dash application instance.
    """

    @app.callback(
        Output('store-selected-marker', 'data'),
        [Input('main-graph', 'clickData'),
         Input({'type': 'marker-button', 'index': ALL}, 'n_clicks')],
        [State('store-selected-marker', 'data')]
    )
    def update_stored_marker(clickData, marker_clicks, stored_data):
        """
        Update the selected marker based on user interactions with the main graph or marker buttons.

        Args:
            clickData (dict): Data related to click events on the main graph.
            marker_clicks (list): List of click counts for each marker button.
            stored_data (dict): Data stored in Dash dcc.Store component, containing the currently selected marker.

        Returns:
            dict: Updated selected marker information.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        # Determine which component triggered the callback
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        selected_marker = stored_data.get('marker', None) if stored_data else None
        # Get the updated selected marker based on the trigger
        marker = get_selected_marker(input_id, clickData, selected_marker)
        return {'marker': marker}
    

def get_selected_marker(input_id, clickData, selected_marker):
    """
    Determine the selected marker based on the input trigger.

    Args:
        input_id (str): ID of the input element that triggered the callback.
        clickData (dict): Data related to click events on the main graph.
        selected_marker (str): The currently selected marker.

    Returns:
        str: The updated selected marker.
    """
    # If a marker button was clicked, use its index as the selected marker
    if 'marker-button' in input_id:
        marker = json.loads(input_id)['index']
    # If a marker was clicked in the main graph, use its ID as the selected marker
    elif clickData and 'points' in clickData and clickData['points'] and 'id' in clickData['points'][0]:
        marker = clickData['points'][0]['id']
    # Otherwise, use the currently selected marker
    else:
        marker = selected_marker
    
    return marker