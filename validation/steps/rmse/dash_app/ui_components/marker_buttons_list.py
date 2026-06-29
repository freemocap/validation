from dash import html

def create_marker_buttons(marker_position_df):
    unique_markers = marker_position_df['keypoint'].unique()
    marker_list = []
    for idx, marker in enumerate(unique_markers):
        marker_list.append(
            html.Button(
                marker, 
                id={'type': 'marker-button', 'index': marker}, 
                className='btn btn-dark', 
                style={'margin': '5px', 'width': '140px', 'height': '40px', 'padding': '2px', 'word-wrap': 'break-word'}
            )
        )
        if (idx + 1) % 2 == 0:
            marker_list.append(html.Br())
    return marker_list
