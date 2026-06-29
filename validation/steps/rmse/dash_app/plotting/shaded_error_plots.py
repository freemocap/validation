import plotly.graph_objects as go
from dash import dcc


def find_continuous_segments(frames):
    segments = []
    start = frames[0]

    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            segments.append((start, frames[i - 1]))
            start = frames[i]

    segments.append((start, frames[-1]))

    return segments

def add_error_shapes(frames, max_value, color):
    shapes = []
    for start, end in frames:
        shapes.append(dict(type="rect",
                x0=start - 0.5,
                x1=end + 0.5,
                y0=0,
                y1=max_value,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,))
    return shapes

def create_shaded_error_plots(marker, dataframe_of_3d_data, absolute_error_dataframe, color_of_cards):
    """
    Plot FreeMoCap trajectories for a specific marker with error shading.
    """

    # Pre-filter the DataFrames
    filtered_df = dataframe_of_3d_data.query("keypoint == @marker and system == 'freemocap'")
    filtered_error_df = absolute_error_dataframe.query("keypoint == @marker")

    graphs = []

    for dimension in ['x', 'y', 'z']:
        fig = go.Figure()

        high_error_frames = find_continuous_segments(filtered_error_df.query(f"{dimension}_error > 50")['frame'].tolist())
        low_error_frames = find_continuous_segments(filtered_error_df.query(f"{dimension}_error < 20")['frame'].tolist())

        bad_shapes = add_error_shapes(high_error_frames, filtered_df[dimension].max(), "Red")
        good_shapes = add_error_shapes(low_error_frames, filtered_df[dimension].max(), "Green")

        fig.add_trace(go.Scatter(x=filtered_df['frame'], y=filtered_df[dimension], mode='lines', name=f'{dimension.upper()} trajectory'))

        fig.update_layout(
            title=f'{dimension.upper()} Trajectory for {marker}',
            xaxis_title='Frame',
            yaxis_title='Position',
            xaxis=dict(showline=True, showgrid=False),
            yaxis=dict(showline=True, showgrid=False),
            paper_bgcolor=color_of_cards,
        )
        fig.update_layout(
            shapes=bad_shapes + good_shapes,
        )

        graphs.append(dcc.Graph(figure=fig))

    return graphs