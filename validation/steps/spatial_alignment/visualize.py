import plotly.graph_objects as go
import numpy as np
from skellymodels.managers.human import Human
def visualize_spatial_alignment(freemocap_actor:Human, qualisys_actor:Human, aligned_freemocap_array: np.ndarray):
    freemocap_3d = freemocap_actor.body.xyz.as_array
    qualisys_3d = qualisys_actor.body.xyz.as_array
    aligned_3d = aligned_freemocap_array

    num_frames = min(freemocap_3d.shape[0], aligned_3d.shape[0], qualisys_3d.shape[0])

    # Stack all data and compute global center + max span
    all_data = np.concatenate([freemocap_3d, aligned_3d, qualisys_3d], axis=1)  # shape: (frames, points, 3)
    all_xyz = all_data.reshape(-1, 3)  # shape: (frames * points, 3)

    center = np.nanmean(all_xyz, axis=0)
    span = np.nanmax(all_xyz, axis=0) - np.nanmin(all_xyz, axis=0)
    max_span = np.max(span) * 0.6  # padding factor

    x_range = [center[0] - max_span, center[0] + max_span]
    y_range = [center[1] - max_span, center[1] + max_span]
    z_range = [center[2] - max_span, center[2] + max_span]

    # Create all frames
    frames = [
        go.Frame(
            name=f'frame{k}',
            data=[
                go.Scatter3d(
                    x=freemocap_3d[k, :, 0], y=freemocap_3d[k, :, 1], z=freemocap_3d[k, :, 2],
                    mode='markers', marker=dict(color='blue', size=4), name='Original FreeMoCap'
                ),
                go.Scatter3d(
                    x=aligned_3d[k, :, 0], y=aligned_3d[k, :, 1], z=aligned_3d[k, :, 2],
                    mode='markers', marker=dict(color='green', size=4), name='Aligned FreeMoCap'
                ),
                go.Scatter3d(
                    x=qualisys_3d[k, :, 0], y=qualisys_3d[k, :, 1], z=qualisys_3d[k, :, 2],
                    mode='markers', marker=dict(color='red', size=4), name='Qualisys'
                ),
            ]
        )
        for k in range(num_frames)
    ]

    # Initialize with first frame data
    initial_data = frames[0].data

    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title='Spatial Alignment',
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
            ),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                }, {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                }]
            }],
            sliders=[{
                'steps': [
                    {'args': [[f'frame{k}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                     'label': str(k), 'method': 'animate'}
                    for k in range(num_frames)
                ],
                'transition': {'duration': 0},
                'x': 0, 'y': 0, 'len': 1.0
            }]
        ),
        frames=frames
    )

    fig.show()
