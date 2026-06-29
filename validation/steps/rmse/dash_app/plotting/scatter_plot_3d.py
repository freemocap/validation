import plotly.express as px

def create_3d_scatter_from_dataframe(dataframe_of_3d_data, ax_range=1200):
    x_mean = dataframe_of_3d_data["x"].mean()
    y_mean = dataframe_of_3d_data["y"].mean()
    z_mean = dataframe_of_3d_data["z"].mean()

    fig = px.scatter_3d(dataframe_of_3d_data, x="x", y="y", z="z", 
                        animation_frame="frame",
                        animation_group="keypoint",
                        color="system",
                        color_discrete_map={'freemocap': 'blue', 'qualisys': 'red'},
                        hover_name="keypoint",
                        hover_data=["x", "y", "z"],
                        range_x=[x_mean - ax_range, x_mean + ax_range], 
                        range_y=[y_mean - ax_range, y_mean + ax_range],
                        range_z=[z_mean - ax_range, z_mean + ax_range]
    )
    fig.update_layout(
        scene_aspectmode='cube',
        autosize=False,
        height=1000,
        width=1100,
        margin={"t": 1, "b": 1, "l": 1, "r": 1},
    )

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000 / 15

    return fig