import plotly.express as px
from dash import dcc

def create_dash_trajectory_plots(marker, dataframe_of_3d_data,color_of_cards):
    """
    Converts the joint trajectory figures into Dash components.
    """
    fig_x, fig_y, fig_z = create_joint_trajectory_plots(marker, dataframe_of_3d_data,color_of_cards)
    return [dcc.Graph(figure=fig_x), dcc.Graph(figure=fig_y), dcc.Graph(figure=fig_z)]


def create_joint_trajectory_plots(marker, dataframe_of_3d_data, color_of_cards):
    df_marker = dataframe_of_3d_data[dataframe_of_3d_data.keypoint == marker]
    
    trajectory_plot_height = 350
    # Your plotting code here. For demonstration, using placeholders.
    fig_x = px.line(df_marker, x='frame', y='x', color='system', color_discrete_map={'freemocap': 'blue', 'qualisys': 'red'}, render_mode='svg')
    fig_y = px.line(df_marker, x='frame', y='y', color='system', color_discrete_map={'freemocap': 'blue', 'qualisys': 'red'}, render_mode='svg')
    fig_z = px.line(df_marker, x='frame', y='z', color='system', color_discrete_map={'freemocap': 'blue', 'qualisys': 'red'}, render_mode='svg')

    fig_x.update_xaxes(title_text='', showticklabels=False)
    fig_x.update_yaxes(title_text='X Position (mm)', title_font=dict(size=18))
    fig_x.update_layout(paper_bgcolor=color_of_cards)

    fig_y.update_xaxes(title_text='', showticklabels=False)
    fig_y.update_yaxes(title_text='Y Position (mm)', title_font=dict(size=18))
    fig_y.update_layout(paper_bgcolor=color_of_cards, height=trajectory_plot_height)

    fig_z.update_xaxes(title_text='Frame', title_font=dict(size=18))
    fig_z.update_yaxes(title_text='Z Position (mm)', title_font=dict(size=18))
    fig_z.update_layout(paper_bgcolor=color_of_cards, height=trajectory_plot_height)

    # Return the list of Plotly figures
    return fig_x, fig_y, fig_z

