import plotly.express as px
from dash import dcc

def create_absolute_error_plots(marker, absolute_error_dataframe, color_of_cards):
    df_marker = absolute_error_dataframe[absolute_error_dataframe.keypoint == marker]

    trajectory_plot_height = 500
    max_error = max(df_marker.x_error.max(), df_marker.y_error.max(), df_marker.z_error.max())

    # Your plotting code for X, Y, and Z error over frames.
    fig_x = px.line(df_marker, x='frame', y='x_error', title='X Error over Time', render_mode='svg')
    fig_y = px.line(df_marker, x='frame', y='y_error', title='Y Error over Time', render_mode='svg')
    fig_z = px.line(df_marker, x='frame', y='z_error', title='Z Error over Time', render_mode='svg')
    
    fig_x.update_xaxes(title_text='', showticklabels=False)
    fig_x.update_yaxes(title_text='X', title_font=dict(size=18))
    fig_x.update_layout(paper_bgcolor=color_of_cards, height = trajectory_plot_height, yaxis=dict(range=[0, max_error+10]))

    fig_y.update_xaxes(title_text='', showticklabels=False)
    fig_y.update_yaxes(title_text='Y', title_font=dict(size=18))
    fig_y.update_layout(paper_bgcolor=color_of_cards, height = trajectory_plot_height, yaxis=dict(range=[0, max_error+10]))

    fig_z.update_xaxes(title_text='Frame', title_font=dict(size=18))
    fig_z.update_yaxes(title_text='Z', title_font=dict(size=18))
    fig_z.update_layout(paper_bgcolor=color_of_cards, height = trajectory_plot_height, yaxis=dict(range=[0, max_error+10]))

    
    return [dcc.Graph(figure=fig_x), dcc.Graph(figure=fig_y), dcc.Graph(figure=fig_z)]
