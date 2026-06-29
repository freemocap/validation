import plotly.graph_objects as go

def create_rmse_joint_bar_plot(df):
    #Creates a bar plot for a single RMSE dataframe 
    dimensions = ['x_error', 'y_error', 'z_error']
    figures = {}
    
    # Calculate the maximum RMSE value across all dimensions
    max_rmse = df['RMSE'].max()
    
    for dim in dimensions:
        filtered_df = df[(df['dimension'] == 'Per Joint') & (df['coordinate'] == dim)]
        fig = go.Figure(data=[
            go.Bar(name=dim, x=filtered_df['keypoint'], y=filtered_df['RMSE'])
        ])
        fig.update_layout(
            title=f'RMSE per marker ({dim})',
            xaxis_title='Marker',
            yaxis_title='RMSE (mm)',
            xaxis=dict(
                tickfont=dict(
                    size=18
                )
            ),
            yaxis=dict(
                range=[0, max_rmse+10]
            )
        )
        figures[dim] = fig
    
    return figures



def create_position_velocity_rmse_joint_bar_plot(df_position, df_velocity):
    dimensions = ['x_error', 'y_error', 'z_error']
    figures = {}

    # Calculate the maximum RMSE value across all dimensions for both position and velocity
    max_rmse_position = df_position['RMSE'].max()
    max_rmse_velocity = df_velocity['RMSE'].max()
    max_rmse = max(max_rmse_position, max_rmse_velocity)
    
    for dim in dimensions:
        filtered_df_position = df_position[(df_position['dimension'] == 'Per Joint') & (df_position['coordinate'] == dim)]
        filtered_df_velocity = df_velocity[(df_velocity['dimension'] == 'Per Joint') & (df_velocity['coordinate'] == dim)]

        fig = go.Figure(data=[
            go.Bar(name=f'Position - {dim}', x=filtered_df_position['keypoint'], y=filtered_df_position['RMSE']),
            go.Bar(name=f'Velocity - {dim}', x=filtered_df_velocity['keypoint'], y=filtered_df_velocity['RMSE'])
        ])
        fig.update_layout(
            barmode='group',
            title=f'RMSE per marker ({dim})',
            xaxis_title='Marker',
            yaxis_title='RMSE',
            xaxis=dict(
                tickfont=dict(
                    size=18
                )
            ),
            yaxis=dict(
                range=[0, max_rmse + 10]
            ),
            legend=dict(
                title='Data Type',
                orientation='h',
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        figures[dim] = fig

    return figures