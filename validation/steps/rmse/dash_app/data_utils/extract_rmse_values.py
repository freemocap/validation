
def extract_overall_rmse_values_from_dataframe(rmse_dataframe):
    """Extract RMSE values for total, x, y, and z."""
    conditions = lambda marker, coordinate: (rmse_dataframe['keypoint'] == marker) & (rmse_dataframe['coordinate'] == coordinate)
    total_rmse = rmse_dataframe.loc[conditions('All', 'All'), 'RMSE'].values[0]
    x_rmse = rmse_dataframe.loc[conditions('All', 'x_error'), 'RMSE'].values[0]
    y_rmse = rmse_dataframe.loc[conditions('All', 'y_error'), 'RMSE'].values[0]
    z_rmse = rmse_dataframe.loc[conditions('All', 'z_error'), 'RMSE'].values[0]

    return {
        'total': total_rmse,
        'x': x_rmse,
        'y': y_rmse,
        'z': z_rmse
    }
