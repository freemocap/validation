def subsample_dataframe(dataframe, frame_skip_interval):
    """
    Subsample a DataFrame by selecting rows whose frame number is a multiple
    of the given subsample_factor.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to be subsampled.
    - frame_skip_interval (int): The factor by which to subsample. 
                              For example, if frame_skip_interval = 3, every third frame will be kept.
                              
    Returns:
    - pd.DataFrame: The subsampled DataFrame.
    """
    
    # Use the modulo operator to find rows where the frame number is a multiple of subsample_factor
    subsampled_df = dataframe[dataframe['frame'] % frame_skip_interval == 0]
    
    return subsampled_df