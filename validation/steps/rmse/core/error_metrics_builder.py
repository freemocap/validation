import pandas as pd
import numpy as np


class ErrorMetricsBuilder:
    """
    A class for calculating error metrics from a dataframe containing data from freemocap and qualisys systems . Can return:
    1) The squared error on each frame for each keypoint
    2) The absolute error on each frame for each keypoint
    3) The RMSE for each keypoint, each dimension, and overall
    """
    def __init__(self, dataframe_of_3d_data):
        self.dataframe_of_3d_data = dataframe_of_3d_data
        self.squared_error_dataframe = None
        self.absolute_error_dataframe = None
        self.rmse_dataframe = None
        self.results = {}

    def calculate_squared_error_dataframe(self,dataframe_of_3d_data):
        """
        Calculate the squared error between the 'freemocap' and 'qualisys' systems
        for each frame and keypoint.

        Parameters:
        - dataframe_of_3d_data (pd.DataFrame): A DataFrame containing 3D motion capture data from both systems.
        The DataFrame should contain columns ['frame', 'keypoint', 'x', 'y', 'z', 'system'].

        Returns:
        - pd.DataFrame: A new DataFrame containing only the squared error calculations.
        """
        original_order_of_keypoints = dataframe_of_3d_data['keypoint'].drop_duplicates().tolist() 

        # Pivot the DataFrame so that the 'system' becomes new columns
        pivot_df = dataframe_of_3d_data.pivot_table(index=['frame', 'keypoint'], columns='system', values=['x', 'y', 'z'])

        pivot_df = pivot_df.reorder_levels(['frame', 'keypoint']).sort_index()
        pivot_df = pivot_df.loc[pd.IndexSlice[:, original_order_of_keypoints], :]
        
        squared_error_dataframe = pd.DataFrame()
        squared_error_dataframe['frame'] = pivot_df.index.get_level_values('frame')
        squared_error_dataframe['keypoint'] = pivot_df.index.get_level_values('keypoint')

        
        # Calculate squared errors for x, y, z dimensions
        for coord in ['x', 'y', 'z']:
            system_A_name = pivot_df.columns.get_level_values(1).unique().tolist()[0]
            system_B_name = pivot_df.columns.get_level_values(1).unique().tolist()[1]
            squared_error_dataframe[f'{coord}_error'] = self._calculate_squared_error(pivot_df[coord][system_A_name], pivot_df[coord][system_B_name]).reset_index(drop=True) #need to reset index here to make sure there's not a dataframe index mismatch
        
        self.squared_error_dataframe = squared_error_dataframe
        return self
    
    def calculate_absolute_error_from_squared_error_dataframe(self):
        """
        Calculate the absolute error based on the squared errors for each keypoint and frame.

        Parameters:
        - squared_error_df (pd.DataFrame): A DataFrame containing squared error calculations.
        The DataFrame should contain columns ['frame', 'keypoint', 'x_error', 'y_error', 'z_error'].

        Returns:
        - pd.DataFrame: A new DataFrame containing only the absolute error calculations.
        """
        if self.squared_error_dataframe is None:
            raise ValueError('Squared error dataframe has not been calculated yet. Please run calculate_squared_error_dataframe() first.')
        
        # Calculate absolute error by taking the square root of squared error for each coordinate
        absolute_error_dataframe = pd.DataFrame()
        absolute_error_dataframe['frame'] = self.squared_error_dataframe['frame']
        absolute_error_dataframe['keypoint'] = self.squared_error_dataframe['keypoint']
        
        # Loop to calculate absolute errors
        for coord in ['x', 'y', 'z']:
            absolute_error_dataframe[f'{coord}_error'] = self._calculate_absolute_error_from_squared_error(self.squared_error_dataframe[f'{coord}_error'])
            
        self.absolute_error_dataframe = absolute_error_dataframe
        return self
    
    def calculate_rmse_from_dataframe(self):
        if self.squared_error_dataframe is None:
            raise ValueError('Squared error dataframe has not been calculated yet. Please run calculate_squared_error_dataframe() first.')
        
        # Create an empty DataFrame to store the RMSE values
        rmse_dataframe = pd.DataFrame()

        # Calculate RMSE for each joint
        #groupby partitions the dataframe into smaller groups based on the keypoint column, keep sort = False to keep the original order of the keypoints
        rmse_joints = self.squared_error_dataframe.groupby('keypoint', sort = False)[['x_error', 'y_error', 'z_error']].apply(self._calculate_rmse_from_squared_error).reset_index()
        rmse_joints['dimension'] = 'Per Joint'
        rmse_joints = rmse_joints.melt(id_vars=['keypoint', 'dimension'], var_name='coordinate', value_name='RMSE')
        
        # Append to rmse_dataframe
        rmse_dataframe = pd.concat([rmse_dataframe, rmse_joints], ignore_index=True)

        # Calculate RMSE for each dimension
        rmse_dimensions = self.squared_error_dataframe[['x_error', 'y_error', 'z_error']].apply(self._calculate_rmse_from_squared_error).reset_index()
        rmse_dimensions.columns = ['coordinate', 'RMSE']
        rmse_dimensions['dimension'] = 'Per Dimension'
        rmse_dimensions['keypoint'] = 'All'
        
        # Append to rmse_dataframe
        rmse_dataframe = pd.concat([rmse_dataframe, rmse_dimensions], ignore_index=True)

        # Calculate overall RMSE (turn the datafrom into a 1d array with all the error values)
        overall_rmse = self._calculate_rmse_from_squared_error(self.squared_error_dataframe[['x_error', 'y_error', 'z_error']].values.flatten())
        overall_rmse = pd.DataFrame({'keypoint': 'All', 'dimension': 'Overall', 'coordinate': 'All', 'RMSE': [overall_rmse]})
        
        # Append to rmse_dataframe
        rmse_dataframe = pd.concat([rmse_dataframe, overall_rmse], ignore_index=True)

        self.rmse_dataframe = rmse_dataframe
        return self

    def _calculate_squared_error(self,value1, value2):
        return (value1 - value2) ** 2
    
    def _calculate_absolute_error_from_squared_error(self,value):
        return np.sqrt(value)
    
    def _calculate_rmse_from_squared_error(self,squared_errors):
        
        mean_squared_error = np.mean(squared_errors, axis = 0)
        root_mean_squared_error = np.sqrt(mean_squared_error)
        return root_mean_squared_error
    
    def build(self):
        return {
            'squared_error_dataframe': self.squared_error_dataframe,
            'absolute_error_dataframe': self.absolute_error_dataframe,
            'rmse_dataframe': self.rmse_dataframe
        }

    
def get_error_metrics(dataframe_of_3d_data):
    error_metrics_builder = ErrorMetricsBuilder(dataframe_of_3d_data=dataframe_of_3d_data)

    return (error_metrics_builder.calculate_squared_error_dataframe(dataframe_of_3d_data=dataframe_of_3d_data)
                                                .calculate_absolute_error_from_squared_error_dataframe()
                                                .calculate_rmse_from_dataframe()
                                                .build())





    