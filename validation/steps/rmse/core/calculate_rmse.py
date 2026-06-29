from skellymodels.managers.human import Human
from skellymodels.models.trajectory import Trajectory
from skellymodels.models.aspect import Aspect
from validation.steps.rmse.config import RMSEConfig
from validation.steps.rmse.core.error_metrics_builder import get_error_metrics
import numpy as np
import pandas as pd

from dataclasses import dataclass
@dataclass
class RMSEResults:
    position_joint_df: pd.DataFrame
    position_rmse: pd.DataFrame
    position_absolute_error:pd.DataFrame
    velocity_joint_df: pd.DataFrame
    velocity_rmse: pd.DataFrame
    velocity_absolute_error:pd.DataFrame

def create_velocity_trajectory(position_trajectory:Trajectory,landmark_names:list[str]):

    velocity_array = np.diff(position_trajectory.as_array, axis = 0)
    velocity_trajectory = Trajectory(
        name =  '3d_velocity_xyz',
        array =  velocity_array,
        landmark_names = landmark_names
    )
    return velocity_trajectory

def combine_system_dataframes_on_common_markers(
    markers_for_comparison: list[str],
    freemocap_trajectory: Trajectory,
    qualisys_trajectory: Trajectory
) -> pd.DataFrame:
    fmc_df = freemocap_trajectory.as_dataframe.query('keypoint in @markers_for_comparison').copy()
    fmc_df['system'] = 'freemocap'

    qtm_df = qualisys_trajectory.as_dataframe.query('keypoint in @markers_for_comparison').copy()
    qtm_df['system'] = 'qualisys'
    return pd.concat([fmc_df, qtm_df], ignore_index=True)

def calculate_rmse(freemocap_actor:Human,
                    qualisys_actor:Human,
                    config: RMSEConfig,
                    frame_range: list[int]|None,
                    use_rigid = True) -> RMSEResults:
    
    #think about using timestamps to get true velocity
    markers_for_comparison = config.markers_for_comparison

    qualisys_trajectory = qualisys_actor.body.xyz

    if use_rigid:
        freemocap_trajectory = freemocap_actor.body.rigid_xyz
    else:
        freemocap_trajectory = freemocap_actor.body.xyz

    freemocap_actor.body.add_trajectory({"3d_velocity_xyz": create_velocity_trajectory(freemocap_trajectory, 
                                                                                       freemocap_actor.body.anatomical_structure.landmark_names)})
    qualisys_actor.body.add_trajectory({"3d_velocity_xyz": create_velocity_trajectory(qualisys_trajectory, 
                                                                                      qualisys_actor.body.anatomical_structure.landmark_names)})

    combined_position_df = combine_system_dataframes_on_common_markers(
                                                                    markers_for_comparison=markers_for_comparison,
                                                                    freemocap_trajectory=freemocap_trajectory,
                                                                    qualisys_trajectory=qualisys_trajectory)
    
    if frame_range is not None:
        start,end = frame_range
    else:
        start = 0
        end = freemocap_trajectory.as_array.shape[0]

    combined_position_df = combined_position_df[
    (combined_position_df['frame'] >= start) &
    (combined_position_df['frame'] <= end)
]

    position_error_metrics_dict = get_error_metrics(dataframe_of_3d_data=combined_position_df)


    combined_velocity_df = combine_system_dataframes_on_common_markers(
        markers_for_comparison=markers_for_comparison,
        freemocap_trajectory=freemocap_actor.body.trajectories['3d_velocity_xyz'],
        qualisys_trajectory=qualisys_actor.body.trajectories['3d_velocity_xyz'])

    combined_velocity_df = combined_velocity_df[
        (combined_velocity_df["frame"] >= start) &
        (combined_velocity_df["frame"] <= end)
    ]

    velocity_error_metrics_dict = get_error_metrics(dataframe_of_3d_data=combined_velocity_df)

    return RMSEResults(
        position_joint_df= combined_position_df,
        position_rmse= position_error_metrics_dict['rmse_dataframe'],
        position_absolute_error= position_error_metrics_dict['absolute_error_dataframe'],
        velocity_joint_df= combined_velocity_df,
        velocity_rmse= velocity_error_metrics_dict['rmse_dataframe'],
        velocity_absolute_error= velocity_error_metrics_dict['absolute_error_dataframe']
    )
    f = 2
