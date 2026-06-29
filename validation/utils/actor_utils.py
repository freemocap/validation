
from skellymodels.models.tracking_model_info import ModelInfo, RTMPoseModelInfo
from skellymodels.managers.human import Human
from skellymodels.models.aspect import TrajectoryNames
from skellymodels.models.trajectory import Trajectory
from validation.pipeline.project_config import ProjectConfig
import numpy as np
from pathlib import Path

def make_qualisys_actor(project_config: ProjectConfig, tracked_points_data:np.ndarray):
    return Human.from_tracked_points_numpy_array(
    name = "qualisys_human",
    model_info = ModelInfo.from_config_path(config_path= project_config.qualisys_model_info_path),
    tracked_points_numpy_array=tracked_points_data)

def get_model_info(freemocap_tracker: str):
    path_to_model_folder = Path(__file__).parent/'freemocap_model_info'
    match freemocap_tracker:
        case "mediapipe":
            model_info = ModelInfo.from_config_path(config_path= path_to_model_folder/'mediapipe_model_info.yaml')
        case "rtmpose":
            model_info = ModelInfo.from_config_path(config_path= path_to_model_folder/'rtmpose_model_info.yaml')
        case "vitpose":
            model_info = ModelInfo.from_config_path(config_path= path_to_model_folder/'vitpose_model_info.yaml')
    return model_info

def make_freemocap_actor_from_tracked_points(freemocap_tracker: str, tracked_points_data:np.ndarray) -> Human:
    model_info = get_model_info(freemocap_tracker)
    return Human.from_tracked_points_numpy_array(
        name = f"{freemocap_tracker}",
        model_info=model_info,
        tracked_points_numpy_array=tracked_points_data
    ) 

def make_freemocap_actor_from_parquet(parquet_path:Path):
    return Human.from_parquet(parquet_path)

def make_freemocap_actor_from_landmarks(freemocap_tracker: str, landmarks:np.ndarray):
    model_info = get_model_info(freemocap_tracker)

    human = Human(
            name="human_one", 
            model_info=model_info
            )
    
    xyz_trajectory =  Trajectory(
                    name = TrajectoryNames.XYZ.value,
                    array = landmarks,
                    landmark_names= human.body.anatomical_structure.landmark_names)

    human.body.add_trajectory({TrajectoryNames.XYZ.value: xyz_trajectory}
                                )
    return human
