from validation.components import (
    FREEMOCAP_PARQUET,
    QUALISYS_PARQUET,
    FREEMOCAP_GAIT_EVENTS,
    QUALISYS_GAIT_EVENTS,
    FREEMOCAP_JOINT_ANGLES,
    QUALISYS_JOINT_ANGLES,
    FREEMOCAP_JOINT_ANGLE_CYCLES,
    QUALISYS_JOINT_ANGLE_CYCLES,
    FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS,
    QUALISYS_JOINT_ANGLE_SUMMARY_STATS,
    POSITIONRMSE,
    VELOCITYRMSE,
    FREEMOCAP_TRAJECTORY_CYCLES,
    QUALISYS_TRAJECTORY_CYCLES,
    FREEMOCAP_TRAJECTORY_SUMMARY_STATS,
    QUALISYS_TRAJECTORY_SUMMARY_STATS,
    TRAJECTORY_RMSE_STATS,
    JOINT_ANGLE_RMSE_STATS,
    QUALISYS_GAIT_METRICS,
    QUALISYS_GAIT_SUMMARY_STATS,
    FREEMOCAP_GAIT_METRICS,
    FREEMOCAP_GAIT_SUMMARY_STATS,
)
from validation.datatypes.data_component import DataComponent


FREEMOCAP_PATH_LENGTH_COM = DataComponent(
    name="path_length_com",
    filename="condition_data.json",
    relative_path=(
        "{tracker}/analysis_outputs/"
        "path_length_analysis"
    ),
)

QUALISYS_PATH_LENGTH_COM = DataComponent(
    name="qualisys_path_length_com",
    filename="condition_data.json",
    relative_path=(
        "qualisys/analysis_outputs/"
        "path_length_analysis"
    ),
)

FREEMOCAP_BALANCE_POSITIONS = DataComponent(
    name="balance_positions",
    filename="condition_positions.csv",
    relative_path=(
        "{tracker}/analysis_outputs/"
        "path_length_analysis"
    ),
)

FREEMOCAP_BALANCE_VELOCITIES = DataComponent(
    name="balance_velocities",
    filename="condition_velocities.csv",
    relative_path=(
        "{tracker}/analysis_outputs/"
        "path_length_analysis"
    ),
)

QUALISYS_BALANCE_POSITIONS = DataComponent(
    name="qualisys_balance_positions",
    filename="condition_positions.csv",
    relative_path=(
        "qualisys/analysis_outputs/"
        "path_length_analysis"
    ),
)

QUALISYS_BALANCE_VELOCITIES = DataComponent(
    name="qualisys_balance_velocities",
    filename="condition_velocities.csv",
    relative_path=(
        "qualisys/analysis_outputs/"
        "path_length_analysis"
    ),
)


BALANCE = {
    "synced_data": [
        FREEMOCAP_PARQUET,
        QUALISYS_PARQUET,
    ],
    "com_analysis": [
        FREEMOCAP_PATH_LENGTH_COM,
        QUALISYS_PATH_LENGTH_COM,
        FREEMOCAP_BALANCE_VELOCITIES,
        QUALISYS_BALANCE_VELOCITIES,
        FREEMOCAP_BALANCE_POSITIONS,
        QUALISYS_BALANCE_POSITIONS,
    ],
}


TREADMILL = {
    "synced_data": [
        FREEMOCAP_PARQUET,
        QUALISYS_PARQUET,
    ],
    "gait_events": [
        FREEMOCAP_GAIT_EVENTS,
        QUALISYS_GAIT_EVENTS,
    ],
    "gait_metrics": [
        QUALISYS_GAIT_METRICS,
        QUALISYS_GAIT_SUMMARY_STATS,
        FREEMOCAP_GAIT_METRICS,
        FREEMOCAP_GAIT_SUMMARY_STATS,
    ],
    "joint_angles": [
        FREEMOCAP_JOINT_ANGLES,
        QUALISYS_JOINT_ANGLES,
    ],
    "joint_angles_per_stride": [
        FREEMOCAP_JOINT_ANGLE_CYCLES,
        QUALISYS_JOINT_ANGLE_CYCLES,
        FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS,
        QUALISYS_JOINT_ANGLE_SUMMARY_STATS,
        JOINT_ANGLE_RMSE_STATS,
    ],
    "rmse_metrics": [
        POSITIONRMSE,
        VELOCITYRMSE,
    ],
    "trajectories_per_stride": [
        FREEMOCAP_TRAJECTORY_CYCLES,
        QUALISYS_TRAJECTORY_CYCLES,
        FREEMOCAP_TRAJECTORY_SUMMARY_STATS,
        QUALISYS_TRAJECTORY_SUMMARY_STATS,
        TRAJECTORY_RMSE_STATS,
    ],
}