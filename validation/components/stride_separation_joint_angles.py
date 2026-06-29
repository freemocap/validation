from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import save_csv, save_plotly_fig

FREEMOCAP_JOINT_ANGLE_CYCLES = DataComponent(
    name = "joint_angles_per_stride",
    filename = "joint_angles_per_stride.csv",
    relative_path = "{tracker}/analysis_outputs/joint_angles",
    saver = save_csv,
    loader= None,
)

QUALISYS_JOINT_ANGLE_CYCLES = DataComponent(
    name = "qualisys_joint_angles_per_stride",
    filename = "joint_angles_per_stride.csv",
    relative_path = "qualisys/analysis_outputs/joint_angles",
    saver = save_csv,
    loader= None,
)

FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS = DataComponent(
    name = "freemocap_joint_angle_summary_stats",
    filename = "joint_angles_per_stride_summary_stats.csv",
    relative_path = "{tracker}/analysis_outputs/joint_angles",
    saver = save_csv,
    loader= None,
)

QUALISYS_JOINT_ANGLE_SUMMARY_STATS = DataComponent(
    name = "qualisys_joint_angle_summary_stats",
    filename = "joint_angles_per_stride_summary_stats.csv",
    relative_path = "qualisys/analysis_outputs/joint_angles",
    saver = save_csv,
    loader= None,
)

JOINT_ANGLE_SUMMARY_FIG = DataComponent(
    name = "joint_angle_summary_figure",
    filename = "joint_angles_mean_stride.html",
    relative_path = "{tracker}/analysis_outputs/joint_angles",
    saver = save_plotly_fig,
    loader= None,
)

JOINT_ANGLE_RMSE_STATS = DataComponent(
    name = "joint_angle_rmse_stats",
    filename = "joint_angles_per_stride_rmse_stats.csv",
    relative_path = "{tracker}/analysis_outputs/joint_angles",
    saver = save_csv,
    loader= None,
)