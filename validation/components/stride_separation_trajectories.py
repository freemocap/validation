from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import save_csv, save_plotly_fig


FREEMOCAP_TRAJECTORY_CYCLES = DataComponent(
    name = "trajectories_per_stride",
    filename = "trajectories_per_stride.csv",
    relative_path = "{tracker}/analysis_outputs/trajectories",
    saver = save_csv,
    loader= None,
)

QUALISYS_TRAJECTORY_CYCLES = DataComponent(
    name = "qualisys_trajectories_per_stride",
    filename = "trajectories_per_stride.csv",
    relative_path = "qualisys/analysis_outputs/trajectories",
    saver = save_csv,
    loader= None,
)

FREEMOCAP_TRAJECTORY_SUMMARY_STATS = DataComponent(
    name = "freemocap_trajectory_summary_stats",
    filename = "trajectories_per_stride_summary_stats.csv",
    relative_path = "{tracker}/analysis_outputs/trajectories",
    saver = save_csv,
    loader= None,
)

QUALISYS_TRAJECTORY_SUMMARY_STATS = DataComponent(
    name = "qualisys_trajectory_summary_stats",
    filename = "trajectories_per_stride_summary_stats.csv",
    relative_path = "qualisys/analysis_outputs/trajectories",
    saver = save_csv,
    loader= None,
)

TRAJECTORY_PER_STRIDE_FIG = DataComponent(
    name = "trajectory_per_stride_figure",
    filename = "trajectories_per_stride.html",
    relative_path = "{tracker}/analysis_outputs/trajectories",
    saver = save_plotly_fig,
    loader= None,
)

TRAJECTORY_MEAN_FIG = DataComponent(
    name = "trajectory_mean_figure",
    filename = "trajectories_mean_stride.html",
    relative_path = "{tracker}/analysis_outputs/trajectories",
    saver = save_plotly_fig,
    loader= None,
)

TRAJECTORY_RMSE_STATS = DataComponent(
    name = "trajectory_rmse_stats",
    filename = "trajectories_per_stride_rmse_stats.csv",
    relative_path = "{tracker}/analysis_outputs/trajectories",
    saver = save_csv,
    loader= None,
)