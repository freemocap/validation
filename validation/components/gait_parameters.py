from validation.components import DataComponent
from validation.utils.io_helpers import save_csv

FREEMOCAP_GAIT_METRICS = DataComponent(
    name = "gait_metrics",
    filename = "gait_metrics.csv",
    relative_path = "{tracker}/analysis_outputs/gait_parameters",
    saver=save_csv,
    loader=None,
)

FREEMOCAP_GAIT_SUMMARY_STATS = DataComponent(
    name = "gait_summary_stats",
    filename = "gait_summary_stats.csv",
    relative_path = "{tracker}/analysis_outputs/gait_parameters",
    saver=save_csv,
    loader=None,
)

QUALISYS_GAIT_METRICS = DataComponent(
    name = "qualisys_gait_metrics",
    filename = "qualisys_gait_metrics.csv",
    relative_path = "qualisys/analysis_outputs/gait_parameters",
    saver=save_csv,
    loader=None,
)

QUALISYS_GAIT_SUMMARY_STATS = DataComponent(
    name = "qualisys_gait_summary_stats",
    filename = "qualisys_gait_summary_stats.csv",
    relative_path = "qualisys/analysis_outputs/gait_parameters",
    saver=save_csv,
    loader=None,
)
