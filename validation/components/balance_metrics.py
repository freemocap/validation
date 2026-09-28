from validation.components import DataComponent
from validation.utils.io_helpers import save_csv, load_csv

FREEMOCAP_BALANCE_METRICS = DataComponent(
    name="freemocap_balance_metrics",
    filename="balance_metrics.csv",
    relative_path="{tracker}/analysis_outputs/balance_metrics",
    loader=load_csv,
    saver=save_csv,
)

QUALISYS_BALANCE_METRICS = DataComponent(
    name="qualisys_balance_metrics",
    filename="balance_metrics.csv",
    relative_path="qualisys/analysis_outputs/balance_metrics",
    loader=load_csv,
    saver=save_csv,
)

FREEMOCAP_BALANCE_VELOCITIES = DataComponent(
    name="freemocap_balance_velocities",
    filename="balance_velocities.csv",
    relative_path="{tracker}/analysis_outputs/balance_metrics",
    loader=load_csv,
    saver=save_csv,
)

QUALISYS_BALANCE_VELOCITIES = DataComponent(
    name="qualisys_balance_velocities",
    filename="balance_velocities.csv",
    relative_path="qualisys/analysis_outputs/balance_metrics",
    loader=load_csv,
    saver=save_csv,
)