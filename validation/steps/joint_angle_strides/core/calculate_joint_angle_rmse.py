import pandas as pd
import numpy as np


def calculate_rmse(freemocap_df: pd.DataFrame, qualisys_df: pd.DataFrame):

    keys = ["cycle", "joint", "side", "component", "percent_gait_cycle"]

    merged = freemocap_df[keys + ["angle"]].merge(
        qualisys_df[keys + ["angle"]],
        on=keys,
        suffixes=("_fmc", "_qtm"),
    )

    merged["sq_err"] = (merged["angle_fmc"] - merged["angle_qtm"]) ** 2

    rmse_df = (
        merged
        .groupby(["cycle", "joint", "side", "component"], observed=True)["sq_err"]
        .mean()
        .pipe(np.sqrt)
        .reset_index(name="rmse_deg")
    )

    return rmse_df
