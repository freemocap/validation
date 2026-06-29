import pandas as pd
import numpy as np


def calculate_rmse(freemocap_df: pd.DataFrame, qualisys_df: pd.DataFrame):

    keys = ["cycle", "marker", "percent_gait_cycle"]

    merged = freemocap_df[keys + ["x", "y", "z"]].merge(
        qualisys_df[keys + ["x", "y", "z"]],
        on=keys,
        suffixes=("_fmc", "_qtm"),
    )

    merged["sq_err_x"] = (merged["x_fmc"] - merged["x_qtm"]) ** 2
    merged["sq_err_y"] = (merged["y_fmc"] - merged["y_qtm"]) ** 2
    merged["sq_err_z"] = (merged["z_fmc"] - merged["z_qtm"]) ** 2
    merged["sq_err_3d"] = merged["sq_err_x"] + merged["sq_err_y"] + merged["sq_err_z"]

    group = merged.groupby(["cycle", "marker"], observed=True)

    rmse_df = pd.DataFrame({
        "rmse_x":  group["sq_err_x"].mean().pipe(np.sqrt),
        "rmse_y":  group["sq_err_y"].mean().pipe(np.sqrt),
        "rmse_z":  group["sq_err_z"].mean().pipe(np.sqrt),
        "rmse_3d": group["sq_err_3d"].mean().pipe(np.sqrt),
    }).reset_index()

    return rmse_df
