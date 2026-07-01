"""
Three-row balance sway figure:
    Row 1: COM path length (mm)
    Row 2: 95% confidence ellipse area (mm^2)
    Row 3: Mean 2D COM velocity (mm/s)
Columns = trackers (Reference / MediaPipe / RTMPose / ViTPose).

Each panel shows individual trial lines (gray) + group mean +/- SD (black),
following the same visual style as the original com_path_length figure.

NOTE ON AGGREGATION
-------------------
The original path-length figure computed mean +/- SD directly across all trials.
The summary *table* (other script) first averages within participant, then takes
mean +/- SD across participants. These give slightly different numbers.

    AGG = "participant"  -> figure means match the summary table
    AGG = "trial"        -> reproduces the original com_path_length figure exactly

The gray individual lines are always per-trial regardless of AGG.
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy.stats import chi2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
AGG = "trial"  # "trial" (matches old fig) or "participant" (matches table)

conn = sqlite3.connect("validation.db")

root_path = Path(
    r"D:\validation_public_release_v1\figures"
)
root_path.mkdir(exist_ok=True, parents=True)

conditions = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]
condition_order = conditions
display_x_short = ["EO-S", "EC-S", "EO-F", "EC-F"]

TRACKERS = ["qualisys", "mediapipe", "rtmpose", "vitpose"]
sub_title = {
    "qualisys": "Reference",
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}

TRACKER_COLORS = {
    "qualisys": "black",       # reference
    "mediapipe": "#006DFC",    # blue
    "vitpose": "#05C936",      # green
    "rtmpose": "#EB7303",      # red
}


col_for = {trk: i + 1 for i, trk in enumerate(TRACKERS)}


# --------------------------------------------------------------------------- #
# Confidence ellipse (for ellipse area)
# --------------------------------------------------------------------------- #
def confidence_ellipse_95(x, y):
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    chi2_val = chi2.ppf(0.95, df=2)
    a = np.sqrt(eigenvalues[0] * chi2_val)
    b = np.sqrt(eigenvalues[1] * chi2_val)
    area = np.pi * a * b
    return area


# --------------------------------------------------------------------------- #
# 1. Path length  (JSON: %path_length_com)
# --------------------------------------------------------------------------- #
pl_query = """
SELECT t.participant_code, t.trial_name, a.path, a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%path_length_com'
ORDER BY t.trial_name, a.path;
"""
pl_dfs = []
for _, row in pd.read_sql_query(pl_query, conn).iterrows():
    sub_df = pd.read_json(row["path"])
    sub_df = (
        sub_df.rename(columns={"Frame Intervals": "frame_interval",
                               "Path Lengths:": "path_length"})
        .reset_index()
        .rename(columns={"index": "condition"})
    )
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    pl_dfs.append(sub_df)
pl_df = pd.concat(pl_dfs, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. Ellipse area  (CSV: %balance_positions)
# --------------------------------------------------------------------------- #
pos_query = """
SELECT t.participant_code, t.trial_name, a.path, a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%balance_positions'
ORDER BY t.trial_name, a.path;
"""
pos_dfs = []
for _, row in pd.read_sql_query(pos_query, conn).iterrows():
    sub_df = pd.read_csv(row["path"])
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    pos_dfs.append(sub_df)
pos_df = pd.concat(pos_dfs, ignore_index=True)

ellipse_results = []
for (participant, trial, tracker), grp in pos_df.groupby(
    ["participant_code", "trial_name", "tracker"]
):
    for condition in conditions:
        x_col, y_col = f"{condition}_x", f"{condition}_y"
        if x_col not in grp.columns:
            continue
        x = grp[x_col].to_numpy()
        y = grp[y_col].to_numpy()
        x = x - np.mean(x)
        y = y - np.mean(y)
        area = confidence_ellipse_95(x, y)
        ellipse_results.append({
            "participant_code": participant,
            "trial_name": trial,
            "tracker": tracker,
            "condition": condition,
            "ellipse_area_mm2": area,
        })
ellipse_df = pd.DataFrame(ellipse_results)


# --------------------------------------------------------------------------- #
# 3. Mean 2D velocity  (CSV: %balance_velocities)
# --------------------------------------------------------------------------- #
vel_query = """
SELECT t.participant_code, t.trial_name, a.path, a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%balance_velocities'
ORDER BY t.trial_name, a.path;
"""
vel_dfs = []
for _, row in pd.read_sql_query(vel_query, conn).iterrows():
    sub_df = pd.read_csv(row["path"])
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    vel_dfs.append(sub_df)
vel_df = pd.concat(vel_dfs, ignore_index=True)

id_cols = ["participant_code", "trial_name", "Frame", "tracker"]
value_cols = [c for c in vel_df.columns if ("Eyes" in c or "Ground" in c or "Foam" in c)]

long_df = vel_df.melt(id_vars=id_cols, value_vars=value_cols,
                      var_name="cond_axis", value_name="velocity")
long_df[["condition", "axis"]] = long_df["cond_axis"].str.rsplit("_", n=1, expand=True)
long_df = long_df.dropna(subset=["velocity"])

xy_df = long_df[long_df["axis"].isin(["x", "y"])].copy()
xy_wide = (
    xy_df.pivot_table(
        index=["participant_code", "trial_name", "Frame", "tracker", "condition"],
        columns="axis", values="velocity", aggfunc="first",
    ).reset_index()
)
xy_wide = xy_wide.dropna(subset=["x", "y"])
xy_wide["velocity_2d"] = np.sqrt(xy_wide["x"] ** 2 + xy_wide["y"] ** 2)

vel_trial = (
    xy_wide.groupby(
        ["participant_code", "trial_name", "tracker", "condition"], as_index=False
    )["velocity_2d"].mean()
    .rename(columns={"velocity_2d": "mean_velocity_2d"})
)

conn.close()


# --------------------------------------------------------------------------- #
# Standardize each metric to columns: tracker, condition, participant_code,
# trial_name, value
# --------------------------------------------------------------------------- #
keep = ["tracker", "condition", "participant_code", "trial_name", "value"]

pl_std = pl_df.rename(columns={"path_length": "value"})[keep]
ell_std = ellipse_df.rename(columns={"ellipse_area_mm2": "value"})[keep]
vel_std = vel_trial.rename(columns={"mean_velocity_2d": "value"})[keep]

metrics = [
    {"df": pl_std,  "ylabel": "Path Length (mm)"},
    {"df": ell_std, "ylabel": "Ellipse Area (mm<sup>2</sup>)"},
    {"df": vel_std, "ylabel": "Mean 2D Velocity (mm/s)"},
]


def aggregate(df, level):
    """Return mean/std per (tracker, condition)."""
    if level == "participant":
        per = (
            df.groupby(["tracker", "condition", "participant_code"])["value"]
            .mean().reset_index()
        )
        return per.groupby(["tracker", "condition"])["value"].agg(["mean", "std"])
    return df.groupby(["tracker", "condition"])["value"].agg(["mean", "std"])


# --------------------------------------------------------------------------- #
# Build 3 x 4 figure
# --------------------------------------------------------------------------- #
subplot_titles = [sub_title[t] for t in TRACKERS] + [""] * 4 + [""] * 4

fig = make_subplots(
    rows=3, cols=len(TRACKERS),
    shared_yaxes=True,          # share y within each row (per metric)
    subplot_titles=subplot_titles,
    horizontal_spacing=0.04,
    vertical_spacing=0.06,
)

for row_idx, m in enumerate(metrics, start=1):
    dfm = m["df"]

    # ---- individual trial lines (gray) ----
    for tracker in TRACKERS:
        dft = dfm[dfm["tracker"] == tracker].copy()
        if dft.empty:
            continue
        dft["trial_id"] = dft["participant_code"] + " | " + dft["trial_name"]
        for trial_id, sub in dft.groupby("trial_id", sort=False):
            s = sub.set_index("condition")["value"].reindex(condition_order)
            fig.add_trace(
                go.Scatter(
                    x=display_x_short, y=s.values,
                    mode="lines+markers",
                    line=dict(color="rgba(150,150,150,0.4)", width=2),
                    marker=dict(size=4, color="rgba(150,150,150,0.5)"),
                    showlegend=False,
                    hovertemplate=(
                        f"{trial_id}<br>%{{x}}<br>"
                        f"value: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row_idx, col=col_for[tracker],
            )

    # ---- group mean +/- SD (black) ----
    agg = aggregate(dfm, AGG)
    for tracker in TRACKERS:
        if tracker not in agg.index.get_level_values(0):
            continue
        sub = agg.loc[tracker].reindex(condition_order)
        means = sub["mean"].to_numpy()
        stds = sub["std"].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=display_x_short, y=means,
                mode="lines+markers",
                line=dict(color=TRACKER_COLORS[tracker], width=2.5),
                marker=dict(color=TRACKER_COLORS[tracker], size=6),
                showlegend=False,
                error_y=dict(type="data", array=stds, visible=True,
                             thickness=2.5, width=4),
                hovertemplate=(
                    "%{x}<br>Mean: %{y:.3f}<br>"
                    "SD: %{customdata:.3f}<extra></extra>"
                ),
                customdata=stds,
            ),
            row=row_idx, col=col_for[tracker],
        )

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
fig.update_layout(
    height=1150, width=1200,
    template="simple_white",
    margin=dict(l=95, r=20, t=45, b=70),
    font=dict(family="Arial", size=14),
)

# subplot titles (top-row tracker names)
for ann in fig.layout.annotations:
    ann.update(font=dict(family="Arial", size=22), xanchor="center")

# per-row y-axis titles (col 1 only)
for r, m in enumerate(metrics, start=1):
    fig.update_yaxes(
        title_text=f"<b>{m['ylabel']}</b>",
        title_font=dict(size=20),
        tickfont=dict(size=16),
        row=r, col=1,
    )

# tick fonts on all y axes
for r in range(1, 4):
    for c in range(1, len(TRACKERS) + 1):
        fig.update_yaxes(tickfont=dict(size=16), row=r, col=c)

# x tick labels only on the bottom row
for c in range(1, len(TRACKERS) + 1):
    fig.update_xaxes(showticklabels=False, row=1, col=c)
    fig.update_xaxes(showticklabels=False, row=2, col=c)
    fig.update_xaxes(tickfont=dict(size=16), row=3, col=c)

fig.show()

fig.write_image(root_path / "balance_sway_metrics.svg", scale=3)
# fig.write_image(root_path / "balance_sway_metrics.png", scale=3)
# # fig.write_image(root_path / "balance_sway_metrics.pdf")