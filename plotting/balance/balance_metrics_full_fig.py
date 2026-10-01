"""
Three-row balance sway figure:
    Row 1: COM path length (mm)
    Row 2: 95% confidence ellipse area (mm^2)
    Row 3: Mean 2D COM velocity (mm/s)

Columns = trackers (Reference / MediaPipe / RTMPose / ViTPose).

Each panel shows individual trial lines (gray) + group mean +/- SD,
following the same visual style as the original com_path_length figure.

NOTE ON AGGREGATION
-------------------
The original path-length figure computed mean +/- SD directly across all trials.
The summary *table* first averages within participant, then takes mean +/- SD
across participants. These give slightly different numbers.

    AGG = "participant"  -> figure means match the summary table
    AGG = "trial"        -> reproduces the original com_path_length figure exactly

The gray individual lines are always per-trial regardless of AGG.
"""

import pandas as pd
import sqlite3
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validation.paths import DATABASE_PATH, output_dir
from validation.figure_display import show_figure


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

AGG = "trial"  # "trial" (matches old fig) or "participant" (matches table)

conn = sqlite3.connect(DATABASE_PATH)

root_path = output_dir("figures", "balance")


condition_order = [
    "eyes_open_solid",
    "eyes_closed_solid",
    "eyes_open_foam",
    "eyes_closed_foam",
]

display_x_short = [
    "EO-S",
    "EC-S",
    "EO-F",
    "EC-F",
]

TRACKERS = [
    "qualisys",
    "mediapipe",
    "rtmpose",
    "vitpose",
]

sub_title = {
    "qualisys": "Reference",
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}

TRACKER_COLORS = {
    "qualisys": "black",
    "mediapipe": "#006DFC",
    "vitpose": "#05C936",
    "rtmpose": "#EB7303",
}

col_for = {
    tracker: i + 1
    for i, tracker in enumerate(TRACKERS)
}


# --------------------------------------------------------------------------- #
# Load balance metrics
# --------------------------------------------------------------------------- #

metrics_query = """
SELECT
    t.participant_code,
    t.trial_name,
    a.path,
    a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name IN (
      "freemocap_balance_metrics",
      "qualisys_balance_metrics"
  )
ORDER BY t.trial_name, a.path;
"""

metric_paths = pd.read_sql_query(
    metrics_query,
    conn,
)

if metric_paths.empty:
    raise RuntimeError(
        "No balance_metrics artifacts were found in validation.db. "
        "Run the balance pipeline and rebuild the database first."
    )


metric_dfs = []

for _, row in metric_paths.iterrows():
    sub_df = pd.read_csv(row["path"])

    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]

    metric_dfs.append(sub_df)


balance_metrics_df = pd.concat(
    metric_dfs,
    ignore_index=True,
)

conn.close()


# --------------------------------------------------------------------------- #
# Standardize each metric to columns:
# tracker, condition, participant_code, trial_name, value
# --------------------------------------------------------------------------- #

keep = [
    "tracker",
    "condition",
    "participant_code",
    "trial_name",
    "value",
]

pl_std = (
    balance_metrics_df
    .rename(columns={"path_length_mm": "value"})[keep]
)

ell_std = (
    balance_metrics_df
    .rename(columns={"ellipse_area_mm2": "value"})[keep]
)

vel_std = (
    balance_metrics_df
    .rename(columns={"mean_2d_velocity_mm_s": "value"})[keep]
)


metrics = [
    {
        "df": pl_std,
        "ylabel": "Path Length (mm)",
    },
    {
        "df": ell_std,
        "ylabel": "Ellipse Area (mm<sup>2</sup>)",
    },
    {
        "df": vel_std,
        "ylabel": "Mean 2D Velocity (mm/s)",
    },
]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(
    df: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """
    Return mean/std per tracker and condition.

    participant:
        First average repeated trials within each participant,
        then calculate group mean +/- SD across participants.

    trial:
        Calculate group mean +/- SD directly across all trials.
    """

    if level == "participant":
        per_participant = (
            df
            .groupby(
                [
                    "tracker",
                    "condition",
                    "participant_code",
                ]
            )["value"]
            .mean()
            .reset_index()
        )

        return (
            per_participant
            .groupby(
                [
                    "tracker",
                    "condition",
                ]
            )["value"]
            .agg(["mean", "std"])
        )

    if level == "trial":
        return (
            df
            .groupby(
                [
                    "tracker",
                    "condition",
                ]
            )["value"]
            .agg(["mean", "std"])
        )

    raise ValueError(
        f"Unknown aggregation level: {level!r}. "
        "Expected 'trial' or 'participant'."
    )


# --------------------------------------------------------------------------- #
# Build 3 x 4 figure
# --------------------------------------------------------------------------- #

subplot_titles = (
    [sub_title[t] for t in TRACKERS]
    + [""] * 4
    + [""] * 4
)

fig = make_subplots(
    rows=3,
    cols=len(TRACKERS),
    shared_yaxes=True,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.04,
    vertical_spacing=0.06,
)


for row_idx, metric in enumerate(
    metrics,
    start=1,
):
    dfm = metric["df"]

    # ----------------------------------------------------------------------- #
    # Individual trial lines
    # ----------------------------------------------------------------------- #

    for tracker in TRACKERS:
        dft = dfm[
            dfm["tracker"] == tracker
        ].copy()

        if dft.empty:
            continue

        dft["trial_id"] = (
            dft["participant_code"]
            + " | "
            + dft["trial_name"]
        )

        for trial_id, sub in dft.groupby(
            "trial_id",
            sort=False,
        ):
            values = (
                sub
                .set_index("condition")["value"]
                .reindex(condition_order)
            )

            fig.add_trace(
                go.Scatter(
                    x=display_x_short,
                    y=values.values,
                    mode="lines+markers",
                    line=dict(
                        color="rgba(150,150,150,0.4)",
                        width=2,
                    ),
                    marker=dict(
                        size=4,
                        color="rgba(150,150,150,0.5)",
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"{trial_id}<br>"
                        "%{x}<br>"
                        "value: %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_for[tracker],
            )

    # ----------------------------------------------------------------------- #
    # Group mean +/- SD
    # ----------------------------------------------------------------------- #

    agg = aggregate(
        dfm,
        AGG,
    )

    for tracker in TRACKERS:
        if tracker not in agg.index.get_level_values(0):
            continue

        sub = (
            agg
            .loc[tracker]
            .reindex(condition_order)
        )

        means = sub["mean"].to_numpy()
        stds = sub["std"].to_numpy()

        fig.add_trace(
            go.Scatter(
                x=display_x_short,
                y=means,
                mode="lines+markers",
                line=dict(
                    color=TRACKER_COLORS[tracker],
                    width=2.5,
                ),
                marker=dict(
                    color=TRACKER_COLORS[tracker],
                    size=6,
                ),
                showlegend=False,
                error_y=dict(
                    type="data",
                    array=stds,
                    visible=True,
                    thickness=2.5,
                    width=4,
                ),
                hovertemplate=(
                    "%{x}<br>"
                    "Mean: %{y:.3f}<br>"
                    "SD: %{customdata:.3f}"
                    "<extra></extra>"
                ),
                customdata=stds,
            ),
            row=row_idx,
            col=col_for[tracker],
        )


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

fig.update_layout(
    height=1150,
    width=1200,
    template="simple_white",
    margin=dict(
        l=95,
        r=20,
        t=45,
        b=70,
    ),
    font=dict(
        family="Arial",
        size=14,
    ),
)


# Subplot titles (top-row tracker names)
for annotation in fig.layout.annotations:
    annotation.update(
        font=dict(
            family="Arial",
            size=22,
        ),
        xanchor="center",
    )


# Per-row y-axis titles (column 1 only)
for row_idx, metric in enumerate(
    metrics,
    start=1,
):
    fig.update_yaxes(
        title_text=f"<b>{metric['ylabel']}</b>",
        title_font=dict(size=20),
        tickfont=dict(size=16),
        row=row_idx,
        col=1,
    )


# Tick fonts on all y axes
for row_idx in range(1, 4):
    for col_idx in range(
        1,
        len(TRACKERS) + 1,
    ):
        fig.update_yaxes(
            tickfont=dict(size=16),
            row=row_idx,
            col=col_idx,
        )


# X tick labels only on the bottom row
for col_idx in range(
    1,
    len(TRACKERS) + 1,
):
    fig.update_xaxes(
        showticklabels=False,
        row=1,
        col=col_idx,
    )

    fig.update_xaxes(
        showticklabels=False,
        row=2,
        col=col_idx,
    )

    fig.update_xaxes(
        tickfont=dict(size=16),
        row=3,
        col=col_idx,
    )


# --------------------------------------------------------------------------- #
# Show and save
# --------------------------------------------------------------------------- #
show_figure(fig)

fig.write_image(
    root_path / "balance_sway_metrics.svg",
    scale=3,
)

# fig.write_image(
#     root_path / "balance_sway_metrics.png",
#     scale=3,
# )

# fig.write_image(
#     root_path / "balance_sway_metrics.pdf",
# )