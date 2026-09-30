"""
COM velocity violin plots.

Creates split violin plots comparing MediaPipe-derived FreeMoCap COM
velocities against the marker-based Qualisys reference for each balance
condition.

Rows:
    1. Mediolateral COM velocity (X)
    2. Anteroposterior COM velocity (Y)
    3. Vertical COM velocity (Z)

The input balance_velocities.csv files are produced directly by BalanceStep
and contain frame-level XYZ COM velocities for each balance condition.
"""

import pandas as pd
import sqlite3
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from validation.paths import DATABASE_PATH, output_dir



DPI = 300

FIG_W_IN = 2
FIG_H_IN = 3

FIG_W_PX = int(
    FIG_W_IN * DPI
)

FIG_H_PX = int(
    FIG_H_IN * DPI
)

EXPORT_BASENAME = (
    "com_velocity_violin"
)


root_path = output_dir("figures", "balance")


# =========================
# Load velocity artifacts
# =========================

conn = sqlite3.connect(DATABASE_PATH)

query = """
SELECT
    t.participant_code,
    t.trial_name,
    a.path,
    a.component_name,
    a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN (
      "mediapipe",
      "qualisys"
  )
  AND a.file_exists = 1
  AND a.component_name IN (
      "freemocap_balance_velocities",
      "qualisys_balance_velocities"
  )
ORDER BY
    t.participant_code,
    t.trial_name,
    a.tracker;
"""


path_df = pd.read_sql_query(
    query,
    conn,
)

conn.close()


if path_df.empty:
    raise RuntimeError(
        "No balance velocity artifacts were found in validation.db. "
        "Run the balance pipeline and rebuild the database first."
    )


# =========================
# Load CSVs
# =========================

dfs = []


for _, row in path_df.iterrows():

    sub_df = pd.read_csv(
        row["path"]
    )

    sub_df["participant_code"] = (
        row["participant_code"]
    )

    sub_df["trial_name"] = (
        row["trial_name"]
    )

    sub_df["tracker"] = (
        row["tracker"]
    )

    dfs.append(
        sub_df
    )


velocity_df = pd.concat(
    dfs,
    ignore_index=True,
)


# =========================
# Validate expected schema
# =========================

required_columns = {
    "condition",
    "frame",
    "velocity_x_mm_s",
    "velocity_y_mm_s",
    "velocity_z_mm_s",
}


missing_columns = (
    required_columns
    - set(velocity_df.columns)
)


if missing_columns:
    raise ValueError(
        "balance_velocities.csv is missing expected columns: "
        f"{sorted(missing_columns)}"
    )


# =========================
# Convert XYZ columns to long form
# =========================

velocity_columns = {
    "velocity_x_mm_s": "x",
    "velocity_y_mm_s": "y",
    "velocity_z_mm_s": "z",
}


id_cols = [
    "participant_code",
    "trial_name",
    "tracker",
    "condition",
    "frame",
]


# Keep the human-readable label if it is present.
if "label" in velocity_df.columns:
    id_cols.append(
        "label"
    )


long_df = velocity_df.melt(
    id_vars=id_cols,
    value_vars=list(
        velocity_columns.keys()
    ),
    var_name="velocity_component",
    value_name="velocity",
)


long_df["axis"] = (
    long_df["velocity_component"]
    .map(velocity_columns)
)


long_df = long_df.dropna(
    subset=["velocity"]
)


# =========================
# Plot configuration
# =========================

colors = {
    "qualisys": "#7A7A7A",
    "mediapipe": "#014E9C",
}


condition_order = [
    "eyes_open_solid",
    "eyes_closed_solid",
    "eyes_open_foam",
    "eyes_closed_foam",
]


tickvals = condition_order


ticktext = [
    "Eyes Open<br>Solid Ground",
    "Eyes Closed<br>Solid Ground",
    "Eyes Open<br>Foam",
    "Eyes Closed<br>Foam",
]


axis_order = [
    "x",
    "y",
    "z",
]


axis_titles = {
    "x": (
        "Mediolateral center-of-mass "
        "velocity (X)"
    ),
    "y": (
        "Anteroposterior center-of-mass "
        "velocity (Y)"
    ),
    "z": (
        "Vertical center-of-mass "
        "velocity (Z)"
    ),
}


# Enforce condition ordering globally.
long_df["condition"] = pd.Categorical(
    long_df["condition"],
    categories=condition_order,
    ordered=True,
)


# =========================
# Build combined figure
# =========================

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[
        axis_titles[axis]
        for axis in axis_order
    ],
)


for row_idx, axis in enumerate(
    axis_order,
    start=1,
):

    df_axis = long_df[
        long_df["axis"] == axis
    ].copy()


    # -------------------
    # Qualisys / reference
    # -------------------

    df_qs = df_axis[
        df_axis["tracker"]
        == "qualisys"
    ]


    fig.add_trace(
        go.Violin(
            x=df_qs["condition"],
            y=df_qs["velocity"],
            legendgroup="qualisys",
            scalegroup=f"axis_{axis}",
            name="Qualisys",
            side="negative",
            line_color=colors[
                "qualisys"
            ],
            width=0.48,
            showlegend=(
                row_idx == 1
            ),
            opacity=0.60,
            spanmode="hard",
        ),
        row=row_idx,
        col=1,
    )


    # -------------------
    # MediaPipe / FreeMoCap
    # -------------------

    df_fmc = df_axis[
        df_axis["tracker"]
        == "mediapipe"
    ]


    fig.add_trace(
        go.Violin(
            x=df_fmc["condition"],
            y=df_fmc["velocity"],
            legendgroup="freemocap",
            scalegroup=f"axis_{axis}",
            name="FreeMoCap (MediaPipe)",
            side="positive",
            line_color=colors[
                "mediapipe"
            ],
            width=0.48,
            showlegend=(
                row_idx == 1
            ),
            opacity=0.80,
            spanmode="hard",
        ),
        row=row_idx,
        col=1,
    )


    fig.update_yaxes(
        title_text=(
            "COM velocity (mm/s)"
        ),
        row=row_idx,
        col=1,
    )


# =========================
# Violin styling
# =========================

fig.update_traces(
    box_visible=True,
    meanline_visible=True,
    points=False,
    scalemode="width",
    meanline=dict(
        width=2
    ),
)


# =========================
# Figure layout
# =========================

fig.update_layout(
    template="simple_white",
    width=FIG_W_PX,
    height=FIG_H_PX,
    margin=dict(
        l=80,
        r=20,
        t=80,
        b=85,
    ),
    font=dict(
        size=12
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.12,
        xanchor="center",
        x=0.5,
        title_text="",
    ),
)


# =========================
# Axes
# =========================

# Only show condition labels on bottom row.
fig.update_xaxes(
    showticklabels=False,
    row=1,
    col=1,
)

fig.update_xaxes(
    showticklabels=False,
    row=2,
    col=1,
)

fig.update_xaxes(
    title_text="Condition",
    row=3,
    col=1,
)


# Preserve the original paper figure's fixed velocity limits.
fig.update_yaxes(
    range=[-75, 75],
    row=1,
    col=1,
)

fig.update_yaxes(
    range=[-75, 75],
    row=2,
    col=1,
)

fig.update_yaxes(
    range=[-75, 75],
    row=3,
    col=1,
)


# Ensure consistent condition order on all panels.
fig.update_xaxes(
    categoryorder="array",
    categoryarray=condition_order,
)


fig.update_xaxes(
    row=3,
    col=1,
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    tickfont=dict(
        size=12
    ),
    automargin=True,
)


fig.update_xaxes(
    tickangle=0
)


# =========================
# Show / export
# =========================

# Uncomment for interactive inspection.
# fig.show()


fig.write_image(
    root_path
    / f"{EXPORT_BASENAME}.png",
    width=FIG_W_PX,
    height=FIG_H_PX,
    scale=3,
)


# Optional PDF export.
# fig.write_image(
#     root_path
#     / f"{EXPORT_BASENAME}.pdf",
#     width=FIG_W_PX,
#     height=FIG_H_PX,
#     scale=1,
# )


print(
    "Figure written to "
    f"{root_path / f'{EXPORT_BASENAME}.png'}"
)