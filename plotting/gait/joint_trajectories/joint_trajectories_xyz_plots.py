import re
import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path
save_root = Path(r"D:\validation_public_release_v1\figures")
save_root.mkdir(exist_ok=True, parents=True)

# ------------------------
# 1) Load data from SQLite
# ------------------------

TRACKERS = ["mediapipe", "rtmpose", "vitpose", "qualisys"]
joint_to_plot = ["hip", "knee", "ankle", "foot_index"]


conn = sqlite3.connect("validation.db")
query = """
SELECT t.participant_code,
       t.trial_name,
       a.path,
       a.component_name,
       a.condition,
       a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "treadmill"
  AND a.category = "trajectories_per_stride"
  AND a.tracker IN ("mediapipe", "rtmpose", "vitpose", "qualisys")
  AND a.file_exists = 1
  AND a.condition LIKE "speed_%"
  AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""
reference_system = "qualisys"

path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"]
    sub["tracker"] = (row["tracker"] or "").lower()
    sub["condition"] = row["condition"] or "none"
    dfs.append(sub)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df = combined_df.copy()

# normalize marker strings first
m = combined_df["marker"].astype(str).str.strip().str.lower()

# side
combined_df["side"] = np.select(
    [m.str.startswith("left_"), m.str.startswith("right_")],
    ["left", "right"],
    default="unknown"
)

# joint = marker name without side prefix
combined_df["joint"] = (
    m
    .str.replace(r"^(left_|right_)", "", regex=True)
)


combined_df["value_mirrored"] = combined_df["value"]

ml_left_mean_mask = (
    (combined_df["axis"] == "x") &
    (combined_df["side"] == "left") &
    (combined_df["stat"] == "mean")
)


combined_df.loc[ml_left_mean_mask, "value_mirrored"] *= -1


df_means = combined_df[combined_df["stat"] == "mean"].copy()

df_trial_lr_mean = (
    df_means
    .groupby(
        ["condition", "tracker", "participant_code", "trial_name", "joint", "axis", "percent_gait_cycle"],
        as_index = False
    )
    .agg(trial_mean_value = ("value_mirrored", "mean"))
)

mean_summary = (
    df_trial_lr_mean
    .groupby(
        ["condition", "tracker", "joint", "axis", "percent_gait_cycle"],
        as_index = False
    )
    .agg(
        mean_value = ("trial_mean_value", "mean"),
        sd_value = ("trial_mean_value", "std"),
    )
)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------
# Helpers (reuse yours)
# ------------------------
def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
def speed_key(cond: str) -> float:
    """
    Parse speed from strings like:
    - 'speed_0_5' -> 0.5
    - 'speed_1_0' -> 1.0
    - 'speed_2'   -> 2.0
    """
    if cond is None:
        return float("inf")

    m = re.search(r"speed_(\d+)[_\.](\d+)", str(cond))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")

    m2 = re.search(r"speed_(\d+)", str(cond))
    if m2:
        return float(m2.group(1))

    return float("inf")
def speed_label(cond: str) -> str:
    k = speed_key(cond)
    return "?" if not np.isfinite(k) else f"{k:g} m/s"
# ------------------------
# Choose what to plot
# ------------------------
TRACKERS = ["mediapipe", "rtmpose", "vitpose", "qualisys"]
DRAW_ORDER = ["qualisys", "mediapipe", "rtmpose", "vitpose"]  # ref on top is fine
JOINT_ORDER = ["hip", "knee", "ankle", "foot_index"]  # or whatever you want
SPEEDS = sorted(combined_df["condition"].unique().tolist(), key=speed_key)

SPEEDS = ['speed_0_5', 'speed_1_0', 'speed_1_5', 'speed_2_0', 'speed_2_5']

AXES_TO_PLOT = ["x", "y", "z"]  # x=ML (mirrored), y=AP, z=Vertical

# Optional nicer axis labels for titles
AXIS_LABEL = {
    "x": "ML (mirrored)",
    "y": "AP",
    "z": "Vertical",
}

# Style dict (reuse yours)
LINE_WIDTH = 2
SD_OPACITY = 0.12
TRACKER_STYLE = {
    "qualisys": {"name": "Reference", "color": "#313131", "dash": "solid",
                 "width": 1.5, "fill_opacity": 0.12, "line_opacity": 0.90},
    "mediapipe": {"name": "MediaPipe", "color": "#0072B2", "dash": "solid",
                  "width": LINE_WIDTH, "fill_opacity": SD_OPACITY, "line_opacity": 0.6},
    "rtmpose": {"name": "RTMPose", "color": "#D55E00", "dash": "solid",
                "width": LINE_WIDTH, "fill_opacity": SD_OPACITY, "line_opacity": 0.6},
    "vitpose": {"name": "ViTPose", "color": "#006D43", "dash": "solid",
                "width": LINE_WIDTH, "fill_opacity": SD_OPACITY, "line_opacity": 0.6},
}

# ------------------------
# Figure sizing (reuse yours)
# ------------------------
SUBPLOT_WIDTH_IN = 1.5
SUBPLOT_HEIGHT_IN = 1.5
DPI = 100

MARGIN_LEFT_IN = 1.6
MARGIN_RIGHT_IN = 0.2
MARGIN_TOP_IN = 0.7
MARGIN_BOTTOM_IN = 0.6

V_SPACING = 0.08
H_SPACING = 0.015

n_rows = len(JOINT_ORDER)
n_cols = len(SPEEDS)

FIG_WIDTH_IN = MARGIN_LEFT_IN + (n_cols * SUBPLOT_WIDTH_IN) + MARGIN_RIGHT_IN
FIG_HEIGHT_IN = MARGIN_TOP_IN + (n_rows * SUBPLOT_HEIGHT_IN) + MARGIN_BOTTOM_IN

FIG_WIDTH_PX = int(FIG_WIDTH_IN * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)

MARGIN_LEFT_PX = int(MARGIN_LEFT_IN * DPI)
MARGIN_RIGHT_PX = int(MARGIN_RIGHT_IN * DPI)
MARGIN_TOP_PX = int(MARGIN_TOP_IN * DPI)
MARGIN_BOTTOM_PX = int(MARGIN_BOTTOM_IN * DPI)

# ------------------------
# Build one figure per axis
# ------------------------
for axis in AXES_TO_PLOT:

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=V_SPACING,
        horizontal_spacing=H_SPACING,
        column_titles=[speed_label(s) for s in SPEEDS],
    )

    # Track y ranges per joint (so each row is consistent across speeds)
    y_minmax = {j: [np.inf, -np.inf] for j in JOINT_ORDER}

    # ------------------------
    # Traces
    # ------------------------
    for cond_idx, cond in enumerate(SPEEDS, start=1):
        for joint in JOINT_ORDER:
            row = JOINT_ORDER.index(joint) + 1
            col = cond_idx

            for tracker in DRAW_ORDER:
                if tracker not in TRACKERS:
                    continue

                style = TRACKER_STYLE[tracker]

                sub = mean_summary[
                    (mean_summary["condition"] == cond) &
                    (mean_summary["tracker"] == tracker) &
                    (mean_summary["joint"] == joint) &
                    (mean_summary["axis"] == axis)
                ].sort_values("percent_gait_cycle")

                if sub.empty:
                    continue

                xgc = sub["percent_gait_cycle"].to_numpy()
                mean = sub["mean_value"].to_numpy()
                sd = sub["sd_value"].to_numpy()

                # If sd is missing (e.g., only 1 trial), treat as 0
                if np.all(np.isnan(sd)):
                    sd = np.zeros_like(mean)

                lower, upper = mean - sd, mean + sd

                # update y-range tracking
                y_minmax[joint][0] = min(y_minmax[joint][0], np.nanmin(lower))
                y_minmax[joint][1] = max(y_minmax[joint][1], np.nanmax(upper))

                fill_color = rgba(style["color"], style["fill_opacity"])
                line_color = rgba(style["color"], style.get("line_opacity", 1.0))

                # SD ribbon (upper then lower with fill=tonexty)
                fig.add_trace(
                    go.Scatter(
                        x=xgc, y=upper, mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=tracker,
                    ),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=xgc, y=lower, mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=fill_color,
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=tracker,
                    ),
                    row=row, col=col
                )

                # Mean line
                showleg = (row == 1 and col == 1)
                fig.add_trace(
                    go.Scatter(
                        x=xgc, y=mean, mode="lines",
                        line=dict(
                            color=line_color,
                            width=style.get("width", LINE_WIDTH),
                            dash=style["dash"],
                        ),
                        name=style["name"],
                        legendgroup=tracker,
                        showlegend=showleg,
                        hovertemplate=(
                            f"{style['name']}<br>"
                            f"{joint} {axis.upper()}<br>"
                            "Value: %{y:.1f} mm<br>"
                            "Gait cycle: %{x:.0f}%<extra></extra>"
                        ),
                    ),
                    row=row, col=col
                )

    # ------------------------
    # Sync y ranges per joint row
    # ------------------------
    for joint in JOINT_ORDER:
        lo, hi = y_minmax[joint]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        pad = (hi - lo) * 0.08 if hi > lo else 1.0
        rng = [lo - pad, hi + pad]
        r = JOINT_ORDER.index(joint) + 1
        for c in range(1, n_cols + 1):
            fig.update_yaxes(range=rng, row=r, col=c)

    # ------------------------
    # Row labels + axis formatting
    # ------------------------
    for joint in JOINT_ORDER:
        r = JOINT_ORDER.index(joint) + 1
        fig.add_annotation(
            x=-0.045, xref="paper",
            y=1 - (r - 0.5) / n_rows, yref="paper",
            text=f"<b>{joint.replace('_',' ').title()} </b> <br>Position (mm)",
            showarrow=False,
            xanchor="right",
            font=dict(size=13, color="#333"),
            align="right",
        ) if not joint.startswith("foot") else fig.add_annotation(
            x=-0.045, xref="paper",
            y=1 - (r - 0.5) / n_rows, yref="paper",
            text=f"<b>{'toe'.title()} </b> <br>Position (mm)</b>",
            showarrow=False,
            xanchor="right",
            font=dict(size=13, color="#333"),
            align="right",
        )

    tickvals = list(range(0, 101, 20))
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(
                title_text="<b>Gait cycle (%)</b>" if r == n_rows else None,
                title_font=dict(size=12, color="#333"),
                title_standoff=5,
                tickvals=tickvals,
                tickfont=dict(size=11),
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="#333",
                mirror=True,
                row=r, col=c
            )
            fig.update_yaxes(
                showticklabels=(c == 1),
                tickfont=dict(size=11),
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="#333",
                mirror=True,
                row=r, col=c
            )

    # ------------------------
    # Layout
    # ------------------------
    fig.update_layout(
        template="plotly_white",
        height=FIG_HEIGHT_PX,
        width=FIG_WIDTH_PX,
        margin=dict(l=MARGIN_LEFT_PX, r=MARGIN_RIGHT_PX, t=MARGIN_TOP_PX, b=MARGIN_BOTTOM_PX),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Bold the column titles
    for ann in fig.layout.annotations:
        if "m/s" in ann.text:
            ann.text = f"<b>{ann.text}</b>"
            ann.font = dict(size=13, color="#333")

    fig.show()

    # Optional exports:
    fig.write_image(save_root / f"trajectories_{axis}.svg", scale=3)
    # fig.write_image(f"trajectories_{axis}.pdf")
