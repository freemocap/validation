"""
Joint angle waveform plots (mean ± SD across trials, by speed).

Key importable functions:
  - load_joint_angle_data(db_path) -> combined_df
  - compute_angle_summary(combined_df) -> (angle_summary, df_trial_lr_mean)

When run as a script, produces the angle-only figure + exports joint_angles_summary.csv.
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


TRACKERS = ["mediapipe", "rtmpose", "vitpose", "qualisys"]

MAJOR = {
    ("hip", "flex_ext"),
    ("knee", "flex_ext"),
    ("ankle", "dorsi_plantar"),
}

JOINT_ORDER = ["hip", "knee", "ankle"]

COMP_LABEL = {
    "flex_ext": "Flex/Ext",
    "dorsi_plantar": "Dorsi/Plantar",
}



def speed_key(cond: str) -> float:
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


def rgba(hex_color, alpha):
    """Convert hex to rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"



def load_joint_angle_data(db_path: str = "validation.db") -> pd.DataFrame:
    """
    Load joint angle per-stride summary_stats from validation.db,
    normalize strings, and filter to major sagittal motions.
    """
    conn = sqlite3.connect(db_path)
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
      AND a.category = "joint_angles_per_stride"
      AND a.tracker IN ("mediapipe", "rtmpose", "vitpose", "qualisys")
      AND a.file_exists = 1
      AND a.condition LIKE "speed_%"
      AND a.component_name LIKE "%summary_stats"
    ORDER BY t.trial_name, a.path
    """
    path_df = pd.read_sql_query(query, conn)
    conn.close()

    if path_df.empty:
        raise RuntimeError("No matching joint angle summary_stats artifacts found.")

    dfs = []
    for _, row in path_df.iterrows():
        sub = pd.read_csv(row["path"])
        sub["participant_code"] = row["participant_code"]
        sub["trial_name"] = row["trial_name"]
        sub["tracker"] = (row["tracker"] or "").lower()
        sub["condition"] = row["condition"] or "none"
        dfs.append(sub)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Normalize strings
    for col in ["joint", "side", "tracker", "stat", "component"]:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).str.lower()

    combined_df["component"] = combined_df["component"].replace(
        {"inversion_eversion": "inv_ev"}
    )

    # Keep only major sagittal motions
    combined_df = combined_df[
        combined_df.apply(lambda r: (r["joint"], r["component"]) in MAJOR, axis=1)
    ].copy()

    return combined_df


def compute_angle_summary(
    combined_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse L/R sides within trial, then summarize (mean ± SD) across trials.

    Returns:
      angle_summary:    group-level mean/std/n per (condition, tracker, joint, component, gait%)
      df_trial_lr_mean: trial-level L/R-averaged waveforms (needed by SPM)
    """
    df_means = combined_df[combined_df["stat"] == "mean"].copy()

    df_trial_lr_mean = (
        df_means.groupby(
            [
                "condition", "tracker", "participant_code", "trial_name",
                "joint", "component", "percent_gait_cycle",
            ],
            as_index=False,
        )
        .agg(trial_mean_angle=("value", "mean"))
    )

    knee_mask = df_trial_lr_mean["joint"] == "knee"
    df_trial_lr_mean.loc[knee_mask, "trial_mean_angle"] *= -1

    angle_summary = (
        df_trial_lr_mean.groupby(
            ["condition", "tracker", "joint", "component", "percent_gait_cycle"],
            as_index=False,
        )
        .agg(
            mean_angle=("trial_mean_angle", "mean"),
            std_angle=("trial_mean_angle", "std"),
            n_trials=("trial_name", "nunique"),
        )
    )


    return angle_summary, df_trial_lr_mean


if __name__ == "__main__":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    root_dir = Path(r"D:\validation\gait\joint_angles")
    root_dir.mkdir(exist_ok=True, parents=True)

    combined_df = load_joint_angle_data()
    angle_summary, df_trial_lr_mean = compute_angle_summary(combined_df)

    SPEEDS = sorted(combined_df["condition"].unique().tolist(), key=speed_key)

    print(
        df_trial_lr_mean.groupby(["condition", "tracker"])["trial_name"]
        .nunique()
        .unstack(fill_value=0)
    )

    # ------------------------
    # Publication-ready styling
    # ------------------------
    SUBPLOT_WIDTH_IN = 1.5
    SUBPLOT_HEIGHT_IN = 1.5
    DPI = 100

    MARGIN_LEFT_IN = 1.5
    MARGIN_RIGHT_IN = 0.2
    MARGIN_TOP_IN = 0.6
    MARGIN_BOTTOM_IN = 0.7

    LINE_WIDTH = 2
    SD_OPACITY = 0.12

    TRACKER_STYLE = {
        "qualisys": {
            "name": "Qualisys",
            "color": "#313131",
            "dash": "solid",
            "width": 1.5,
            "fill_opacity": 0.3,
            "line_opacity": 0.5,
        },
        "mediapipe": {
            "name": "MediaPipe",
            "color": "#0072B2",
            "dash": "solid",
            "width": LINE_WIDTH,
            "fill_opacity": SD_OPACITY,
            "line_opacity": 0.7,
        },
        "rtmpose": {
            "name": "RTMPose",
            "color": "#D55E00",
            "dash": "solid",
            "width": LINE_WIDTH,
            "fill_opacity": SD_OPACITY,
            "line_opacity": 0.7,
        },
        "vitpose": {
            "name": "ViTPose",
            "color": "#006D43",
            "dash": "solid",
            "width": LINE_WIDTH,
            "fill_opacity": SD_OPACITY,
            "line_opacity": 0.7,
        },
    }

    DRAW_ORDER = ["qualisys", "mediapipe", "rtmpose", "vitpose"]

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

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=V_SPACING,
        horizontal_spacing=H_SPACING,
        column_titles=[speed_label(s) for s in SPEEDS],
    )

    y_minmax = {j: [np.inf, -np.inf] for j in JOINT_ORDER}

    for cond_idx, cond in enumerate(SPEEDS, start=1):
        for joint in JOINT_ORDER:
            component = "flex_ext" if joint in ("hip", "knee") else "dorsi_plantar"
            row = JOINT_ORDER.index(joint) + 1
            col = cond_idx

            for tracker in DRAW_ORDER:
                if tracker not in TRACKERS:
                    continue
                style = TRACKER_STYLE[tracker]

                sub = angle_summary[
                    (angle_summary["condition"] == cond)
                    & (angle_summary["tracker"] == tracker)
                    & (angle_summary["joint"] == joint)
                    & (angle_summary["component"] == component)
                ]
                if sub.empty:
                    continue

                sub = sub.sort_values("percent_gait_cycle")
                x = sub["percent_gait_cycle"].to_numpy()
                mean = sub["mean_angle"].to_numpy()
                sd = sub["std_angle"].to_numpy()

                if np.all(np.isnan(sd)):
                    sd = np.zeros_like(mean)

                lower, upper = mean - sd, mean + sd

                y_minmax[joint][0] = min(y_minmax[joint][0], np.nanmin(lower))
                y_minmax[joint][1] = max(y_minmax[joint][1], np.nanmax(upper))

                fill_color = rgba(style["color"], style["fill_opacity"])
                line_color = rgba(style["color"], style.get("line_opacity", 1.0))

                # SD ribbon
                fig.add_trace(
                    go.Scatter(
                        x=x, y=upper, mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False, legendgroup=tracker,
                    ),
                    row=row, col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x, y=lower, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=fill_color,
                        hoverinfo="skip", showlegend=False, legendgroup=tracker,
                    ),
                    row=row, col=col,
                )

                # Mean line
                showleg = row == 1 and col == 1
                fig.add_trace(
                    go.Scatter(
                        x=x, y=mean, mode="lines",
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
                            "Angle: %{y:.1f}°<br>"
                            "Gait cycle: %{x:.0f}%<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )

    # Sync y ranges per joint
    for joint in JOINT_ORDER:
        lo, hi = y_minmax[joint]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = -30, 30
        pad = (hi - lo) * 0.08
        rng = [lo - pad, hi + pad]
        r = JOINT_ORDER.index(joint) + 1
        for c in range(1, n_cols + 1):
            fig.update_yaxes(range=rng, row=r, col=c)

    # Row labels and axis formatting
    for joint in JOINT_ORDER:
        component = "flex_ext" if joint in ("hip", "knee") else "dorsi_plantar"
        label = COMP_LABEL[component]
        r = JOINT_ORDER.index(joint) + 1
        fig.add_annotation(
            x=-0.045, xref="paper",
            y=1 - (r - 0.5) / n_rows, yref="paper",
            text=f"<b>{joint.title()}</b><br>{label} (°)",
            showarrow=False, xanchor="right",
            font=dict(size=12, color="#333"), align="right",
        )

    tickvals = list(range(0, 101, 20))
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(
                title_text="<b>Gait cycle (%)</b>" if r == n_rows else None,
                title_font=dict(size=12, color="#333"),
                title_standoff=5, tickvals=tickvals, tickfont=dict(size=9),
                showgrid=False, zeroline=False, showline=True,
                linecolor="#333", mirror=True, row=r, col=c,
            )
            fig.update_yaxes(
                showticklabels=(c == 1), tickfont=dict(size=9),
                showgrid=False, zeroline=False, showline=True,
                linecolor="#333", mirror=True, row=r, col=c,
            )

    fig.update_layout(
        template="plotly_white",
        height=FIG_HEIGHT_PX,
        width=FIG_WIDTH_PX,
        margin=dict(l=MARGIN_LEFT_PX, r=MARGIN_RIGHT_PX, t=MARGIN_TOP_PX, b=MARGIN_BOTTOM_PX),
        title=dict(
            text="<b>Sagittal Plane Joint Angles Across Treadmill Speeds</b>",
            font=dict(size=14),
            y=0.97, x=0.5, xanchor="center", yanchor="top",
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.12,
            xanchor="center", x=0.5, font=dict(size=11), itemwidth=40,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    for annotation in fig.layout.annotations:
        if "m/s" in annotation.text:
            annotation.text = f"<b>{annotation.text}</b>"
            annotation.font = dict(size=11, color="#333")

    fig.show()

    # Export
    angle_summary.to_csv(root_dir / "joint_angles_summary.csv", index=False)
    fig.write_image(root_dir / "joint_angles_by_speed.png", scale=3)
    print("Saved:", root_dir / "joint_angles_summary.csv")