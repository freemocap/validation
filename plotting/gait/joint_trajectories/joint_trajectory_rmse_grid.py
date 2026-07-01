import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
JOINT_ORDER = ["hip", "knee", "ankle", "foot_index"]
JOINT_LABELS = {"hip": "Hip", "knee": "Knee", "ankle": "Ankle", "foot_index": "Toe"}

TRACKER_ORDER = ["mediapipe", "rtmpose", "vitpose"]
TRACKER_LABELS = {
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}

AXIS_ORDER = ["x", "y", "z"]
AXIS_LABELS = {"x": "ML", "y": "AP", "z": "Vertical"}

AXIS_STYLE = {
    "x": {"name": "X (Mediolateral)",      "color": "#0072B2", "symbol": "square",  "dash": "solid"},
    "y": {"name": "Y (Anteroposterior)",    "color": "#D55E00", "symbol": "square",  "dash": "solid"},
    "z": {"name": "Z (Vertical)",           "color": "#006D43", "symbol": "diamond", "dash": "solid"},
}

LINE_WIDTH = 2


# ── Helpers ───────────────────────────────────────────────────────────
def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def parse_speed_float(cond: str) -> float:
    s = str(cond).replace("speed_", "").replace("_", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


# ── Figure sizing ─────────────────────────────────────────────────────
SUBPLOT_WIDTH_IN = 2.2
SUBPLOT_HEIGHT_IN = 1.6
DPI = 100

MARGIN_LEFT_IN = 1.2
MARGIN_RIGHT_IN = 0.3
MARGIN_TOP_IN = 0.7
MARGIN_BOTTOM_IN = 0.8

V_SPACING = 0.06
H_SPACING = 0.02

n_rows = len(JOINT_ORDER)    # 4 joints
n_cols = len(TRACKER_ORDER)  # 3 trackers

FIG_WIDTH_PX = int((MARGIN_LEFT_IN + n_cols * SUBPLOT_WIDTH_IN + MARGIN_RIGHT_IN) * DPI)
FIG_HEIGHT_PX = int((MARGIN_TOP_IN + n_rows * SUBPLOT_HEIGHT_IN + MARGIN_BOTTOM_IN) * DPI)


def generate_trajectory_rmse_grid(
    df: pd.DataFrame,
    title: str = "Trajectory RMSE trends by joint and tracker",
    show: bool = True,
    save_path: str | Path | None = None,
) -> go.Figure:
    """
    4×3 grid: rows = joints, columns = trackers.
    Each panel has 3 lines (ML / AP / Vertical) with error bars.
    All panels share the same y-axis range for direct comparison.
    """
    df = df.copy()
    if "speed" not in df.columns:
        df["speed"] = df["condition"].map(parse_speed_float)

    speeds = sorted(df["speed"].dropna().unique())

    # ── Subplot grid ──────────────────────────────────────────────
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=V_SPACING,
        horizontal_spacing=H_SPACING,
        column_titles=[f"<b>{TRACKER_LABELS[t]}</b>" for t in TRACKER_ORDER],
    )

    # ── Global y-range tracking ───────────────────────────────────
    y_min, y_max = np.inf, -np.inf

    # ── Traces ────────────────────────────────────────────────────
    for r_idx, joint in enumerate(JOINT_ORDER):
        row = r_idx + 1
        for c_idx, tracker in enumerate(TRACKER_ORDER):
            col = c_idx + 1
            panel_data = df[(df["tracker"] == tracker) & (df["joint"] == joint)]

            for axis in AXIS_ORDER:
                style = AXIS_STYLE[axis]
                adata = (
                    panel_data[panel_data["axis"] == axis]
                    .sort_values("speed")
                )
                if adata.empty:
                    continue

                means = adata["mean"].to_numpy()
                stds = adata["std"].to_numpy()

                upper = means + stds
                lower = means - stds
                y_min = min(y_min, np.nanmin(lower))
                y_max = max(y_max, np.nanmax(upper))

                show_legend = (r_idx == 0 and c_idx == 0)

                fig.add_trace(
                    go.Scatter(
                        x=adata["speed"],
                        y=means,
                        error_y=dict(
                            type="data",
                            array=stds,
                            visible=True,
                            color=rgba(style["color"], 0.45),
                            thickness=1.2,
                            width=3,
                        ),
                        mode="lines+markers",
                        name=style["name"],
                        legendgroup=axis,
                        showlegend=show_legend,
                        line=dict(
                            color=rgba(style["color"], 0.85),
                            width=LINE_WIDTH,
                            dash=style["dash"],
                        ),
                        marker=dict(
                            color=style["color"],
                            size=6,
                            symbol=style["symbol"],
                            line=dict(width=0),
                        ),
                        hovertemplate=(
                            f"<b>{TRACKER_LABELS[tracker]} — {JOINT_LABELS[joint]}</b><br>"
                            f"{style['name']}<br>"
                            f"Speed: %{{x:.1f}} m/s<br>"
                            f"RMSE: %{{y:.1f}} ± %{{error_y.array:.1f}} mm"
                            f"<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

    # ── Shared y-range with padding ───────────────────────────────
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    y_range = [max(0, y_min - pad), y_max + pad]

    # ── Row labels (joint names on the left) ──────────────────────
    for r_idx, joint in enumerate(JOINT_ORDER):
        row = r_idx + 1
        label = JOINT_LABELS[joint]
        fig.add_annotation(
            x=-0.06,
            xref="paper",
            y=1 - (row - 0.5) / n_rows,
            yref="paper",
            text=f"<b>{label}</b><br>RMSE (mm)",
            showarrow=False,
            xanchor="right",
            font=dict(size=12, color="#333"),
            align="right",
        )

    # ── Axis formatting ───────────────────────────────────────────
    speed_ticks = list(speeds)
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            is_bottom = (r == n_rows)
            is_left = (c == 1)

            fig.update_xaxes(
                title_text="<b>Speed (m/s)</b>" if is_bottom else None,
                title_font=dict(size=12, color="#333"),
                title_standoff=5,
                tickvals=speed_ticks,
                ticktext=[f"{s:g}" for s in speed_ticks],
                tickfont=dict(size=11),
                showticklabels=is_bottom,
                range=[min(speeds) - 0.2, max(speeds) + 0.2],
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="#333",
                mirror=True,
                row=r, col=c,
            )

            fig.update_yaxes(
                range=y_range,
                showticklabels=is_left,
                tickfont=dict(size=11),
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="#333",
                mirror=True,
                row=r, col=c,
            )

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_white",
        # title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14, color="#333")),
        width=FIG_WIDTH_PX,
        height=FIG_HEIGHT_PX,
        margin=dict(
            l=int(MARGIN_LEFT_IN * DPI),
            r=int(MARGIN_RIGHT_IN * DPI),
            t=int(MARGIN_TOP_IN * DPI),
            b=int(MARGIN_BOTTOM_IN * DPI),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
    )

    # Bold column titles
    for ann in fig.layout.annotations:
        if ann.text and any(t in ann.text for t in TRACKER_LABELS.values()):
            ann.font = dict(size=13, color="#333")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.write_image(str(save_path.with_suffix(".svg")), scale=3)
        fig.write_image(str(save_path.with_suffix(".png")), scale=3)
        print(f"Saved: {save_path.with_suffix('.svg')}")
        print(f"Saved: {save_path.with_suffix('.png')}")

    if show:
        fig.show()

    return fig


# ── Direct execution ──────────────────────────────────────────────
if __name__ == "__main__":
    from plotting.gait.joint_trajectories.joint_trajectory_rmse_tables import (
        load_trajectory_summary_stats,
        combine_left_and_right_side,
        calculate_total_mean_and_std_rmse,
    )

    DB_PATH = "validation.db"
    TRACKERS_ALL = ["mediapipe", "rtmpose", "vitpose", "qualisys"]
    REFERENCE_SYSTEM = "qualisys"

    FIGURE_OUT_DIR = Path(r"D:\validation_public_release_v1\figures")
    FIGURE_OUT_DIR.mkdir(exist_ok=True, parents=True)

    database_data = load_trajectory_summary_stats(DB_PATH)
    df_trial_lr_mean = combine_left_and_right_side(database_data, JOINT_ORDER)
    df_total = calculate_total_mean_and_std_rmse(
        df_trial_lr_mean,
        tracker_list=TRACKERS_ALL,
        reference_system=REFERENCE_SYSTEM,
    )


    generate_trajectory_rmse_grid(
        df_total,
        save_path=FIGURE_OUT_DIR / "trajectory_rmse_grid",
    )

    print("\nDone!")