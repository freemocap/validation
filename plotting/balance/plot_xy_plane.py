"""
Representative horizontal-plane COM trajectories and 95% ellipses.

The aligned FreeMoCap/Qualisys parquet files are the canonical source of
the COM trajectories. Balance condition frame intervals are read directly
from the version-controlled trial YAML.

"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from scipy.stats import chi2

from validation.utils.actor_utils import make_freemocap_actor_from_parquet

from validation.paths import DATABASE_PATH, REPO_ROOT, output_dir
# =========================
# Paths
# =========================


root_path = output_dir("figures", "balance")


# =========================
# Paper-ready figure params
# =========================

DPI = 300

FIG_W_IN = 4
FIG_H_IN = 4

FIG_W_PX = int(
    FIG_W_IN * DPI
)

FIG_H_PX = int(
    FIG_H_IN * DPI
)

EXPORT_BASENAME = "com_xy_plane"


# =========================
# Representative trial
# =========================

participant_to_use = "sub-001"

trial_config_name = (
    "task-balance_trial-02.yaml"
)

config_path = (
    REPO_ROOT
    / "configs"
    / participant_to_use
    / trial_config_name
)


# =========================
# Tracker configuration
# =========================

tracker_order = [
    "qualisys",
    "mediapipe",
    "rtmpose",
    "vitpose",
]

tracker_labels = {
    "qualisys": "Reference",
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}

tracker_colors = {
    "qualisys": "black",
    "mediapipe": "#006DFC",
    "rtmpose": "#EB7303",
    "vitpose": "#05C936",
}


# =========================
# Condition configuration
# =========================

condition_order = [
    "eyes_open_solid",
    "eyes_closed_solid",
    "eyes_open_foam",
    "eyes_closed_foam",
]

short_titles = {
    "eyes_open_solid": "EO / Solid",
    "eyes_closed_solid": "EC / Solid",
    "eyes_open_foam": "EO / Foam",
    "eyes_closed_foam": "EC / Foam",
}


# =========================
# Helpers
# =========================

def prediction_ellipse_95(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Calculate the 95% prediction ellipse for centered X/Y COM data.
    """

    cov_matrix = np.cov(
        x,
        y,
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            cov_matrix
        )
    )

    order = (
        eigenvalues
        .argsort()[::-1]
    )

    eigenvalues = (
        eigenvalues[order]
    )

    eigenvectors = (
        eigenvectors[:, order]
    )

    chi2_val = chi2.ppf(
        0.95,
        df=2,
    )

    a = np.sqrt(
        eigenvalues[0]
        * chi2_val
    )

    b = np.sqrt(
        eigenvalues[1]
        * chi2_val
    )

    area = (
        np.pi
        * a
        * b
    )

    theta = np.arctan2(
        eigenvectors[1, 0],
        eigenvectors[0, 0],
    )

    t = np.linspace(
        0,
        2 * np.pi,
        100,
    )

    ellipse_x = (
        a
        * np.cos(t)
    )

    ellipse_y = (
        b
        * np.sin(t)
    )

    rotation_matrix = np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta),
            ],
        ]
    )

    ellipse_points = (
        rotation_matrix
        @ np.array(
            [
                ellipse_x,
                ellipse_y,
            ]
        )
    )

    return (
        ellipse_points,
        a,
        b,
        theta,
        area,
    )


def load_total_body_com(
    parquet_path: Path,
) -> np.ndarray:
    """
    Load total-body COM from an aligned FreeMoCap-format parquet.

    If COM is not already stored in the parquet, calculate it using
    the same actor calculation path used by BalanceStep.
    """

    actor = (
        make_freemocap_actor_from_parquet(
            parquet_path=parquet_path
        )
    )

    if actor.body.total_body_com is None:
        actor.calculate()

    return (
        actor
        .body
        .total_body_com
        .as_array
    )


# =========================
# Load trial YAML
# =========================

if not config_path.exists():
    raise FileNotFoundError(
        f"Trial config not found: {config_path}"
    )


with config_path.open(
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(
        file
    )


trial_path = config.get(
    "trial_path"
)

if not trial_path:
    raise ValueError(
        f"No trial_path found in {config_path}"
    )


# Database uses:
#
# sub-001_task-balance_trial-02
#
# while YAML uses:
#
# sub-001/task-balance_trial-02

trial_name_to_use = (
    trial_path
    .replace("/", "_")
    .replace("\\", "_")
)


project_config = (
    config.get("ProjectConfig")
    or {}
)

conditions = (
    project_config.get("conditions")
    or {}
)


missing_conditions = [
    condition
    for condition in condition_order
    if condition not in conditions
]


if missing_conditions:
    raise ValueError(
        "Trial config is missing expected balance conditions: "
        f"{missing_conditions}"
    )


print(
    f"Representative trial: {trial_name_to_use}"
)

print(
    f"Config: {config_path}"
)


# =========================
# Load aligned parquet paths
# from database
# =========================

conn = sqlite3.connect(
    DATABASE_PATH
)


query = """
SELECT
    t.participant_code,
    t.trial_name,
    a.path,
    a.component_name,
    a.tracker
FROM artifacts a
JOIN trials t
    ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND t.participant_code = ?
  AND t.trial_name = ?
  AND a.category = "synced_data"
  AND a.tracker IN (
      "mediapipe",
      "qualisys",
      "rtmpose",
      "vitpose"
  )
  AND a.file_exists = 1
  AND a.component_name IN (
      "freemocap_parquet",
      "qualisys_parquet"
  )
ORDER BY
    a.tracker;
"""


path_df = pd.read_sql_query(
    query,
    conn,
    params=[
        participant_to_use,
        trial_name_to_use,
    ],
)


conn.close()


if path_df.empty:
    raise RuntimeError(
        "No aligned parquet artifacts were found for "
        f"{trial_name_to_use}. "
        "Rebuild validation.db and confirm the aligned parquet "
        "files exist."
    )


# Make sure every expected tracker is present.

found_trackers = set(
    path_df["tracker"]
)


missing_trackers = (
    set(tracker_order)
    - found_trackers
)


if missing_trackers:
    raise RuntimeError(
        "Missing aligned parquet artifacts for trackers: "
        f"{sorted(missing_trackers)}"
    )


# =========================
# Load COM from each parquet
# =========================

com_by_tracker = {}


for tracker in tracker_order:

    tracker_rows = path_df[
        path_df["tracker"] == tracker
    ]


    if len(tracker_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one aligned parquet for "
            f"{tracker!r}, found {len(tracker_rows)}."
        )


    parquet_path = Path(
        tracker_rows.iloc[0]["path"]
    )


    print(
        f"Loading {tracker}: {parquet_path}"
    )


    com_by_tracker[tracker] = (
        load_total_body_com(
            parquet_path
        )
    )


# =========================
# Build centered condition data
# =========================

plot_data = {}


for tracker in tracker_order:

    com_data = (
        com_by_tracker[tracker]
    )


    for condition in condition_order:

        condition_info = (
            conditions[condition]
        )

        start_frame, end_frame = (
            condition_info["frames"]
        )


        # IMPORTANT:
        #
        # Preserve the historical position/ellipse slicing exactly.
        # The old condition_positions.csv used:
        #
        #     com_data[start_frame:end_frame]
        #
        # Unlike path length and velocity, there is NO end_frame - 1
        # adjustment here.

        condition_com = (
            com_data[
                start_frame:end_frame
            ]
        )


        # total_body_com has shape:
        #
        #     (frames, 1, 3)
        #
        # X = mediolateral
        # Y = anteroposterior

        x_raw = (
            condition_com[
                :,
                0,
                0,
            ]
        )

        y_raw = (
            condition_com[
                :,
                0,
                1,
            ]
        )


        # Center each condition around its own mean,
        # matching the historical figure.

        x = (
            x_raw
            - np.mean(x_raw)
        )

        y = (
            y_raw
            - np.mean(y_raw)
        )


        (
            ellipse_points,
            a,
            b,
            theta,
            area,
        ) = prediction_ellipse_95(
            x,
            y,
        )


        plot_data[
            (
                tracker,
                condition,
            )
        ] = {
            "x": x,
            "y": y,
            "start_x": x[0],
            "start_y": y[0],
            "ellipse_x": (
                ellipse_points[0]
            ),
            "ellipse_y": (
                ellipse_points[1]
            ),
            "area": area,
        }


# =========================
# Shared symmetric axis limits
# =========================

all_values = np.concatenate(
    [
        plot_data[key]["x"]
        for key in plot_data
    ]
    + [
        plot_data[key]["y"]
        for key in plot_data
    ]
    + [
        plot_data[key]["ellipse_x"]
        for key in plot_data
    ]
    + [
        plot_data[key]["ellipse_y"]
        for key in plot_data
    ]
)


axis_limit = np.ceil(
    np.max(
        np.abs(
            all_values
        )
    )
    + 1
)


# =========================
# Plot: 4 rows x 4 columns
# =========================

fig = make_subplots(
    rows=4,
    cols=4,
    horizontal_spacing=0.01,
    vertical_spacing=0.03,
    subplot_titles=[
        short_titles[condition]
        for condition in condition_order
    ],
)


for row_idx, tracker in enumerate(
    tracker_order,
    start=1,
):

    for col_idx, condition in enumerate(
        condition_order,
        start=1,
    ):

        key = (
            tracker,
            condition,
        )

        data = (
            plot_data[key]
        )


        # -------------------
        # COM trajectory
        # -------------------

        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="lines",
                line=dict(
                    width=2,
                    color=(
                        "rgba(150,150,150,0.9)"
                    ),
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_idx,
            col=col_idx,
        )


        # -------------------
        # 95% ellipse
        # -------------------

        fig.add_trace(
            go.Scatter(
                x=data["ellipse_x"],
                y=data["ellipse_y"],
                mode="lines",
                line=dict(
                    width=2,
                    dash="dash",
                    color=(
                        tracker_colors[
                            tracker
                        ]
                    ),
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_idx,
            col=col_idx,
        )


        # -------------------
        # Crosshairs
        # -------------------

        fig.add_hline(
            y=0,
            line_width=0.5,
            line_dash="dot",
            row=row_idx,
            col=col_idx,
        )

        fig.add_vline(
            x=0,
            line_width=0.5,
            line_dash="dot",
            row=row_idx,
            col=col_idx,
        )


        # -------------------
        # Axes
        # -------------------

        fig.update_xaxes(
            range=[
                -axis_limit,
                axis_limit,
            ],
            zeroline=False,
            showticklabels=(
                row_idx == 4
            ),
            row=row_idx,
            col=col_idx,
        )


        axis_number = (
            (row_idx - 1) * 4
            + col_idx
        )


        fig.update_yaxes(
            range=[
                -axis_limit,
                axis_limit,
            ],
            scaleanchor=(
                f"x{axis_number}"
            ),
            scaleratio=1,
            zeroline=False,
            showticklabels=(
                col_idx == 1
            ),
            row=row_idx,
            col=col_idx,
        )


    # Tracker label on leftmost y-axis.

    fig.update_yaxes(
        title_text=(
            f"<b>{tracker_labels[tracker]}</b>"
            "<br>AP (mm)"
        ),
        row=row_idx,
        col=1,
    )


# =========================
# Bottom-row X labels
# =========================

for col_idx in range(
    1,
    5,
):
    fig.update_xaxes(
        title_text="ML (mm)",
        row=4,
        col=col_idx,
    )


# =========================
# Figure layout
# =========================

fig.update_layout(
    width=FIG_W_PX,
    height=FIG_H_PX,
    template="simple_white",
    margin=dict(
        l=60,
        r=20,
        t=40,
        b=50,
    ),
)


for annotation in (
    fig.layout.annotations
):
    annotation.update(
        font=dict(
            size=24
        )
    )


fig.update_xaxes(
    tickfont=dict(
        size=20
    ),
    title_font=dict(
        size=24
    ),
)


fig.update_yaxes(
    tickfont=dict(
        size=20
    ),
    title_font=dict(
        size=24
    ),
)


# =========================
# Show / export
# =========================

fig.show()


fig.write_image(
    root_path
    / f"{EXPORT_BASENAME}.png",
    scale=3,
)


# Optional SVG export.
#
# fig.write_image(
#     root_path
#     / f"{EXPORT_BASENAME}.svg",
#     scale=3,
# )


print(
    "Figure written to "
    f"{root_path / f'{EXPORT_BASENAME}.png'}"
)