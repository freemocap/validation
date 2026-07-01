import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import chi2

# =========================
# Paper-ready figure params
# =========================
DPI = 300
FIG_W_IN = 4
FIG_H_IN = 4
FIG_W_PX = int(FIG_W_IN * DPI)
FIG_H_PX = int(FIG_H_IN * DPI)

EXPORT_BASENAME = "com_xy_plane"
root_path = Path(r"D:\validation_public_release_v1\figures")
root_path.mkdir(exist_ok=True, parents=True)

participant_to_use = "sub-001"
trial_name_to_use = "sub-001_task-balance_trial-02"

tracker_colors = {
    "qualisys": "black",
    "mediapipe": "#006DFC",
    "rtmpose": "#EB7303",
    "vitpose": "#05C936",
}

# -------------------
# Helpers
# -------------------
def prediction_ellipse_95(x, y):
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    chi2_val = chi2.ppf(0.95, df=2)

    a = np.sqrt(eigenvalues[0] * chi2_val)
    b = np.sqrt(eigenvalues[1] * chi2_val)
    area = np.pi * a * b

    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a * np.cos(t)
    ellipse_y = b * np.sin(t)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    ellipse_points = R @ np.array([ellipse_x, ellipse_y])

    return ellipse_points, a, b, theta, area


# -------------------
# Load data from DB
# -------------------
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
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%balance_positions'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)
conn.close()

dfs = []
for _, row in path_df.iterrows():
    sub_df = pd.read_csv(row["path"])
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["condition"] = row.get("condition") or ""
    sub_df["tracker"] = row["tracker"]
    dfs.append(sub_df)

final_df = pd.concat(dfs, ignore_index=True)

conditions = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

short_titles = ["EO / Solid", "EC / Solid", "EO / Foam", "EC / Foam"]

tracker_order = ["qualisys", "mediapipe", "rtmpose", "vitpose"]
tracker_labels = {
    "qualisys": "Reference",
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}

# -------------------
# Build centered data per tracker
# -------------------
plot_data = {}

for tracker in tracker_order:
    tracker_df = final_df.query(
        "participant_code == @participant_to_use "
        "and trial_name == @trial_name_to_use "
        "and tracker == @tracker"
    ).copy()

    for condition in conditions:
        x_col = f"{condition}_x"
        y_col = f"{condition}_y"

        x_raw = tracker_df[x_col].to_numpy()
        y_raw = tracker_df[y_col].to_numpy()

        x = x_raw - np.mean(x_raw)
        y = y_raw - np.mean(y_raw)

        ellipse_points, a, b, theta, area = prediction_ellipse_95(x, y)

        plot_data[(tracker, condition)] = {
            "x": x,
            "y": y,
            "start_x": x[0],
            "start_y": y[0],
            "ellipse_x": ellipse_points[0],
            "ellipse_y": ellipse_points[1],
            "area": area,
        }

# Shared symmetric axis limits across all panels
all_vals = np.concatenate(
    [plot_data[k]["x"] for k in plot_data] +
    [plot_data[k]["y"] for k in plot_data] +
    [plot_data[k]["ellipse_x"] for k in plot_data] +
    [plot_data[k]["ellipse_y"] for k in plot_data]
)
axis_limit = np.ceil(np.max(np.abs(all_vals)) + 1)

# -------------------
# Plot: 4 rows x 4 cols
# -------------------
subplot_titles = []
for tracker in tracker_order:
    for condition in conditions:
        # Only show condition title on first row
        subplot_titles.append("")

fig = make_subplots(
    rows=4,
    cols=4,
    horizontal_spacing=0.01,
    vertical_spacing=0.03,
    subplot_titles=short_titles,  # top row only
)

for row_idx, tracker in enumerate(tracker_order, start=1):
    for col_idx, condition in enumerate(conditions, start=1):
        key = (tracker, condition)
        d = plot_data[key]

        # COM path
        fig.add_trace(
            go.Scatter(
                x=d["x"], y=d["y"],
                mode="lines",
                line=dict(width=2, color="rgba(150,150,150,0.9)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_idx, col=col_idx,
        )

        # 95% prediction ellipse
        fig.add_trace(
            go.Scatter(
                x=d["ellipse_x"], y=d["ellipse_y"],
                mode="lines",
                line=dict(width=2, dash="dash", color=tracker_colors[tracker]),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_idx, col=col_idx,
        )

        # Mean center marker
        # fig.add_trace(
        #     go.Scatter(
        #         x=[0], y=[0],
        #         mode="markers",
        #         marker=dict(symbol="square", size=5),
        #         showlegend=False,
        #         hoverinfo="skip",
        #     ),
        #     row=row_idx, col=col_idx,
        # )

        # Crosshairs
        fig.add_hline(y=0, line_width=0.5, line_dash="dot", row=row_idx, col=col_idx)
        fig.add_vline(x=0, line_width=0.5, line_dash="dot", row=row_idx, col=col_idx)

        # Axes
        fig.update_xaxes(
            range=[-axis_limit, axis_limit],
            zeroline=False,
            showticklabels=(row_idx == 4),
            row=row_idx, col=col_idx,
        )

        fig.update_yaxes(
            range=[-axis_limit, axis_limit],
            scaleanchor=f"x{(row_idx - 1) * 4 + col_idx}",
            scaleratio=1,
            zeroline=False,
            showticklabels=(col_idx == 1),
            row=row_idx, col=col_idx,
        )

    # Row label (tracker name) on leftmost y-axis
    fig.update_yaxes(
        title_text=f"<b>{tracker_labels[tracker]}</b><br>AP (mm)",
        row=row_idx, col=1,
    )

# Bottom row x-axis label
for col_idx in range(1, 5):
    fig.update_xaxes(
        title_text="ML (mm)",
        row=4, col=col_idx,
    )

fig.update_layout(
    width=FIG_W_PX,
    height=FIG_H_PX,
    template="simple_white",
    margin=dict(l=60, r=20, t=40, b=50),
)

for ann in fig.layout.annotations:
    ann.update(font=dict(size=24))

fig.update_xaxes(tickfont=dict(size=20), title_font=dict(size=24))
fig.update_yaxes(tickfont=dict(size=20), title_font=dict(size=24))


fig.show()

# fig.write_image(root_path / f"{EXPORT_BASENAME}.svg", scale=3)
fig.write_image(root_path / f"{EXPORT_BASENAME}.png", scale=3)