

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gait_ba_utils import (
    load_paired_gait_data, ba_stats,
    TRACKERS, TRACKER_LABELS,
    SPEED_ORDER, SPEED_STYLE,
    inches_to_px, style_paperish,
)


from pathlib import Path
save_root = Path(r"D:\validation_public_release_v1\figures")
save_root.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# Config 
# ------------------------------------------------------------------
METRICS = [
    {
        "key": "stride_duration",
        "label": "Stride Duration",
        "y_scale": 1000,
        "y_units": "ms",
        "x_units": "s",
        "y_range": 200, 
    },
    {
        "key": "stride_length",
        "label": "Stride Length",
        "y_scale": 1.0,
        "y_units": "mm",
        "x_units": "mm",
        "y_range": 200,  # auto from data
    },
]
# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
paired_df = load_paired_gait_data("validation.db")

# ------------------------------------------------------------------
# Build figure — 3 rows (trackers) × 2 cols (metrics)
# ------------------------------------------------------------------
nrows = len(TRACKERS)
ncols = len(METRICS)

# Subplot titles: metric label on top row, tracker label on each row
# Plotly fills subplot_titles left-to-right, top-to-bottom
subplot_titles = []
for row_idx, tracker in enumerate(TRACKERS):
    for col_idx, metric in enumerate(METRICS):
        subplot_titles.append(TRACKER_LABELS[tracker])

fig = make_subplots(
    rows=nrows, cols=ncols,
    shared_xaxes="columns",
    shared_yaxes=False,
    vertical_spacing=0.06,
    horizontal_spacing=0.08,
    subplot_titles=subplot_titles,
)

# Track y-values per column for symmetric ranging
all_y_per_col = {c: [] for c in range(ncols)}

for col_idx, metric in enumerate(METRICS, start=1):
    metric_key = metric["key"]
    y_scale = metric["y_scale"]
    y_units = metric["y_units"]
    x_units = metric["x_units"]

    for row_idx, tracker in enumerate(TRACKERS, start=1):
        tracker_label = TRACKER_LABELS[tracker]

        df_met = paired_df.query("metric == @metric_key and tracker == @tracker")
        if df_met.empty:
            continue

        diffs_scaled = df_met["ba_diff"].to_numpy() * y_scale
        means_scaled = df_met["ba_mean"].to_numpy()
        all_y_per_col[col_idx - 1].append(diffs_scaled)

        # Compute BA lines on ALL data
        stats = ba_stats(diffs_scaled)

        # Reference lines
        # Plotly axis indexing: row 1 col 1 -> (x, y), row 1 col 2 -> (x2, y2), etc.
        ax_num = (row_idx - 1) * ncols + col_idx
        xref = f"x{ax_num} domain" if ax_num > 1 else "x domain"
        yref = f"y{ax_num}" if ax_num > 1 else "y"
        line_kw = dict(xref=xref, x0=0, x1=1, yref=yref)

        fig.add_shape(type="line", y0=stats["bias"], y1=stats["bias"],
                      line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"), **line_kw)
        fig.add_shape(type="line", y0=stats["loa_upper"], y1=stats["loa_upper"],
                      line=dict(color="black", width=1, dash="dashdot"), **line_kw)
        fig.add_shape(type="line", y0=stats["loa_lower"], y1=stats["loa_lower"],
                      line=dict(color="black", width=1, dash="dashdot"), **line_kw)

        # Plot each speed as separate trace
        for spd in SPEED_ORDER:
            ds = df_met[df_met["condition"] == spd]
            if ds.empty:
                continue
            sty = SPEED_STYLE[spd]
            # Show legend only from first row, first column
            show_legend = (row_idx == 1 and col_idx == 1)
            fig.add_trace(
                go.Scatter(
                    x=ds["ba_mean"].values,
                    y=ds["ba_diff"].values * y_scale,
                    mode="markers",
                    name=sty["label"],
                    legendgroup=spd,
                    showlegend=show_legend,
                    marker=dict(
                        size=7, opacity=0.5, color=sty["color"],
                        line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
                    ),
                ),
                row=row_idx, col=col_idx,
            )

        print(
            f"[{tracker_label} — {metric['label']}] "
            f"Bias={stats['bias']:+.2f} {y_units}, "
            f"LoA=[{stats['loa_lower']:+.2f}, {stats['loa_upper']:+.2f}] {y_units}"
        )

# ------------------------------------------------------------------
# Y-range: symmetric per column
# ------------------------------------------------------------------
for col_idx, metric in enumerate(METRICS, start=1):
    y_arrays = all_y_per_col[col_idx - 1]
    if y_arrays:
        y_all = np.concatenate(y_arrays)
        y_finite = y_all[np.isfinite(y_all)]
        y_absmax = float(np.max(np.abs(y_finite))) if len(y_finite) > 0 else 10.0
    else:
        y_absmax = 10.0
        
    if metric.get("y_range") is not None:
        y_pad = metric["y_range"]
    else:
        y_pad = y_absmax * 1.05

    y_units = metric["y_units"]
    x_units = metric["x_units"]

    for r in range(1, nrows + 1):
        fig.update_yaxes(range=[-y_pad, y_pad], autorange=False, row=r, col=col_idx)
        fig.update_yaxes(title_text=f"<b>Difference ({y_units})</b>", row=r, col=col_idx)

    # X-axis label only on bottom row
    fig.update_xaxes(title_text=f"<b>Mean ({x_units})</b>", row=nrows, col=col_idx)

# ------------------------------------------------------------------
# Column headers (metric labels) as annotations
# ------------------------------------------------------------------
# Get x-domain centers from the first row's subplots
for col_idx, metric in enumerate(METRICS):
    ax_key = f"xaxis{col_idx + 1}" if col_idx > 0 else "xaxis"
    domain = fig.layout[ax_key].domain
    x_center = (domain[0] + domain[1]) / 2 if domain else 0.5

    x_center = 0.18 if metric["key"] == "stride_duration" else 0.82
    fig.add_annotation(
        text=f"<b>{metric['label']}</b>",
        xref="paper", yref="paper",
        x=x_center, y=1.06,
        showarrow=False,
        font=dict(size=16),
    )

# Bold subplot titles (tracker names)
for ann in fig.layout.annotations:
    if hasattr(ann, "text") and ann.text in [TRACKER_LABELS[t] for t in TRACKERS]:
        ann.font = dict(size=15, weight="bold")

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
FIG_W_IN = 4.0
FIG_H_IN = 1 * nrows

style_paperish(fig, width_px=inches_to_px(FIG_W_IN), height_px=inches_to_px(FIG_H_IN))

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.07,
        xanchor="center", x=0.5,
        font=dict(size=18),
        tracegroupgap=2,
    ),
    margin=dict(l=80, r=10, t=60, b=75),
)

# Force y-range after styling
for i in range(1, nrows * ncols + 1):
    key = f"yaxis{i}" if i > 1 else "yaxis"
    # odd axes = col 1 (duration), even axes = col 2 (length)
    col = (i - 1) % ncols
    metric = METRICS[col]
    
    y_arrays = all_y_per_col[col]
    if y_arrays:
        y_all = np.concatenate(y_arrays)
        y_finite = y_all[np.isfinite(y_all)]
        y_absmax = float(np.max(np.abs(y_finite))) if len(y_finite) > 0 else 10.0
    else:
        y_absmax = 10.0
    
    if metric.get("y_range") is not None:
        y_pad = metric["y_range"]
    else:
        y_pad = y_absmax * 1.05
    
    fig.layout[key].autorange = False
    fig.layout[key].range = [-y_pad, y_pad]



fig.show()
fig.write_image(save_root / "ba_stride_both.png", scale=3)
print(f"\nFigure saved to: {save_root / 'ba_stride_both.png'}")