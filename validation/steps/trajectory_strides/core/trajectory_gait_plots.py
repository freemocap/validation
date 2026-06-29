import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_trajectory_cycles_grid(cycles: pd.DataFrame, marker_order=None):
    """
    Rows = Left/Right per joint (2 rows per joint)
    Cols = X/Y/Z
    trackers overlaid (solid colors), single legend.
    """
    # ---- checks ----
    required = {'marker','x','y','z','cycle','percent_gait_cycle','tracker'}
    missing = required - set(cycles.columns)
    if missing:
        raise ValueError(f"cycles DataFrame missing columns: {missing}")

    axes  = ['x','y','z']
    sides = ['left','right']

    all_markers = cycles['marker'].astype(str).unique().tolist()
    if marker_order is None:
        joints = sorted({m.replace('left_','').replace('right_','') for m in all_markers})
    else:
        joints = list(marker_order)

    n_rows = len(joints) * 2
    n_cols = len(axes)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=True, shared_yaxes=False,
        vertical_spacing=0.025,
        horizontal_spacing=0.09,      # was 0.05 → more room between columns
        column_titles=[ax.upper() for ax in axes],
    )

    # ---- visuals: solid colors per tracker, single legend ----
    trackers = list(cycles['tracker'].unique())

    TRACKER_COLORS = {
        "qualisys": "#d62728",   # red
        "mediapipe": "#1f77b4",  # blue
        "rtmpose": "#1f77b4",    # blue
    }

    DEFAULT_COLORS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

    tracker_color = {}
    fallback_idx = 0

    for tracker in trackers:
        if tracker in TRACKER_COLORS:
            tracker_color[tracker] = TRACKER_COLORS[tracker]
        else:
            tracker_color[tracker] = DEFAULT_COLORS[fallback_idx % len(DEFAULT_COLORS)]
            fallback_idx += 1


    def ax_key(kind, row, col):
        idx = (row - 1) * n_cols + col
        return f"{kind}axis" + ("" if idx == 1 else str(idx))

    # collect y-range per (joint,axis) to sync left/right
    y_minmax = {(j,a): [np.inf, -np.inf] for j in joints for a in axes}

    # ---- pre-group cycles by (marker, tracker) for fast lookup ----
    grouped = cycles.groupby(['marker', 'tracker'])

    # ---- traces ----
    first_legend = set()  # track which trackers have had their legend entry added
    row_idx = 0
    for j_idx, joint in enumerate(joints):
        for side in sides:
            row_idx += 1
            marker_name = f"{side}_{joint}"
            if marker_name not in all_markers:
                continue

            for c, axis in enumerate(axes, start=1):
                for tracker in trackers:
                    key = (marker_name, tracker)
                    if key not in grouped.groups:
                        continue
                    df_s = grouped.get_group(key)

                    # update y-range
                    vals = df_s[axis].to_numpy()
                    if vals.size:
                        y_minmax[(joint, axis)][0] = min(y_minmax[(joint, axis)][0], np.nanmin(vals))
                        y_minmax[(joint, axis)][1] = max(y_minmax[(joint, axis)][1], np.nanmax(vals))

                    # concatenate all cycles with NaN separators for a single trace
                    x_parts, y_parts = [], []
                    for _, df_cyc in df_s.groupby('cycle', sort=True):
                        x_parts.append(df_cyc['percent_gait_cycle'].values)
                        x_parts.append([np.nan])
                        y_parts.append(df_cyc[axis].values)
                        y_parts.append([np.nan])

                    show = tracker not in first_legend
                    if show:
                        first_legend.add(tracker)

                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate(x_parts),
                            y=np.concatenate(y_parts),
                            mode='lines',
                            line=dict(color=tracker_color[tracker], width=1.3),
                            opacity=0.55,
                            name=tracker,
                            legendgroup=tracker,
                            showlegend=show,
                        ),
                        row=row_idx, col=c
                    )

    # ---- sync y-ranges for left/right per joint/axis ----
    for j_idx, joint in enumerate(joints):
        left_row  = j_idx*2 + 1
        right_row = j_idx*2 + 2
        for c, axis in enumerate(axes, start=1):
            lo, hi = y_minmax[(joint, axis)]
            if np.isfinite(lo) and np.isfinite(hi):
                pad = (hi - lo) * 0.05 if hi > lo else 1.0
                rng = [lo - pad, hi + pad]
                fig.update_yaxes(range=rng, row=left_row, col=c)
                fig.update_yaxes(range=rng, row=right_row, col=c)

    # ---- ticks/labels ----
    tickvals = list(range(0, 101, 20))
    # X-ticks on every column's bottom row
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            showticklabels=True,
            tickmode='array', tickvals=tickvals, ticks='outside', automargin=True,
            title=dict(text="Percent gait cycle", standoff=6),
            row=n_rows, col=c
        )
    # Show y-ticks on ALL subplots
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_yaxes(showticklabels=True, ticks='outside', automargin=True, row=r, col=c)

    # ---- alternating row backgrounds (Left vs Right) ----
    fig.update_layout(margin=dict(l=120, r=40, t=70, b=90))  # a bit more left room
    for r in range(1, n_rows + 1):
        y0, y1 = fig.layout[ax_key("y", r, 1)].domain
        fill = "rgba(0,0,0,0.03)" if (r % 2 == 1) else "rgba(0,0,0,0.015)"
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.0, x1=1.0, y0=y0, y1=y1,
            layer="below", line=dict(width=0), fillcolor=fill
        )

    # ---- Left/Right labels like y-axis tags for EVERY subplot ----
    # place for each (row, col) just left of that subplot's y-axis
    side_text = {True: "Left", False: "Right"}
    for r in range(1, n_rows + 1):
        is_left_row = (r % 2 == 1)
        for c in range(1, n_cols + 1):
            xdom = fig.layout[ax_key("x", r, c)].domain
            ydom = fig.layout[ax_key("y", r, c)].domain
            x_pos = xdom[0] - 0.04   # small gutter before each subplot
            y_pos = 0.5 * (ydom[0] + ydom[1])
            fig.add_annotation(
                x=x_pos, xref="paper",
                y=y_pos, yref="paper",
                text=side_text[is_left_row],
                textangle=-90,
                showarrow=False,
                xanchor="right",  # hug the left edge
                yanchor="middle",
                font=dict(size=13, color="#444")
            )

    # ---- joint titles centered between Left/Right rows ----
    for j_idx, joint in enumerate(joints):
        top_row = j_idx*2 + 1
        bot_row = j_idx*2 + 2
        y_top = fig.layout[ax_key("y", top_row, 1)].domain[1]
        y_bot = fig.layout[ax_key("y", bot_row, 1)].domain[0]
        y_mid = 0.5 * (y_top + y_bot)
        fig.add_annotation(
            x=0.5, xref="paper",
            y=y_mid, yref="paper",
            text=f"<b>{joint.upper()}</b>",
            showarrow=False,
            xanchor="center", yanchor="middle",
            font=dict(size=13, color="#000"),
            bgcolor="rgba(255,255,255,0.96)", borderpad=2
        )

    for j_idx, joint in enumerate(joints[:-1]):  # skip after the last one
        bot_row = j_idx * 2 + 2
        y_bot = fig.layout[ax_key("y", bot_row, 1)].domain[0]

        fig.add_shape(
            type="line",
            x0=0.0, x1=1.0, xref="paper",
            y0=y_bot - 0.012, y1=y_bot - 0.012, yref="paper",
            line=dict(width=1, color="rgba(0,0,0,0.65)")
        )
    # ---- single Y-axis label on first column ----
    y_top = fig.layout[ax_key("y", 1, 1)].domain[1]
    y_bot = fig.layout[ax_key("y", n_rows, 1)].domain[0]
    y_center = (y_top + y_bot) / 2
    x_left = fig.layout[ax_key("x", 1, 1)].domain[0]
    fig.add_annotation(
        x=x_left - 0.070, xref="paper",
        y=y_center, yref="paper",
        text="<b>Position (mm)</b>",
        textangle=-90,
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=13, color="#000")
    )

    # ---- final layout ----
    fig.update_layout(
        height=max(440, 120 * n_rows),
        width=1120,
        template="plotly_white",
        title="Trajectory cycles per marker (FreeMoCap vs Qualisys)",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1.0, orientation="h"),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    return fig

def plot_trajectory_summary_grid(summary: pd.DataFrame, marker_order=None, title=None):
    """
    Expects the long 'summary' dataframe from get_trajectory_summary(), with columns:
      ['tracker', 'marker', 'percent_gait_cycle', 'axis', 'stat', 'value']
    where 'stat' ∈ {'mean','std'} and 'axis' ∈ {'x','y','z'}.

    Layout:
      rows  = Left/Right per joint (2 rows per joint)
      cols  = X / Y / Z
      traces= trackers overlaid; mean lines + shaded ±1 SD bands
    """
    # ---- checks ----
    need = {'tracker', 'marker', 'percent_gait_cycle', 'axis', 'stat', 'value'}
    missing = need - set(summary.columns)
    if missing:
        raise ValueError(f"summary DataFrame missing columns: {missing}")

    axes  = ['x','y','z']
    sides = ['left','right']

    all_markers = summary['marker'].astype(str).unique().tolist()
    if marker_order is None:
        joints = sorted({m.replace('left_','').replace('right_','') for m in all_markers})
    else:
        joints = list(marker_order)

    n_rows = len(joints) * 2
    n_cols = len(axes)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=True, shared_yaxes=False,
        vertical_spacing=0.025,
        horizontal_spacing=0.09,
        column_titles=[ax.upper() for ax in axes],
    )

    # ---- visuals: solid colors per tracker, single legend ----
    trackers = list(summary['tracker'].unique())

    TRACKER_COLORS = {
        "qualisys": "#d62728",   # red
        "mediapipe": "#1f77b4",  # blue
        "rtmpose": "#1f77b4",    # blue
    }

    DEFAULT_COLORS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

    tracker_color = {}
    fallback_idx = 0

    for tracker in trackers:
        if tracker in TRACKER_COLORS:
            tracker_color[tracker] = TRACKER_COLORS[tracker]
        else:
            tracker_color[tracker] = DEFAULT_COLORS[fallback_idx % len(DEFAULT_COLORS)]
            fallback_idx += 1

    def ax_key(kind, row, col):
        idx = (row - 1) * n_cols + col
        return f"{kind}axis" + ("" if idx == 1 else str(idx))

    # collect y-range per (joint,axis) to sync left/right (use mean±std)
    y_minmax = {(j,a): [np.inf, -np.inf] for j in joints for a in axes}

    # ---- pre-pivot the entire summary once instead of inside nested loops ----
    pivoted = summary.pivot_table(
        index=['tracker', 'marker', 'axis', 'percent_gait_cycle'],
        columns='stat', values='value'
    )
    # Ensure mean and std columns exist
    if not {'mean', 'std'}.issubset(pivoted.columns):
        raise ValueError("summary must contain both 'mean' and 'std' stat values")
    pivoted = pivoted.sort_index()

    tickvals = list(range(0, 101, 20))

    first_legend = set()
    row_idx = 0
    for j_idx, joint in enumerate(joints):
        for side in sides:
            row_idx += 1
            marker_name = f"{side}_{joint}"
            if marker_name not in all_markers:
                continue

            for c, axis in enumerate(axes, start=1):

                for tracker in trackers:
                    try:
                        df_w = pivoted.loc[(tracker, marker_name, axis)]
                    except KeyError:
                        continue
                    if df_w.empty:
                        continue

                    pct   = df_w.index.values
                    mu    = df_w['mean'].values
                    sig   = df_w['std'].values
                    upper = mu + sig
                    lower = mu - sig

                    # Update min/max
                    if upper.size:
                        lo_now = np.nanmin(lower)
                        hi_now = np.nanmax(upper)
                        y_minmax[(joint, axis)][0] = min(y_minmax[(joint, axis)][0], lo_now)
                        y_minmax[(joint, axis)][1] = max(y_minmax[(joint, axis)][1], hi_now)

                    # Shaded band: two traces (upper then lower with fill='tonexty')
                    fig.add_trace(
                        go.Scatter(
                            x=pct, y=upper,
                            mode='lines',
                            line=dict(color=tracker_color[tracker], width=0),
                            hoverinfo='skip',
                            opacity=0.4,
                            showlegend=False
                        ),
                        row=row_idx, col=c
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=pct, y=lower,
                            mode='lines',
                            line=dict(color=tracker_color[tracker], width=0),
                            fill='tonexty',
                            hoverinfo='skip',
                            opacity=0.4,
                            showlegend=False
                        ),
                        row=row_idx, col=c
                    )

                    # Mean line
                    show = tracker not in first_legend
                    if show:
                        first_legend.add(tracker)

                    fig.add_trace(
                        go.Scatter(
                            x=pct, y=mu,
                            mode='lines',
                            line=dict(color=tracker_color[tracker], width=2),
                            name=tracker,
                            legendgroup=tracker,
                            showlegend=show,
                            hovertemplate=(
                                f"{marker_name} | {axis}<br>"
                                "tracker=%{meta}<br>"
                                "pct=%{x}<br>"
                                "mean=%{y}<extra></extra>"
                            ),
                            meta=tracker,
                        ),
                        row=row_idx, col=c
                    )

    # ---- sync y-ranges for left/right per joint/axis ----
    for j_idx, joint in enumerate(joints):
        left_row  = j_idx*2 + 1
        right_row = j_idx*2 + 2
        for c, axis in enumerate(axes, start=1):
            lo, hi = y_minmax[(joint, axis)]
            if np.isfinite(lo) and np.isfinite(hi):
                pad = (hi - lo) * 0.05 if hi > lo else 1.0
                rng = [lo - pad, hi + pad]
                fig.update_yaxes(range=rng, row=left_row, col=c)
                fig.update_yaxes(range=rng, row=right_row, col=c)

    # ---- ticks/labels ----
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            showticklabels=True,
            tickmode='array', tickvals=tickvals, ticks='outside', automargin=True,
            title=dict(text="Percent gait cycle", standoff=6),
            row=n_rows, col=c
        )
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_yaxes(showticklabels=True, ticks='outside', automargin=True, row=r, col=c)

    # ---- alternating row backgrounds (Left vs Right) ----
    fig.update_layout(margin=dict(l=120, r=40, t=70, b=90))
    for r in range(1, n_rows + 1):
        y0, y1 = fig.layout[ax_key("y", r, 1)].domain
        fill = "rgba(0,0,0,0.03)" if (r % 2 == 1) else "rgba(0,0,0,0.015)"
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.0, x1=1.0, y0=y0, y1=y1,
            layer="below", line=dict(width=0), fillcolor=fill
        )

    # ---- Left/Right strip labels beside every subplot ----
    side_text = {True: "Left", False: "Right"}
    for r in range(1, n_rows + 1):
        is_left_row = (r % 2 == 1)
        for c in range(1, n_cols + 1):
            xdom = fig.layout[ax_key("x", r, c)].domain
            ydom = fig.layout[ax_key("y", r, c)].domain
            x_pos = xdom[0] - 0.04
            y_pos = 0.5 * (ydom[0] + ydom[1])
            fig.add_annotation(
                x=x_pos, xref="paper",
                y=y_pos, yref="paper",
                text=side_text[is_left_row],
                textangle=-90,
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=13, color="#444")
            )

    # ---- joint titles centered between Left/Right rows ----
    for j_idx, joint in enumerate(joints):
        top_row = j_idx*2 + 1
        bot_row = j_idx*2 + 2
        y_top = fig.layout[ax_key("y", top_row, 1)].domain[1]
        y_bot = fig.layout[ax_key("y", bot_row, 1)].domain[0]
        y_mid = 0.5 * (y_top + y_bot)
        fig.add_annotation(
            x=0.5, xref="paper",
            y=y_mid, yref="paper",
            text=f"<b>{joint.upper()}</b>",
            showarrow=False,
            xanchor="center", yanchor="middle",
            font=dict(size=13, color="#000"),
            bgcolor="rgba(255,255,255,0.96)", borderpad=2
        )

    # separator lines between joints
    for j_idx, joint in enumerate(joints[:-1]):
        bot_row = j_idx * 2 + 2
        y_bot = fig.layout[ax_key("y", bot_row, 1)].domain[0]
        fig.add_shape(
            type="line",
            x0=0.0, x1=1.0, xref="paper",
            y0=y_bot - 0.012, y1=y_bot - 0.012, yref="paper",
            line=dict(width=1, color="rgba(0,0,0,0.65)")
        )

    # ---- single Y-axis master label on first column ----
    y_top = fig.layout[ax_key("y", 1, 1)].domain[1]
    y_bot = fig.layout[ax_key("y", n_rows, 1)].domain[0]
    y_center = (y_top + y_bot) / 2
    x_left = fig.layout[ax_key("x", 1, 1)].domain[0]
    fig.add_annotation(
        x=x_left - 0.070, xref="paper",
        y=y_center, yref="paper",
        text="<b>Position (mm)</b>",
        textangle=-90,
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=13, color="#000")
    )

    # ---- final layout ----
    fig.update_layout(
        height=max(440, 120 * n_rows),
        width=1120,
        template="plotly_white",
        title=title or "Trajectory summary (mean ± SD) per marker",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1.0, orientation="h"),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    return fig