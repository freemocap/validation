import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Your preferred component order by joint
COMPONENTS_BY_JOINT_DEFAULT = {
    'hip':   ['flex_ext','abd_add','int_ext'],
    'knee':  ['flex_ext','abd_add','int_ext'],
    'ankle': ['dorsi_plantar','inv_ev','int_ext'],
    '_default': ['c1','c2','c3'],
}

# Optional pretty labels for subplot subtitles / legends
COMP_LABEL = {
    'flex_ext':     'Flex / Ext (°)',
    'abd_add':      'Abd / Add (°)',
    'int_ext':      'Int / Ext Rot (°)',
    'dorsi_plantar':'Dorsi / Plantar (°)',
    'inv_ev':       'Inversion / Eversion (°)',
    'c1': 'C1 (°)', 'c2': 'C2 (°)', 'c3': 'C3 (°)',
}

def plot_angle_summary_grid(
    summary: pd.DataFrame,
    components_by_joint: dict[str, list[str]] | None = None,
    title: str | None = None,
    joint_order: list[str] | None = None,
):
    """
    Expects a tidy long 'summary' DataFrame with columns:
      ['tracker','joint','side','component','percent_gait_cycle','stat','value']
    where:
      - 'stat' ∈ {'mean','std'}
      - 'side' ∈ {'left','right'}
      - 'percent_gait_cycle' is 0..100 (commonly 101 samples)
      - 'component' is one of the biomechanical components per joint

    Layout:
      rows  = Left / Right per joint (2 rows per joint)
      cols  = per-joint component order (e.g., hip: flex/ext, abd/add, int/ext)
      traces= trackers overlaid; mean lines + shaded ±1 SD bands
    """
    # ---- checks ----
    need = {'tracker','joint','side','component','percent_gait_cycle','stat','value'}
    missing = need - set(summary.columns)
    if missing:
        raise ValueError(f"summary DataFrame missing columns: {missing}")

    if components_by_joint is None:
        components_by_joint = COMPONENTS_BY_JOINT_DEFAULT

    # Figure out joints present and order them hip → knee → ankle → others
    joints_present = list(summary['joint'].astype(str).unique())
    canonical = ['hip','knee','ankle']
    ordered = [j for j in canonical if j in joints_present] + [j for j in joints_present if j not in canonical]
    if joint_order is not None:
        # Respect explicit order if given
        ordered = [j for j in joint_order if j in joints_present]
    joints = ordered

    # Columns are the union of components we want per joint (3 typical)
    # We build per-joint column labels dynamically
    per_joint_components = {j: components_by_joint.get(j, components_by_joint.get('_default', [])) for j in joints}
    # Infer maximum number of columns across joints (usually 3)
    n_cols = max(len(v) for v in per_joint_components.values()) if joints else 0
    n_rows = len(joints) * 2

    # Create subplots with generic titles (we'll add component subtitles inside each subplot)
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=True, shared_yaxes=False,
        vertical_spacing=0.025,
        horizontal_spacing=0.09,
        column_titles=None,
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
    # Helpers to access subplot axis domains
    def ax_key(kind, row, col):
        idx = (row - 1) * n_cols + col
        return f"{kind}axis" + ("" if idx == 1 else str(idx))

    # For syncing y-range per (joint, component)
    y_minmax = {(j, c): [np.inf, -np.inf] for j in joints for c in sum((per_joint_components[j] for j in joints), [])}

    tickvals = list(range(0, 101, 20))  # assume 0..100%

    # ---- pre-pivot the entire summary once instead of inside nested loops ----
    pivoted = summary.pivot_table(
        index=['tracker', 'joint', 'side', 'component', 'percent_gait_cycle'],
        columns='stat', values='value'
    )
    if not {'mean', 'std'}.issubset(pivoted.columns):
        raise ValueError("summary must contain both 'mean' and 'std' stat values")
    pivoted = pivoted.sort_index()

    first_legend = set()

    # Iterate over joints, then sides (Left row, Right row)
    for j_idx, joint in enumerate(joints):
        comps = per_joint_components[joint]
        left_row  = j_idx * 2 + 1
        right_row = j_idx * 2 + 2

        for side in ['left','right']:
            row = left_row if side == 'left' else right_row

            for c_idx, comp in enumerate(comps, start=1):

                for tracker in trackers:
                    try:
                        df_w = pivoted.loc[(tracker, joint, side, comp)]
                    except KeyError:
                        continue
                    if df_w.empty:
                        continue

                    pct  = df_w.index.values
                    mu   = df_w['mean'].values
                    sig  = df_w['std'].values
                    upper = mu + sig
                    lower = mu - sig

                    # --- UPDATE Y-RANGE *HERE* ---
                    if upper.size:
                        lo_now = float(np.nanmin(lower))
                        hi_now = float(np.nanmax(upper))
                        y_minmax[(joint, comp)][0] = min(y_minmax[(joint, comp)][0], lo_now)
                        y_minmax[(joint, comp)][1] = max(y_minmax[(joint, comp)][1], hi_now)

                    # Shaded band (upper then lower with fill='tonexty')
                    fig.add_trace(
                        go.Scatter(
                            x=pct, y=upper,
                            mode='lines',
                            line=dict(color=tracker_color[tracker], width=0),
                            hoverinfo='skip',
                            opacity=0.18,
                            showlegend=False
                        ),
                        row=row, col=c_idx
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=pct, y=lower,
                            mode='lines',
                            line=dict(color=tracker_color[tracker], width=0),
                            fill='tonexty',
                            hoverinfo='skip',
                            opacity=0.18,
                            showlegend=False
                        ),
                        row=row, col=c_idx
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
                                f"{side}_{joint} | {COMP_LABEL.get(comp, comp)}<br>"
                                "tracker=%{meta}<br>"
                                "pct=%{x:.0f}%<br>"
                                "mean=%{y:.2f}°<extra></extra>"
                            ),
                            meta=tracker,
                        ),
                        row=row, col=c_idx
                    )

        
        for c_idx, comp in enumerate(comps, start=1):
            lo, hi = y_minmax[(joint, comp)]
            if np.isfinite(lo) and np.isfinite(hi):
                span = hi - lo
                pad = max(0.12 * span, 1.0)  # 12% or ≥ 1°
                rng = [lo - pad, hi + pad]
                fig.update_yaxes(range=rng, row=left_row,  col=c_idx)
                fig.update_yaxes(range=rng, row=right_row, col=c_idx)

    # ---- ticks / labels ----
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_yaxes(showticklabels=True, ticks='outside', automargin=True, row=r, col=c)
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            showticklabels=True,
            tickmode='array', tickvals=tickvals, ticks='outside', automargin=True,
            title=dict(text="Percent gait cycle", standoff=6),
            row=n_rows, col=c
        )

    # ---- alternating row backgrounds (Left vs Right) ----
    fig.update_layout(margin=dict(l=120, r=40, t=70, b=90))
    for r in range(1, n_rows + 1):
        key = ax_key("y", r, 1)
        if key in fig.layout:
            y0, y1 = fig.layout[key].domain
            fill = "rgba(0,0,0,0.03)" if (r % 2 == 1) else "rgba(0,0,0,0.015)"
            fig.add_shape(
                type="rect", xref="paper", yref="paper",
                x0=0.0, x1=1.0, y0=y0, y1=y1,
                layer="below", line=dict(width=0), fillcolor=fill
            )

    # ---- Left/Right strip labels ----
    side_text = {True: "Left", False: "Right"}
    for r in range(1, n_rows + 1):
        is_left_row = (r % 2 == 1)
        for c in range(1, n_cols + 1):
            kx, ky = ax_key("x", r, c), ax_key("y", r, c)
            if kx in fig.layout and ky in fig.layout:
                xdom = fig.layout[kx].domain
                ydom = fig.layout[ky].domain
                x_pos = xdom[0] - 0.04
                y_pos = 0.5 * (ydom[0] + ydom[1])
                fig.add_annotation(
                    x=x_pos, xref="paper",
                    y=y_pos,  yref="paper",
                    text=side_text[is_left_row],
                    textangle=-90,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    font=dict(size=13, color="#444")
                )

    # ---- Joint titles centered between Left/Right rows ----
    for j_idx, joint in enumerate(joints):
        top_row = j_idx*2 + 1
        bot_row = j_idx*2 + 2
        ky_top = ax_key("y", top_row, 1)
        ky_bot = ax_key("y", bot_row, 1)
        if ky_top in fig.layout and ky_bot in fig.layout:
            y_top = fig.layout[ky_top].domain[1]
            y_bot = fig.layout[ky_bot].domain[0]
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
    for j_idx in range(len(joints) - 1):
        bot_row = j_idx * 2 + 2
        ky = ax_key("y", bot_row, 1)
        if ky in fig.layout:
            y_bot = fig.layout[ky].domain[0]
            fig.add_shape(
                type="line",
                x0=0.0, x1=1.0, xref="paper",
                y0=y_bot - 0.012, y1=y_bot - 0.012, yref="paper",
                line=dict(width=1, color="rgba(0,0,0,0.65)")
                )
    # ---- per-column headers (once per joint, above the top row) ----
    for j_idx, joint in enumerate(joints):
        comps = per_joint_components[joint]
        top_row = j_idx * 2 + 1

        # y position just above the top row for this joint
        ky_top = ax_key("y", top_row, 1)
        if ky_top not in fig.layout:
            continue
        y_top = fig.layout[ky_top].domain[1]
        y_hdr = y_top   # a bit above the subplot; adjust if needed

        for c_idx, comp in enumerate(comps, start=1):
            kx = ax_key("x", top_row, c_idx)
            if kx not in fig.layout:
                continue
            xdom = fig.layout[kx].domain
            x_mid = 0.5 * (xdom[0] + xdom[1])

            fig.add_annotation(
                x=x_mid, xref="paper",
                y=y_hdr, yref="paper",
                text=f"<b>{COMP_LABEL.get(comp, comp)}</b>",
                showarrow=False,
                xanchor="center", yanchor="bottom",
                font=dict(size=12, color="#333")
            )

    # give a bit more top margin for the new headers
    fig.update_layout(margin=dict(l=120, r=40, t=100, b=90))
    # ---- single Y-axis master label on first column ----
    if n_rows > 0 and n_cols > 0:
        ky1 = ax_key("y", 1, 1)
        kx1 = ax_key("x", 1, 1)
        if ky1 in fig.layout and kx1 in fig.layout:
            y_top = fig.layout[ky1].domain[1]
            y_bot = fig.layout[ax_key("y", n_rows, 1)].domain[0]
            y_center = (y_top + y_bot) / 2
            x_left = fig.layout[kx1].domain[0]
            fig.add_annotation(
                x=x_left - 0.070, xref="paper",
                y=y_center, yref="paper",
                text="<b>Angle (°)</b>",
                textangle=-90,
                showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(size=13, color="#000")
            )

    # ---- final layout ----
    fig.update_layout(
        height=max(1300, 120 * n_rows),
        width=1300,
        template="plotly_white",
        title=title or "Joint angles (mean ± SD) per component",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1.0, orientation="h"),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    # X-axis ticks: 0..100%
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(tickmode='array', tickvals=tickvals, row=r, col=c)

    return fig