import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_gait_events_over_time(
    q_hs: np.ndarray,
    q_to: np.ndarray,
    fmc_hs: np.ndarray,
    fmc_to: np.ndarray,
    sampling_rate: float,
    title: str = "Gait events over time (one foot)",
    xlim: tuple | None = None,
    separation: float = 0.03,
):
    frame_interval = 1.0 / sampling_rate
    q_hs, q_to = np.asarray(q_hs), np.asarray(q_to)
    fmc_hs, fmc_to = np.asarray(fmc_hs), np.asarray(fmc_to)

    # consistent colors
    qual_color = "#d62728"   # red
    fmc_color  = "#1f77b4"   # blue

    # y positions
    q_hs_y   = np.full(q_hs.shape, 0.3 + separation/2)
    fmc_hs_y = np.full(fmc_hs.shape, 0.3 - separation/2)
    q_to_y   = np.full(q_to.shape, 0.0 + separation/2)
    fmc_to_y = np.full(fmc_to.shape, 0.0 - separation/2)

    # convert to seconds
    q_hs_t, q_to_t = q_hs * frame_interval, q_to * frame_interval
    fmc_hs_t, fmc_to_t = fmc_hs * frame_interval, fmc_to * frame_interval

    fig = go.Figure()

    # heel strikes
    fig.add_trace(go.Scatter(
        x=q_hs_t, y=q_hs_y, mode="markers",
        marker=dict(symbol="x", size=8, color=qual_color),
        name="Qualisys HS"
    ))
    fig.add_trace(go.Scatter(
        x=fmc_hs_t, y=fmc_hs_y, mode="markers",
        marker=dict(symbol="x", size=8, color=fmc_color),
        name="FreeMoCap HS"
    ))

    # toe offs – same color as each system’s HS
    fig.add_trace(go.Scatter(
        x=q_to_t, y=q_to_y, mode="markers",
        marker=dict(symbol="circle-open", size=8, color=qual_color,
                    line=dict(width=1.5, color=qual_color)),
        name="Qualisys TO"
    ))
    fig.add_trace(go.Scatter(
        x=fmc_to_t, y=fmc_to_y, mode="markers",
        marker=dict(symbol="circle-open", size=8, color = fmc_color,
                    line=dict(width=1.5, color=fmc_color)),
        name="FreeMoCap TO"
    ))

    # axes / layout
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0.3, 0.0],
        ticktext=["Heel Strike", "Toe Off"],
        range=[-0.3, 0.6],
        showgrid=False
    )
    fig.update_xaxes(title="Time (seconds)", showgrid=True, gridcolor="rgba(0,0,0,0.15)")
    if xlim:
        fig.update_xaxes(range=list(xlim))

    fig.update_layout(
        title=title,
        legend=dict(orientation="h", y=1.08, yanchor="bottom", x=1.0, xanchor="right"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )

    return fig

def plot_gait_events_over_time_both_feet(
    # left foot
    q_left_hs: np.ndarray,
    q_left_to: np.ndarray,
    fmc_left_hs: np.ndarray,
    fmc_left_to: np.ndarray,
    # right foot
    q_right_hs: np.ndarray,
    q_right_to: np.ndarray,
    fmc_right_hs: np.ndarray,
    fmc_right_to: np.ndarray,
    sampling_rate: float,
    title: str = "Gait events over time (left & right feet)",
    xlim: tuple | None = None,
    separation: float = 0.03,
):
    """
    Two-row plot:
      - Row 1: left foot
      - Row 2: right foot
    Qualisys events in red, FreeMoCap in blue.
    Heel strikes and toe offs on separate y-bands (like your one-foot plot).
    """

    frame_interval = 1.0 / sampling_rate

    # ensure arrays
    q_left_hs   = np.asarray(q_left_hs)
    q_left_to   = np.asarray(q_left_to)
    fmc_left_hs = np.asarray(fmc_left_hs)
    fmc_left_to = np.asarray(fmc_left_to)

    q_right_hs   = np.asarray(q_right_hs)
    q_right_to   = np.asarray(q_right_to)
    fmc_right_hs = np.asarray(fmc_right_hs)
    fmc_right_to = np.asarray(fmc_right_to)

    qual_color = "#d62728"   # red
    fmc_color  = "#1f77b4"   # blue

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Left foot", "Right foot"),
    )

    def _foot_row(
        row: int,
        q_hs: np.ndarray,
        q_to: np.ndarray,
        fmc_hs: np.ndarray,
        fmc_to: np.ndarray,
    ):
        # y positions (same as your one-foot version)
        q_hs_y   = np.full(q_hs.shape, 0.3 + separation/2)
        fmc_hs_y = np.full(fmc_hs.shape, 0.3 - separation/2)
        q_to_y   = np.full(q_to.shape, 0.0 + separation/2)
        fmc_to_y = np.full(fmc_to.shape, 0.0 - separation/2)

        # times in seconds
        q_hs_t, q_to_t       = q_hs * frame_interval, q_to * frame_interval
        fmc_hs_t, fmc_to_t   = fmc_hs * frame_interval, fmc_to * frame_interval

        # Qualisys HS
        fig.add_trace(go.Scatter(
            x=q_hs_t, y=q_hs_y, mode="markers",
            marker=dict(symbol="x", size=8, color=qual_color),
            name="Qualisys HS" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # FreeMoCap HS
        fig.add_trace(go.Scatter(
            x=fmc_hs_t, y=fmc_hs_y, mode="markers",
            marker=dict(symbol="x", size=8, color=fmc_color),
            name="FreeMoCap HS" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # Qualisys TO
        fig.add_trace(go.Scatter(
            x=q_to_t, y=q_to_y, mode="markers",
            marker=dict(symbol="circle-open", size=8, color=qual_color,
                        line=dict(width=1.5, color=qual_color)),
            name="Qualisys TO" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # FreeMoCap TO
        fig.add_trace(go.Scatter(
            x=fmc_to_t, y=fmc_to_y, mode="markers",
            marker=dict(symbol="circle-open", size=8, color=fmc_color,
                        line=dict(width=1.5, color=fmc_color)),
            name="FreeMoCap TO" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # y-axis config for this row
        fig.update_yaxes(
            row=row, col=1,
            tickmode="array",
            tickvals=[0.3, 0.0],
            ticktext=["Heel Strike", "Toe Off"],
            range=[-0.3, 0.6],
            showgrid=False,
        )

    # row 1: left foot
    _foot_row(
        row=1,
        q_hs=q_left_hs,
        q_to=q_left_to,
        fmc_hs=fmc_left_hs,
        fmc_to=fmc_left_to,
    )

    # row 2: right foot
    _foot_row(
        row=2,
        q_hs=q_right_hs,
        q_to=q_right_to,
        fmc_hs=fmc_right_hs,
        fmc_to=fmc_right_to,
    )

    # shared x-axis
    fig.update_xaxes(
        title="Time (seconds)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        row=2, col=1,   # only label bottom row
    )
    if xlim:
        fig.update_xaxes(range=list(xlim), row=1, col=1)
        fig.update_xaxes(range=list(xlim), row=2, col=1)

    fig.update_layout(
        title=title,
        legend=dict(
            orientation="h",
            y=1.1, yanchor="bottom",
            x=1.0, xanchor="right",
        ),
        margin=dict(l=40, r=20, t=80, b=40),
        height=650,
    )

    return fig

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from validation.steps.step_finder.core.models import GaitResults     # adjust import path if needed
from validation.steps.step_finder.core.calculate_kinematics import FootKinematics  # adjust path if needed


def plot_stepfinder_mega_debug(
    freemocap_kinematics: FootKinematics,
    qualisys_kinematics: FootKinematics,
    freemocap_gait_events: GaitResults,
    qualisys_gait_events: GaitResults,
    flagged_events: GaitResults,
    sampling_rate: float,
    title: str = "Step finder mega debug",
    xlim: Optional[tuple] = None,
) -> go.Figure:
    """
    One big debug figure with 3x2 subplots:

        Row 1: AP (Y) position vs time  (left, right)
        Row 2: Vertical (Z) position vs time (left, right)
        Row 3: HS/TO event timelines vs Qualisys (left, right)

    FreeMoCap events flagged as suspicious are highlighted in orange.
    """

    frame_interval = 1.0 / sampling_rate

    # ------------------------------------------------------------------
    # Helper: build masks for flagged HS/TO events (per foot)
    # ------------------------------------------------------------------
    def _flag_masks(all_events: np.ndarray, flagged: np.ndarray) -> np.ndarray:
        all_events = np.asarray(all_events, dtype=int)
        flagged = np.asarray(flagged, dtype=int)
        return np.isin(all_events, flagged)

    # left foot masks
    left_hs_all = np.asarray(freemocap_gait_events.left_foot.heel_strikes, dtype=int)
    left_to_all = np.asarray(freemocap_gait_events.left_foot.toe_offs, dtype=int)
    left_hs_flag = _flag_masks(left_hs_all, flagged_events.left_foot.heel_strikes)
    left_to_flag = _flag_masks(left_to_all, flagged_events.left_foot.toe_offs)

    # right foot masks
    right_hs_all = np.asarray(freemocap_gait_events.right_foot.heel_strikes, dtype=int)
    right_to_all = np.asarray(freemocap_gait_events.right_foot.toe_offs, dtype=int)
    right_hs_flag = _flag_masks(right_hs_all, flagged_events.right_foot.heel_strikes)
    right_to_flag = _flag_masks(right_to_all, flagged_events.right_foot.toe_offs)

    # ------------------------------------------------------------------
    # Helper: add a position panel (Y or Z) for one foot
    # ------------------------------------------------------------------
    def _add_position_panel(
        fig: go.Figure,
        row: int,
        col: int,
        heel_pos: np.ndarray,
        toe_pos: np.ndarray,
        hs_idx: np.ndarray,
        to_idx: np.ndarray,
        hs_flag: np.ndarray,
        to_flag: np.ndarray,
        sampling_rate: float,
        axis: int,
        ylabel: str,
        panel_title: str,
    ):
        n = heel_pos.shape[0]
        t = np.arange(n) / sampling_rate

        heel_axis = heel_pos[:, axis]
        toe_axis = toe_pos[:, axis]

        # base lines
        fig.add_trace(
            go.Scatter(
                x=t,
                y=heel_axis,
                mode="lines",
                name="Heel",
                line=dict(color="#d62728"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=toe_axis,
                mode="lines",
                name="Toe",
                line=dict(color="#1f77b4"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # HS markers
        if hs_idx.size:
            hs_idx = hs_idx.astype(int)
            normal = ~hs_flag
            fig.add_trace(
                go.Scatter(
                    x=t[hs_idx[normal]],
                    y=heel_axis[hs_idx[normal]],
                    mode="markers",
                    name="HS (normal)",
                    marker=dict(symbol="circle-open", size=7, color="red"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=t[hs_idx[hs_flag]],
                    y=heel_axis[hs_idx[hs_flag]],
                    mode="markers",
                    name="HS (flagged)",
                    marker=dict(symbol="circle", size=9, color="orange"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # TO markers
        if to_idx.size:
            to_idx = to_idx.astype(int)
            normal = ~to_flag
            fig.add_trace(
                go.Scatter(
                    x=t[to_idx[normal]],
                    y=toe_axis[to_idx[normal]],
                    mode="markers",
                    name="TO (normal)",
                    marker=dict(symbol="x", size=7, color="blue"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=t[to_idx[to_flag]],
                    y=toe_axis[to_idx[to_flag]],
                    mode="markers",
                    name="TO (flagged)",
                    marker=dict(symbol="x", size=9, color="orange"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_yaxes(title_text=ylabel, row=row, col=col)
        fig.update_xaxes(showgrid=True, row=row, col=col)
        fig.layout.annotations[(row - 1) * 2 + (col - 1)].update(text=panel_title)

    # ------------------------------------------------------------------
    # Helper: add the HS/TO timeline panel for one foot
    # ------------------------------------------------------------------
    def _add_event_timeline_panel(
        fig: go.Figure,
        row: int,
        col: int,
        q_hs: np.ndarray,
        q_to: np.ndarray,
        fmc_hs: np.ndarray,
        fmc_to: np.ndarray,
        hs_flag: np.ndarray,
        to_flag: np.ndarray,
        frame_interval: float,
        foot_label: str,
    ):
        qual_color = "#d62728"
        fmc_color = "#1f77b4"
        cluster_color = "orange"

        q_hs = np.asarray(q_hs, dtype=int)
        q_to = np.asarray(q_to, dtype=int)
        fmc_hs = np.asarray(fmc_hs, dtype=int)
        fmc_to = np.asarray(fmc_to, dtype=int)

        # y offsets for HS / TO rows
        q_hs_y = np.full(q_hs.shape, 0.3)
        fmc_hs_y = np.full(fmc_hs.shape, 0.2)
        q_to_y = np.full(q_to.shape, 0.0)
        fmc_to_y = np.full(fmc_to.shape, -0.1)

        q_hs_t, q_to_t = q_hs * frame_interval, q_to * frame_interval
        fmc_hs_t, fmc_to_t = fmc_hs * frame_interval, fmc_to * frame_interval

        # Qualisys HS/TO
        fig.add_trace(
            go.Scatter(
                x=q_hs_t,
                y=q_hs_y,
                mode="markers",
                marker=dict(symbol="x", size=7, color=qual_color),
                name="Qualisys HS" if (row == 3 and col == 1) else "",
                showlegend=(row == 3 and col == 1),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=q_to_t,
                y=q_to_y,
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=7,
                    color=qual_color,
                    line=dict(width=1.5, color=qual_color),
                ),
                name="Qualisys TO" if (row == 3 and col == 1) else "",
                showlegend=(row == 3 and col == 1),
            ),
            row=row,
            col=col,
        )

        # FreeMoCap HS/TO: normal vs flagged
        hs_normal = ~hs_flag
        to_normal = ~to_flag

        fig.add_trace(
            go.Scatter(
                x=fmc_hs_t[hs_normal],
                y=fmc_hs_y[hs_normal],
                mode="markers",
                marker=dict(symbol="x", size=7, color=fmc_color),
                name="FreeMoCap HS" if (row == 3 and col == 1) else "",
                showlegend=(row == 3 and col == 1),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=fmc_hs_t[hs_flag],
                y=fmc_hs_y[hs_flag],
                mode="markers",
                marker=dict(symbol="x", size=9, color=cluster_color),
                name="FreeMoCap HS (flagged)" if (row == 3 and col == 1) else "",
                showlegend=(row == 3 and col == 1),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=fmc_to_t[to_normal],
                y=fmc_to_y[to_normal],
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=7,
                    color=fmc_color,
                    line=dict(width=1.5, color=fmc_color),
                ),
                name="FreeMoCap TO" if (row == 3 and col == 1) else "",
                showlegend=(row == 3 and col == 1),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=fmc_to_t[to_flag],
                y=fmc_to_y[to_flag],
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=9,
                    color=cluster_color,
                    line=dict(width=1.5, color=cluster_color),
                ),
                name="FreeMoCap TO (flagged)" if (row == 3 and col == 1) else "",
                showlegend=(row == 3 and col == 1),
            ),
            row=row,
            col=col,
        )

        fig.update_yaxes(
            row=row,
            col=col,
            tickmode="array",
            tickvals=[0.3, 0.2, 0.0, -0.1],
            ticktext=[
                f"{foot_label} HS (Qualisys)",
                f"{foot_label} HS (FMC)",
                f"{foot_label} TO (Qualisys)",
                f"{foot_label} TO (FMC)",
            ],
            range=[-0.2, 0.4],
            showgrid=False,
        )

    # ------------------------------------------------------------------
    # Create mega figure layout
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=(
            "Left foot AP position",
            "Right foot AP position",
            "Left foot vertical position",
            "Right foot vertical position",
            "Left foot gait events",
            "Right foot gait events",
        ),
    )

    # Row 1: AP (Y) position
    _add_position_panel(
        fig,
        row=1,
        col=1,
        heel_pos=freemocap_kinematics.left_heel_pos,
        toe_pos=freemocap_kinematics.left_toe_pos,
        hs_idx=left_hs_all,
        to_idx=left_to_all,
        hs_flag=left_hs_flag,
        to_flag=left_to_flag,
        sampling_rate=sampling_rate,
        axis=1,  # Y / AP
        ylabel="AP position (Y)",
        panel_title="Left foot AP (Y)",
    )

    _add_position_panel(
        fig,
        row=1,
        col=2,
        heel_pos=freemocap_kinematics.right_heel_pos,
        toe_pos=freemocap_kinematics.right_toe_pos,
        hs_idx=right_hs_all,
        to_idx=right_to_all,
        hs_flag=right_hs_flag,
        to_flag=right_to_flag,
        sampling_rate=sampling_rate,
        axis=1,
        ylabel="AP position (Y)",
        panel_title="Right foot AP (Y)",
    )

    # Row 2: Vertical (Z) position
    _add_position_panel(
        fig,
        row=2,
        col=1,
        heel_pos=freemocap_kinematics.left_heel_pos,
        toe_pos=freemocap_kinematics.left_toe_pos,
        hs_idx=left_hs_all,
        to_idx=left_to_all,
        hs_flag=left_hs_flag,
        to_flag=left_to_flag,
        sampling_rate=sampling_rate,
        axis=2,  # Z / height
        ylabel="Vertical position (Z)",
        panel_title="Left foot vertical (Z)",
    )

    _add_position_panel(
        fig,
        row=2,
        col=2,
        heel_pos=freemocap_kinematics.right_heel_pos,
        toe_pos=freemocap_kinematics.right_toe_pos,
        hs_idx=right_hs_all,
        to_idx=right_to_all,
        hs_flag=right_hs_flag,
        to_flag=right_to_flag,
        sampling_rate=sampling_rate,
        axis=2,
        ylabel="Vertical position (Z)",
        panel_title="Right foot vertical (Z)",
    )

    # Row 3: Event timelines vs Qualisys
    _add_event_timeline_panel(
        fig,
        row=3,
        col=1,
        q_hs=qualisys_gait_events.left_foot.heel_strikes,
        q_to=qualisys_gait_events.left_foot.toe_offs,
        fmc_hs=left_hs_all,
        fmc_to=left_to_all,
        hs_flag=left_hs_flag,
        to_flag=left_to_flag,
        frame_interval=frame_interval,
        foot_label="Left",
    )

    _add_event_timeline_panel(
        fig,
        row=3,
        col=2,
        q_hs=qualisys_gait_events.right_foot.heel_strikes,
        q_to=qualisys_gait_events.right_foot.toe_offs,
        fmc_hs=right_hs_all,
        fmc_to=right_to_all,
        hs_flag=right_hs_flag,
        to_flag=right_to_flag,
        frame_interval=frame_interval,
        foot_label="Right",
    )

    # Shared x-axis & layout
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)

    if xlim is not None:
        for r in range(1, 4):
            for c in range(1, 3):
                fig.update_xaxes(range=list(xlim), row=r, col=c)

    fig.update_layout(
        title=title,
        height=900,
        margin=dict(l=60, r=40, t=80, b=50),
        legend=dict(orientation="h", y=1.08, yanchor="bottom", x=1.0, xanchor="right"),
    )

    return fig
