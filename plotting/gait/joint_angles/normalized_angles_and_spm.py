"""
Self-contained joint-angle waveform + SPM{t} combined figure.

Imports data-loading and computation from:
  - joint_angles_plots.load_joint_angle_data()
  - joint_angles_plots.compute_angle_summary()
  - joint_angle_spm.run_spm_paired_ttests()

Outputs:
  - joint_angles_with_spm.png
  - joint_angles_summary.csv
  - spm_paired_ttest_clusters.csv
  - spm_paired_ttest_curves.csv
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotting.gait.joint_angles.joint_angle_spm_utils.joint_angles_plots import (
    load_joint_angle_data,
    compute_angle_summary,
    speed_key,
    speed_label,
    rgba,
    JOINT_ORDER,
    COMP_LABEL,
)
from plotting.gait.joint_angles.joint_angle_spm_utils.joint_angle_spm import run_spm_paired_ttests

# ============================================================
# Config
# ============================================================
root_dir = Path(r"D:\validation_public_release_v1\analyses")
root_dir.mkdir(exist_ok=True, parents=True)

plot_dir = Path(r"D:\validation_public_release_v1\figures")
plot_dir.mkdir(exist_ok=True, parents=True)


REFERENCE = "qualisys"
COMPARE = ["mediapipe", "rtmpose", "vitpose"]

ALPHA = 0.05
TWO_TAILED = True
Q_EXPECTED = 100

COMPONENT_BY_JOINT = {"hip": "flex_ext", "knee": "flex_ext", "ankle": "dorsi_plantar"}

COLORS = {
    "qualisys": "#313131",
    "mediapipe": "#0072B2",
    "rtmpose": "#D55E00",
    "vitpose": "#006D43",
}

# ============================================================
# 1) Load + compute
# ============================================================
print("Loading data from validation.db...")
combined_df = load_joint_angle_data()

print("Computing angle summary...")
angle_summary, df_trial_lr_mean = compute_angle_summary(combined_df) #note - this function flips the knee angles to make flexion positive (as is normal)

print("Running SPM paired t-tests...")
spm_clusters, spm_curves = run_spm_paired_ttests(
    df_trial_lr_mean=df_trial_lr_mean,
    trackers=COMPARE,
    reference=REFERENCE,
    alpha=ALPHA,
    two_tailed=TWO_TAILED,
    q_expected=Q_EXPECTED,
)

# Save intermediate CSVs (so downstream scripts still work)
angle_summary.to_csv(root_dir / "joint_angles_summary.csv", index=False)
spm_clusters.to_csv(root_dir / "spm_paired_ttest_clusters.csv", index=False)
spm_curves.to_csv(root_dir / "spm_paired_ttest_curves.csv", index=False)
print("Saved CSVs to:", root_dir)

# Normalize strings for consistent filtering
for df in (angle_summary, spm_clusters, spm_curves):
    for col in ["condition", "joint", "component", "tracker", "reference"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

SPEEDS = sorted(angle_summary["condition"].unique().tolist(), key=speed_key)

# ============================================================
# 2) Plotting helpers
# ============================================================
def contiguous_true_runs(mask: np.ndarray):
    if mask.size == 0:
        return []
    m = mask.astype(bool)
    d = np.diff(m.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size - 1]
    return list(zip(starts, ends))


def add_suprathreshold_fill(fig, *, x, y, ythr, color_rgba, row, col, showlegend, name):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ythr = float(ythr)
    mask = np.isfinite(x) & np.isfinite(y) & (y > ythr)
    runs = contiguous_true_runs(mask)
    if showlegend:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=10, color=color_rgba),
                       name=name, showlegend=True),
            row=row, col=col,
        )
    for i0, i1 in runs:
        xs = x[i0 : i1 + 1]
        ys = y[i0 : i1 + 1]
        ybase = np.full_like(ys, ythr, dtype=float)
        fig.add_trace(
            go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[ys, ybase[::-1]],
                       mode="lines", line=dict(width=0), fill="toself",
                       fillcolor=color_rgba, hoverinfo="skip", showlegend=False),
            row=row, col=col,
        )


def add_subthreshold_fill(fig, *, x, y, ythr, color_rgba, row, col, showlegend, name):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ythr = float(ythr)
    mask = np.isfinite(x) & np.isfinite(y) & (y < ythr)
    runs = contiguous_true_runs(mask)
    if showlegend:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=10, color=color_rgba),
                       name=name, showlegend=True),
            row=row, col=col,
        )
    for i0, i1 in runs:
        xs = x[i0 : i1 + 1]
        ys = y[i0 : i1 + 1]
        ybase = np.full_like(ys, ythr, dtype=float)
        fig.add_trace(
            go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[ys, ybase[::-1]],
                       mode="lines", line=dict(width=0), fill="toself",
                       fillcolor=color_rgba, hoverinfo="skip", showlegend=False),
            row=row, col=col,
        )


# ============================================================
# 3) Build figure
# ============================================================
n_cols = len(SPEEDS)

SUBPLOT_WIDTH_IN = 1.5
SUBPLOT_HEIGHT_IN = 1.5
SPM_HEIGHT_IN = SUBPLOT_HEIGHT_IN / 1.5
SPACER_HEIGHT_IN = 0.1

DPI = 100

MARGIN_LEFT_IN = 2.0
MARGIN_RIGHT_IN = 0.2
MARGIN_TOP_IN = 0.8
MARGIN_BOTTOM_IN = 1.15

row_kinds = []
for j in range(len(JOINT_ORDER)):
    row_kinds += ["angle", "spm"]
    if j < len(JOINT_ORDER) - 1:
        row_kinds += ["spacer"]

n_rows = len(row_kinds)

FIG_WIDTH_IN = MARGIN_LEFT_IN + n_cols * SUBPLOT_WIDTH_IN + MARGIN_RIGHT_IN
FIG_HEIGHT_IN = (
    MARGIN_TOP_IN
    + len(JOINT_ORDER) * (SUBPLOT_HEIGHT_IN + SPM_HEIGHT_IN)
    + (len(JOINT_ORDER) - 1) * SPACER_HEIGHT_IN
    + MARGIN_BOTTOM_IN
)

FIG_WIDTH_PX = int(FIG_WIDTH_IN * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)

row_heights = []
for kind in row_kinds:
    if kind == "angle":
        row_heights.append(SUBPLOT_HEIGHT_IN)
    elif kind == "spm":
        row_heights.append(SPM_HEIGHT_IN)
    else:
        row_heights.append(SPACER_HEIGHT_IN)
row_heights = (np.array(row_heights) / np.sum(row_heights)).tolist()

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    shared_xaxes=True,
    shared_yaxes=False,
    vertical_spacing=0.02,
    horizontal_spacing=0.02,
    column_titles=[speed_label(s) for s in SPEEDS],
    row_heights=row_heights,
)

joint_to_rows = {}
r = 1
for joint in JOINT_ORDER:
    joint_to_rows[joint] = (r, r + 1)
    r += 2
    if r <= n_rows and row_kinds[r - 1] == "spacer":
        r += 1

# -----------------------
# Traces
# -----------------------
for c, cond in enumerate(SPEEDS, start=1):
    for joint in JOINT_ORDER:
        comp = COMPONENT_BY_JOINT[joint]
        row_angles, row_spm = joint_to_rows[joint]

        # ---- ANGLE waveforms ----
        sub = angle_summary[
            (angle_summary["condition"] == cond)
            & (angle_summary["joint"] == joint)
            & (angle_summary["component"] == comp)
        ].copy()

        for trk in [REFERENCE] + COMPARE:
            s = sub[sub["tracker"] == trk].sort_values("percent_gait_cycle")
            if s.empty:
                continue

            x = s["percent_gait_cycle"].to_numpy(float)
            m = s["mean_angle"].to_numpy(float)
            sd = s["std_angle"].to_numpy(float)
            if np.all(np.isnan(sd)):
                sd = np.zeros_like(m)

            fill_alpha = 0.12 if trk == REFERENCE else 0.18
            line_alpha = 0.60 if trk == REFERENCE else 0.90
            lw = 1.6 if trk == REFERENCE else 2.2

            fig.add_trace(
                go.Scatter(
                    x=x, y=m + sd, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ),
                row=row_angles, col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=m - sd, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=rgba(COLORS[trk], fill_alpha),
                    showlegend=False, hoverinfo="skip",
                ),
                row=row_angles, col=c,
            )

            showleg = row_angles == joint_to_rows["hip"][0] and c == 1
            fig.add_trace(
                go.Scatter(
                    x=x, y=m, mode="lines",
                    line=dict(color=rgba(COLORS[trk], line_alpha), width=lw),
                    name="Reference" if trk == REFERENCE else trk.capitalize(),
                    showlegend=showleg,
                    hovertemplate=f"{trk.capitalize()}<br>%{{x:.0f}}% GC<br>%{{y:.1f}}°<extra></extra>",
                ),
                row=row_angles, col=c,
            )

        # ---- SPM{t} panels ----
        sub_curves = spm_curves[
            (spm_curves["condition"] == cond)
            & (spm_curves["joint"] == joint)
            & (spm_curves["component"] == comp)
            & (spm_curves["reference"] == REFERENCE)
            & (spm_curves["tracker"].isin(COMPARE))
        ].copy()

        if not sub_curves.empty:
            for trk in COMPARE:
                sc = sub_curves[sub_curves["tracker"] == trk].sort_values("percent_gait_cycle")
                if sc.empty:
                    continue

                x = sc["percent_gait_cycle"].to_numpy(float)
                z = sc["spm_t"].to_numpy(float)
                zstar = float(sc["t_star"].iloc[0])

                fig.add_trace(
                    go.Scatter(
                        x=x, y=z, mode="lines",
                        line=dict(color=rgba(COLORS[trk], 0.90), width=2),
                        name=f"{trk.capitalize()} SPM{{t}}",
                        showlegend=False,
                        hovertemplate=f"{trk.capitalize()}<br>%{{x:.0f}}% GC<br>SPM(t): %{{y:.2f}}<extra></extra>",
                    ),
                    row=row_spm, col=c,
                )

                add_suprathreshold_fill(
                    fig, x=x, y=z, ythr=zstar,
                    color_rgba=rgba(COLORS[trk], 0.18),
                    row=row_spm, col=c, showlegend=False,
                    name=f"{trk.capitalize()} significant",
                )

                fig.add_hline(
                    y=zstar,
                    line=dict(color=rgba(COLORS[trk], 0.90), width=1.2, dash="dash"),
                    opacity=0.6, row=row_spm, col=c,
                )

                add_subthreshold_fill(
                    fig, x=x, y=z, ythr=-zstar,
                    color_rgba=rgba(COLORS[trk], 0.18),
                    row=row_spm, col=c, showlegend=False,
                    name=f"{trk.capitalize()} significant (neg)",
                )

                fig.add_hline(
                    y=-zstar,
                    line=dict(color=rgba(COLORS[trk], 0.90), width=1.2, dash="dash"),
                    opacity=0.6, row=row_spm, col=c,
                )

# -----------------------
# Spacer rows: hide axes
# -----------------------
for rr, kind in enumerate(row_kinds, start=1):
    if kind != "spacer":
        continue
    for cc in range(1, n_cols + 1):
        fig.update_xaxes(visible=False, row=rr, col=cc)
        fig.update_yaxes(visible=False, row=rr, col=cc)

# -----------------------
# Row labels
# -----------------------
for joint in JOINT_ORDER:
    comp = COMPONENT_BY_JOINT[joint]
    label = COMP_LABEL[comp]
    row_angles, row_spm = joint_to_rows[joint]
    y_center = 1 - (((row_angles - 0.5) + (row_spm - 0.5)) / (2 * n_rows))
    fig.add_annotation(
        x=-0.10, xref="paper",
        y=y_center, yref="paper",
        text=f"<b>{joint.title()}</b><br>{label} (°)",
        showarrow=False, xanchor="right", align="right",
        font=dict(size=13, color="#333"),
    )

# -----------------------
# Axes formatting
# -----------------------
tickvals = list(range(0, 101, 20))
bottom_row = max(i for i, k in enumerate(row_kinds, start=1) if k != "spacer")

for rr, kind in enumerate(row_kinds, start=1):
    if kind == "spacer":
        continue
    for cc in range(1, n_cols + 1):
        show_xticks = rr == bottom_row
        fig.update_xaxes(
            tickvals=tickvals, tickfont=dict(size=11),
            showticklabels=show_xticks, showgrid=False,
            zeroline=False, showline=True, linecolor="#333", mirror=True,
            row=rr, col=cc,
        )
        fig.update_yaxes(
            showticklabels=(cc == 1), tickfont=dict(size=11),
            showgrid=False, zeroline=False, showline=True,
            linecolor="#333", mirror=True, row=rr, col=cc,
        )

for cc in range(1, n_cols + 1):
    fig.update_xaxes(
        title_text="<b>Gait cycle (%)</b>",
        title_font=dict(size=13, color="#333"),
        title_standoff=6, row=bottom_row, col=cc,
    )

for joint in JOINT_ORDER:
    row_angles, row_spm = joint_to_rows[joint]
    fig.update_yaxes(title_text="Angle (deg)", row=row_angles, col=1)
    fig.update_yaxes(title_text="SPM(t)", row=row_spm, col=1)

# Bold column titles
for ann in fig.layout.annotations:
    if "m/s" in ann.text:
        ann.text = f"<b>{ann.text}</b>"
        ann.font = dict(size=13, color="#333")

# -----------------------
# Shared y-axes per joint
# -----------------------
Y_PAD_FRAC = 0.05

for joint in JOINT_ORDER:
    comp = COMPONENT_BY_JOINT[joint]
    row_angles, row_spm = joint_to_rows[joint]

    sub_ang = angle_summary[
        (angle_summary["joint"] == joint) & (angle_summary["component"] == comp)
    ]
    if not sub_ang.empty:
        ang_lo = (sub_ang["mean_angle"] - sub_ang["std_angle"].fillna(0)).min()
        ang_hi = (sub_ang["mean_angle"] + sub_ang["std_angle"].fillna(0)).max()
        ang_pad = (ang_hi - ang_lo) * Y_PAD_FRAC
        ang_range = [ang_lo - ang_pad, ang_hi + ang_pad]
        for cc in range(1, n_cols + 1):
            fig.update_yaxes(range=ang_range, row=row_angles, col=cc)

    sub_spm = spm_curves[
        (spm_curves["joint"] == joint)
        & (spm_curves["component"] == comp)
        & (spm_curves["reference"] == REFERENCE)
        & (spm_curves["tracker"].isin(COMPARE))
    ]
    if not sub_spm.empty:
        spm_lo = sub_spm["spm_t"].min()
        spm_hi = sub_spm["spm_t"].max()
        tstar_max = sub_spm["t_star"].max()
        spm_lo = min(spm_lo, -tstar_max)
        spm_hi = max(spm_hi, tstar_max)
        max_abs = max(abs(spm_lo), abs(spm_hi))
        spm_range = [-max_abs, max_abs]
        for cc in range(1, n_cols + 1):
            fig.update_yaxes(range=spm_range, row=row_spm, col=cc)

# -----------------------
# Layout
# -----------------------
fig.update_layout(
    template="plotly_white",
    width=FIG_WIDTH_PX,
    height=FIG_HEIGHT_PX,
    margin=dict(
        l=int(MARGIN_LEFT_IN * DPI),
        r=int(MARGIN_RIGHT_IN * DPI),
        t=int(MARGIN_TOP_IN * DPI),
        b=int(MARGIN_BOTTOM_IN * DPI),
    ),
    legend=dict(
        orientation="h", x=0.5, y=-0.08,
        xanchor="center", yanchor="top",
        font=dict(size=14),
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)

fig.show()

# Export
fig.write_image(str(plot_dir / "joint_angles_with_spm.svg"), scale=3)
print("Saved:", plot_dir / "joint_angles_with_spm.svg")