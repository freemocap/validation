import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# =========================
# Paper-ready figure params
# =========================
DPI = 300
FIG_W_IN = 2         # typical single-column ~3.5", double-column ~7"
FIG_H_IN = 3         # adjust as needed
FIG_W_PX = int(FIG_W_IN * DPI)
FIG_H_PX = int(FIG_H_IN * DPI)

EXPORT_BASENAME = "com_velocity_violin"  # writes PNG + PDF

root_path = Path(r"D:\validation_public_release_v1\figures")
root_path.mkdir(exist_ok=True, parents=True)
# -------------------
# Load paths from DB
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
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE '%balance_velocities'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    condition = row.get("condition") or ""
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_csv(path)

    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["condition"] = condition
    sub_df["tracker"] = tracker
    dfs.append(sub_df)

final_df = pd.concat(dfs, ignore_index=True)

id_cols = ["participant_code", "trial_name", "Frame", "tracker"]

# all the condition+axis columns
value_cols = [c for c in final_df.columns if ("Eyes" in c or "Ground" in c or "Foam" in c)]

long_df = final_df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="cond_axis",
    value_name="velocity",
)

# split "Eyes Open/Solid Ground_x" → condition="Eyes Open/Solid Ground", axis="x"
long_df[["condition", "axis"]] = long_df["cond_axis"].str.rsplit("_", n=1, expand=True)

# drop NaNs (frames outside that condition)
long_df = long_df.dropna(subset=["velocity"])

# -------------------
# Plot configuration
# -------------------
colors = {
    "qualisys":  "#7A7A7A",   # neutral reference gray
    "mediapipe": "#014E9C",   # FreeMoCap blue
}

condition_order = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

tickvals = condition_order
ticktext = [
    "Eyes Open<br>Solid Ground",
    "Eyes Closed<br>Solid Ground",
    "Eyes Open<br>Foam",
    "Eyes Closed<br>Foam",
]

legend_labels = [
    "Eyes Open <br<"
]

axis_order = ["x", "y", "z"]
axis_titles = {
    "x": "Mediolateral center-of-mass velocity (X)",
    "y": "Anteroposterior center-of-mass velocity (Y)",
    "z": "Vertical center-of-mass velocity (Z)",
}

# -------------------
# Build combined figure
# -------------------
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[axis_titles[a] for a in axis_order],
)

# enforce condition ordering globally
long_df["condition"] = pd.Categorical(long_df["condition"], categories=condition_order, ordered=True)

for r, axis in enumerate(axis_order, start=1):
    df_axis = long_df[long_df["axis"] == axis].copy()

    # Qualisys on the left
    df_qs = df_axis[df_axis["tracker"] == "qualisys"]

    fig.add_trace(
        go.Violin(
            x=df_qs["condition"],
            y=df_qs["velocity"],
            legendgroup="qualisys",
            scalegroup=f"axis_{axis}",
            name="Qualisys",
            side="negative",
            line_color=colors["qualisys"],
            width=0.48,
            showlegend=(r == 1),
            opacity=0.60,
            spanmode="hard",
        ),
        row=r,
        col=1,
    )

    # FreeMoCap on the right
    df_fmc = df_axis[df_axis["tracker"] == "mediapipe"]

    fig.add_trace(
        go.Violin(
            x=df_fmc["condition"],
            y=df_fmc["velocity"],
            legendgroup="freemocap",
            scalegroup=f"axis_{axis}",
            name="FreeMoCap",
            side="positive",
            line_color=colors["mediapipe"],
            width=0.48,
            showlegend=(r == 1),
            opacity=0.80,
            spanmode="hard",
        ),
        row=r,
        col=1,
    )

    fig.update_yaxes(
        title_text="COM velocity (mm/s)",
        row=r,
        col=1,
    )

    # Symmetric limits with padding
    axis_values = df_axis["velocity"].dropna()
    max_abs = axis_values.abs().max()

    fig.update_yaxes(
        range=[-max_abs * 1.12, max_abs * 1.12],
        row=r,
        col=1,
    )

fig.update_traces(
    box_visible=True,
    meanline_visible=True,
    points=False,
    scalemode="width",
    meanline=dict(width=2),
)

# Layout: paper-like
fig.update_layout(
    template="simple_white",     # cleaner than plotly_white
    width=FIG_W_PX,
    height=FIG_H_PX,
    margin=dict(l=80, r=20, t=80, b=85),
    font=dict(size=12),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.12,
        xanchor="center",
        x=0.5,
        title_text="",
    ),
)

# Share x tick labels only on bottom row
fig.update_xaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(showticklabels=False, row=2, col=1)
fig.update_xaxes(title_text="Condition", row=3, col=1)

fig.update_yaxes(range=[-75, 75], row=1, col=1)
fig.update_yaxes(range=[-75, 75], row=2, col=1)
fig.update_yaxes(range=[-75, 75], row=3, col=1)

# Ensure condition order on x-axis
fig.update_xaxes(categoryorder="array", categoryarray=condition_order)
fig.update_xaxes(
    row=3, col=1,
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    tickfont=dict(size=12),   # try 11 if still tight
    automargin=True,
)


# Optional: tighten the whitespace between category labels
fig.update_xaxes(tickangle=0)

# fig.show()


# -------------------
# Export at 300 dpi
# -------------------
# pip install -U kaleido
fig.write_image(root_path / f"{EXPORT_BASENAME}.png", width=FIG_W_PX, height=FIG_H_PX, scale=3)
# fig.write_image(f"{EXPORT_BASENAME}.pdf", width=FIG_W_PX, height=FIG_H_PX, scale=1)

