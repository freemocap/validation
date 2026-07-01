import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import chi2



from pathlib import Path

table_path = Path(r"D:\validation_public_release_v1\tables")
table_path.mkdir(exist_ok=True, parents=True)


conn = sqlite3.connect("validation.db")
out_file = table_path / "balance_metrics_table.typ"


conditions = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

# --- Confidence ellipse ---
def confidence_ellipse_95(x, y):
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


# =====================
# 1. Ellipse area
# =====================
pos_query = """
SELECT t.participant_code, t.trial_name, a.path, a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%balance_positions'
ORDER BY t.trial_name, a.path;
"""
pos_path_df = pd.read_sql_query(pos_query, conn)

pos_dfs = []
for _, row in pos_path_df.iterrows():
    sub_df = pd.read_csv(row["path"])
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    pos_dfs.append(sub_df)

pos_df = pd.concat(pos_dfs, ignore_index=True)

ellipse_results = []
for (participant, trial, tracker), grp in pos_df.groupby(
    ["participant_code", "trial_name", "tracker"]
):
    for condition in conditions:
        x_col = f"{condition}_x"
        y_col = f"{condition}_y"
        if x_col not in grp.columns:
            continue

        x = grp[x_col].to_numpy()
        y = grp[y_col].to_numpy()

        x = x - np.mean(x)
        y = y - np.mean(y)

        _, a, b, theta, area = confidence_ellipse_95(x, y)

        ellipse_results.append({
            "participant": participant,
            "trial": trial,
            "tracker": tracker,
            "condition": condition,
            "ellipse_area_mm2": area,
        })

ellipse_df = pd.DataFrame(ellipse_results)


# =====================
# 2. Path length
# =====================
pl_query = """
SELECT t.participant_code, t.trial_name, a.path, a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%path_length_com'
ORDER BY t.trial_name, a.path;
"""
pl_path_df = pd.read_sql_query(pl_query, conn)

pl_dfs = []
for _, row in pl_path_df.iterrows():
    sub_df = pd.read_json(row["path"])
    sub_df = sub_df.rename(columns={
        "Frame Intervals": "frame_interval",
        "Path Lengths:": "path_length",
    }).reset_index().rename(columns={"index": "condition"})

    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    pl_dfs.append(sub_df)

pl_df = pd.concat(pl_dfs, ignore_index=True)


# =====================
# 3. Mean 2D velocity
# =====================
vel_query = """
SELECT t.participant_code, t.trial_name, a.path, a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%balance_velocities'
ORDER BY t.trial_name, a.path;
"""
vel_path_df = pd.read_sql_query(vel_query, conn)

vel_dfs = []
for _, row in vel_path_df.iterrows():
    sub_df = pd.read_csv(row["path"])
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    vel_dfs.append(sub_df)

vel_df = pd.concat(vel_dfs, ignore_index=True)

id_cols = ["participant_code", "trial_name", "Frame", "tracker"]
value_cols = [c for c in vel_df.columns if ("Eyes" in c or "Ground" in c or "Foam" in c)]

long_df = vel_df.melt(id_vars=id_cols, value_vars=value_cols,
                       var_name="cond_axis", value_name="velocity")
long_df[["condition", "axis"]] = long_df["cond_axis"].str.rsplit("_", n=1, expand=True)
long_df = long_df.dropna(subset=["velocity"])

xy_df = long_df[long_df["axis"].isin(["x", "y"])].copy()
xy_wide = (
    xy_df.pivot_table(
        index=["participant_code", "trial_name", "Frame", "tracker", "condition"],
        columns="axis", values="velocity", aggfunc="first",
    ).reset_index()
)
xy_wide = xy_wide.dropna(subset=["x", "y"])
xy_wide["velocity_2d"] = np.sqrt(xy_wide["x"]**2 + xy_wide["y"]**2)

vel_trial = (
    xy_wide
    .groupby(["participant_code", "trial_name", "tracker", "condition"],
             as_index=False)["velocity_2d"]
    .mean()
    .rename(columns={"velocity_2d": "mean_velocity_2d"})
)

conn.close()


# =====================
# Combined summary table
# =====================
def summarize(df, value_col, participant_col="participant"):
    return (
        df.groupby(["tracker", "condition", participant_col])[value_col]
        .mean()
        .reset_index()
        .groupby(["tracker", "condition"])[value_col]
        .agg(["mean", "std"])
        .reset_index()
    )

ea_summ = summarize(ellipse_df, "ellipse_area_mm2")
pl_summ = summarize(pl_df, "path_length", "participant_code")
vel_summ = summarize(vel_trial, "mean_velocity_2d", "participant_code")

summary = (
    pl_summ.rename(columns={"mean": "pl_mean", "std": "pl_std"})
    .merge(ea_summ.rename(columns={"mean": "ea_mean", "std": "ea_std"}),
           on=["tracker", "condition"])
    .merge(vel_summ.rename(columns={"mean": "vel_mean", "std": "vel_std"}),
           on=["tracker", "condition"])
)

summary["Path Length (mm)"] = summary.apply(
    lambda r: f"{r['pl_mean']:.1f} ± {r['pl_std']:.1f}", axis=1)
summary["Ellipse Area (mm²)"] = summary.apply(
    lambda r: f"{r['ea_mean']:.1f} ± {r['ea_std']:.1f}", axis=1)
summary["Mean 2D Velocity (mm/s)"] = summary.apply(
    lambda r: f"{r['vel_mean']:.2f} ± {r['vel_std']:.2f}", axis=1)

print(summary[["tracker", "condition", "Path Length (mm)",
               "Ellipse Area (mm²)", "Mean 2D Velocity (mm/s)"]].to_string(index=False))

tracker_order = ["qualisys", "mediapipe", "rtmpose", "vitpose"]
tracker_labels = {
    "qualisys": "Reference",
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}
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

condition_labels = {
    "Eyes Open/Solid Ground": "EO / Solid",
    "Eyes Closed/Solid Ground": "EC / Solid",
    "Eyes Open/Foam": "EO / Foam",
    "Eyes Closed/Foam": "EC / Foam",
}


def fmt1(mean: float, std: float) -> str:
    return f"{mean:.1f} $plus.minus$ {std:.1f}"


def fmt2(mean: float, std: float) -> str:
    return f"{mean:.2f} $plus.minus$ {std:.2f}"


def generate_balance_metrics_table_typst(
    summary_df: pd.DataFrame,
) -> str:
    """Generate an importable Typst balance-metrics table."""

    lines = [
        "#let balance-metrics = {",
        "  set text(size: 9pt)",
        "  table(",
        "    columns: (auto, auto, auto, auto, auto),",
        "    align: (left, left, center, center, center),",
        "    stroke: none,",
        "    table.hline(stroke: 1pt),",
        "    table.header(",
        "      [*Condition*],",
        "      [*Backend*],",
        "      [*Path Length* \\ (mm)],",
        "      [*Ellipse Area* \\ (mm#super[2])],",
        "      [*Mean Velocity* \\ (mm/s)],",
        "    ),",
        "    table.hline(stroke: 0.5pt),",
    ]

    for condition_index, condition in enumerate(conditions):
        condition_label = condition_labels[condition]

        for tracker_index, tracker in enumerate(tracker_order):
            matching_rows = summary_df[
                (summary_df["tracker"] == tracker)
                & (summary_df["condition"] == condition)
            ]

            if matching_rows.empty:
                raise ValueError(
                    "No summary row found for "
                    f"tracker={tracker!r}, "
                    f"condition={condition!r}"
                )

            row = matching_rows.iloc[0]

            condition_cell = (
                f"[{condition_label}]"
                if tracker_index == 0
                else "[]"
            )

            backend = tracker_labels[tracker]
            path_length = fmt1(
                row["pl_mean"],
                row["pl_std"],
            )
            ellipse_area = fmt1(
                row["ea_mean"],
                row["ea_std"],
            )
            mean_velocity = fmt2(
                row["vel_mean"],
                row["vel_std"],
            )

            lines.append(
                f"    {condition_cell}, "
                f"[{backend}], "
                f"[{path_length}], "
                f"[{ellipse_area}], "
                f"[{mean_velocity}],"
            )

        if condition_index < len(conditions) - 1:
            lines.append(
                "    table.hline(stroke: 0.3pt),"
            )

    lines.extend(
        [
            "    table.hline(stroke: 1pt),",
            "  )",
            "}",
        ]
    )

    return "\n".join(lines) + "\n"


typst_table = generate_balance_metrics_table_typst(
    summary,
)

out_file.write_text(
    typst_table,
    encoding="utf-8",
)

print(f"Table written to {out_file}")