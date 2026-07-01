"""
Joint-angle RMSE tables (trial/participant-level RMSE -> summarized mean±SD)

This script:
1) Loads joint angle per-stride summary_stats CSVs from validation.db artifacts
2) Keeps major sagittal motions: hip/knee flex-ext + ankle dorsi-plantar
3) Collapses L/R within each trial (mean)
4) Computes RMSE across gait-cycle samples for each TRIAL vs reference (Qualisys)
5) Optionally collapses to PARTICIPANT-level per speed (mean across that participant's trials)
6) Summarizes across participants: mean RMSE, SD RMSE, N
7) Prints slide-ready tables for hip/knee/ankle

No plots. Just prints and (optionally) CSV exports.
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------
# Config
# ------------------------
TRACKERS = ["mediapipe", "rtmpose", "vitpose", "qualisys"]
REFERENCE_SYSTEM = "qualisys"

# If True: average RMSE within participant (across multiple trials at same speed) before group summary
# This is usually what you want to avoid participants with more trials dominating the mean.
COLLAPSE_TO_PARTICIPANT_LEVEL = False

# Optional: save tables
EXPORT_TABLES = True
EXPORT_DIR = Path(r"D:\validation_public_release_v1\analyses")
EXPORT_DIR.mkdir(exist_ok=True, parents=True)


TYPST_OUT_DIR = Path(r"D:\validation_public_release_v1\tables")
TYPST_OUT_DIR.mkdir(exist_ok=True, parents=True)


# ------------------------
# 1) Load data from SQLite
# ------------------------
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
WHERE t.trial_type = "treadmill"
  AND a.category = "joint_angles_per_stride"
  AND a.tracker IN ("mediapipe", "rtmpose", "vitpose", "qualisys")
  AND a.file_exists = 1
  AND a.condition LIKE "speed_%"
  AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""
path_df = pd.read_sql_query(query, conn)
if path_df.empty:
    raise RuntimeError("No matching joint angle summary_stats artifacts found from validation.db query.")

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"]
    sub["tracker"] = (row["tracker"] or "").lower()
    sub["condition"] = row["condition"] or "none"
    dfs.append(sub)

combined_df = pd.concat(dfs, ignore_index=True)

# ------------------------
# 2) Normalize + keep only major motions
# ------------------------
for col in ["joint", "side", "tracker", "stat", "component"]:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].astype(str).str.lower()

# normalize component naming
combined_df["component"] = combined_df["component"].replace({"inversion_eversion": "inv_ev"})

MAJOR = {
    ("hip", "flex_ext"),
    ("knee", "flex_ext"),
    ("ankle", "dorsi_plantar"),
}
combined_df = combined_df[
    combined_df.apply(lambda r: (r["joint"], r["component"]) in MAJOR, axis=1)
].copy()

# sanity
required_cols = [
    "participant_code", "trial_name", "tracker", "condition",
    "joint", "component", "percent_gait_cycle", "stat", "value"
]
missing = [c for c in required_cols if c not in combined_df.columns]
if missing:
    raise ValueError(f"Missing required columns in combined_df: {missing}")

# ------------------------
# 3) Parse numeric speed
# ------------------------
def parse_speed(cond: str) -> float:
    """
    Accepts: speed_0_5, speed_1_0, speed_2, speed_2_5, speed_1.5
    Returns float speed in m/s
    """
    s = str(cond)
    m = re.search(r"speed_(\d+)[_\.](\d+)", s)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"speed_(\d+)", s)
    if m2:
        return float(m2.group(1))
    return np.nan

combined_df["speed"] = combined_df["condition"].apply(parse_speed).astype(float)

# ------------------------
# 4) Collapse sides within trial (L/R mean), keep waveform per trial
# ------------------------
# We use stat == "mean" rows as the waveform values (matches your existing approach)
df_means = combined_df[combined_df["stat"] == "mean"].copy()

# trial_mean_angle: mean across side (L/R) at each gait% sample
df_trial_lr_mean = (
    df_means
    .groupby(
        ["speed", "condition", "tracker", "participant_code", "trial_name",
         "joint", "component", "percent_gait_cycle"],
        as_index=False
    )
    .agg(trial_mean_angle=("value", "mean"))
)

# Quick visibility into how many trials per speed/tracker
print("\n# Trials per speed/tracker (n unique trial_name)")
print(
    df_trial_lr_mean
    .groupby(["speed", "tracker"])["trial_name"]
    .nunique()
    .unstack(fill_value=0)
    .sort_index()
)

# ------------------------
# 5) Compute RMSE per TRIAL vs reference across gait-cycle samples
# ------------------------
# Pivot to wide per timepoint so we can compare tracker vs reference within each trial waveform
trial_wide = (
    df_trial_lr_mean
    .pivot_table(
        index=["speed", "condition", "participant_code", "trial_name", "joint", "component", "percent_gait_cycle"],
        columns="tracker",
        values="trial_mean_angle",
        aggfunc="first"
    )
    .reset_index()
)

if REFERENCE_SYSTEM not in trial_wide.columns:
    raise ValueError(f"Reference system '{REFERENCE_SYSTEM}' not present in data columns: {trial_wide.columns.tolist()}")

tracker_cols_present = [t for t in TRACKERS if t in trial_wide.columns]
compare_trackers = [t for t in tracker_cols_present if t != REFERENCE_SYSTEM]
if not compare_trackers:
    raise ValueError("No non-reference trackers found to compare against reference.")

# Melt to paired long format: each row = one gait% sample for one tracker within one trial
trial_paired = (
    trial_wide
    .melt(
        id_vars=["speed", "condition", "participant_code", "trial_name", "joint", "component", "percent_gait_cycle", REFERENCE_SYSTEM],
        value_vars=compare_trackers,
        var_name="tracker",
        value_name="tracker_value"
    )
    .rename(columns={REFERENCE_SYSTEM: "reference_value"})
)

def rmse_1d(tracker_vals: pd.Series, ref_vals: pd.Series) -> float:
    a = np.asarray(tracker_vals, dtype=float)
    b = np.asarray(ref_vals, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

# RMSE per trial: computed over gait-cycle samples (percent_gait_cycle)
rmse_per_trial = (
    trial_paired
    .groupby(["speed", "condition", "participant_code", "trial_name", "joint", "component", "tracker"], as_index=False)
    .apply(lambda g: rmse_1d(g["tracker_value"], g["reference_value"]))
    .rename(columns={None: "rmse"})
)

print("\n# RMSE per trial (head)")
print(rmse_per_trial.head(10).to_string(index=False))

# ------------------------
# 6) Optional: collapse to participant-level per speed (mean across that participant's trials)
# ------------------------
if COLLAPSE_TO_PARTICIPANT_LEVEL:
    rmse_unit = (
        rmse_per_trial
        .groupby(["speed", "condition", "participant_code", "joint", "component", "tracker"], as_index=False)
        .agg(rmse=("rmse", "mean"), n_trials=("trial_name", "nunique"))
    )
    unit_label = "participant"
else:
    rmse_unit = rmse_per_trial.copy()
    rmse_unit["n_trials"] = 1
    unit_label = "trial"

# ------------------------
# 7) Group summary across units (participants or trials)
# ------------------------
rmse_summary = (
    rmse_unit
    .groupby(["speed", "joint", "component", "tracker"], as_index=False)
    .agg(
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        n_units=("rmse", "count"),
        total_trials=("n_trials", "sum"),
    )
)

def export_paste_ready(df: pd.DataFrame, stem: str, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) TSV (best for direct paste + preservuv es columns cleanly)
    tsv_path = out_dir / f"{stem}.tsv"
    df.to_csv(tsv_path, sep="\t", index=True, encoding="utf-8-sig")  # utf-8-sig helps Excel

    # 2) CSV (Excel-friendly encoding)
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=True, encoding="utf-8-sig")

    # 3) Markdown (optional: paste into docs / slides that support it)
    md_path = out_dir / f"{stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown())

    # 4) Copy to clipboard as an Excel-ready tab-separated block (super convenient)
    #    Then you can just Ctrl+V straight into Excel.
    df.to_clipboard(excel=True, sep="\t")

    print(f"Saved:\n  {tsv_path}\n  {csv_path}\n  {md_path}\n  (Also copied to clipboard as TSV)")

# ------------------------
# 8) Slide-ready tables per joint (rows=tracker, cols=speed)
# ------------------------
def joint_rmse_table_mean_sd(rmse_summary_df: pd.DataFrame, joint_name: str) -> pd.DataFrame:
    sub = rmse_summary_df[rmse_summary_df["joint"] == joint_name].copy()
    if sub.empty:
        return pd.DataFrame()

    # Mean table
    mean_tbl = (
        sub.pivot_table(index="tracker", columns="speed", values="rmse_mean", aggfunc="first")
        .sort_index(axis=1)
    )
    # SD table
    sd_tbl = (
        sub.pivot_table(index="tracker", columns="speed", values="rmse_sd", aggfunc="first")
        .sort_index(axis=1)
    )
    # N table
    n_tbl = (
        sub.pivot_table(index="tracker", columns="speed", values="n_units", aggfunc="first")
        .sort_index(axis=1)
    )

    # Format columns (speed labels)
    mean_tbl.columns = [f"{c:g}" for c in mean_tbl.columns]
    sd_tbl.columns = mean_tbl.columns
    n_tbl.columns = mean_tbl.columns

    # Nicely format values
    mean_tbl = mean_tbl.round(1)
    sd_tbl = sd_tbl.round(1)

    # Build "mean (sd)" strings
    fmt = mean_tbl.copy().astype(object)
    for trk in fmt.index:
        for spd in fmt.columns:
            m = mean_tbl.loc[trk, spd]
            s = sd_tbl.loc[trk, spd]
            n = n_tbl.loc[trk, spd]
            if pd.isna(m):
                fmt.loc[trk, spd] = ""
            else:
                # example: 4.1  ± .2
                fmt.loc[trk, spd] = f"{m:.1f} ± {0.0 if pd.isna(s) else s:.1f}"

    # Presentation polish
    fmt.index = fmt.index.str.capitalize()
    fmt.insert(0, "Speed (m/s)", fmt.index)
    fmt = fmt.set_index("Speed (m/s)")

    # Overall across-speed mean/sd of the *speed-level means* (optional quick summary)
    # NOTE: this is across speeds, NOT across participants/trials.
    numeric_means = (
        sub.pivot_table(index="tracker", columns="speed", values="rmse_mean", aggfunc="first")
        .sort_index(axis=1)
    )
    fmt["RMSE Mean"] = numeric_means.mean(axis=1).round(1).values
    fmt["RMSE SD"] = numeric_means.std(axis=1, ddof=1).round(1).values

    return fmt

hip_table   = joint_rmse_table_mean_sd(rmse_summary, "hip")
knee_table  = joint_rmse_table_mean_sd(rmse_summary, "knee")
ankle_table = joint_rmse_table_mean_sd(rmse_summary, "ankle")

print(f"\n==============================")
print(f"RMSE tables (unit = {unit_label}; RMSE computed per {unit_label} vs {REFERENCE_SYSTEM})")
print(f"Cells show: mean (SD) [n=units] at each speed")
print(f"==============================\n")

print("Hip RMSE (°)")
print(hip_table.to_string())

print("\nKnee RMSE (°)")
print(knee_table.to_string())

print("\nAnkle RMSE (°)")
print(ankle_table.to_string())

# ------------------------
# 9) Optional exports
# ------------------------
if EXPORT_TABLES:
    export_paste_ready(hip_table,   "hip_rmse_table",   EXPORT_DIR)
    export_paste_ready(knee_table,  "knee_rmse_table",  EXPORT_DIR)
    export_paste_ready(ankle_table, "ankle_rmse_table", EXPORT_DIR)

    # rmse_per_trial.to_csv(EXPORT_DIR / "rmse_per_trial.csv", index=False)
    # rmse_unit.to_csv(EXPORT_DIR / f"rmse_per_{unit_label}.csv", index=False)
    # rmse_summary.to_csv(EXPORT_DIR / "rmse_summary_across_units.csv", index=False)

    print(f"\nSaved tables + RMSE breakdowns to: {EXPORT_DIR}")

# ------------------------
# Typst table generation
# ------------------------
TRACKER_DISPLAY = {
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}

JOINT_DISPLAY = {
    "hip": "Hip",
    "knee": "Knee",
    "ankle": "Ankle",
}

COMPONENT_DISPLAY = {
    "flex_ext": "Flex/Ext",
    "dorsi_plantar": "Dorsi/Plantar",
}

def generate_typst_joint_angle_rmse_table(rmse_summary: pd.DataFrame) -> str:
    """
    Generate a single Typst file with one table containing all joints.
    Rows: joint (rowspan) x tracker. Columns: speeds.
    Cell values: mean ± SD RMSE (°).
    """
    speeds = sorted(rmse_summary["speed"].dropna().unique())
    speed_labels = [f"{s:g}" for s in speeds]

    trackers_ordered = [t for t in ["mediapipe", "rtmpose", "vitpose"]
                        if t in rmse_summary["tracker"].unique()]
    n_trackers = len(trackers_ordered)

    n_speed_cols = len(speeds)
    col_spec = f"(1fr, 1.5fr, {'0.9fr, ' * (n_speed_cols - 1)}0.9fr)"
    align_spec = f"(left, left, {'center, ' * (n_speed_cols - 1)}center)"

    # Header cells
    header_cells = ["[*Joint*]", "[*Backend*]"]
    for sl in speed_labels:
        header_cells.append(f"[*{sl} m/s*]")

    # Body rows grouped by joint
    body_lines = []
    joints_in_data = [j for j in ["hip", "knee", "ankle"] if j in rmse_summary["joint"].unique()]

    for joint in joints_in_data:
        joint_sub = rmse_summary[rmse_summary["joint"] == joint]
        joint_label = JOINT_DISPLAY.get(joint, joint.title())

        # Get the component for the label suffix
        comp = joint_sub["component"].iloc[0] if "component" in joint_sub.columns else ""
        comp_label = COMPONENT_DISPLAY.get(comp, comp.replace("_", " ").title())
        full_label = f"{joint_label} {comp_label}"

        for i, tracker in enumerate(trackers_ordered):
            row_data = joint_sub[joint_sub["tracker"] == tracker]
            display_name = TRACKER_DISPLAY.get(tracker, tracker.title())

            if i == 0:
                body_lines.append(f"      table.cell(rowspan: {n_trackers}, align: horizon)[{full_label}],")

            body_lines.append(f"      [{display_name}],")
            for spd in speeds:
                vals = row_data[row_data["speed"] == spd]
                if len(vals) > 0:
                    m = vals["rmse_mean"].iloc[0]
                    s = vals["rmse_sd"].iloc[0]
                    if np.isfinite(m):
                        sd_str = f"{s:.1f}" if np.isfinite(s) else "0.0"
                        body_lines.append(f"      [{m:.1f} ± {sd_str}],")
                    else:
                        body_lines.append("      [--],")
                else:
                    body_lines.append("      [--],")

        body_lines.append("      table.hline(stroke: 0.5pt),")

    # Assemble
    lines = []
    lines.append("#figure(")
    lines.append("  {")
    lines.append("    set text(size: 9pt)")
    lines.append("    table(")
    lines.append(f"      columns: {col_spec},")
    lines.append(f"      align: {align_spec},")
    lines.append("      stroke: none,")
    lines.append("      table.hline(stroke: 1pt),")
    lines.append("      table.header(")
    for cell in header_cells:
        lines.append(f"        {cell},")
    lines.append("      ),")
    lines.append("      table.hline(stroke: 0.5pt),")
    lines.extend(body_lines)
    lines.append("      table.hline(stroke: 1pt),")
    lines.append("    )")
    lines.append("  },")
    unit_desc = "trials" if not COLLAPSE_TO_PARTICIPANT_LEVEL else "participants"
    lines.append(f"  caption: [Joint angle RMSE (°) by speed. Values represent mean ± SD across {unit_desc} compared to Qualisys.],")
    lines.append(") <tbl-joint-angle-rmse>")

    return "\n".join(lines) + "\n"


typst_content = generate_typst_joint_angle_rmse_table(rmse_summary)
typst_path = TYPST_OUT_DIR / "joint_angle_rmse_table.typ"
typst_path.write_text(typst_content, encoding="utf-8")
print(f"Saved Typst table to: {typst_path}")