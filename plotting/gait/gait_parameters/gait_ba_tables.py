"""
Outputs:
  1. Main table   — pooled across all speeds 
  2. Supplementary — stratified by speed condition
"""

from gait_ba_utils import (
    load_paired_gait_data, ba_stats, compute_icc_2_1,
    ALL_METRICS, TRACKERS, TRACKER_LABELS,
    SPEED_ORDER, SPEED_LABELS,
)
import numpy as np
import pingouin as pg
import pandas as pd

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
paired_df = load_paired_gait_data("validation.db")

# ------------------------------------------------------------------
# Compute stats for a given slice
# ------------------------------------------------------------------
def compute_row(df_slice, metric_key, units, scale, tracker):
    """Compute BA + ICC stats for one metric/tracker combination."""
    sub:pd.DataFrame = df_slice.query("metric == @metric_key and tracker == @tracker").dropna(
        subset=["tracker_value", "reference_value"]
    )
    if sub.empty:
        return None

    

    diffs = sub["ba_diff"].values * scale
    stats = ba_stats(diffs)
    tracker_rows = sub[['participant_code', 'trial_name', 'condition', 'side', 'metric', 'event_index', 'tracker']].copy()
    tracker_rows['value'] = sub['tracker_value']

    ref_rows = sub[['participant_code', 'trial_name', 'condition', 'side', 'metric', 'event_index']].copy()
    ref_rows['tracker'] = 'qualisys'
    ref_rows['value'] = sub['reference_value']

    long_df = pd.concat([tracker_rows, ref_rows], ignore_index=True)

    long_df['target'] = (
        long_df['trial_name'] + '|' +
        long_df['condition'] + '|' +
        long_df['side'] + '|' +
        long_df['event_index'].astype(str)
    )
    icc_overall = pg.intraclass_corr(
        data = long_df,
        targets = "target",
        raters = "tracker",
        ratings = "value"
    )
    icc_overall = icc_overall.set_index("Type")
    icc_result = icc_overall.loc["ICC(A,1)", ["ICC", "CI95"]]

    return dict(
        n=stats["n"],
        bias=stats["bias"],
        sd=stats["sd"],
        loa_lower=stats["loa_lower"],
        loa_upper=stats["loa_upper"],
        icc=icc_result["ICC"],
        icc_lb=icc_result["CI95"][0],
        icc_ub=icc_result["CI95"][1],
        units=units,
    )


# ------------------------------------------------------------------
# 1. Pooled table (all speeds)
# ------------------------------------------------------------------
print("=" * 80)
print("MAIN TABLE — Pooled across all speeds")
print("=" * 80)

pooled_rows = []
for m_key, m_label, m_units, m_scale in ALL_METRICS:
    for tracker in TRACKERS:
        row = compute_row(paired_df, m_key, m_units, m_scale, tracker)
        if row is None:
            continue
        row["metric"] = m_label
        row["tracker"] = TRACKER_LABELS[tracker]
        pooled_rows.append(row)

# Print readable summary
fmt_header = f"{'Metric':<20s} {'Tracker':<12s} {'N':>5s}  {'Bias':>8s} {'SD':>8s} {'LoA Low':>8s} {'LoA Up':>8s}  {'ICC':>6s} ({'LB':>5s}, {'UB':>5s})"
print(fmt_header)
print("-" * len(fmt_header))
for r in pooled_rows:
    print(
        f"{r['metric']:<20s} {r['tracker']:<12s} {r['n']:>5d}  "
        f"{r['bias']:>+8.2f} {r['sd']:>8.2f} {r['loa_lower']:>+8.2f} {r['loa_upper']:>+8.2f}  "
        f"{r['icc']:>6.3f} ({r['icc_lb']:>5.3f}, {r['icc_ub']:>5.3f})  {r['units']}"
    )


# ------------------------------------------------------------------
# 2. Supplementary table — stratified by speed
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUPPLEMENTARY TABLE — Stratified by speed")
print("=" * 80)

speed_rows = []
for m_key, m_label, m_units, m_scale in ALL_METRICS:
    for speed in SPEED_ORDER:
        df_speed = paired_df[paired_df["condition"] == speed]
        for tracker in TRACKERS:
            row = compute_row(df_speed, m_key, m_units, m_scale, tracker)
            if row is None:
                continue
            row["metric"] = m_label
            row["speed"] = SPEED_LABELS[speed]
            row["tracker"] = TRACKER_LABELS[tracker]
            speed_rows.append(row)

fmt_header2 = f"{'Metric':<20s} {'Speed':<8s} {'Tracker':<12s} {'N':>5s}  {'Bias':>8s} {'SD':>8s} {'LoA Low':>8s} {'LoA Up':>8s}  {'ICC':>6s} ({'LB':>5s}, {'UB':>5s})"
print(fmt_header2)
print("-" * len(fmt_header2))
for r in speed_rows:
    print(
        f"{r['metric']:<20s} {r['speed']:<8s} {r['tracker']:<12s} {r['n']:>5d}  "
        f"{r['bias']:>+8.2f} {r['sd']:>8.2f} {r['loa_lower']:>+8.2f} {r['loa_upper']:>+8.2f}  "
        f"{r['icc']:>6.3f} ({r['icc_lb']:>5.3f}, {r['icc_ub']:>5.3f})  {r['units']}"
    )


# ------------------------------------------------------------------
# 3. Export as Typst tables
# ------------------------------------------------------------------
from pathlib import Path

save_root = Path(r"D:\validation_public_release_v1\tables")
save_root.mkdir(exist_ok=True, parents=True)

supp_root = Path(r"D:\validation_public_release_v1\tables")

def fmt_val(v, decimals=2):
    """Format a float with sign for bias, or plain for others."""
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def fmt_bias(v, decimals=2):
    if np.isnan(v):
        return "—"
    return f"{v:+.{decimals}f}"


def fmt_icc(v, decimals=3):
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


# --- Main table (pooled) ---
typst_lines = []
typst_lines.append('#figure(')
typst_lines.append('  {')
typst_lines.append('    set text(size: 9pt)')
typst_lines.append('    table(')
typst_lines.append('      columns: (auto, auto, auto, auto, auto, auto),')
typst_lines.append('      align: (left, left, right, right, right, right),')
typst_lines.append('      stroke: none,')
typst_lines.append('      table.hline(stroke: 1pt),')
typst_lines.append('      table.header(')
typst_lines.append('        [*Metric*], [*Tracker*], [*Bias*], [*Lower LoA*], [*Upper LoA*], [*ICC (95% CI)*],')
typst_lines.append('      ),')
typst_lines.append('      table.hline(stroke: 0.5pt),')

prev_metric = None
for r in pooled_rows:
    if prev_metric is not None and r["metric"] != prev_metric:
        typst_lines.append('      table.hline(stroke: 0.3pt),')
    prev_metric = r["metric"]

    metric_cell = f'[{r["metric"]} ({r["units"]})]' if r["tracker"] == "MediaPipe" else '[]'
    icc_str = f'{fmt_icc(r["icc"])} ({fmt_icc(r["icc_lb"])}, {fmt_icc(r["icc_ub"])})'

    typst_lines.append(
        f'      {metric_cell}, [{r["tracker"]}], '
        f'[{fmt_bias(r["bias"])}], '
        f'[{fmt_bias(r["loa_lower"])}], [{fmt_bias(r["loa_upper"])}], '
        f'[{icc_str}],'
    )

typst_lines.append('      table.hline(stroke: 1pt),')
typst_lines.append('    )')
typst_lines.append('  },')
typst_lines.append('  caption: [Bland-Altman agreement statistics for spatiotemporal gait parameters across all walking speeds. Bias and limits of agreement (LoA) are reported in the native units of each metric. ICC = intraclass correlation coefficient (2,1) with 95% confidence interval.],')
typst_lines.append(') <tbl-ba-gait-pooled>')

typst_main = "\n".join(typst_lines)
main_path = save_root / "ba_gait_pooled.typ"
main_path.write_text(typst_main, encoding="utf-8")
print(f"\nMain table written to: {main_path}")

# --- Supplementary tables (split: spatial vs temporal) ---

SPATIAL_METRICS = {"Stride Length", "Step Length", "Stride Length"}
TEMPORAL_METRICS = {"Stance Duration", "Swing Duration", "Stride Duration"}

def write_speed_table(rows, filename, label, caption):
    """Write a by-speed Typst table from a filtered subset of speed_rows."""
    lines = []
    lines.append('#figure(')
    lines.append('  {')
    lines.append('    set text(size: 9pt)')
    lines.append('    table(')
    lines.append('      columns: (auto, auto, auto, auto, auto, auto, auto),')
    lines.append('      align: (left, left, left, right, right, right, right),')
    lines.append('      stroke: none,')
    lines.append('      table.hline(stroke: 1pt),')
    lines.append('      table.header(')
    lines.append('        [*Metric*], [*Speed*], [*Backend*], [*Bias*], [*Lower LoA*], [*Upper LoA*], [*ICC (95% CI)*],')
    lines.append('      ),')
    lines.append('      table.hline(stroke: 0.5pt),')

    prev_metric = None
    prev_speed = None
    for r in rows:
        if prev_metric is not None and r["metric"] != prev_metric:
            lines.append('      table.hline(stroke: 0.5pt),')
        elif prev_speed is not None and r["speed"] != prev_speed:
            lines.append('      table.hline(stroke: 0.3pt),')

        metric_cell = f'[{r["metric"]} ({r["units"]})]' if r["metric"] != prev_metric else '[]'
        speed_cell = f'[{r["speed"]}]' if r["speed"] != prev_speed or r["metric"] != prev_metric else '[]'

        prev_metric = r["metric"]
        prev_speed = r["speed"]

        icc_str = f'{fmt_icc(r["icc"])} ({fmt_icc(r["icc_lb"])}, {fmt_icc(r["icc_ub"])})'

        lines.append(
            f'      {metric_cell}, {speed_cell}, [{r["tracker"]}], '
            f'[{fmt_bias(r["bias"])}], '
            f'[{fmt_bias(r["loa_lower"])}], [{fmt_bias(r["loa_upper"])}], '
            f'[{icc_str}],'
        )

    lines.append('      table.hline(stroke: 1pt),')
    lines.append('    )')
    lines.append('  },')
    lines.append(f'  caption: [{caption}],')
    lines.append(f') <{label}>')

    out_path = supp_root / filename
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out_path}")


spatial_rows = [r for r in speed_rows if r["metric"] in SPATIAL_METRICS]
temporal_rows = [r for r in speed_rows if r["metric"] in TEMPORAL_METRICS]

write_speed_table(
    spatial_rows,
    "ba_gait_by_speed_spatial.typ",
    "tbl-ba-gait-by-speed-spatial",
    "Bland-Altman agreement statistics for spatial gait parameters (stride length, step length) stratified by walking speed. Bias and limits of agreement (LoA) are reported in millimeters. ICC = intraclass correlation coefficient (2,1) with 95% confidence interval.",
)

write_speed_table(
    temporal_rows,
    "ba_gait_by_speed_temporal.typ",
    "tbl-ba-gait-by-speed-temporal",
    "Bland-Altman agreement statistics for temporal gait parameters (stance duration, swing duration) stratified by walking speed. Bias and limits of agreement (LoA) are reported in milliseconds. ICC = intraclass correlation coefficient (2,1) with 95% confidence interval.",
)