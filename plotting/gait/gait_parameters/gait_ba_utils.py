"""
Shared utilities for gait metric Bland-Altman analysis.

Handles: DB query, pivot/melt to paired format, BA statistics,
         ICC calculation, and Plotly figure styling.
"""

import sqlite3
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
REFERENCE_SYSTEM = "qualisys"
TRACKERS = ["mediapipe", "rtmpose", "vitpose"]

TRACKER_LABELS = {
    "mediapipe": "MediaPipe",
    "rtmpose":   "RTMPose",
    "vitpose":   "ViTPose",
}

# All gait metrics we compute BA stats for
ALL_METRICS = [
    # spatial (mm)
    ("stride_length", "Stride Length",    "mm", 1.0),
    ("step_length",   "Step Length",      "mm", 1.0),
    # temporal (s → ms)
    ("stance_duration", "Stance Duration", "ms", 1000.0),
    ("swing_duration",  "Swing Duration",  "ms", 1000.0),
    ("stride_duration", "Stride Duration", "ms", 1000.0),
]

# Speed conditions — viridis sequential palette (perceptually uniform, colorblind-safe)
SPEED_ORDER = ["speed_0_5", "speed_1_0", "speed_1_5", "speed_2_0", "speed_2_5"]
SPEED_STYLE = {
    "speed_0_5": dict(label="0.5 m/s", color="#440154"),  # purple
    "speed_1_0": dict(label="1.0 m/s", color="#3b528b"),  # blue
    "speed_1_5": dict(label="1.5 m/s", color="#21918c"),  # teal
    "speed_2_0": dict(label="2.0 m/s", color="#5ec962"),  # green
    "speed_2_5": dict(label="2.5 m/s", color="#F4A143"),  # orange
}

SPEED_LABELS = {k: v["label"] for k, v in SPEED_STYLE.items()}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
GAIT_METRICS_QUERY = """
SELECT t.participant_code,
       t.trial_name,
       a.path,
       a.component_name,
       a.condition,
       a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "treadmill"
  AND a.category = "gait_metrics"
  AND a.tracker IN ("mediapipe", "rtmpose", "vitpose", "qualisys")
  AND a.file_exists = 1
  AND a.component_name LIKE "%gait_metrics"
ORDER BY t.trial_name, a.path
"""


def load_paired_gait_data(db_path="validation.db"):
    """Load gait metrics from DB, pivot to wide, melt to paired format.

    Returns a DataFrame with columns:
        participant_code, trial_name, condition, side, metric, event_index,
        reference_value, tracker, tracker_value, ba_mean, ba_diff
    """
    conn = sqlite3.connect(db_path)
    path_df = pd.read_sql_query(GAIT_METRICS_QUERY, conn)
    conn.close()

    dfs = []
    for _, row in path_df.iterrows():
        sub = pd.read_csv(row["path"])
        sub["participant_code"] = row["participant_code"]
        sub["trial_name"] = row["trial_name"].lower()
        sub["condition"] = row["condition"] or "none"
        dfs.append(sub)

    df = pd.concat(dfs, ignore_index=True)

    id_cols = [
        "participant_code", "trial_name", "condition",
        "side", "metric", "event_index",
    ]

    wide = (
        df.pivot_table(
            index=id_cols, columns="system",
            values="value", aggfunc="first",
        )
        .reset_index()
    )
    wide = wide.rename(columns={REFERENCE_SYSTEM: "reference_value"})

    tracker_cols = [t for t in TRACKERS if t in wide.columns]

    paired = wide.melt(
        id_vars=id_cols + ["reference_value"],
        value_vars=tracker_cols,
        var_name="tracker",
        value_name="tracker_value",
    )

    paired["ba_mean"] = (paired["tracker_value"] + paired["reference_value"]) / 2
    paired["ba_diff"] = paired["tracker_value"] - paired["reference_value"]

    return paired


# ------------------------------------------------------------------
# Bland-Altman statistics
# ------------------------------------------------------------------
def ba_stats(diffs):
    """Compute bias, SD, and 95% LoA from an array of differences.

    Returns dict with keys: n, bias, sd, loa_lower, loa_upper
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return dict(n=0, bias=np.nan, sd=np.nan, loa_lower=np.nan, loa_upper=np.nan)
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    return dict(
        n=n,
        bias=bias,
        sd=sd,
        loa_lower=bias - 1.96 * sd,
        loa_upper=bias + 1.96 * sd,
    )


def compute_icc_2_1(reference, test):
    """Compute ICC(2,1) — two-way random, single measures, absolute agreement.

    Parameters
    ----------
    reference, test : array-like, same length

    Returns
    -------
    dict with keys: icc, lbound, ubound (95% CI)
    """
    from scipy import stats as sp_stats

    ref = np.asarray(reference, dtype=float)
    tst = np.asarray(test, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(tst)
    ref, tst = ref[mask], tst[mask]
    n = len(ref)

    if n < 3:
        return dict(icc=np.nan, lbound=np.nan, ubound=np.nan)

    k = 2  # two raters (reference + test)
    # Stack as (n, k) matrix
    data = np.column_stack([ref, tst])
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # Sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)      # BMS
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)       # JMS
    ss_error = ss_total - ss_rows - ss_cols                    # EMS

    ms_rows = ss_rows / (n - 1)       # BMS
    ms_cols = ss_cols / (k - 1)       # JMS
    ms_error = ss_error / ((n - 1) * (k - 1))  # EMS

    # ICC(2,1) formula
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)

    # F-test for confidence interval (Shrout & Fleiss, 1979)
    F_val = ms_rows / ms_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    F_lo = F_val / sp_stats.f.ppf(0.975, df1, df2)
    F_hi = F_val / sp_stats.f.ppf(0.025, df1, df2)

    lbound = (F_lo - 1) / (F_lo + k - 1)
    ubound = (F_hi - 1) / (F_hi + k - 1)

    return dict(icc=float(icc), lbound=float(lbound), ubound=float(ubound))


# ------------------------------------------------------------------
# Plotly figure helpers
# ------------------------------------------------------------------
def inches_to_px(inches, dpi=300):
    return int(inches * dpi)


def style_paperish(fig, *, width_px, height_px):
    BASE, TICK = 14, 12
    fig.update_layout(
        template="simple_white",
        width=width_px, height=height_px,
        font=dict(family="Arial", size=BASE, color="black"),
        margin=dict(l=90, r=10, t=32, b=58),
    )
    fig.update_xaxes(
        tickfont=dict(size=TICK), title_font=dict(size=BASE),
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=3, showgrid=False, zeroline=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=TICK), title_font=dict(size=BASE),
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=3, showgrid=False, zeroline=False,
    )
    return fig
