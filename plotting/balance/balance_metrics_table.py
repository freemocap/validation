import pandas as pd
import sqlite3

from pathlib import Path

from validation.paths import DATABASE_PATH, output_dir



# --------------------------------------------------------------------------- #
# Paths / database
# --------------------------------------------------------------------------- #

root_path = output_dir("tables", "balance")


conn = sqlite3.connect(
    DATABASE_PATH
)

out_file = (
    root_path
    / "balance_metrics_table.typ"
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

conditions = [
    "eyes_open_solid",
    "eyes_closed_solid",
    "eyes_open_foam",
    "eyes_closed_foam",
]

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
    "eyes_open_solid": "EO / Solid",
    "eyes_closed_solid": "EC / Solid",
    "eyes_open_foam": "EO / Foam",
    "eyes_closed_foam": "EC / Foam",
}


# --------------------------------------------------------------------------- #
# Load balance metrics
# --------------------------------------------------------------------------- #

metrics_query = """
SELECT
    t.participant_code,
    t.trial_name,
    a.path,
    a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name IN (
      "freemocap_balance_metrics",
      "qualisys_balance_metrics"
  )
ORDER BY t.trial_name, a.path;
"""

metric_paths = pd.read_sql_query(
    metrics_query,
    conn,
)

if metric_paths.empty:
    raise RuntimeError(
        "No balance_metrics artifacts were found in validation.db. "
        "Run the balance pipeline and rebuild the database first."
    )


metric_dfs = []

for _, row in metric_paths.iterrows():
    sub_df = pd.read_csv(
        row["path"]
    )

    sub_df["participant_code"] = (
        row["participant_code"]
    )

    sub_df["trial_name"] = (
        row["trial_name"]
    )

    sub_df["tracker"] = (
        row["tracker"]
    )

    metric_dfs.append(
        sub_df
    )


balance_metrics_df = pd.concat(
    metric_dfs,
    ignore_index=True,
)

conn.close()


# --------------------------------------------------------------------------- #
# Combined summary table
# --------------------------------------------------------------------------- #

def summarize(
    df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """
    First average repeated trials within participant, then calculate
    mean +/- SD across participants for each tracker and condition.
    """

    per_participant = (
        df
        .groupby(
            [
                "tracker",
                "condition",
                "participant_code",
            ]
        )[value_col]
        .mean()
        .reset_index()
    )

    return (
        per_participant
        .groupby(
            [
                "tracker",
                "condition",
            ]
        )[value_col]
        .agg(["mean", "std"])
        .reset_index()
    )


pl_summ = summarize(
    balance_metrics_df,
    "path_length_mm",
)

ea_summ = summarize(
    balance_metrics_df,
    "ellipse_area_mm2",
)

vel_summ = summarize(
    balance_metrics_df,
    "mean_2d_velocity_mm_s",
)


summary = (
    pl_summ
    .rename(
        columns={
            "mean": "pl_mean",
            "std": "pl_std",
        }
    )
    .merge(
        ea_summ.rename(
            columns={
                "mean": "ea_mean",
                "std": "ea_std",
            }
        ),
        on=[
            "tracker",
            "condition",
        ],
    )
    .merge(
        vel_summ.rename(
            columns={
                "mean": "vel_mean",
                "std": "vel_std",
            }
        ),
        on=[
            "tracker",
            "condition",
        ],
    )
)


# --------------------------------------------------------------------------- #
# Human-readable summary
# --------------------------------------------------------------------------- #

summary["Path Length (mm)"] = summary.apply(
    lambda row: (
        f"{row['pl_mean']:.1f} ± "
        f"{row['pl_std']:.1f}"
    ),
    axis=1,
)

summary["Ellipse Area (mm²)"] = summary.apply(
    lambda row: (
        f"{row['ea_mean']:.1f} ± "
        f"{row['ea_std']:.1f}"
    ),
    axis=1,
)

summary["Mean 2D Velocity (mm/s)"] = summary.apply(
    lambda row: (
        f"{row['vel_mean']:.2f} ± "
        f"{row['vel_std']:.2f}"
    ),
    axis=1,
)


print(
    summary[
        [
            "tracker",
            "condition",
            "Path Length (mm)",
            "Ellipse Area (mm²)",
            "Mean 2D Velocity (mm/s)",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------------------------------- #
# Typst formatting helpers
# --------------------------------------------------------------------------- #

def fmt1(
    mean: float,
    std: float,
) -> str:
    return (
        f"{mean:.1f} "
        f"$plus.minus$ "
        f"{std:.1f}"
    )


def fmt2(
    mean: float,
    std: float,
) -> str:
    return (
        f"{mean:.2f} "
        f"$plus.minus$ "
        f"{std:.2f}"
    )


# --------------------------------------------------------------------------- #
# Generate Typst table
# --------------------------------------------------------------------------- #

def generate_balance_metrics_table_typst(
    summary_df: pd.DataFrame,
) -> str:
    """
    Generate an importable Typst balance-metrics table.
    """

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

    for condition_index, condition in enumerate(
        conditions
    ):
        condition_label = (
            condition_labels[condition]
        )

        for tracker_index, tracker in enumerate(
            tracker_order
        ):
            matching_rows = summary_df[
                (
                    summary_df["tracker"]
                    == tracker
                )
                & (
                    summary_df["condition"]
                    == condition
                )
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

            backend = (
                tracker_labels[tracker]
            )

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

        if (
            condition_index
            < len(conditions) - 1
        ):
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

    return (
        "\n".join(lines)
        + "\n"
    )


# --------------------------------------------------------------------------- #
# Write table
# --------------------------------------------------------------------------- #

typst_table = (
    generate_balance_metrics_table_typst(
        summary
    )
)

out_file.write_text(
    typst_table,
    encoding="utf-8",
)

print(
    f"Table written to {out_file}"
)