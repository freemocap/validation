import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TRACKERS = ["mediapipe", "rtmpose", "vitpose", "qualisys"]
REFERENCE_SYSTEM = "qualisys"

JOINT_ORDER = ["hip", "knee", "ankle", "foot_index"]
AXES = ["x", "y", "z"]

AXIS_PRETTY = {
    "x": "ML",
    "y": "AP",
    "z": "Vertical",
}

TRACKER_DISPLAY = {
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}

JOINT_DISPLAY = {
    "hip": "Hip",
    "knee": "Knee",
    "ankle": "Ankle",
    "foot_index": "Toe",
}

DB_PATH = Path("validation.db")

TYPST_OUT_DIR = Path(r"D:\validation_public_release_v1\tables")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_trajectory_summary_stats(database_path: Path | str) -> pd.DataFrame:
    """Load treadmill trajectory summary-stat CSVs registered in the database."""
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
      AND a.category = "trajectories_per_stride"
      AND a.tracker IN ("mediapipe", "rtmpose", "vitpose", "qualisys")
      AND a.file_exists = 1
      AND a.condition LIKE "speed_%"
      AND a.component_name LIKE "%summary_stats"
    ORDER BY t.trial_name, a.path
    """

    with sqlite3.connect(database_path) as connection:
        path_df = pd.read_sql_query(query, connection)

    dataframes = []

    for _, row in path_df.iterrows():
        dataframe = pd.read_csv(row["path"])
        dataframe["participant_code"] = row["participant_code"]
        dataframe["trial_name"] = row["trial_name"]
        dataframe["tracker"] = (row["tracker"] or "").lower()
        dataframe["condition"] = row["condition"] or "none"
        dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError(
            "No trajectory summary_stats CSVs found from the query."
        )

    return pd.concat(dataframes, ignore_index=True)


# -----------------------------------------------------------------------------
# Trajectory preparation and RMSE calculation
# -----------------------------------------------------------------------------
def combine_left_and_right_side(
    dataframe: pd.DataFrame,
    joints_to_use: list[str],
) -> pd.DataFrame:
    """Mirror left-side ML values and average left/right joint trajectories."""
    dataframe = dataframe.copy()

    marker_names = dataframe["marker"].astype(str).str.strip().str.lower()

    dataframe["side"] = np.select(
        [
            marker_names.str.startswith("left_"),
            marker_names.str.startswith("right_"),
        ],
        ["left", "right"],
        default="unknown",
    )

    dataframe["joint"] = marker_names.str.replace(
        r"^(left_|right_)",
        "",
        regex=True,
    )

    dataframe = dataframe[
        dataframe["joint"].isin(joints_to_use)
        & (dataframe["stat"].astype(str).str.lower() == "mean")
    ].copy()

    dataframe["value_mirrored"] = dataframe["value"].astype(float)

    left_ml = (
        (dataframe["axis"] == "x")
        & (dataframe["side"] == "left")
    )
    dataframe.loc[left_ml, "value_mirrored"] *= -1

    return (
        dataframe.groupby(
            [
                "condition",
                "tracker",
                "participant_code",
                "trial_name",
                "joint",
                "axis",
                "percent_gait_cycle",
            ],
            as_index=False,
        )
        .agg(trial_mean_value=("value_mirrored", "mean"))
    )


def calculate_total_mean_and_std_rmse(
    dataframe: pd.DataFrame,
    tracker_list: list[str],
    reference_system: str,
) -> pd.DataFrame:
    """Calculate trial-level trajectory RMSE, then mean and SD across trials."""
    trial_wide = dataframe.pivot(
        index=[
            "condition",
            "participant_code",
            "trial_name",
            "joint",
            "axis",
            "percent_gait_cycle",
        ],
        columns="tracker",
        values="trial_mean_value",
    ).reset_index()

    markerless_trackers = [
        tracker
        for tracker in tracker_list
        if tracker != reference_system
    ]

    trial_long = trial_wide.melt(
        id_vars=[
            "condition",
            "participant_code",
            "trial_name",
            "joint",
            "axis",
            "percent_gait_cycle",
            reference_system,
        ],
        value_vars=markerless_trackers,
        var_name="tracker",
        value_name="mean_trajectory",
    )

    grouped_trials = trial_long.groupby(
        ["condition", "trial_name", "axis", "tracker", "joint"]
    )

    rows = []

    print(
    trial_long[
        ["participant_code", "trial_name"]
    ]
    .drop_duplicates()
    .groupby("trial_name")
    .size()
)

    for (
        condition,
        trial_name,
        axis,
        tracker,
        joint,
    ), trial_group in grouped_trials:
        rmse = np.sqrt(
            np.mean(
                (
                    trial_group["mean_trajectory"]
                    - trial_group[reference_system]
                )
                ** 2
            )
        )

        rows.append(
            {
                "condition": condition,
                "trial_name": trial_name,
                "axis": axis,
                "tracker": tracker,
                "joint": joint,
                "rmse": rmse,
            }
        )

    trial_level_rmse = pd.DataFrame(rows)

    return (
        trial_level_rmse.groupby(
            ["condition", "axis", "tracker", "joint"]
        )
        .agg(
            mean=("rmse", "mean"),
            std=("rmse", "std"),
        )
        .reset_index()
    )


# -----------------------------------------------------------------------------
# Typst table generation
# -----------------------------------------------------------------------------
def parse_speed_float(condition: str) -> float:
    speed_string = (
        str(condition)
        .replace("speed_", "")
        .replace("_", ".")
    )

    try:
        return float(speed_string)
    except Exception:
        return np.nan


def generate_typst_trajectory_rmse_table(
    rmse_df: pd.DataFrame,
    axis: str,
) -> str:
    """
    Generate the Typst table content.

    The table structure, labels, captions, precision, and formatting are kept
    identical to the original script.
    """
    rmse_df = rmse_df.copy()
    rmse_df["speed"] = rmse_df["condition"].map(parse_speed_float)

    speeds = sorted(rmse_df["speed"].dropna().unique())
    speed_labels = [f"{speed:g}" for speed in speeds]

    trackers_ordered = [
        tracker
        for tracker in ["mediapipe", "rtmpose", "vitpose"]
        if tracker in rmse_df["tracker"].unique()
    ]
    n_trackers = len(trackers_ordered)

    n_speed_cols = len(speeds)
    col_spec = (
        f"(1fr, 1.5fr, "
        f"{'1.2fr, ' * (n_speed_cols - 1)}"
        f"1.2fr)"
    )
    align_spec = (
        f"(left, left, "
        f"{'center, ' * (n_speed_cols - 1)}"
        f"center)"
    )

    axis_label = AXIS_PRETTY.get(axis, axis.upper())
    label = f"tbl-traj-rmse-{axis}"
    caption = (
        f"Trajectory RMSE per pose estimation backend across speeds — "
        f"{axis_label} (mm). "
        f"Values represent mean ± SD RMSE across all participants and trials "
        f"compared to the marker-based reference."
    )

    subset_df = rmse_df[rmse_df["axis"] == axis]

    header_cells = ["[*Joint*]", "[*Backend*]"]
    for speed_label in speed_labels:
        header_cells.append(f"[*{speed_label} m/s*]")

    body_lines = []

    for joint in JOINT_ORDER:
        subset = subset_df[subset_df["joint"] == joint]

        if subset.empty:
            continue

        joint_label = JOINT_DISPLAY.get(joint, joint.title())

        for tracker_index, tracker in enumerate(trackers_ordered):
            row_data = subset[subset["tracker"] == tracker]
            display_name = TRACKER_DISPLAY.get(tracker, tracker.title())

            if tracker_index == 0:
                body_lines.append(
                    f"      table.cell(rowspan: {n_trackers}, "
                    f"align: horizon)[{joint_label}],"
                )

            body_lines.append(f"      [{display_name}],")

            for speed in speeds:
                value = row_data.loc[row_data["speed"] == speed]

                if len(value) > 0:
                    mean = value["mean"].iloc[0]
                    std = value["std"].iloc[0]

                    if np.isfinite(mean) and np.isfinite(std):
                        body_lines.append(
                            f"      [{mean:.1f} ± {std:.1f}],"
                        )
                    elif np.isfinite(mean):
                        body_lines.append(f"      [{mean:.1f}],")
                    else:
                        body_lines.append("      [--],")
                else:
                    body_lines.append("      [--],")

        body_lines.append("      table.hline(stroke: 0.5pt),")

    export_name = f"traj-rmse-{axis}"

    lines = [
        f"#let {export_name} = [",
        "  #figure(",
        "    {",
        "    set text(size: 9pt)",
        "    table(",
        f"      columns: {col_spec},",
        f"      align: {align_spec},",
        "      stroke: none,",
        "      table.hline(stroke: 1pt),",
        "      table.header(",
    ]

    for cell in header_cells:
        lines.append(f"        {cell},")

    lines.extend(
        [
            "      ),",
            "      table.hline(stroke: 0.5pt),",
        ]
    )
    lines.extend(body_lines)
    lines.extend(
        [
            "      table.hline(stroke: 1pt),",
            "    )",
            "    },",
            f"    caption: [{caption}],",
            f"  ) <{label}>",
            "]",
        ]
    )

    return "\n".join(lines) + "\n"


def write_typst_tables(rmse_df: pd.DataFrame) -> None:
    """Write the x, y, and z trajectory RMSE Typst tables."""
    TYPST_OUT_DIR.mkdir(exist_ok=True, parents=True)

    for axis in AXES:
        typst_content = generate_typst_trajectory_rmse_table(
            rmse_df,
            axis,
        )
        typst_path = TYPST_OUT_DIR / f"trajectory_rmse_{axis}.typ"
        typst_path.write_text(typst_content, encoding="utf-8")
        print(f"Saved: {typst_path}")


def main() -> None:
    database_data = load_trajectory_summary_stats(DB_PATH)

    trial_left_right_mean = combine_left_and_right_side(
        database_data,
        JOINT_ORDER,
    )

    total_means_and_stds = calculate_total_mean_and_std_rmse(
        trial_left_right_mean,
        tracker_list=TRACKERS,
        reference_system=REFERENCE_SYSTEM,
    )

    write_typst_tables(total_means_and_stds)


if __name__ == "__main__":
    main()
