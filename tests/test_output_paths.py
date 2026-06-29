from pathlib import Path

from validation.components.freemocap import (
    FREEMOCAP_PARQUET,
    FREEMOCAP_JOINT_ANGLES,
)
from validation.components.qualisys import (
    QUALISYS_PARQUET,
    QUALISYS_JOINT_ANGLES,
)
from validation.components.stride_separation_trajectories import FREEMOCAP_TRAJECTORY_CYCLES


ROOT = Path("dataset/sub-001/task-treadmill_run-01")


def test_freemocap_parquet_public_path() -> None:
    assert FREEMOCAP_PARQUET.full_path(
        ROOT,
        tracker="mediapipe",
    ) == (
        ROOT
        / "mediapipe"
        / "aligned_3d_data"
        / "freemocap_data_by_frame.parquet"
    )


def test_qualisys_parquet_public_path() -> None:
    assert QUALISYS_PARQUET.full_path(ROOT) == (
        ROOT
        / "qualisys"
        / "aligned_3d_data"
        / "freemocap_data_by_frame.parquet"
    )


def test_freemocap_joint_angles_output_path() -> None:
    assert FREEMOCAP_JOINT_ANGLES.full_path(
        ROOT,
        tracker="mediapipe",
    ) == (
        ROOT
        / "mediapipe"
        / "analysis_outputs"
        / "joint_angles"
        / "mediapipe_joint_angles.csv"
    )


def test_qualisys_joint_angles_output_path() -> None:
    assert QUALISYS_JOINT_ANGLES.full_path(ROOT) == (
        ROOT
        / "qualisys"
        / "analysis_outputs"
        / "joint_angles"
        / "qualisys_joint_angles.csv"
    )


def test_condition_prefixed_trajectory_path() -> None:
    component = FREEMOCAP_TRAJECTORY_CYCLES.clone_with_prefix("speed_1_0")

    assert component.full_path(
        ROOT,
        tracker="mediapipe",
    ) == (
        ROOT
        / "mediapipe"
        / "analysis_outputs"
        / "trajectories"
        / "speed_1_0"
        / "trajectories_per_stride.csv"
    )