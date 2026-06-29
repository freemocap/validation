from pathlib import Path

from validation.components.freemocap import FREEMOCAP_PARQUET
from validation.components.qualisys import QUALISYS_PARQUET


def test_freemocap_parquet_public_path() -> None:
    recording_root = Path("dataset/sub-001/task-treadmill_run-01")

    expected = (
        recording_root
        / "mediapipe"
        / "aligned_3d_data"
        / "freemocap_data_by_frame.parquet"
    )

    actual = FREEMOCAP_PARQUET.full_path(
        recording_root,
        tracker="mediapipe",
    )

    assert actual == expected


def test_qualisys_parquet_public_path() -> None:
    recording_root = Path("dataset/sub-001/task-treadmill_run-01")

    expected = (
        recording_root
        / "qualisys"
        / "aligned_3d_data"
        / "freemocap_data_by_frame.parquet"
    )

    actual = QUALISYS_PARQUET.full_path(recording_root)

    assert actual == expected