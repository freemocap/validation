from pathlib import Path

import pytest

from database.trial_identity import (
    parse_trial_identity,
    resolve_recording_dir,
)


def test_parse_treadmill_trial_identity() -> None:
    identity = parse_trial_identity(
        "sub-001/task-treadmill_run-01"
    )

    assert identity.participant_code == "sub-001"
    assert identity.trial_name == "task-treadmill_run-01"
    assert identity.trial_type == "treadmill"
    assert identity.trial_number == 1
    assert identity.trial_path == "sub-001/task-treadmill_run-01"


def test_parse_balance_trial_identity() -> None:
    identity = parse_trial_identity(
        "sub-006/task-balance_run-02"
    )

    assert identity.participant_code == "sub-006"
    assert identity.trial_name == "task-balance_run-02"
    assert identity.trial_type == "balance"
    assert identity.trial_number == 2


def test_windows_separator_is_normalized() -> None:
    identity = parse_trial_identity(
        r"sub-001\task-treadmill_run-01"
    )

    assert identity.trial_path == "sub-001/task-treadmill_run-01"


def test_trial_path_requires_two_parts() -> None:
    with pytest.raises(ValueError, match="exactly two parts"):
        parse_trial_identity("task-treadmill_run-01")


def test_invalid_participant_id_fails() -> None:
    with pytest.raises(ValueError, match="Invalid participant"):
        parse_trial_identity(
            "participant-001/task-treadmill_run-01"
        )


def test_invalid_trial_name_fails() -> None:
    with pytest.raises(ValueError, match="Invalid trial name"):
        parse_trial_identity(
            "sub-001/treadmill-trial-1"
        )


def test_zero_trial_number_fails() -> None:
    with pytest.raises(ValueError, match="1 or greater"):
        parse_trial_identity(
            "sub-001/task-treadmill_run-00"
        )


def test_resolve_recording_dir(tmp_path: Path) -> None:
    recording_dir = resolve_recording_dir(
        dataset_root=tmp_path,
        trial_path="sub-001/task-treadmill_run-01",
    )

    expected = (
        tmp_path
        / "sub-001"
        / "task-treadmill_run-01"
    ).resolve()

    assert recording_dir == expected