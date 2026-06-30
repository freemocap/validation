from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PARTICIPANT_ID_PATTERN = re.compile(
    r"^sub-(?P<number>\d+)$"
)

TRIAL_NAME_PATTERN = re.compile(
    r"^task-(?P<trial_type>[a-z0-9-]+)_trial-(?P<trial_number>\d+)$"
)


@dataclass(frozen=True)
class TrialIdentity:
    participant_code: str
    trial_name: str
    trial_type: str
    trial_number: int
    trial_path: str

    @property
    def trial_key(self) -> str:
        return self.trial_path


def parse_trial_identity(
    trial_path: str | Path,
) -> TrialIdentity:
    normalized_path = _normalize_trial_path(trial_path)
    path = PurePosixPath(normalized_path)

    if path.is_absolute():
        raise ValueError(
            "trial_path must be relative to the dataset root, "
            f"not absolute: {trial_path!r}"
        )

    if len(path.parts) != 2:
        raise ValueError(
            "trial_path must contain exactly two parts: "
            "<participant_id>/<trial_name>. "
            "For example: sub-001/task-treadmill_run-01. "
            f"Received: {trial_path!r}"
        )

    participant_code, trial_name = path.parts

    participant_match = PARTICIPANT_ID_PATTERN.fullmatch(
        participant_code
    )
    if participant_match is None:
        raise ValueError(
            "Invalid participant identifier in trial_path. "
            "Expected a value such as 'sub-001'. "
            f"Received: {participant_code!r}"
        )

    trial_match = TRIAL_NAME_PATTERN.fullmatch(
        trial_name
    )
    if trial_match is None:
        raise ValueError(
            "Invalid trial name in trial_path. "
            "Expected a value such as "
            "'task-treadmill_run-01'. "
            f"Received: {trial_name!r}"
        )

    trial_type = trial_match.group("trial_type")
    trial_number = int(
        trial_match.group("trial_number")
    )

    if trial_number < 1:
        raise ValueError(
            "trial_number must be 1 or greater. "
            f"Received: {trial_number}"
        )

    return TrialIdentity(
        participant_code=participant_code,
        trial_name=trial_name,
        trial_type=trial_type,
        trial_number=trial_number,
        trial_path=path.as_posix(),
    )


def resolve_recording_dir(
    dataset_root: Path,
    trial_path: str | Path,
) -> Path:
    identity = parse_trial_identity(trial_path)

    dataset_root = dataset_root.expanduser().resolve()

    recording_dir = (
        dataset_root
        / Path(identity.trial_path)
    ).resolve()

    try:
        recording_dir.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            "Resolved recording directory escaped the dataset root: "
            f"dataset_root={dataset_root}, "
            f"recording_dir={recording_dir}"
        ) from exc

    return recording_dir


def _normalize_trial_path(
    trial_path: str | Path,
) -> str:
    value = str(trial_path).strip()

    if not value:
        raise ValueError("trial_path cannot be empty.")

    return value.replace("\\", "/")