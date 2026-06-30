from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from database.expected import (
    BALANCE,
    TREADMILL,
)
from database.trial_identity import (
    TrialIdentity,
    parse_trial_identity,
    resolve_recording_dir,
)
from validation.datatypes.data_component import DataComponent


QUALISYS_TRACKER = "qualisys"

# These categories contain one trial-level output rather than one output
# for every condition.
TREADMILL_UNCONDITIONED_CATEGORIES = {
    "synced_data",
    "gait_events",
    "joint_angles",
}

BALANCE_UNCONDITIONED_CATEGORIES = {
    "synced_data",
    "com_analysis",
}


@dataclass(frozen=True)
class TrialScan:
    config_path: Path
    identity: TrialIdentity
    recording_dir: Path
    trackers: tuple[str, ...]
    conditions: tuple[str, ...]
    config: dict[str, Any]


@dataclass(frozen=True)
class ArtifactScan:
    category: str
    condition: str
    tracker: str
    component_name: str

    path: Path
    relative_path: str

    file_exists: bool
    size_bytes: int | None
    mtime_utc: float | None


def load_trial_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load one public trial YAML.
    """
    config_path = config_path.expanduser().resolve()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a YAML mapping in {config_path}"
        )

    return config


def scan_trial_config(
    config_path: Path,
    dataset_root: Path,
) -> TrialScan:
    """
    Load one config and derive its public trial identity.
    """
    config = load_trial_config(config_path)

    trial_path = config.get("trial_path")

    if not trial_path:
        raise ValueError(
            f"Config does not define trial_path: {config_path}"
        )

    identity = parse_trial_identity(
        trial_path
    )

    recording_dir = resolve_recording_dir(
        dataset_root=dataset_root,
        trial_path=trial_path,
    )

    trackers = tuple(
        str(tracker)
        for tracker in (
            config.get("trackers") or []
        )
    )

    if not trackers:
        raise ValueError(
            f"No trackers listed in config: {config_path}"
        )

    project_config = (
        config.get("ProjectConfig") or {}
    )

    if not isinstance(project_config, dict):
        raise ValueError(
            "ProjectConfig must be a mapping or null in "
            f"{config_path}"
        )

    conditions_config = (
        project_config.get("conditions") or {}
    )

    if not isinstance(conditions_config, dict):
        raise ValueError(
            "ProjectConfig.conditions must be a mapping or null in "
            f"{config_path}"
        )

    conditions = tuple(
        str(condition_name)
        for condition_name in conditions_config.keys()
    )

    return TrialScan(
        config_path=config_path,
        identity=identity,
        recording_dir=recording_dir,
        trackers=trackers,
        conditions=conditions,
        config=config,
    )


def discover_trial_configs(
    config_root: Path,
) -> list[Path]:
    """
    Find all public trial YAMLs recursively.
    """
    config_root = config_root.expanduser().resolve()

    configs = sorted(
        config_root.rglob("*.yaml")
    )

    if not configs:
        raise FileNotFoundError(
            f"No YAML configs found under {config_root}"
        )

    return configs


def scan_trial_artifacts(
    trial_scan: TrialScan,
    dataset_root: Path,
    only_existing: bool = False,
) -> list[ArtifactScan]:
    """
    Resolve all expected artifacts for one trial.

    Parameters
    ----------
    trial_scan
        Parsed trial config and identity.
    dataset_root
        Root containing sub-001, sub-002, and so on.
    only_existing
        When True, omit expected files that do not exist.
        When False, preserve missing expected files in the database.
    """
    if trial_scan.identity.trial_type == "treadmill":
        return scan_treadmill_artifacts(
            trial_scan=trial_scan,
            dataset_root=dataset_root,
            only_existing=only_existing,
        )

    if trial_scan.identity.trial_type == "balance":
        return scan_balance_artifacts(
            trial_scan=trial_scan,
            dataset_root=dataset_root,
            only_existing=only_existing,
        )

    raise ValueError(
        "Unsupported trial type "
        f"{trial_scan.identity.trial_type!r} in "
        f"{trial_scan.config_path}"
    )


def scan_treadmill_artifacts(
    trial_scan: TrialScan,
    dataset_root: Path,
    only_existing: bool = False,
) -> list[ArtifactScan]:
    """
    Resolve expected treadmill artifacts using the old database-style
    category and component naming.
    """
    rows: list[ArtifactScan] = []

    for tracker in trial_scan.trackers:
        context = _component_context(
            trial_scan=trial_scan,
            tracker=tracker,
        )

        for category, components in TREADMILL.items():
            if category in TREADMILL_UNCONDITIONED_CATEGORIES:
                rows.extend(
                    _resolve_components(
                        components=components,
                        trial_scan=trial_scan,
                        dataset_root=dataset_root,
                        context=context,
                        category=category,
                        condition="",
                        requested_tracker=tracker,
                        only_existing=only_existing,
                    )
                )
                continue

            if not trial_scan.conditions:
                rows.extend(
                    _resolve_components(
                        components=components,
                        trial_scan=trial_scan,
                        dataset_root=dataset_root,
                        context=context,
                        category=category,
                        condition="",
                        requested_tracker=tracker,
                        only_existing=only_existing,
                    )
                )
                continue

            for condition in trial_scan.conditions:
                conditioned_components = [
                    component.clone_with_prefix(
                        condition
                    )
                    for component in components
                ]

                rows.extend(
                    _resolve_components(
                        components=conditioned_components,
                        trial_scan=trial_scan,
                        dataset_root=dataset_root,
                        context=context,
                        category=category,
                        condition=condition,
                        requested_tracker=tracker,
                        only_existing=only_existing,
                    )
                )

    return _deduplicate_artifacts(rows)


def scan_balance_artifacts(
    trial_scan: TrialScan,
    dataset_root: Path,
    only_existing: bool = False,
) -> list[ArtifactScan]:
    """
    Resolve expected balance artifacts.
    """
    rows: list[ArtifactScan] = []

    for tracker in trial_scan.trackers:
        context = _component_context(
            trial_scan=trial_scan,
            tracker=tracker,
        )

        for category, components in BALANCE.items():
            rows.extend(
                _resolve_components(
                    components=components,
                    trial_scan=trial_scan,
                    dataset_root=dataset_root,
                    context=context,
                    category=category,
                    condition="",
                    requested_tracker=tracker,
                    only_existing=only_existing,
                )
            )

    return _deduplicate_artifacts(rows)


def _resolve_components(
    components: Iterable[DataComponent],
    trial_scan: TrialScan,
    dataset_root: Path,
    context: dict[str, str],
    category: str,
    condition: str,
    requested_tracker: str,
    only_existing: bool,
) -> list[ArtifactScan]:
    rows: list[ArtifactScan] = []

    for component in components:
        row = make_artifact_scan(
            component=component,
            trial_scan=trial_scan,
            dataset_root=dataset_root,
            context=context,
            category=category,
            condition=condition,
            requested_tracker=requested_tracker,
        )

        if only_existing and not row.file_exists:
            continue

        rows.append(row)

    return rows


def make_artifact_scan(
    component: DataComponent,
    trial_scan: TrialScan,
    dataset_root: Path,
    context: dict[str, str],
    category: str,
    condition: str,
    requested_tracker: str,
) -> ArtifactScan:
    """
    Convert one expected DataComponent into a database artifact row.
    """
    path = component.full_path(
        trial_scan.recording_dir,
        **context,
    )

    resolved_tracker = resolve_component_tracker(
        component=component,
        requested_tracker=requested_tracker,
    )

    exists = path.exists()
    stat = path.stat() if exists else None

    dataset_root = dataset_root.expanduser().resolve()
    resolved_path = path.resolve()

    try:
        relative_path = resolved_path.relative_to(
            dataset_root
        ).as_posix()
    except ValueError:
        relative_path = resolved_path.as_posix()

    return ArtifactScan(
        category=category,
        condition=condition,
        tracker=resolved_tracker,
        component_name=component.name,
        path=resolved_path,
        relative_path=relative_path,
        file_exists=exists,
        size_bytes=(
            stat.st_size
            if stat is not None
            else None
        ),
        mtime_utc=(
            stat.st_mtime
            if stat is not None
            else None
        ),
    )


def resolve_component_tracker(
    component: DataComponent,
    requested_tracker: str,
) -> str:
    """
    Assign Qualisys components to the Qualisys tracker.

    Markerless components inherit the tracker currently being scanned.
    """
    relative_path = (
        component.relative_path or ""
    ).lower()

    component_name = (
        component.name or ""
    ).lower()

    if (
        "qualisys" in relative_path
        or component_name.startswith("qualisys")
    ):
        return QUALISYS_TRACKER

    return requested_tracker


def _component_context(
    trial_scan: TrialScan,
    tracker: str,
) -> dict[str, str]:
    return {
        "tracker": tracker,
        "recording_name": trial_scan.recording_dir.name,
    }


def _deduplicate_artifacts(
    artifacts: list[ArtifactScan],
) -> list[ArtifactScan]:
    """
    Remove duplicate logical artifact rows.

    Qualisys components may be encountered once per markerless tracker.
    They should appear only once in the database for a given trial.
    """
    unique: dict[
        tuple[str, str, str, str],
        ArtifactScan,
    ] = {}

    for artifact in artifacts:
        key = (
            artifact.category,
            artifact.condition,
            artifact.tracker,
            artifact.component_name,
        )

        existing = unique.get(key)

        if existing is None:
            unique[key] = artifact
            continue

        if existing.path != artifact.path:
            raise ValueError(
                "Two different artifact paths resolved to the same "
                "logical database key:\n"
                f"  key: {key}\n"
                f"  first: {existing.path}\n"
                f"  second: {artifact.path}"
            )

    return list(unique.values())