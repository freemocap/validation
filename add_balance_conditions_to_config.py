from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml


CONDITION_MAP = {
    "Eyes Open/Solid Ground": "eyes_open_solid",
    "Eyes Closed/Solid Ground": "eyes_closed_solid",
    "Eyes Open/Foam": "eyes_open_foam",
    "Eyes Closed/Foam": "eyes_closed_foam",
}


def find_condition_json(
    trial_dir: Path,
    analysis_name: str,
) -> Path:
    """
    Find the Qualisys condition_data.json containing the manually
    annotated balance-condition frame intervals.

    Supports both:
        qualisys/analysis_outputs/path_length_analysis/analysis_manuscript/
        qualisys/path_length_analysis/analysis_manuscript/

    and a flattened public-release path.
    """
    candidates = [
        trial_dir
        / "qualisys"
        / "analysis_outputs"
        / "path_length_analysis"
        / analysis_name
        / "condition_data.json",

        trial_dir
        / "qualisys"
        / "path_length_analysis"
        / analysis_name
        / "condition_data.json",

        trial_dir
        / "qualisys"
        / "analysis_outputs"
        / "path_length_analysis"
        / "condition_data.json",
    ]

    existing = [path for path in candidates if path.exists()]

    if len(existing) == 0:
        raise FileNotFoundError(
            f"Could not find condition_data.json for:\n"
            f"  {trial_dir}\n\n"
            f"Checked:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )

    # Prefer explicitly named analysis over flattened fallback.
    return existing[0]


def load_and_validate_intervals(
    condition_json: Path,
) -> dict[str, list[int]]:
    with open(condition_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "Frame Intervals" not in data:
        raise ValueError(
            f"'Frame Intervals' missing from {condition_json}"
        )

    intervals = data["Frame Intervals"]

    expected = set(CONDITION_MAP)
    actual = set(intervals)

    if actual != expected:
        raise ValueError(
            f"Condition mismatch in {condition_json}\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(actual)}"
        )

    for condition, frames in intervals.items():
        if (
            not isinstance(frames, list)
            or len(frames) != 2
            or not all(isinstance(x, int) for x in frames)
        ):
            raise ValueError(
                f"Invalid frame interval for {condition}: {frames}"
            )

        start, end = frames

        if start >= end:
            raise ValueError(
                f"Invalid frame interval for {condition}: "
                f"start={start}, end={end}"
            )

    return intervals


def make_conditions_block(
    intervals: dict[str, list[int]],
) -> str:
    lines = ["  conditions:"]

    for display_name, config_name in CONDITION_MAP.items():
        start, end = intervals[display_name]

        lines.extend(
            [
                f"    {config_name}:",
                f"      frames: [{start}, {end}]",
                f'      label: "{display_name}"',
            ]
        )

    return "\n".join(lines)


def prepare_updated_config(
    config_path: Path,
    intervals: dict[str, list[int]],
) -> str:
    original = config_path.read_text(encoding="utf-8")

    # Deliberately only allow replacement of the current null conditions.
    pattern = r"(?m)^  conditions:\s*(?:~|null)\s*$"

    matches = re.findall(pattern, original)

    if len(matches) != 1:
        raise ValueError(
            f"{config_path}: expected exactly one "
            f"'  conditions: ~' or '  conditions: null' line, "
            f"found {len(matches)}.\n"
            f"Refusing to modify this file."
        )

    replacement = make_conditions_block(intervals)

    updated = re.sub(
        pattern,
        replacement,
        original,
        count=1,
    )

    # Validate resulting YAML before touching the original file.
    parsed = yaml.safe_load(updated)

    conditions = parsed.get("ProjectConfig", {}).get("conditions")

    if not conditions:
        raise ValueError(
            f"Generated YAML failed validation for {config_path}"
        )

    for display_name, config_name in CONDITION_MAP.items():
        expected_frames = intervals[display_name]
        actual_frames = conditions[config_name]["frames"]

        if actual_frames != expected_frames:
            raise ValueError(
                f"Validation failed for {config_path}: "
                f"{config_name}: expected {expected_frames}, "
                f"got {actual_frames}"
            )

    return updated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing sub-001, sub-002, etc.",
    )

    parser.add_argument(
        "--configs-root",
        type=Path,
        default=Path("configs"),
    )

    parser.add_argument(
        "--analysis-name",
        default="analysis_manuscript",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify configs. Without this flag, dry-run only.",
    )

    args = parser.parse_args()

    config_paths = sorted(
        args.configs_root.glob(
            "sub-*/task-balance_trial-*.yaml"
        )
    )

    if not config_paths:
        raise RuntimeError(
            f"No balance configs found under {args.configs_root}"
        )

    print(f"Found {len(config_paths)} balance configs.")
    print()

    prepared: dict[Path, str] = {}

    # ---------------------------------------------------------
    # PHASE 1
    # Validate EVERYTHING before touching a single config.
    # ---------------------------------------------------------
    for config_path in config_paths:
        config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        trial_path = config.get("trial_path")

        if not trial_path:
            raise ValueError(
                f"{config_path} has no trial_path"
            )

        trial_dir = args.dataset_root / trial_path

        if not trial_dir.exists():
            raise FileNotFoundError(
                f"Trial directory does not exist:\n"
                f"  {trial_dir}"
            )

        condition_json = find_condition_json(
            trial_dir,
            args.analysis_name,
        )

        intervals = load_and_validate_intervals(
            condition_json
        )

        updated = prepare_updated_config(
            config_path,
            intervals,
        )

        prepared[config_path] = updated

        print(config_path)
        print(f"  source: {condition_json}")

        for display_name in CONDITION_MAP:
            print(
                f"  {display_name:<26} "
                f"{intervals[display_name]}"
            )

        print()

    print(
        f"✓ All {len(prepared)} configs passed validation."
    )

    # ---------------------------------------------------------
    # DRY RUN STOPS HERE
    # ---------------------------------------------------------
    if not args.apply:
        print()
        print("DRY RUN ONLY — no files were modified.")
        print()
        print(
            "Run again with --apply to write the annotations."
        )
        return

    # ---------------------------------------------------------
    # PHASE 2
    # Backup + write
    # ---------------------------------------------------------
    backup_root = (
        args.configs_root.parent
        / "config_backups_before_balance_annotations"
    )

    for config_path, updated in prepared.items():
        relative = config_path.relative_to(
            args.configs_root
        )

        backup_path = backup_root / relative
        backup_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            config_path,
            backup_path,
        )

        temp_path = config_path.with_suffix(
            ".yaml.tmp"
        )

        temp_path.write_text(
            updated,
            encoding="utf-8",
        )

        temp_path.replace(config_path)

    print()
    print(
        f"✓ Updated {len(prepared)} balance configs."
    )
    print(
        f"✓ Original configs backed up to:\n"
        f"  {backup_root}"
    )


if __name__ == "__main__":
    main()