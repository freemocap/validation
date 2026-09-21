from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

from validation.runners.run_trial import run_trial

@dataclass(frozen=True)
class TrialRunResult:
    config_path: Path
    succeeded: bool
    error_message: str | None = None


def run_trial_worker(
    config_path: Path,
    dataset_root: Path,
    trackers: list[str] | None,
    start_at: int,
    use_rigid: bool,
) -> TrialRunResult:
    """
    Run one trial config.

    This function must remain at module level so it can be used by
    ProcessPoolExecutor on Windows.
    """
    try:
        run_trial(
            config_path=config_path,
            dataset_root=dataset_root,
            trackers=trackers,
            start_at=start_at,
            use_rigid=use_rigid,
        )

        return TrialRunResult(
            config_path=config_path,
            succeeded=True,
        )

    except Exception:
        return TrialRunResult(
            config_path=config_path,
            succeeded=False,
            error_message=traceback.format_exc(),
        )


def find_config_files(
    config_root: Path,
    pattern: str = "*.yaml",
) -> list[Path]:
    """
    Find trial YAML files recursively beneath config_root.
    """
    config_files = sorted(config_root.rglob(pattern))

    if not config_files:
        raise FileNotFoundError(
            f"No config files matching {pattern!r} were found "
            f"under {config_root}"
        )

    return config_files

def run_batch(
    config_root: Path,
    dataset_root: Path,
    trackers: list[str] | None = None,
    start_at: int = 0,
    use_rigid: bool = False,
    config_pattern: str = "*.yaml",
) -> list[TrialRunResult]:

    config_root = config_root.resolve()
    dataset_root = dataset_root.resolve()

    config_files = find_config_files(
        config_root=config_root,
        pattern=config_pattern,
    )

    print(f"Found {len(config_files)} trial configs.")
    print(f"Dataset root: {dataset_root}")

    results = []

    for completed_count, config_path in enumerate(
        config_files,
        start=1,
    ):
        print()
        print(
            f"[{completed_count}/{len(config_files)}] "
            f"Running: {config_path}"
        )

        try:
            run_trial(
                config_path=config_path,
                dataset_root=dataset_root,
                trackers=trackers,
                start_at=start_at,
                use_rigid=use_rigid,
            )

            result = TrialRunResult(
                config_path=config_path,
                succeeded=True,
            )

            print(
                f"[{completed_count}/{len(config_files)}] "
                f"PASSED: {config_path}"
            )

        except Exception:
            error_message = traceback.format_exc()

            result = TrialRunResult(
                config_path=config_path,
                succeeded=False,
                error_message=error_message,
            )

            print(
                f"[{completed_count}/{len(config_files)}] "
                f"FAILED: {config_path}"
            )
            print(error_message)

        results.append(result)

    successful = [r for r in results if r.succeeded]
    failed = [r for r in results if not r.succeeded]

    print()
    print("Batch complete")
    print(f"Successful trials: {len(successful)}")
    print(f"Failed trials:     {len(failed)}")

    return results

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Run the FreeMoCap validation pipeline across all configured trials."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the public dataset 'data' directory containing sub-001, sub-002, etc.",
    )

    parser.add_argument(
        "--config-root",
        type=Path,
        default=repo_root / "configs",
        help="Directory containing trial YAML configs. Defaults to repo/configs.",
    )

    parser.add_argument(
        "--tracker",
        action="append",
        dest="trackers",
        default=None,
        help=(
            "Optional tracker to run. Repeat for multiple trackers. "
            "If omitted, all trackers listed in each YAML are run."
        ),
    )

    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Pipeline step index to start at. Defaults to 0.",
    )

    parser.add_argument(
        "--use-rigid",
        action="store_true",
        help="Use rigid-body trajectories where supported.",
    )

    parser.add_argument(
        "--config-pattern",
        default="*.yaml",
        help="Config filename pattern. Defaults to '*.yaml'.",
    )

    args = parser.parse_args()

    batch_results = run_batch(
        config_root=args.config_root,
        dataset_root=args.dataset_root,
        trackers=args.trackers,
        start_at=args.start_at,
        use_rigid=args.use_rigid,
        config_pattern=args.config_pattern,
    )

    failed_results = [
        result
        for result in batch_results
        if not result.succeeded
    ]

    if failed_results:
        raise SystemExit(1)


    