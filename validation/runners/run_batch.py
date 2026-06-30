from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    max_workers: int = 1,
    config_pattern: str = "*.yaml",
) -> list[TrialRunResult]:
    """
    Run all trial YAMLs beneath config_root.

    Parallelization happens across trials. Trackers within each trial
    are run sequentially by run_trial().
    """
    config_root = config_root.resolve()
    dataset_root = dataset_root.resolve()

    config_files = find_config_files(
        config_root=config_root,
        pattern=config_pattern,
    )

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(config_files), cpu_count)

    max_workers = max(1, min(max_workers, len(config_files)))

    print(f"Found {len(config_files)} trial configs.")
    print(f"Using {max_workers} worker processes.")
    print(f"Dataset root: {dataset_root}")

    results: list[TrialRunResult] = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
    ) as executor:
        future_to_config = {
            executor.submit(
                run_trial_worker,
                config_path,
                dataset_root,
                trackers,
                start_at,
                use_rigid,
            ): config_path
            for config_path in config_files
        }

        for completed_count, future in enumerate(
            as_completed(future_to_config),
            start=1,
        ):
            config_path = future_to_config[future]

            try:
                result = future.result()
            except Exception:
                result = TrialRunResult(
                    config_path=config_path,
                    succeeded=False,
                    error_message=traceback.format_exc(),
                )

            results.append(result)

            status = "PASSED" if result.succeeded else "FAILED"

            print(
                f"[{completed_count}/{len(config_files)}] "
                f"{status}: {config_path}"
            )

            if result.error_message:
                print(result.error_message)

    successful = [
        result
        for result in results
        if result.succeeded
    ]
    failed = [
        result
        for result in results
        if not result.succeeded
    ]

    print()
    print("Batch complete")
    print(f"Successful trials: {len(successful)}")
    print(f"Failed trials:     {len(failed)}")

    if failed:
        print()
        print("Failed configs:")

        for result in failed:
            print(f"  - {result.config_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[2]

    # -----------------------------------------------------------------
    # USER SETTINGS
    # -----------------------------------------------------------------

    config_root = repo_root / "configs"

    dataset_root = Path(
        r"D:\validation_public_release_v1\data"
    )

    # None runs every tracker listed in each YAML.
    trackers_to_run = None

    # Examples:
    # trackers_to_run = ["mediapipe"]
    # trackers_to_run = ["mediapipe", "vitpose"]

    start_at_step = 0
    use_rigid = False

    config_pattern = "*.yaml"

    # -----------------------------------------------------------------

    batch_results = run_batch(
        config_root=config_root,
        dataset_root=dataset_root,
        trackers=trackers_to_run,
        start_at=start_at_step,
        use_rigid=use_rigid,
        config_pattern=config_pattern,
    )

    failed_results = [
        result
        for result in batch_results
        if not result.succeeded
    ]

    if failed_results:
        raise SystemExit(1)