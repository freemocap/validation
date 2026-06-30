
import logging
from pathlib import Path

from validation.pipeline.base import ValidationPipeline
from validation.pipeline.builder import (
    build_pipeline,
    load_pipeline_config,
)


LOGGER = logging.getLogger("validation.runner")


def resolve_recording_dir(
    dataset_root: Path,
    trial_path: str,
) -> Path:
    """
    Combine the machine-specific dataset root with the public relative
    trial path from the YAML.
    """
    return (dataset_root / Path(trial_path)).resolve()


def run_trial(
    config_path: Path,
    dataset_root: Path,
    trackers: list[str] | None = None,
    start_at: int = 0,
    use_rigid: bool = False,
) -> None:
    config = load_pipeline_config(config_path)

    trial_path = config.get("trial_path")
    if not trial_path:
        raise ValueError(
            f"Config does not define trial_path: {config_path}"
        )

    recording_dir = resolve_recording_dir(
        dataset_root=dataset_root,
        trial_path=trial_path,
    )

    if not recording_dir.exists():
        raise FileNotFoundError(
            f"Trial directory does not exist: {recording_dir}"
        )

    configured_trackers = config.get("trackers", [])
    selected_trackers = trackers or configured_trackers

    if not selected_trackers:
        raise ValueError(
            "No trackers were provided and the config contains "
            "no trackers list."
        )

    for tracker in selected_trackers:
        LOGGER.info(
            "Running tracker=%s for trial=%s",
            tracker,
            recording_dir,
        )

        context, step_classes = build_pipeline(
            config=config,
            recording_dir=recording_dir,
            tracker=tracker,
        )

        pipeline = ValidationPipeline(
            context=context,
            steps=step_classes,
            logger=LOGGER,
        )

        pipeline.run(start_at=start_at)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # -----------------------------------------------------------------
    # USER SETTINGS
    # -----------------------------------------------------------------

    dataset_root = Path(
        r"D:\validation_public_release_v1\data"
    )

    config_path = Path(
        r"configs\sub-004\task-balance_trial-01.yaml"
    )

    # None means use every tracker listed in the YAML.
    trackers_to_run = None

    # To test only one tracker:
    # trackers_to_run = ["mediapipe"]

    start_at_step = 0
    use_rigid = False

    # -----------------------------------------------------------------

    run_trial(
        config_path=config_path,
        dataset_root=dataset_root,
        trackers=trackers_to_run,
        start_at=start_at_step,
        use_rigid=use_rigid,
    )