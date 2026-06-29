from pathlib import Path
import yaml
from validation.step_registry import STEP_REGISTRY
from validation.pipeline.project_config import ProjectConfig
from validation.pipeline.base import PipelineContext
from typing import Any

RESERVED_CONFIG_KEYS = {
    "trial_path",
    "trackers",
    "pipeline",
    "ProjectConfig",
}

def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    """
    Load one trial-level pipeline YAML.
    """
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a YAML mapping in config file: {config_path}"
        )

    return config

def build_pipeline(
    config: dict[str, Any],
    recording_dir: Path,
    tracker: str,
):
    project_config_data = dict(
        config.get("ProjectConfig", {})
    )

    project_config_data["freemocap_tracker"] = tracker

    project_config = ProjectConfig(
        **project_config_data
    )

    context = PipelineContext(
        recording_dir=recording_dir,
        project_config=project_config,
    )

    for key, value in config.items():
        if key in RESERVED_CONFIG_KEYS:
            continue

        # At this point, key should be a step name.
        context.put(
            f"{key}.config",
            value,
        )

    step_names = config["pipeline"]

    step_classes = [
        STEP_REGISTRY[step_name]
        for step_name in step_names
    ]

    return context, step_classes