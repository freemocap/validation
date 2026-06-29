from pathlib import Path
import yaml
from validation.step_registry import STEP_REGISTRY
from validation.pipeline.project_config import ProjectConfig
from validation.pipeline.base import PipelineContext

def build_pipeline(config_path:Path,
                   step_list: list[str]|None = None, 
                   use_rigid:bool = False) -> PipelineContext:
    """
    Build the pipeline from the config file and recording directory.
    """
    with open(config_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)
    if pipeline_config["path_to_recording"] is None:
        raise ValueError("path_to_recording must be specified in the pipeline configuration YAML file.")
    recording_dir = Path(pipeline_config.pop("path_to_recording"))

    project_cfg = ProjectConfig(**pipeline_config.pop("ProjectConfig"))
    ctx = PipelineContext(recording_dir=recording_dir,
                          project_config=project_cfg,
                          use_rigid=use_rigid)
    


    for step_name, step_parameters in pipeline_config.items():
        if step_name == "pipeline":
            continue
        ctx.put(f"{step_name}.config", step_parameters)


    step_classes = []
    step_names = step_list if step_list else pipeline_config["pipeline"]
    step_classes = [STEP_REGISTRY[name] for name in step_names]
    
    return ctx, step_classes