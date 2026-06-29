
from dataclasses import dataclass, field
from pathlib import Path
from validation.pipeline.project_config import ProjectConfig
from typing import Any

@dataclass
class PipelineContext:
    recording_dir: Path
    project_config: ProjectConfig
    use_rigid:bool
    backpack: dict[str, Any] = field(default_factory = dict)

    def get(self, name:str) -> Any:
        return self.backpack.get(name)
    
    def put(self, name:str, value: Any) -> Any:
        self.backpack[name] = value

    @property
    def data_component_context(self) -> dict:
        return {
            "tracker": self.project_config.freemocap_tracker,
            "recording_name": self.recording_dir.stem,
        }
    
    @property
    def freemocap_path(self) -> Path:
        return self.recording_dir / "validation" / self.project_config.freemocap_tracker
    
    @property
    def qualisys_path(self) -> Path:
        return self.recording_dir / "validation" / "qualisys"

    @property
    def conditions(self) -> dict:
        return self.project_config.conditions or {}
