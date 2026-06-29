# validation/project_config.py
from pathlib import Path
from pydantic import BaseModel, Field

class ProjectConfig(BaseModel):
    qualisys_model_info_path: Path
    freemocap_tracker: str 
    conditions: dict|None