from pydantic import BaseModel, Field
from typing import Optional, Tuple

class JointAnglesConfig(BaseModel):
    neutral_frames: Optional[Tuple[int, int]] = Field(default=None)