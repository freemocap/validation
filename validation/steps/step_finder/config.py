from pydantic import BaseModel, Field
from typing import Optional, Tuple

class StepFinderConfig(BaseModel):
    sampling_rate: float
    frames_of_interest: Optional[Tuple[int, int]] = Field(default=None)
    