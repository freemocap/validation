from typing import List
from pydantic import BaseModel, Field

class SpatialAlignmentConfig(BaseModel):
    markers_for_alignment: List[str]
    frames_to_sample: int = Field(20, gt=0, description="Number of frames to sample in each RANSAC iteration")
    max_iterations: int = Field(20, gt=0, description="Maximum number of RANSAC iterations")
    inlier_threshold: float = Field(50, gt=0, description="Inlier threshold for RANSAC")

