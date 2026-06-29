from pydantic import BaseModel

class TrajectoryStridesConfig(BaseModel):
    loop_over_conditions: bool = False