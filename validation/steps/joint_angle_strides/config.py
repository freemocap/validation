from pydantic import BaseModel

class JointAnglesStridesConfig(BaseModel):
    loop_over_conditions: bool = False