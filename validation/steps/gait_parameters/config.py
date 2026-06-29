from pydantic import BaseModel

class GaitParametersConfig(BaseModel):
    loop_over_conditions: bool = False
    sampling_rate: float = 30.0