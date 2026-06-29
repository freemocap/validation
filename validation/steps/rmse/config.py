from pydantic import BaseModel, Field
from typing import List, Optional

class RMSEConfig(BaseModel):
    markers_for_comparison: List[str]