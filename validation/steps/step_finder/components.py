
from validation.components import (FREEMOCAP_PARQUET, 
                                   QUALISYS_PARQUET, 
                                   FREEMOCAP_GAIT_EVENTS, 
                                   QUALISYS_GAIT_EVENTS,
                                   STEPS_FIG,
                                    GAIT_EVENTS_DEBUG)

REQUIRES = [QUALISYS_PARQUET, FREEMOCAP_PARQUET]
PRODUCES = [FREEMOCAP_GAIT_EVENTS, QUALISYS_GAIT_EVENTS, STEPS_FIG, GAIT_EVENTS_DEBUG]