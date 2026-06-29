from validation.components import (QUALISYS_GAIT_EVENTS, 
                                   FREEMOCAP_GAIT_EVENTS, 
                                   QUALISYS_GAIT_METRICS, 
                                   FREEMOCAP_GAIT_METRICS, 
                                   QUALISYS_GAIT_SUMMARY_STATS, 
                                   FREEMOCAP_GAIT_SUMMARY_STATS,
                                   QUALISYS_PARQUET,
                                   FREEMOCAP_PARQUET)


REQUIRES = [QUALISYS_GAIT_EVENTS, FREEMOCAP_GAIT_EVENTS, QUALISYS_PARQUET, FREEMOCAP_PARQUET]
PRODUCES = [QUALISYS_GAIT_METRICS, FREEMOCAP_GAIT_METRICS, QUALISYS_GAIT_SUMMARY_STATS, FREEMOCAP_GAIT_SUMMARY_STATS]