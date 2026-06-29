from validation.components import (TRANSFORMATION_MATRIX, 
                                   FREEMOCAP_PRE_SYNC_JOINT_CENTERS, 
                                   QUALISYS_SYNCED_JOINT_CENTERS, 
                                   FREEMOCAP_PARQUET,
                                   QUALISYS_PARQUET)

REQUIRES = [FREEMOCAP_PRE_SYNC_JOINT_CENTERS, QUALISYS_SYNCED_JOINT_CENTERS]
PRODUCES = [TRANSFORMATION_MATRIX, FREEMOCAP_PARQUET, QUALISYS_PARQUET]
