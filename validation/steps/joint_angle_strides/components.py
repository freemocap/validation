from validation.components import (QUALISYS_GAIT_EVENTS,
                                   FREEMOCAP_JOINT_ANGLES,
                                   QUALISYS_JOINT_ANGLES,
                                   FREEMOCAP_JOINT_ANGLE_CYCLES,
                                   QUALISYS_JOINT_ANGLE_CYCLES,
                                   FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS,
                                   QUALISYS_JOINT_ANGLE_SUMMARY_STATS,
                                   JOINT_ANGLE_SUMMARY_FIG,
                                   JOINT_ANGLE_RMSE_STATS
                                    )

REQUIRES = [QUALISYS_GAIT_EVENTS, FREEMOCAP_JOINT_ANGLES, QUALISYS_JOINT_ANGLES]
PRODUCES = [FREEMOCAP_JOINT_ANGLE_CYCLES, 
            QUALISYS_JOINT_ANGLE_CYCLES, 
            FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS, 
            QUALISYS_JOINT_ANGLE_SUMMARY_STATS, 
            JOINT_ANGLE_SUMMARY_FIG,
            JOINT_ANGLE_RMSE_STATS]
