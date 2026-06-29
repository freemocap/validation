from validation.components import   (QUALISYS_GAIT_EVENTS, 
                                    FREEMOCAP_PARQUET, 
                                    QUALISYS_PARQUET,
                                    FREEMOCAP_TRAJECTORY_CYCLES,
                                    QUALISYS_TRAJECTORY_CYCLES,
                                    FREEMOCAP_TRAJECTORY_SUMMARY_STATS,
                                    QUALISYS_TRAJECTORY_SUMMARY_STATS, 
                                    TRAJECTORY_PER_STRIDE_FIG,
                                    TRAJECTORY_MEAN_FIG,
                                    TRAJECTORY_RMSE_STATS)

REQUIRES = [QUALISYS_GAIT_EVENTS, FREEMOCAP_PARQUET, QUALISYS_PARQUET]
PRODUCES = [FREEMOCAP_TRAJECTORY_CYCLES, 
            QUALISYS_TRAJECTORY_CYCLES, 
            FREEMOCAP_TRAJECTORY_SUMMARY_STATS, 
            QUALISYS_TRAJECTORY_SUMMARY_STATS, 
            TRAJECTORY_PER_STRIDE_FIG, 
            TRAJECTORY_MEAN_FIG,
            TRAJECTORY_RMSE_STATS]