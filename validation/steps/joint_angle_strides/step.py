from validation.pipeline.base import ValidationStep
from validation.components import   (QUALISYS_GAIT_EVENTS,
                                    FREEMOCAP_JOINT_ANGLES,
                                    QUALISYS_JOINT_ANGLES,
                                    FREEMOCAP_JOINT_ANGLE_CYCLES,
                                    QUALISYS_JOINT_ANGLE_CYCLES,
                                    FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS,
                                    QUALISYS_JOINT_ANGLE_SUMMARY_STATS,
                                    JOINT_ANGLE_SUMMARY_FIG,
                                    JOINT_ANGLE_RMSE_STATS
                                    )
from validation.steps.joint_angle_strides.components import REQUIRES, PRODUCES
from validation.steps.joint_angle_strides.core.joint_angle_cycles import create_angle_cycles, get_angle_summary
from validation.steps.joint_angle_strides.core.stride_slices import get_heel_strike_slices
from validation.steps.joint_angle_strides.core.angle_gait_plots import plot_angle_summary_grid
from validation.steps.joint_angle_strides.config import JointAnglesStridesConfig
from validation.steps.joint_angle_strides.core.calculate_joint_angle_rmse import calculate_rmse

class JointAnglesStridesStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = JointAnglesStridesConfig

    def calculate(self, condition_frame_range:list[int] = None):
        frame_range = range(*condition_frame_range) if condition_frame_range is not None else None

        gait_events = self.data[QUALISYS_GAIT_EVENTS.name]
        fmc_joint_angles = self.data[FREEMOCAP_JOINT_ANGLES.name]
        qual_joint_angles = self.data[QUALISYS_JOINT_ANGLES.name]

        heel_strikes:dict[str, list[slice]] = get_heel_strike_slices(
            gait_events=gait_events, 
            frame_range=frame_range)

        self.logger.info("Separating joint angles into strides")
        angles_per_stride = create_angle_cycles(
            freemocap_df=fmc_joint_angles,
            qualisys_df=qual_joint_angles,
            gait_events=heel_strikes,
            freemocap_tracker_name=self.ctx.project_config.freemocap_tracker,
            n_points=100
        )
        split_ja_dfs = {tracker: df_t for tracker, df_t in angles_per_stride.groupby("tracker")}

        freemocap_stride_df = split_ja_dfs[self.ctx.project_config.freemocap_tracker]
        qualisys_stride_df = split_ja_dfs["qualisys"]
        rmse_df = calculate_rmse(
            freemocap_df=freemocap_stride_df,
            qualisys_df=qualisys_stride_df
        )
        self.outputs[JOINT_ANGLE_RMSE_STATS.name] = rmse_df
        self.outputs[FREEMOCAP_JOINT_ANGLE_CYCLES.name] = freemocap_stride_df
        self.outputs[QUALISYS_JOINT_ANGLE_CYCLES.name] = qualisys_stride_df

        self.logger.info("Computing summary statistics for joint angles per stride")
        angle_summary_stats = get_angle_summary(angles_per_stride)
        split_ja_summary_dfs = {tracker: df_t for tracker, df_t in angle_summary_stats.groupby("tracker")}
        self.outputs[FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS.name] = split_ja_summary_dfs[self.ctx.project_config.freemocap_tracker]
        self.outputs[QUALISYS_JOINT_ANGLE_SUMMARY_STATS.name] = split_ja_summary_dfs["qualisys"]

        self.logger.info("Generating joint angle summary statistics plots")
        fig_angle_summary = plot_angle_summary_grid(angle_summary_stats)
        self.outputs[JOINT_ANGLE_SUMMARY_FIG.name] = fig_angle_summary