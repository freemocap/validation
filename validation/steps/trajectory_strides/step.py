from validation.pipeline.base import ValidationStep
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

from validation.steps.trajectory_strides.components import REQUIRES, PRODUCES
from validation.steps.trajectory_strides.config import TrajectoryStridesConfig
from validation.steps.trajectory_strides.core.stride_slices import get_heel_strike_slices
from validation.steps.trajectory_strides.core.trajectory_cycles import create_trajectory_cycles, get_trajectory_summary
from validation.steps.trajectory_strides.core.trajectory_gait_plots import plot_trajectory_cycles_grid, plot_trajectory_summary_grid
from validation.steps.trajectory_strides.core.calculate_trajectory_rmse import calculate_rmse


from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from skellymodels.managers.human import Human


class TrajectoryStridesStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = TrajectoryStridesConfig

    def calculate(self, condition_frame_range:list[int]=None):

        use_rigid = self.ctx.use_rigid
        gait_events = self.data[QUALISYS_GAIT_EVENTS.name]
        frame_range = range(*condition_frame_range) if condition_frame_range is not None else None

        freemocap_actor:Human = make_freemocap_actor_from_parquet(parquet_path=self.data[FREEMOCAP_PARQUET.name])
        qualisys_actor:Human = make_freemocap_actor_from_parquet(parquet_path=self.data[QUALISYS_PARQUET.name])

        heel_strikes:dict[str, list[slice]] = get_heel_strike_slices(gait_events=gait_events, frame_range=frame_range)

        for side, slices in heel_strikes.items():
            self.logger.info(f"Found {len(slices)} strides for the {side} foot")


        markers = ["hip", "knee", "ankle", "heel", "foot_index"]
        
        qtm = qualisys_actor.body.xyz.as_dict
        
        
        fmc_body = freemocap_actor.body

        if use_rigid:
            fmc = fmc_body.rigid_xyz.as_dict
        else:
            fmc = fmc_body.xyz.as_dict

        self.logger.info(f"Separating trajectories into strides")
        trajectory_per_stride = create_trajectory_cycles(
            freemocap_dict=fmc,
            qualisys_dict=qtm,
            marker_list=markers,
            gait_events=heel_strikes,
            freemocap_tracker_name = self.ctx.project_config.freemocap_tracker,
            n_points=100
        )

        self.logger.info(f"Computing summary statistics for trajectories per stride")
        trajectory_summary_stats = get_trajectory_summary(trajectory_per_stride)

        split_t_stride_dfs = {tracker: df_t for tracker, df_t in trajectory_per_stride.groupby("tracker")}

        freemocap_stride_df = split_t_stride_dfs[self.ctx.project_config.freemocap_tracker]
        qualisys_stride_df = split_t_stride_dfs["qualisys"]

        rmse_df = calculate_rmse(freemocap_df=freemocap_stride_df, qualisys_df=qualisys_stride_df)
        self.outputs[TRAJECTORY_RMSE_STATS.name] = rmse_df
        self.outputs[FREEMOCAP_TRAJECTORY_CYCLES.name] = split_t_stride_dfs[self.ctx.project_config.freemocap_tracker]
        self.outputs[QUALISYS_TRAJECTORY_CYCLES.name] = split_t_stride_dfs["qualisys"]

        split_t_summary_dfs = {tracker: df_t for tracker, df_t in trajectory_summary_stats.groupby("tracker")}
        self.outputs[FREEMOCAP_TRAJECTORY_SUMMARY_STATS.name] = split_t_summary_dfs[self.ctx.project_config.freemocap_tracker]
        self.outputs[QUALISYS_TRAJECTORY_SUMMARY_STATS.name] = split_t_summary_dfs["qualisys"]

        self.logger.info(f"Generating trajectories per gait cycle plots")
        fig = plot_trajectory_cycles_grid(trajectory_per_stride, marker_order=markers)

        self.logger.info(f"Generating trajectory summary statistics plots")
        fig_summary = plot_trajectory_summary_grid(trajectory_summary_stats, marker_order=markers)
        
        self.outputs[TRAJECTORY_PER_STRIDE_FIG.name] = fig
        self.outputs[TRAJECTORY_MEAN_FIG.name] = fig_summary

