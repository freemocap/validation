from validation.pipeline.base import ValidationStep
from validation.components import FREEMOCAP_PARQUET, QUALISYS_PARQUET, FREEMOCAP_GAIT_EVENTS, QUALISYS_GAIT_EVENTS, STEPS_FIG, GAIT_EVENTS_DEBUG
from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from validation.steps.step_finder.components import REQUIRES, PRODUCES
from validation.steps.step_finder.core.step_finding import detect_gait_events
from validation.steps.step_finder.core.cleanup.detection import find_suspicious_events
from validation.steps.step_finder.core.cleanup.removal import remove_flagged_events_from_gait_results, filter_double_detections_from_gait_results
from validation.steps.step_finder.core.calculate_kinematics import get_foot_kinematics, FootKinematics
from validation.steps.step_finder.core.models import GaitResults
from validation.steps.step_finder.core.steps_plot import plot_stepfinder_mega_debug, plot_gait_events_over_time_both_feet
from validation.steps.step_finder.config import StepFinderConfig
import numpy as np



class StepFinderStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = StepFinderConfig

    def calculate(self):
        self.logger.info("Starting step finding")
        sampling_rate = self.cfg.sampling_rate

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        qualisys_parquet_path = self.data[QUALISYS_PARQUET.name]

        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        freemocap_foot_kinematics:FootKinematics = get_foot_kinematics(freemocap_actor, sampling_rate, use_rigid=self.ctx.use_rigid)

        qualisys_actor = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)
        qualisys_foot_kinematics:FootKinematics = get_foot_kinematics(qualisys_actor, sampling_rate, use_rigid=False)


        freemocap_gait_events:GaitResults = detect_gait_events(
            left_heel_velocity=freemocap_foot_kinematics.left_heel_vel,
            left_toe_velocity=freemocap_foot_kinematics.left_toe_vel,
            right_heel_velocity=freemocap_foot_kinematics.right_heel_vel,
            right_toe_velocity=freemocap_foot_kinematics.right_toe_vel,
            frames_of_interest=self.cfg.frames_of_interest,
        )

        qualisys_gait_events:GaitResults = detect_gait_events(
            left_heel_velocity=qualisys_foot_kinematics.left_heel_vel,
            left_toe_velocity=qualisys_foot_kinematics.left_toe_vel,
            right_heel_velocity=qualisys_foot_kinematics.right_heel_vel,
            right_toe_velocity=qualisys_foot_kinematics.right_toe_vel,
            frames_of_interest=self.cfg.frames_of_interest,
        )

        cleanup_enabled = False  

        freemocap_gait_events = filter_double_detections_from_gait_results(freemocap_gait_events, min_gap=15)

        # qualisys_gait_events = filter_double_detections_from_gait_results(qualisys_gait_events, min_gap=15)
        freemocap_flagged_events = find_suspicious_events(
            foot_kinematics=freemocap_foot_kinematics,
            gait_events=freemocap_gait_events,
        )

        if cleanup_enabled:
            freemocap_cleaned_gait_events = remove_flagged_events_from_gait_results(
                gait_events=freemocap_gait_events,
                flagged_events=freemocap_flagged_events,
            )
        else:
            freemocap_cleaned_gait_events = freemocap_gait_events

        

        freemocap_cleaned_gait_events_df = freemocap_cleaned_gait_events.to_dataframe()
        qualisys_events_df = qualisys_gait_events.to_dataframe()

        fig_debug = plot_stepfinder_mega_debug(
            freemocap_kinematics=freemocap_foot_kinematics,
            qualisys_kinematics=qualisys_foot_kinematics,
            freemocap_gait_events=freemocap_gait_events,
            qualisys_gait_events=qualisys_gait_events,
            flagged_events=freemocap_flagged_events,
            sampling_rate=self.cfg.sampling_rate,
        )

        fig_steps = plot_gait_events_over_time_both_feet(
            q_left_hs=qualisys_gait_events.left_foot.heel_strikes,
            q_left_to=qualisys_gait_events.left_foot.toe_offs,
            fmc_left_hs=freemocap_cleaned_gait_events.left_foot.heel_strikes,
            fmc_left_to=freemocap_cleaned_gait_events.left_foot.toe_offs,
            q_right_hs=qualisys_gait_events.right_foot.heel_strikes,
            q_right_to=qualisys_gait_events.right_foot.toe_offs,
            fmc_right_hs=freemocap_cleaned_gait_events.right_foot.heel_strikes,
            fmc_right_to=freemocap_cleaned_gait_events.right_foot.toe_offs,
            sampling_rate=self.cfg.sampling_rate,
            title=f"Gait events for {self.ctx.recording_dir.stem}",
            xlim=None
        )

        path_to_save = self.ctx.recording_dir / "validation" / self.ctx.project_config.freemocap_tracker / "gait_events"
        path_to_save.mkdir(parents=True, exist_ok=True)

        self.outputs[STEPS_FIG.name] = fig_steps
        self.outputs[GAIT_EVENTS_DEBUG.name] = fig_debug

        self.outputs[FREEMOCAP_GAIT_EVENTS.name] = freemocap_cleaned_gait_events_df
        self.outputs[QUALISYS_GAIT_EVENTS.name] = qualisys_events_df
        

