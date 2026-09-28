from validation.pipeline.base import ValidationStep
from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from validation.components import FREEMOCAP_PARQUET, QUALISYS_PARQUET, FREEMOCAP_BALANCE_METRICS, QUALISYS_BALANCE_METRICS, FREEMOCAP_BALANCE_VELOCITIES, QUALISYS_BALANCE_VELOCITIES
from validation.steps.balance_metrics.components import REQUIRES, PRODUCES
from validation.steps.balance_metrics.core.calculate_path_length import calculate_path_lengths
from validation.steps.balance_metrics.core.calculate_95_confidence_ellipse import calculate_ellipse_area
from validation.steps.balance_metrics.core.calculate_velocity import calculate_mean_2d_velocity, calculate_condition_velocities

import pandas as pd

class BalanceStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES


    def calculate(self): 
        self.logger.info("Starting balance_metrics calculation")

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        qualisys_parquet_path = self.data[QUALISYS_PARQUET.name]

        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        qualisys_actor = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)

        if freemocap_actor.body.total_body_com is None:
            freemocap_actor.calculate()

        if qualisys_actor.body.total_body_com is None:
            qualisys_actor.calculate()

        freemocap_com = freemocap_actor.body.total_body_com.as_array
        qualisys_com = qualisys_actor.body.total_body_com.as_array


        freemocap_path_lengths = calculate_path_lengths(
            freemocap_com, 
            condition_info_dict=self.ctx.conditions
            )
        qualisys_path_lengths = calculate_path_lengths(
            qualisys_com, 
            condition_info_dict=self.ctx.conditions
            )

        freemocap_ellipse_areas = calculate_ellipse_area(
            freemocap_com, 
            condition_info_dict=self.ctx.conditions
            )
        qualisys_ellipse_areas = calculate_ellipse_area(
            qualisys_com, 
            condition_info_dict=self.ctx.conditions
            )

    
        freemocap_com_velocities = calculate_condition_velocities(
            com_data=freemocap_com,
            condition_info_dict=self.ctx.conditions,
            sampling_rate=self.ctx.project_config.sampling_rate
        )

        qualisys_com_velocities = calculate_condition_velocities(
            com_data=qualisys_com,
            condition_info_dict=self.ctx.conditions,
            sampling_rate=self.ctx.project_config.sampling_rate
        )

        freemocap_mean_2d_velocities = calculate_mean_2d_velocity(
            freemocap_com_velocities
        )
        qualisys_mean_2d_velocities = calculate_mean_2d_velocity(
            qualisys_com_velocities
        )

        freemocap_balance_metrics = create_balance_metrics_dataframe(
             condition_info_dict=self.ctx.conditions,
             path_lengths=freemocap_path_lengths,
             ellipse_areas = freemocap_ellipse_areas,
             mean_2d_velocities = freemocap_mean_2d_velocities
        )

        qualisys_balance_metrics = create_balance_metrics_dataframe(
             condition_info_dict=self.ctx.conditions,
            path_lengths=qualisys_path_lengths,
            ellipse_areas=qualisys_ellipse_areas,
            mean_2d_velocities=qualisys_mean_2d_velocities
        )

        freemocap_balance_velocity_df = create_balance_velocity_dataframe(
            condition_velocities=freemocap_com_velocities,
            condition_info_dict=self.ctx.conditions
        )

        qualisys_balance_velocity_df = create_balance_velocity_dataframe(
            condition_velocities=qualisys_com_velocities,
            condition_info_dict=self.ctx.conditions
        )


        self.outputs[FREEMOCAP_BALANCE_METRICS.name] = freemocap_balance_metrics
        self.outputs[QUALISYS_BALANCE_METRICS.name] = qualisys_balance_metrics
        self.outputs[FREEMOCAP_BALANCE_VELOCITIES.name] = freemocap_balance_velocity_df
        self.outputs[QUALISYS_BALANCE_VELOCITIES.name] = qualisys_balance_velocity_df

        f = 2


def create_balance_metrics_dataframe(
        condition_info_dict: dict,
        path_lengths: dict,
        ellipse_areas: dict,
        mean_2d_velocities: dict
):
        rows = []

        for condition_name, condition_data in condition_info_dict.items():
            start_frame, end_frame = condition_data["frames"]
            label = condition_data["label"]

            rows.append({
                "condition": condition_name,
                "label": label,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "path_length_mm": path_lengths[condition_name],
                "ellipse_area_mm2": ellipse_areas[condition_name],
                "mean_2d_velocity_mm_s": mean_2d_velocities[condition_name],
            })
             
        return pd.DataFrame(rows)

def create_balance_velocity_dataframe(
    condition_velocities: dict,
    condition_info_dict: dict,
):
    rows = []

    for condition_name, velocity in condition_velocities.items():
        condition_data = condition_info_dict[condition_name]

        start_frame, _ = condition_data["frames"]
        label = condition_data["label"]

        for frame_offset, (velocity_x, velocity_y, velocity_z) in enumerate(velocity):
            rows.append({
                "condition": condition_name,
                "label": label,
                "frame": start_frame + frame_offset,
                "velocity_x_mm_s": velocity_x,
                "velocity_y_mm_s": velocity_y,
                "velocity_z_mm_s": velocity_z,
            })

    return pd.DataFrame(rows)