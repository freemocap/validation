from validation.pipeline.base import ValidationStep
from validation.steps.rmse.components import REQUIRES, PRODUCES
from validation.steps.rmse.config import RMSEConfig
from validation.steps.rmse.core.calculate_rmse import calculate_rmse
from validation.components import (
    QUALISYS_PARQUET, FREEMOCAP_PARQUET,
    POSITIONABSOLUTEERROR, POSITIONRMSE,
    VELOCITYABSOLUTEERROR, VELOCITYRMSE
)
from validation.utils.actor_utils import make_freemocap_actor_from_parquet

from validation.steps.rmse.dash_app.run_dash_app import run_dash_app


class RMSEStep(ValidationStep):

    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = RMSEConfig
    
    def calculate(self, condition_frame_range:list[int]=None):
        self.logger.info("Starting RMSE calculation")

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        qualisys_parquet_path = self.data[QUALISYS_PARQUET.name]

        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        qualisys_actor = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)

        
        self.rmse_results = calculate_rmse(freemocap_actor=freemocap_actor,
                       qualisys_actor=qualisys_actor,
                       config = self.cfg,
                       frame_range = condition_frame_range,
                       use_rigid=self.ctx.use_rigid
                       )
        
        self.outputs[POSITIONRMSE.name] = self.rmse_results.position_rmse
        self.outputs[POSITIONABSOLUTEERROR.name] = self.rmse_results.position_absolute_error
        self.outputs[VELOCITYRMSE.name] = self.rmse_results.velocity_rmse
        self.outputs[VELOCITYABSOLUTEERROR.name] = self.rmse_results.velocity_absolute_error

    # def visualize(self):
    #     # pass
    #     run_dash_app(
    #         data_and_error=self.rmse_results,
    #         recording_name = self.ctx.recording_dir.stem
    #     )
