from validation.pipeline.base import ValidationStep
from validation.steps.joint_angles.components import REQUIRES, PRODUCES
from validation.steps.joint_angles.core.calculate_joint_angles import calculate_joint_angles
from validation.steps.joint_angles.config import JointAnglesConfig
from validation.components import FREEMOCAP_PARQUET, QUALISYS_PARQUET, FREEMOCAP_JOINT_ANGLES, QUALISYS_JOINT_ANGLES
from validation.utils.actor_utils import make_freemocap_actor_from_parquet

class JointAnglesStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = JointAnglesConfig

    def calculate(self):
        self.logger.info("Starting joint angles calculation")

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        qualisys_parquet_path = self.data[QUALISYS_PARQUET.name]
        
        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        qualisys_actor = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)
        
        freemocap_joint_angles = calculate_joint_angles(
            human=freemocap_actor,
            neutral_stance_frames= range(*self.cfg.neutral_frames) if self.cfg.neutral_frames else None,
            use_rigid=self.ctx.use_rigid
        )

        self.logger.info("FreeMoCap joint angles calculated")

        qualisys_joint_angles = calculate_joint_angles(
            human=qualisys_actor,
            neutral_stance_frames= range(*self.cfg.neutral_frames) if self.cfg.neutral_frames else None,
            use_rigid=False
        )
        self.logger.info("Qualisys joint angles calculated")

        self.outputs[FREEMOCAP_JOINT_ANGLES.name] = freemocap_joint_angles
        self.outputs[QUALISYS_JOINT_ANGLES.name] = qualisys_joint_angles
