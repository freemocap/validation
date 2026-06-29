
from validation.steps.temporal_alignment.step import TemporalAlignmentStep
from validation.steps.spatial_alignment.step   import SpatialAlignmentStep
from validation.steps.rmse.step                import RMSEStep
from validation.steps.trc_conversion.step import TRCConversionStep
from validation.steps.joint_angles.step import JointAnglesStep
from validation.steps.step_finder.step import StepFinderStep
from validation.steps.trajectory_strides.step import TrajectoryStridesStep
from validation.steps.joint_angle_strides.step import JointAnglesStridesStep
from validation.steps.gait_parameters.step import GaitParametersStep

STEP_REGISTRY = {
    "TemporalAlignmentStep": TemporalAlignmentStep,
    "SpatialAlignmentStep":  SpatialAlignmentStep,
    "RMSEStep":              RMSEStep,
    "TRCConversionStep":     TRCConversionStep,
    "JointAnglesStep":       JointAnglesStep,
    "StepFinderStep":        StepFinderStep,
    "TrajectoryStridesStep":  TrajectoryStridesStep,
    "JointAnglesStridesStep": JointAnglesStridesStep,
    "GaitParametersStep":     GaitParametersStep
}