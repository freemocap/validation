from validation.pipeline.base import ValidationStep
from validation.steps.gait_parameters.config import GaitParametersConfig
from validation.steps.gait_parameters.components import *
from validation.steps.gait_parameters.components import REQUIRES, PRODUCES

from validation.utils.actor_utils import make_freemocap_actor_from_parquet

from validation.steps.gait_parameters.core.get_slices import get_heel_strike_frames, get_toe_off_frames
from skellymodels.managers.human import Human
import numpy as np
import pandas as pd
from dataclasses import dataclass 
from itertools import pairwise

@dataclass
class ParamStats:
    data: np.ndarray
    mean: float
    std: float

@dataclass
class StanceSwingStats:
    stance: ParamStats
    swing: ParamStats

@dataclass
class GaitParameters:
    left_stride_duration: ParamStats
    right_stride_duration: ParamStats
    left_stance_duration: ParamStats
    right_stance_duration: ParamStats
    left_swing_duration: ParamStats
    right_swing_duration: ParamStats
    left_stance_percentage: ParamStats
    right_stance_percentage: ParamStats
    left_swing_percentage: ParamStats
    right_swing_percentage: ParamStats
    left_step_length: ParamStats
    right_step_length: ParamStats
    left_stride_length: ParamStats
    right_stride_length: ParamStats


def calculate_stride_duration(heel_strikes:list,sampling_rate:float):
    stride_duration = np.diff(heel_strikes)/sampling_rate
    mean_stride_duration = np.mean(stride_duration)
    std_stride_duration = np.std(stride_duration)

    return ParamStats(
        data = stride_duration,
        mean = mean_stride_duration,
        std = std_stride_duration
    )

def calculate_stance_swing_time(toe_offs:list, heel_strikes:list, sampling_rate:float):
    stance_times = []
    swing_times = []
    for i in range(len(heel_strikes)-1):
        hs_1 = heel_strikes[i]
        hs_2 = heel_strikes[i+1]

        toe_offs_within_step = [x for x in toe_offs if x > hs_1 and x < hs_2]

        if not toe_offs_within_step:
            print(f"No toe offs found between heel strikes at frames {hs_1, hs_2}. Inserting NaN.")
            stance_times.append(np.nan)
            swing_times.append(np.nan)
            continue

        if len(toe_offs_within_step) > 1:
            print(f"Found multiple toe offs between heel strikes at frames {hs_1,hs_2}, using the first")

        toe_off = toe_offs_within_step[0]
        stance_time = (toe_off - hs_1)/sampling_rate
        stance_times.append(stance_time)

        swing_time = (hs_2 - toe_off)/sampling_rate
        swing_times.append(swing_time)

    stance_times = np.array(stance_times)
    mean_stance_time = np.nanmean(stance_times)
    std_stance_time = np.nanstd(stance_times)

    swing_times = np.array(swing_times)
    mean_swing_time = np.nanmean(swing_times)
    std_swing_time = np.nanstd(swing_times)

    stance_data = ParamStats(
        data=stance_times,
        mean=mean_stance_time,
        std=std_stance_time
    )

    swing_data = ParamStats(
        data=swing_times,
        mean=mean_swing_time,
        std=std_swing_time
    )

    return StanceSwingStats(
        stance=stance_data,
        swing=swing_data
    )

def calculate_stance_swing_percentages(stance_swing_times:StanceSwingStats, stride_durations:ParamStats):
    stance_percentages = (stance_swing_times.stance.data / stride_durations.data) * 100.0
    mean_stance_percentage = np.nanmean(stance_percentages)
    std_stance_percentage = np.nanstd(stance_percentages)

    swing_percentages = (stance_swing_times.swing.data / stride_durations.data) * 100.0
    mean_swing_percentage = np.nanmean(swing_percentages)
    std_swing_percentage = np.nanstd(swing_percentages)

    stance_percentage_data = ParamStats(
        data=stance_percentages,
        mean=mean_stance_percentage,
        std=std_stance_percentage
    )

    swing_percentage_data = ParamStats(
        data=swing_percentages,
        mean=mean_swing_percentage,
        std=std_swing_percentage
    )

    return StanceSwingStats(
        stance=stance_percentage_data,
        swing=swing_percentage_data
    )

def calculate_step_length(
    heel_strikes: dict[str, np.ndarray],
    ankle: dict[str, np.ndarray],
    treadmill_belt_speed: float,
    sampling_rate: float,
    side: str,
):
    AP_AXIS = 1  #whatever +y forward is for your convention

    ipsi = side
    contra = "left" if side == "right" else "right"

    ipsi_hs = np.asarray(heel_strikes[ipsi], dtype=int)
    contra_hs = np.asarray(heel_strikes[contra], dtype=int)

    step_lengths = []

    for ipsi_frame in ipsi_hs:
        previous_contra = contra_hs[contra_hs < ipsi_frame]
        
        
        if previous_contra.size == 0:
            continue

        contra_frame = previous_contra[-1]

        frame_duration = ipsi_frame - contra_frame
        dt = frame_duration / sampling_rate
        belt_distance = treadmill_belt_speed * dt

        contra_pos = ankle[f"{contra}_ankle"][contra_frame, AP_AXIS]
        ipsi_pos = ankle[f"{ipsi}_ankle"][ipsi_frame, AP_AXIS]

        step_length = np.abs(ipsi_pos - contra_pos) + belt_distance
        step_lengths.append(step_length)

    step_lengths = np.asarray(step_lengths, dtype=float)

    return ParamStats(
        data=step_lengths,
        mean=np.nanmean(step_lengths),
        std=np.nanstd(step_lengths),
    )

def calculate_stride_length(
        heel_strikes: dict[str, np.ndarray],
        ankle: dict[str, np.ndarray],
        treadmill_belt_speed: float,
        sampling_rate: float,
        side: str
):
    AP_AXIS = 1
    ipsi = side
    ipsi_hs = heel_strikes[ipsi]
    ipsi_ankle = ankle[f"{ipsi}_ankle"]

    stride_lengths = []
    for pair in pairwise(ipsi_hs):
        hs, next_hs = pair

        frame_duration = next_hs - hs
        dt = frame_duration/sampling_rate
        belt_distance = dt*treadmill_belt_speed

        ankle_pos = ipsi_ankle[hs,AP_AXIS]
        ankle_next_pos = ipsi_ankle[next_hs, AP_AXIS]

        stride_length = np.abs(ankle_next_pos-ankle_pos) + belt_distance
        stride_lengths.append(stride_length)

    stride_lengths = np.array(stride_lengths, dtype=float)
    stride_length_mean = np.mean(stride_lengths)
    stride_length_std = np.std(stride_lengths)

    return ParamStats(
        data = stride_lengths,
        mean = stride_length_mean,
        std = stride_length_std
    )


def estimate_belt_speed(ankle, heel_strikes, toe_offs, AP_AXIS, sampling_rate):
    """Estimate belt speed from foot velocity during mid-stance, both sides."""
    velocities = []
    for side in ["left", "right"]:
        side_ankle = ankle[f"{side}_ankle"]
        side_hs = heel_strikes[side]
        side_to = toe_offs[side]
        
        for i in range(len(side_hs) - 1):
            hs = side_hs[i]
            to = [t for t in side_to if hs < t < side_hs[i + 1]]
            if not to:
                continue
            midstance = (hs + to[0]) // 2
            dt = 1.0 / sampling_rate
            vel = (side_ankle[midstance + 1, AP_AXIS] - side_ankle[midstance - 1, AP_AXIS]) / (2 * dt)
            velocities.append(abs(vel))
    
    return np.mean(velocities)

def calculate_gait_parameters(
    gait_events: pd.DataFrame,
    sampling_rate: float,
    frame_range: range | None,
    ankle: dict[str, np.ndarray],
    treadmill_belt_speed: float
) -> GaitParameters:
    heel_strikes: dict[str, list[int]] = get_heel_strike_frames(gait_events, frame_range)
    toe_offs: dict[str, list[int]] = get_toe_off_frames(gait_events, frame_range)

    left_stride_duration = calculate_stride_duration(
        heel_strikes=heel_strikes["left"],
        sampling_rate=sampling_rate,
    )
    right_stride_duration = calculate_stride_duration(
        heel_strikes=heel_strikes["right"],
        sampling_rate=sampling_rate,
    )

    left_stance_swing_stats = calculate_stance_swing_time(
        toe_offs=toe_offs["left"],
        heel_strikes=heel_strikes["left"],
        sampling_rate=sampling_rate,
    )
    right_stance_swing_stats = calculate_stance_swing_time(
        toe_offs=toe_offs["right"],
        heel_strikes=heel_strikes["right"],
        sampling_rate=sampling_rate,
    )

    left_stance_swing_percentages = calculate_stance_swing_percentages(
        stance_swing_times=left_stance_swing_stats,
        stride_durations=left_stride_duration,
    )
    right_stance_swing_percentages = calculate_stance_swing_percentages(
        stance_swing_times=right_stance_swing_stats,
        stride_durations=right_stride_duration,
    )

    left_step_length = calculate_step_length(
        heel_strikes=heel_strikes,
        ankle=ankle,
        treadmill_belt_speed = treadmill_belt_speed,
        sampling_rate=sampling_rate,
        side = "left"
    )
    right_step_length = calculate_step_length(
        heel_strikes=heel_strikes,
        ankle=ankle,
        treadmill_belt_speed = treadmill_belt_speed,
        sampling_rate=sampling_rate,
        side = "right"
    )

    left_stride_length = calculate_stride_length(
        heel_strikes=heel_strikes,
        ankle=ankle,
        treadmill_belt_speed=treadmill_belt_speed,
        sampling_rate=sampling_rate,
        side = "left"
    )

    right_stride_length = calculate_stride_length(
        heel_strikes=heel_strikes,
        ankle=ankle,
        treadmill_belt_speed=treadmill_belt_speed,
        sampling_rate=sampling_rate,
        side = "right"
    )

    f = 2

    return GaitParameters(
        left_stride_duration=left_stride_duration,
        right_stride_duration=right_stride_duration,
        left_stance_duration=left_stance_swing_stats.stance,
        right_stance_duration=right_stance_swing_stats.stance,
        left_swing_duration=left_stance_swing_stats.swing,
        right_swing_duration=right_stance_swing_stats.swing,
        left_stance_percentage=left_stance_swing_percentages.stance,
        right_stance_percentage=right_stance_swing_percentages.stance,
        left_swing_percentage=left_stance_swing_percentages.swing,
        right_swing_percentage=right_stance_swing_percentages.swing,
        left_step_length=left_step_length,
        right_step_length=right_step_length,
        left_stride_length=left_stride_length,
        right_stride_length=right_stride_length
    )


def _param_to_gait_metrics_df(
    param: ParamStats,
    metric: str,
    side: str,
    system: str,
) -> pd.DataFrame:
    """
    Flatten a ParamStats.data array into a per-stride long-format DataFrame.
    """
    n = param.data.shape[0]
    return pd.DataFrame(
        {
            "system": system,
            "side": side,
            "metric": metric,           # e.g. "stride_duration", "stance_pct"
            "event_index": np.arange(n, dtype=int),
            "value": param.data,        # may contain NaNs
        }
    )


def gait_metrics_to_long_df(
    gp: GaitParameters,
    system: str,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    dfs.append(_param_to_gait_metrics_df(gp.left_stride_duration,  "stride_duration",  "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_stride_duration, "stride_duration",  "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_stance_duration,  "stance_duration",  "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_stance_duration, "stance_duration",  "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_swing_duration,   "swing_duration",   "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_swing_duration,  "swing_duration",   "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_stance_percentage, "stance_pct", "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_stance_percentage,"stance_pct","right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_swing_percentage,  "swing_pct", "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_swing_percentage, "swing_pct", "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_step_length,  "step_length",  "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_step_length, "step_length",  "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_stride_length,  "stride_length",  "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_stride_length, "stride_length",  "right", system))

    return pd.concat(dfs, ignore_index=True)


def _param_to_summary_row(
    param: ParamStats,
    metric: str,
    side: str,
    system: str,
) -> dict:
    """
    Build a single summary row (mean, std, n_valid) for a given ParamStats.
    """
    data = param.data
    n_valid = int(np.isfinite(data).sum())
    return {
        "system": system,
        "side": side,
        "metric": metric,
        "mean": float(param.mean),
        "std": float(param.std),
        "n_valid": n_valid,
    }


def gait_parameters_to_summary_df(
    gp: GaitParameters,
    system: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    rows.append(_param_to_summary_row(gp.left_stride_duration,  "stride_duration",  "left",  system))
    rows.append(_param_to_summary_row(gp.right_stride_duration, "stride_duration",  "right", system))

    rows.append(_param_to_summary_row(gp.left_stance_duration,  "stance_duration",  "left",  system))
    rows.append(_param_to_summary_row(gp.right_stance_duration, "stance_duration",  "right", system))

    rows.append(_param_to_summary_row(gp.left_swing_duration,   "swing_duration",   "left",  system))
    rows.append(_param_to_summary_row(gp.right_swing_duration,  "swing_duration",   "right", system))

    rows.append(_param_to_summary_row(gp.left_stance_percentage, "stance_pct", "left",  system))
    rows.append(_param_to_summary_row(gp.right_stance_percentage,"stance_pct","right", system))

    rows.append(_param_to_summary_row(gp.left_swing_percentage,  "swing_pct", "left",  system))
    rows.append(_param_to_summary_row(gp.right_swing_percentage, "swing_pct", "right", system))

    rows.append(_param_to_summary_row(gp.left_step_length, "step_length", "left", system))
    rows.append(_param_to_summary_row(gp.right_step_length, "step_length", "right", system))

    rows.append(_param_to_summary_row(gp.left_stride_length, "stride_length", "left", system))
    rows.append(_param_to_summary_row(gp.right_stride_length, "stride_length", "right", system))

    return pd.DataFrame(rows)

class GaitParametersStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = GaitParametersConfig

    def calculate(self, condition_frame_range:list[int]=None):
        self.logger.info("Calculating gait parameters")
        sampling_rate = self.cfg.sampling_rate

        qualisys_gait_events = self.data[QUALISYS_GAIT_EVENTS.name]
        freemocap_gait_events = self.data[FREEMOCAP_GAIT_EVENTS.name]

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        qualisys_parquet_path = self.data[QUALISYS_PARQUET.name]

        freemocap_actor:Human = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        qualisys_actor:Human = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)
        
        #need ankle joint centers for step/stride length
        keys = ['left_ankle', 'right_ankle']
        freemocap_ankle = {k: freemocap_actor.body.xyz.as_dict[k] for k in keys}
        qualisys_ankle = {k: qualisys_actor.body.xyz.as_dict[k] for k in keys}

        frame_range = range(*condition_frame_range) if condition_frame_range is not None else None

        AP_AXIS = 1  
        treadmill_belt_speed = np.round(estimate_belt_speed(
            ankle=qualisys_ankle,
            heel_strikes=get_heel_strike_frames(qualisys_gait_events, frame_range),
            toe_offs= get_toe_off_frames(qualisys_gait_events, frame_range),
            AP_AXIS=AP_AXIS,
            sampling_rate=sampling_rate
        ),-2)

        self.logger.info(f"ESTIMATED treadmill belt speed: {treadmill_belt_speed} m/s")

        q_gp: GaitParameters = calculate_gait_parameters(
            gait_events=qualisys_gait_events,
            sampling_rate=sampling_rate,
            frame_range=frame_range,
            ankle = qualisys_ankle,
            treadmill_belt_speed = treadmill_belt_speed
        )
        f_gp: GaitParameters = calculate_gait_parameters(
            gait_events=freemocap_gait_events,
            sampling_rate=sampling_rate,
            frame_range=frame_range,
            ankle = freemocap_ankle,
            treadmill_belt_speed = treadmill_belt_speed
        )

        q_gait_metrics_df = gait_metrics_to_long_df(q_gp, system="qualisys")
        f_gait_metrics_df = gait_metrics_to_long_df(f_gp, system=self.ctx.project_config.freemocap_tracker)
        per_gait_metrics_df = pd.concat([q_gait_metrics_df, f_gait_metrics_df], ignore_index=True)

        q_summary_df = gait_parameters_to_summary_df(q_gp, system="qualisys")
        f_summary_df = gait_parameters_to_summary_df(f_gp, system=self.ctx.project_config.freemocap_tracker)
        summary_df = pd.concat([q_summary_df, f_summary_df], ignore_index=True)

        self.outputs[QUALISYS_GAIT_METRICS.name] = q_gait_metrics_df
        self.outputs[FREEMOCAP_GAIT_METRICS.name] = f_gait_metrics_df
        self.outputs[QUALISYS_GAIT_SUMMARY_STATS.name] = q_summary_df
        self.outputs[FREEMOCAP_GAIT_SUMMARY_STATS.name] = f_summary_df


        f = 2 