
from validation.steps.step_finder.core.models import GaitEvents, GaitResults
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_heel_strike_and_toe_off_events(heel_velocity:np.ndarray, 
                                       toe_velocity:np.ndarray,
                                       frames_of_interest:tuple[int,int]|None = None,):
    
    heel_strike_candidates = np.where((heel_velocity[:-1,1]>0) & (heel_velocity[1:, 1] <= 0)) [0] + 1
    toe_off_candidates = np.where((toe_velocity[:-1,1]<=0) & (toe_velocity[1:,1]>0))[0] + 1
    
    heel_strikes = heel_strike_candidates    
    toe_offs = toe_off_candidates

    if frames_of_interest is not None:
        start_frame, end_frame = frames_of_interest
        heel_strikes = heel_strikes[(heel_strikes >= start_frame) & (heel_strikes <= end_frame)]
        toe_offs = toe_offs[(toe_offs >= start_frame) & (toe_offs <= end_frame)]
    
    
    return GaitEvents(heel_strikes=heel_strikes, toe_offs=toe_offs)



def detect_gait_events(left_heel_velocity:np.ndarray,
                       left_toe_velocity:np.ndarray,
                       right_heel_velocity:np.ndarray,
                       right_toe_velocity:np.ndarray,
                       frames_of_interest:tuple[int,int]|None = None,):

    right_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=right_heel_velocity,
        toe_velocity=right_toe_velocity,
        frames_of_interest=frames_of_interest,
    )
    
    left_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=left_heel_velocity,
        toe_velocity=left_toe_velocity,
        frames_of_interest=frames_of_interest,
    )

    return GaitResults(right_foot=right_foot_gait_events, left_foot=left_foot_gait_events)
