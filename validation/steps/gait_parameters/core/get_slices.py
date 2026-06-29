import pandas as pd
from itertools import pairwise 

def get_heel_strike_frames(
    gait_events: pd.DataFrame,
    frame_range: range | None,
) -> dict[str, list[int]]:
    heelstrikes_dict: dict[str, list[int]] = {}

    for side in ["left", "right"]:
        mask = (
            (gait_events["foot"] == side) &
            (gait_events["event"] == "heel_strike")
        )

        heel_strikes = gait_events.loc[mask, "frame"].sort_values()

        if frame_range is not None:
            heel_strikes = heel_strikes[
                (heel_strikes >= frame_range.start) &
                (heel_strikes < frame_range.stop)
            ]

        # convert to plain Python list of ints
        heelstrikes_dict[side] = heel_strikes.astype(int).to_list()

    return heelstrikes_dict
    
def get_toe_off_frames(
    gait_events: pd.DataFrame,
    frame_range: range | None,
) -> dict[str, list[int]]:
    toeoff_dict: dict[str, list[int]] = {}

    for side in ["left", "right"]:
        mask = (
            (gait_events["foot"] == side) &
            (gait_events["event"] == "toe_off")
        )

        toe_offs = gait_events.loc[mask, "frame"].sort_values()

        if frame_range is not None:
            toe_offs = toe_offs[
                (toe_offs >= frame_range.start) &
                (toe_offs < frame_range.stop)
            ]

        toeoff_dict[side] = toe_offs.astype(int).to_list()

    return toeoff_dict
