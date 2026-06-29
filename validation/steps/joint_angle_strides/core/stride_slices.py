import pandas as pd
from itertools import pairwise 

def get_heel_strike_slices(gait_events:pd.DataFrame, frame_range: range|None) -> dict[str, list[slice]]:
    heelstrikes_dict = {}

    for side in ["left", "right"]:
        mask = (
            (gait_events["foot"] == side) &
            (gait_events["event"] == "heel_strike")
        )

        heel_strikes = gait_events.loc[mask, "frame"].sort_values()

        if frame_range is not None:
            heel_strikes = heel_strikes[(heel_strikes >= frame_range.start) & (heel_strikes < frame_range.stop)].to_list()

        slices: list[slice] = []

        for a,b in pairwise(heel_strikes):
            if a < b:
                slices.append(slice(a,b))

        heelstrikes_dict[side] = slices
    
    return heelstrikes_dict
    