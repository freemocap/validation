from dataclasses import dataclass
import numpy as np
import pandas as pd
@dataclass
class GaitEvents:
    heel_strikes: np.ndarray
    toe_offs: np.ndarray

@dataclass
class GaitResults:
    right_foot: GaitEvents
    left_foot: GaitEvents
    
    def to_dataframe(self):
        rows = []

        for frame in self.left_foot.heel_strikes:
            rows.append({'foot':'left', 'event': 'heel_strike', 'frame': frame})
        for frame in self.right_foot.heel_strikes:
            rows.append({'foot': 'right', 'event': 'heel_strike', 'frame':frame})
        for frame in self.left_foot.toe_offs:
            rows.append({'foot':'left', 'event': 'toe_off', 'frame': frame})
        for frame in self.right_foot.toe_offs:
            rows.append({'foot': 'right', 'event': 'toe_off', 'frame':frame})
        return pd.DataFrame(rows)
        f = 2


@dataclass
class GaitEventsFlagged:
    right_foot: GaitEvents
    left_foot: GaitEvents
    


 


