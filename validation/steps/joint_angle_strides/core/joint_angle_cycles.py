import pandas as pd
import numpy as np

def normalize_gait_cycle(gait_cycle_data:np.ndarray, n_points: int) -> np.ndarray:
    num_frames = gait_cycle_data.shape[0]
    
    x = np.linspace(0, 1.0, num=num_frames)
    x_new = np.linspace(0, 1.0, num=n_points, endpoint=True)

    result = np.interp(x_new, x, gait_cycle_data)
    
    return result

def get_cycles_for_tracker(
        angle_df: pd.DataFrame,
        gait_events: dict[str, list[slice]],
        tracker: str, 
        n_points:int = 100
    ) -> pd.DataFrame:
    dfs = []

    for side, slices in gait_events.items():
        side_df = angle_df.query(f"side == '{side}'")
        for (joint, comp),g in side_df.groupby(['joint', 'component'], observed=True):
            angle = g['angle'].to_numpy()
            angle_cycles = np.stack([normalize_gait_cycle(angle[s], n_points = n_points) for s in slices])
            
            num_cycles = angle_cycles.shape[0]
            cycle_count = np.repeat(np.arange(1, num_cycles+1), n_points)
            percent_gait_cycle = np.tile(np.arange(n_points), num_cycles)
            angle_cycles_1d = angle_cycles.ravel()
            
            df = pd.DataFrame({
                "joint": joint,
                "side": side,
                "angle": angle_cycles_1d,
                "component": comp,
                "cycle": cycle_count,
                "percent_gait_cycle": percent_gait_cycle,
                "tracker": tracker
            })
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def create_angle_cycles(
        freemocap_df: pd.DataFrame,
        qualisys_df: pd.DataFrame,
        gait_events: dict[str, list[slice]],
        freemocap_tracker_name: str,
        n_points: int = 100,
):
    fmc_angle_cycles = get_cycles_for_tracker(freemocap_df,
                           gait_events=gait_events,
                           tracker=freemocap_tracker_name,
                           n_points=n_points)
    
    qtm_angle_cycles = get_cycles_for_tracker(qualisys_df,
                           gait_events=gait_events,
                           tracker='qualisys',
                           n_points=n_points)

    return pd.concat([fmc_angle_cycles, qtm_angle_cycles], ignore_index=True)

def get_angle_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    return (
        cycles
        .groupby(["tracker", "joint", "side", "component", "percent_gait_cycle"], observed=True)
        .agg(mean=("angle", "mean"),
             std=("angle", "std"))
        .reset_index()
        .melt(
            id_vars=["tracker", "joint", "side", "component", "percent_gait_cycle"],
            value_vars=["mean", "std"],
            var_name="stat",
            value_name="value"
        )
    )