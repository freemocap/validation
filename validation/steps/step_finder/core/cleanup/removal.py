from validation.steps.step_finder.core.models import GaitEvents, GaitResults
import numpy as np

def _remove_flagged(events: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Remove any events that appear in `flagged`."""
    if flagged is None or flagged.size == 0:
        return events
    keep_mask = ~np.isin(events, flagged)
    return events[keep_mask]

def _clean_gait_events(
    events: GaitEvents,
    flagged: GaitEvents,
) -> GaitEvents:
    """Return a new GaitEvents with flagged events removed."""
    return GaitEvents(
        heel_strikes=_remove_flagged(events.heel_strikes, flagged.heel_strikes),
        toe_offs=_remove_flagged(events.toe_offs, flagged.toe_offs),
    )

def filter_double_detections(events: np.ndarray, min_gap: int) -> np.ndarray:
    if len(events) < 2:
        return events
    filtered = [events[-1]]
    for e in reversed(events[:-1]):
        if (filtered[-1] - e) >= min_gap:
            filtered.append(e)
    filtered.reverse()
    return np.array(filtered, dtype=int)

def filter_double_detections_from_gait_results(gait_events: GaitResults, min_gap: int = 15) -> GaitResults:
    return GaitResults(
        left_foot=GaitEvents(
            heel_strikes=filter_double_detections(gait_events.left_foot.heel_strikes, min_gap),
            toe_offs=filter_double_detections(gait_events.left_foot.toe_offs, min_gap),
        ),
        right_foot=GaitEvents(
            heel_strikes=filter_double_detections(gait_events.right_foot.heel_strikes, min_gap),
            toe_offs=filter_double_detections(gait_events.right_foot.toe_offs, min_gap),
        ),
    )

def remove_flagged_events_from_gait_results(
    gait_events: GaitResults,
    flagged_events: GaitResults,
) -> GaitResults:

    clean_left = _clean_gait_events(
        gait_events.left_foot,
        flagged_events.left_foot,
    )

    clean_right = _clean_gait_events(
        gait_events.right_foot,
        flagged_events.right_foot,
    )

    return GaitResults(
        left_foot=clean_left,
        right_foot=clean_right,
    )

