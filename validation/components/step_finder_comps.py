from validation.components import DataComponent
from validation.utils.io_helpers import save_plotly_fig

STEPS_FIG = DataComponent(
    name="steps_figure",
    filename="gait_events.html",
    relative_path="{tracker}/analysis_outputs/gait_events",
    saver=save_plotly_fig,
)

GAIT_EVENTS_DEBUG = DataComponent(
    name="gait_events_debug_figure",
    filename="gait_events_debug.html",
    relative_path="{tracker}/analysis_outputs/gait_events/debug",
    saver=save_plotly_fig,
)
