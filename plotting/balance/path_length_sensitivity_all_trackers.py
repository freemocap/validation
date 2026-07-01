import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass, field

@dataclass
class PlotConfig:
    reference_system: str = "qualisys"
    freemocap_trackers: tuple[str] = ("mediapipe", "vitpose", "rtmpose")

    plot_height=400
    plot_width=1000

    plot_order_and_titles = {
        "eyes_on_solid": "<b> Visual <br> Perturbation </b>",
        "foam_with_open": "<b> Proprioceptive <br> Perturbation </b>",
        "hardest_vs_easiest": "<b> Visual + Proprioceptive  <br> Perturbation </b>",
    }

    zero_reference_line_style = dict(color="darkgrey", width=1.5, dash="dot")

    tracker_styles = {
        "mediapipe": dict(color="#1f77b4", symbol="circle"),
        "rtmpose": dict(color="#d62728", symbol="diamond"),
    }

    x_axis_title = "<b> Reference <br> ΔCOM path length (mm) </b>"
    y_axis_title = "<b> MediaPipe <br> ΔCOM path length (mm) </b>"

    axis_title_font = dict(family="Arial", size=20)
    axis_tickfont = dict(size=16)

    subplot_title_font = dict(size=22)

    tracker_display_names: dict = field(default_factory=lambda: {
        "mediapipe": "MediaPipe",
        "rtmpose": "RTMPose",
        "vitpose": "ViTPose",
        "qualisys": "Qualisys",
    })

    def display_name(self, tracker: str) -> str:
        return self.tracker_display_names.get(tracker, tracker)


    def __post_init__(self):
        self.all_trackers = list(self.freemocap_trackers) + [self.reference_system]


def query_df(path_to_db:str, trackers:list[str]):
    conn = sqlite3.connect(path_to_db)


    placeholders = ",".join(["?"]*len(trackers))

    query = f"""
    SELECT t.participant_code, 
            t.trial_name,
            a.path,
            a.component_name,
            a.condition,
            a.tracker
    FROM artifacts a
    JOIN trials t ON a.trial_id = t.id
    WHERE t.trial_type = "balance"
        AND a.category = "com_analysis"
        AND a.tracker IN ({placeholders})
        AND a.file_exists = 1
        AND a.component_name LIKE '%path_length_com'
    ORDER BY t.trial_name, a.path;
    """
    path_df = pd.read_sql_query(query, conn, params = trackers)
    conn.close()

    return path_df

def parse_db_dataframe(db_dataframe:pd.DataFrame):
    dfs = []

    for _, row in db_dataframe.iterrows():
        path = row["path"]
        tracker = row["tracker"]
        condition = row.get("condition") or ""  # handle None/empty
        participant = row["participant_code"]
        trial = row["trial_name"]

        # Load file — autodetect type
        sub_df = pd.read_json(path)
        sub_df = sub_df.rename(columns={"Frame Intervals": "frame_interval",
                                        "Path Lengths:": "path_length"}).reset_index()
        sub_df = sub_df.rename(columns={"index": "condition"})
        # Add metadata columns
        sub_df["participant_code"] = participant
        sub_df["trial_name"] = trial
        sub_df["tracker"] = tracker

        dfs.append(sub_df)

    # Concatenate all into one tidy DataFrame
    return pd.concat(dfs, ignore_index=True)


def manipulation_eyes_effect(df:pd.DataFrame, surface:str) -> pd.DataFrame:
    """
    Gets the effect of visual manipulation (eyes open vs closed) for the specified surface.
    """
    if surface == "solid":
        surface_df = df.query("surface == 'Solid Ground'").copy()
    elif surface == "foam":
        surface_df = df.query("surface == 'Foam'").copy()
    else:
        raise ValueError(f"Unknown surface '{surface}' for manipulation_eyes_effect")

    id_cols = [
        "participant_code",
        "trial_name",
        "tracker"
    ]

    surface_wide = (
        surface_df.pivot_table(
            index = id_cols,
            columns = "eyes",
            values = "path_length",
            aggfunc = "first"
        )
        .reset_index()
    )

    surface_wide["difference"] = surface_wide["Closed"] - surface_wide["Open"]
    surface_wide["manipulation"] = f"eyes_on_{surface}"

    return surface_wide[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]


def manipulation_foam_effect(df: pd.DataFrame, eyes:str) -> pd.DataFrame:
    """
    Gets the effect of surface manipulation (foam vs solid) for the specified visual condition (eyes open vs closed).
    """
    if eyes == "open":
        eyes_df = df.query("eyes == 'Open'").copy()
    elif eyes == "closed":
        eyes_df = df.query("eyes == 'Closed'").copy()
    else:
        raise ValueError(f"Unknown eye condition '{eyes}")
    
    id_cols = [
        "participant_code",
        "trial_name",
        "tracker"
    ]

    eyes_wide = (
        eyes_df.pivot_table(
            index = id_cols,
            columns = "surface",
            values = "path_length",
            aggfunc = "first"
        )
        .reset_index()
    )

    eyes_wide["difference"] = eyes_wide["Foam"] - eyes_wide["Solid Ground"]
    eyes_wide["manipulation"] = f"foam_with_{eyes}"
    return eyes_wide[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]
    f = 2

def manipulation_hardest_easiest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the effect of the hardest (eyes closed on foam) vs easiest (eyes open on solid ground) condition.
    """

    easy_condition = df.query(("eyes == 'Open' & surface == 'Solid Ground'")).copy()
    hard_condition = df.query(("eyes == 'Closed' & surface == 'Foam'")).copy()

    contrast_df = easy_condition.merge(
        hard_condition,
        on = ["participant_code", "trial_name", "tracker"],
        suffixes=("_easy", "_hard")
    )

    contrast_df["difference"] = contrast_df["path_length_hard"] - contrast_df["path_length_easy"]
    contrast_df["manipulation"] = "hardest_vs_easiest"
    return contrast_df[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]

id_cols = [
    "participant_code",
    "condition",
    "trial_name",
]

def calculate_regression_equation(manipulation_df:pd.DataFrame, trackers:list[str], reference:str) -> dict[str, tuple[float, float]]:
        """ Calculates the slope and intercept of the linear regression line between each tracker and the reference system for every manipulation.
        Returns a dictionary with the manipulations tagged with tracker names as keys and tupples of the slope and intercept as values, e.g. {"mediapipe_eyes_on_solid": (slope, intercept)}."""
        r_df = manipulation_df.pivot(
        index = ["participant_code", "trial_name", "manipulation"],
        columns= "tracker",
        values = "difference"
        ).reset_index()

        manipulations = r_df["manipulation"].dropna().unique()

        r_value_dict = {}
        for tracker in trackers:
            for manipulation in manipulations:
                r_queried = r_df.query(f"manipulation == '{manipulation}'")
                x = r_queried[reference].to_numpy()
                y = r_queried[tracker].to_numpy()
                m, b = np.polyfit(x, y, 1)  
                r_value_dict[f"{tracker}_{manipulation}"] = (m, b)
        return r_value_dict

def calculate_pearsons_r(manipulation_df:pd.DataFrame, trackers:list[str], reference:str) -> dict[str, float]:
    """Calculates Pearson's r between each tracker and the reference system for each manipulation.
    Returns a dictionary with keys like 'mediapipe_eyes_on_solid' and values being the corresponding r value."""

    r_df = manipulation_df.pivot(
        index = ["participant_code", "trial_name", "manipulation"],
        columns= "tracker",
        values = "difference"
    ).reset_index()
    
    manipulations = r_df["manipulation"].dropna().unique()

    r_value_dict = {}
    for tracker in trackers:
        for manipulation in manipulations:
            r_queried = r_df.query(f"manipulation == '{manipulation}'")
            r_value_dict[f"{tracker}_{manipulation}"] = r_queried[tracker].corr(r_queried[reference])
        
    return r_value_dict


def limits_zoom_positive(
    sub: pd.DataFrame,
    cols,
    *,
    neg_buffer_frac: float = 0.15,   # show a bit below 0
    margin_frac: float = 0.08,       # padding around data
    min_neg_buffer: float = 0.02,    # absolute minimum negative space
):
    """
    Data-driven limits that keep 0 visible but don't waste 50% of the panel.
    Uses combined x/y values (Qualisys + trackers) so the y=x line is meaningful.
    """
    vals = sub[cols].to_numpy().astype(float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1, 1)

    vmin = float(vals.min())
    vmax = float(vals.max())
    span = max(vmax - vmin, 1e-9)

    # positive-focused negative buffer based on overall positive magnitude
    pos_scale = max(abs(vmax), 1e-9)
    neg_buffer = max(min_neg_buffer, neg_buffer_frac * pos_scale)

    # upper padding
    upper = vmax + margin_frac * span

    # lower: include observed negatives if present, else small negative buffer
    lower = min(vmin - margin_frac * span, -neg_buffer)

    # ensure 0 is inside
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)

    # make x and y share same limits (square-ish interpretability)
    lim = max(abs(lower), abs(upper))
    # but *not* symmetric: we want more positive room; keep lower as computed
    # We'll return lower/upper, and you set both x and y to these.
    return (lower, upper)

def plot_mediapipe_sensitivity(
    manipulation_df: pd.DataFrame,
    pearson_r_dict: dict,
    regression_dict: dict,
    cfg: PlotConfig,
    height: int = 400,
    width: int = 1000,
) -> go.Figure:
    """Primary figure: single-row MediaPipe sensitivity across manipulations."""
    tracker = "mediapipe"
    reference = cfg.reference_system

    plotting_df = (
        manipulation_df
        .pivot_table(
            index=["participant_code", "trial_name", "manipulation"],
            columns="tracker",
            values="difference",
            aggfunc="first",
        )
        .reset_index()
    )

    manip_order = list(cfg.plot_order_and_titles.keys())

    fig = make_subplots(
        rows=1,
        cols=len(manip_order),
        subplot_titles=tuple(cfg.plot_order_and_titles.values()),
        shared_xaxes=False,
        shared_yaxes=False,
    )

    for col, manipulation in enumerate(manip_order, start=1):
        sub = plotting_df.query("manipulation == @manipulation")
        lower, upper = limits_zoom_positive(sub, [reference, tracker])

        xaxis_ref = f"x{col}" if col > 1 else "x"
        yaxis_ref = f"y{col}" if col > 1 else "y"

        key = f"{tracker}_{manipulation}"
        m, b = regression_dict[key]
        r = pearson_r_dict[key]
        r2 = r ** 2

        # Regression line
        x_line = np.array([lower, upper])
        y_line = m * x_line + b
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode="lines", line=dict(color="red", width=1.5),
                showlegend=False, opacity=0.7,
            ),
            row=1, col=col,
        )

        # Annotation
        fig.add_annotation(
            x=0.95, y=0.05,
            xref=f"x{col if col > 1 else ''} domain",
            yref=f"y{col if col > 1 else ''} domain",
            text=f"<i>r²</i> = {r2:.2f}<br>slope = {m:.2f}",
            showarrow=False, font=dict(size=12),
            xanchor="right", yanchor="bottom",
        )

        # Zero reference lines
        fig.add_shape(type="line", x0=lower, x1=upper, y0=0, y1=0,
                      line=cfg.zero_reference_line_style,
                      xref=xaxis_ref, yref=yaxis_ref)
        fig.add_shape(type="line", x0=0, x1=0, y0=lower, y1=upper,
                      line=cfg.zero_reference_line_style,
                      xref=xaxis_ref, yref=yaxis_ref)

        # Identity line
        fig.add_trace(
            go.Scatter(
                x=[lower, upper], y=[lower, upper],
                mode="lines", line=dict(color="black", dash="dash"),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=col,
        )

        # Data points
        fig.add_trace(
            go.Scatter(
                x=sub[reference], y=sub[tracker],
                mode="markers",
                marker=dict(size=9, opacity=0.7, color="#1f77b4", symbol="circle"),
                showlegend=False,
            ),
            row=1, col=col,
        )

    for c in range(1, len(manip_order) + 1):
        fig.update_xaxes(
            title_text=cfg.x_axis_title,
            title_font=cfg.axis_title_font,
            tickfont=cfg.axis_tickfont,
            scaleanchor=f"y{c if c > 1 else ''}",
            scaleratio=1,
            row=1, col=c,
        )
        fig.update_yaxes(
            title_text=f"<b>{cfg.display_name(tracker)}<br>ΔCOM path length (mm)</b>" if c == 1 else "",
            title_font=cfg.axis_title_font,
            tickfont=cfg.axis_tickfont,
            row=1, col=c,
        )

    fig.update_layout(
        height=height, width=width,
        template="simple_white",
        font=dict(family="Arial", size=14),
        margin=dict(l=120, r=80, t=60, b=80),
    )
    fig.update_annotations(font_size=cfg.subplot_title_font["size"])

    return fig


def plot_all_trackers_sensitivity(
    manipulation_df: pd.DataFrame,
    pearson_r_dict: dict,
    regression_dict: dict,
    cfg: PlotConfig,
    trackers: list[str] = None,
    row_height: int = 400,
    width: int = 1000,
) -> go.Figure:
    """Supplementary figure: one row per tracker, columns are manipulations."""
    if trackers is None:
        trackers = list(cfg.freemocap_trackers)

    reference = cfg.reference_system
    manip_order = list(cfg.plot_order_and_titles.keys())
    n_trackers = len(trackers)
    n_manips = len(manip_order)

    # Only show manipulation titles on the first row, empty for the rest
    subplot_titles = []
    for i, t in enumerate(trackers):
        for manip_title in cfg.plot_order_and_titles.values():
            if i == 0:
                subplot_titles.append(manip_title)
            else:
                subplot_titles.append("")

    tracker_colors = {
        "mediapipe": "#1f77b4",
        "vitpose": "#ff7f0e",
        "rtmpose": "#2ca02c",
    }

    fig = make_subplots(
        rows=n_trackers,
        cols=n_manips,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    plotting_df = (
        manipulation_df
        .pivot_table(
            index=["participant_code", "trial_name", "manipulation"],
            columns="tracker",
            values="difference",
            aggfunc="first",
        )
        .reset_index()
    )

    for i, tracker in enumerate(trackers):
        row = i + 1

        for j, manipulation in enumerate(manip_order):
            col = j + 1
            sub = plotting_df.query("manipulation == @manipulation")
            lower, upper = limits_zoom_positive(sub, [reference, tracker])

            # Plotly axis indexing
            ax_idx = (row - 1) * n_manips + col
            xaxis_ref = "x" if ax_idx == 1 else f"x{ax_idx}"
            yaxis_ref = "y" if ax_idx == 1 else f"y{ax_idx}"
            x_domain = "x domain" if ax_idx == 1 else f"x{ax_idx} domain"
            y_domain = "y domain" if ax_idx == 1 else f"y{ax_idx} domain"

            key = f"{tracker}_{manipulation}"
            m, b = regression_dict[key]
            r = pearson_r_dict[key]
            r2 = r ** 2

            # Regression line
            x_line = np.array([lower, upper])
            y_line = m * x_line + b
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines", line=dict(color="red", width=1.5),
                    showlegend=False, opacity=0.7,
                ),
                row=row, col=col,
            )

            # Annotation
            fig.add_annotation(
                x=0.95, y=0.05,
                xref=x_domain, yref=y_domain,
                text=f"<i>r²</i> = {r2:.2f}<br>slope = {m:.2f}",
                showarrow=False, font=dict(size=12),
                xanchor="right", yanchor="bottom",
            )

            # Zero reference lines
            fig.add_shape(type="line", x0=lower, x1=upper, y0=0, y1=0,
                          line=cfg.zero_reference_line_style,
                          xref=xaxis_ref, yref=yaxis_ref)
            fig.add_shape(type="line", x0=0, x1=0, y0=lower, y1=upper,
                          line=cfg.zero_reference_line_style,
                          xref=xaxis_ref, yref=yaxis_ref)

            # Identity line
            fig.add_trace(
                go.Scatter(
                    x=[lower, upper], y=[lower, upper],
                    mode="lines", line=dict(color="black", dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ),
                row=row, col=col,
            )

            # Data points
            fig.add_trace(
                go.Scatter(
                    x=sub[reference], y=sub[tracker],
                    mode="markers",
                    marker=dict(size=9, opacity=0.7, color=tracker_colors.get(tracker, "#1f77b4")),
                    showlegend=False,
                ),
                row=row, col=col,
            )

            # Axes
            fig.update_xaxes(
                title_text=cfg.x_axis_title if row == n_trackers else "",
                title_font=cfg.axis_title_font,
                tickfont=cfg.axis_tickfont,
                scaleanchor=yaxis_ref,
                scaleratio=1,
                row=row, col=col,
            )
            fig.update_yaxes(
                title_text=f"<b>{cfg.display_name(tracker)}<br>ΔCOM path length (mm)</b>" if col == 1 else "",
                title_font=cfg.axis_title_font,
                tickfont=cfg.axis_tickfont,
                row=row, col=col,
            )

    fig.update_layout(
        height=row_height * n_trackers,
        width=width,
        template="simple_white",
        font=dict(family="Arial", size=14),
        margin=dict(l=120, r=80, t=60, b=80),
    )
    fig.update_annotations(font_size=cfg.subplot_title_font["size"])

    return fig

f = 2

def generate_sensitivity_table_typst(
    pearson_r_dict: dict,
    regression_dict: dict,
    cfg: PlotConfig,
    trackers: list[str] | None = None,
) -> str:
    """Generate an importable Typst sensitivity table."""

    if trackers is None:
        trackers = list(cfg.freemocap_trackers)

    manip_order = list(cfg.plot_order_and_titles.keys())

    manip_labels = {
        "eyes_on_solid": "Visual \\ Perturbation",
        "foam_with_open": "Proprioceptive \\ Perturbation",
        "hardest_vs_easiest": (
            "Visual + Proprioceptive \\ Perturbation"
        ),
    }

    n_manips = len(manip_order)

    lines = [
        "#let path-length-sensitivity = {",
        "  set text(size: 9pt)",
        "  table(",
        (
            f'    columns: '
            f'(1.2fr, {", ".join(["1.5fr"] * n_manips)}),'
        ),
        (
            f'    align: '
            f'(left, {", ".join(["center"] * n_manips)}),'
        ),
        "    stroke: none,",
        "    table.hline(stroke: 1pt),",
        "    table.header(",
        "      [*System*],",
    ]

    for manipulation in manip_order:
        lines.append(
            f"      [*{manip_labels[manipulation]}*],"
        )

    lines.extend(
        [
            "    ),",
            "    table.hline(stroke: 0.5pt),",
        ]
    )

    for tracker in trackers:
        lines.append(
            f"    [{cfg.display_name(tracker)}],"
        )

        for manipulation in manip_order:
            key = f"{tracker}_{manipulation}"

            slope, _ = regression_dict[key]
            r_value = pearson_r_dict[key]
            r_squared = r_value**2

            lines.append(
                f"    [slope = {slope:.2f}, "
                f"_r_#super[2] = {r_squared:.2f}],"
            )

        lines.append(
            "    table.hline(stroke: 0.5pt),"
        )

    lines.extend(
        [
            "    table.hline(stroke: 1pt),",
            "  )",
            "}",
        ]
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cfg = PlotConfig()    
    path_to_database = "validation.db"
    root_path = Path(r"D:\validation_public_release_v1")
    root_path.mkdir(exist_ok=True, parents=True)


    db_df = query_df(path_to_db=path_to_database, trackers=cfg.all_trackers)
    
    balance_data = parse_db_dataframe(db_df)
    categorized_conditions = balance_data["condition"].str.extract(r"Eyes\s+(Open|Closed)\s*/\s*(Solid Ground|Foam)") #separate out into eyes and ground conditions
    
    #update the df with the separated conditions
    balance_data["eyes"] = categorized_conditions[0]
    balance_data["surface"] = categorized_conditions[1]

    #Compute all the manipulation effects and combine
    eyes_on_solid_effect = manipulation_eyes_effect(balance_data, "solid")
    eyes_on_foam_effect = manipulation_eyes_effect(balance_data, "foam")
    foam_with_eyes_open = manipulation_foam_effect(balance_data, eyes = "open")
    foam_with_eyes_closed = manipulation_foam_effect(balance_data, eyes = "closed")
    hardest_vs_easiest = manipulation_hardest_easiest(balance_data)
    manipulation_df = pd.concat(
    [
        eyes_on_solid_effect,
        foam_with_eyes_open,
        hardest_vs_easiest
    ],
    ignore_index=True
)


    r_value_dict = calculate_pearsons_r(manipulation_df=manipulation_df,
                                        trackers=cfg.freemocap_trackers,
                                        reference=cfg.reference_system)
    
    
    slope_intercept_dict = calculate_regression_equation(manipulation_df=manipulation_df,
                              trackers=cfg.all_trackers,
                              reference=cfg.reference_system)

    # fig_mp = plot_mediapipe_sensitivity(
    #     manipulation_df=manipulation_df,
    #     pearson_r_dict=r_value_dict,
    #     regression_dict=slope_intercept_dict,
    #     cfg=cfg,
    # )
    # fig_mp.show()
    # fig_mp.write_image(root_path / "com_sensitivity.svg", scale=3)

    # Supplementary figure
    fig_all = plot_all_trackers_sensitivity(
        manipulation_df=manipulation_df,
        pearson_r_dict=r_value_dict,
        regression_dict=slope_intercept_dict,
        cfg=cfg,
    )
    fig_all.show()
    fig_all.write_image(root_path / "figures" /"sensitivity_all_trackers.svg", scale=3)

    typst_table = generate_sensitivity_table_typst(
    pearson_r_dict=r_value_dict,
    regression_dict=slope_intercept_dict,
    cfg=cfg,
    )
    (root_path / "tables" / "path_length_sensitivity_table.typ").write_text(typst_table, encoding="utf-8")


f = 2