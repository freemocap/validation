import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from validation.paths import DATABASE_PATH, output_dir
from validation.figure_display import show_figure


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PlotConfig:
    reference_system: str = "qualisys"

    freemocap_trackers: tuple[str, ...] = (
        "mediapipe",
        "vitpose",
        "rtmpose",
    )

    plot_height: int = 400
    plot_width: int = 1000

    plot_order_and_titles: dict = field(
        default_factory=lambda: {
            "eyes_on_solid": (
                "<b> Visual <br> Perturbation </b>"
            ),
            "foam_with_open": (
                "<b> Proprioceptive <br> Perturbation </b>"
            ),
            "hardest_vs_easiest": (
                "<b> Visual + Proprioceptive "
                "<br> Perturbation </b>"
            ),
        }
    )

    zero_reference_line_style: dict = field(
        default_factory=lambda: dict(
            color="darkgrey",
            width=1.5,
            dash="dot",
        )
    )

    tracker_styles: dict = field(
        default_factory=lambda: {
            "mediapipe": dict(
                color="#1f77b4",
                symbol="circle",
            ),
            "vitpose": dict(
                color="#ff7f0e",
                symbol="square",
            ),
            "rtmpose": dict(
                color="#2ca02c",
                symbol="diamond",
            ),
        }
    )

    x_axis_title: str = (
        "<b> Reference <br> "
        "ΔCOM path length (mm) </b>"
    )

    axis_title_font: dict = field(
        default_factory=lambda: dict(
            family="Arial",
            size=20,
        )
    )

    axis_tickfont: dict = field(
        default_factory=lambda: dict(
            size=16
        )
    )

    subplot_title_font: dict = field(
        default_factory=lambda: dict(
            size=22
        )
    )

    tracker_display_names: dict = field(
        default_factory=lambda: {
            "mediapipe": "MediaPipe",
            "rtmpose": "RTMPose",
            "vitpose": "ViTPose",
            "qualisys": "Reference",
        }
    )

    def __post_init__(self):
        self.all_trackers = (
            list(self.freemocap_trackers)
            + [self.reference_system]
        )

    def display_name(
        self,
        tracker: str,
    ) -> str:
        return self.tracker_display_names.get(
            tracker,
            tracker,
        )


# =============================================================================
# Condition metadata
# =============================================================================

CONDITION_METADATA = {
    "eyes_open_solid": {
        "eyes": "Open",
        "surface": "Solid Ground",
    },
    "eyes_closed_solid": {
        "eyes": "Closed",
        "surface": "Solid Ground",
    },
    "eyes_open_foam": {
        "eyes": "Open",
        "surface": "Foam",
    },
    "eyes_closed_foam": {
        "eyes": "Closed",
        "surface": "Foam",
    },
}


# =============================================================================
# Database loading
# =============================================================================

def query_df(
    path_to_db: str | Path,
    trackers: list[str],
) -> pd.DataFrame:

    placeholders = ",".join(
        ["?"] * len(trackers)
    )

    query = f"""
        SELECT
            t.participant_code,
            t.trial_name,
            a.path,
            a.component_name,
            a.tracker
        FROM artifacts a
        JOIN trials t
            ON a.trial_id = t.id
        WHERE
            t.trial_type = "balance"
            AND a.category = "com_analysis"
            AND a.tracker IN ({placeholders})
            AND a.file_exists = 1
            AND a.component_name IN (
                "freemocap_balance_metrics",
                "qualisys_balance_metrics"
            )
        ORDER BY
            t.participant_code,
            t.trial_name,
            a.tracker;
    """

    conn = sqlite3.connect(
        path_to_db
    )

    try:
        return pd.read_sql_query(
            query,
            conn,
            params=trackers,
        )
    finally:
        conn.close()


def load_balance_metrics_csv(
    csv_path: str | Path,
) -> pd.DataFrame:

    path = Path(
        csv_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "condition",
        "path_length_mm",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing)}"
        )

    keep = [
        "condition",
        "path_length_mm",
    ]

    if "label" in df.columns:
        keep.append(
            "label"
        )

    out = df[
        keep
    ].copy()

    out = out.rename(
        columns={
            "path_length_mm": "path_length",
        }
    )

    out["path_length"] = pd.to_numeric(
        out["path_length"],
        errors="coerce",
    )

    return out


def parse_db_dataframe(
    db_dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dfs = []

    for _, row in db_dataframe.iterrows():

        sub_df = load_balance_metrics_csv(
            row["path"]
        )

        sub_df["participant_code"] = (
            row["participant_code"]
        )

        sub_df["trial_name"] = (
            row["trial_name"]
        )

        sub_df["tracker"] = (
            row["tracker"]
        )

        dfs.append(
            sub_df
        )

    if not dfs:
        raise RuntimeError(
            "No balance_metrics.csv artifacts were found."
        )

    return pd.concat(
        dfs,
        ignore_index=True,
    )


def add_condition_metadata(
    balance_data: pd.DataFrame,
) -> pd.DataFrame:

    balance_data = (
        balance_data.copy()
    )

    unknown_conditions = (
        set(
            balance_data[
                "condition"
            ].dropna()
        )
        - set(
            CONDITION_METADATA
        )
    )

    if unknown_conditions:
        raise ValueError(
            "Unknown balance condition keys: "
            f"{sorted(unknown_conditions)}"
        )

    balance_data["eyes"] = (
        balance_data[
            "condition"
        ].map(
            lambda condition:
            CONDITION_METADATA[
                condition
            ]["eyes"]
        )
    )

    balance_data["surface"] = (
        balance_data[
            "condition"
        ].map(
            lambda condition:
            CONDITION_METADATA[
                condition
            ]["surface"]
        )
    )

    return balance_data


# =============================================================================
# Manipulation effects
# =============================================================================

def manipulation_eyes_effect(
    df: pd.DataFrame,
    surface: str,
) -> pd.DataFrame:
    """
    Visual perturbation:
        eyes closed - eyes open

    Calculated separately for a specified surface.
    """

    surface_lookup = {
        "solid": "Solid Ground",
        "foam": "Foam",
    }

    if surface not in surface_lookup:
        raise ValueError(
            f"Unknown surface: {surface}"
        )

    surface_name = (
        surface_lookup[
            surface
        ]
    )

    surface_df = df[
        df["surface"]
        == surface_name
    ].copy()

    id_cols = [
        "participant_code",
        "trial_name",
        "tracker",
    ]

    surface_wide = (
        surface_df
        .pivot_table(
            index=id_cols,
            columns="eyes",
            values="path_length",
            aggfunc="first",
        )
        .reset_index()
    )

    surface_wide["difference"] = (
        surface_wide["Closed"]
        - surface_wide["Open"]
    )

    surface_wide["manipulation"] = (
        f"eyes_on_{surface}"
    )

    return surface_wide[
        [
            "participant_code",
            "trial_name",
            "tracker",
            "difference",
            "manipulation",
        ]
    ]


def manipulation_foam_effect(
    df: pd.DataFrame,
    eyes: str,
) -> pd.DataFrame:
    """
    Proprioceptive perturbation:
        foam - solid

    Calculated separately for a specified visual condition.
    """

    eyes_lookup = {
        "open": "Open",
        "closed": "Closed",
    }

    if eyes not in eyes_lookup:
        raise ValueError(
            f"Unknown eye condition: {eyes}"
        )

    eyes_name = (
        eyes_lookup[
            eyes
        ]
    )

    eyes_df = df[
        df["eyes"]
        == eyes_name
    ].copy()

    id_cols = [
        "participant_code",
        "trial_name",
        "tracker",
    ]

    eyes_wide = (
        eyes_df
        .pivot_table(
            index=id_cols,
            columns="surface",
            values="path_length",
            aggfunc="first",
        )
        .reset_index()
    )

    eyes_wide["difference"] = (
        eyes_wide["Foam"]
        - eyes_wide["Solid Ground"]
    )

    eyes_wide["manipulation"] = (
        f"foam_with_{eyes}"
    )

    return eyes_wide[
        [
            "participant_code",
            "trial_name",
            "tracker",
            "difference",
            "manipulation",
        ]
    ]


def manipulation_hardest_easiest(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combined perturbation:

        Eyes Closed / Foam
            -
        Eyes Open / Solid Ground
    """

    easy_condition = df[
        (
            df["eyes"] == "Open"
        )
        & (
            df["surface"]
            == "Solid Ground"
        )
    ].copy()

    hard_condition = df[
        (
            df["eyes"] == "Closed"
        )
        & (
            df["surface"]
            == "Foam"
        )
    ].copy()

    contrast_df = (
        easy_condition.merge(
            hard_condition,
            on=[
                "participant_code",
                "trial_name",
                "tracker",
            ],
            suffixes=(
                "_easy",
                "_hard",
            ),
        )
    )

    contrast_df["difference"] = (
        contrast_df[
            "path_length_hard"
        ]
        - contrast_df[
            "path_length_easy"
        ]
    )

    contrast_df["manipulation"] = (
        "hardest_vs_easiest"
    )

    return contrast_df[
        [
            "participant_code",
            "trial_name",
            "tracker",
            "difference",
            "manipulation",
        ]
    ]


# =============================================================================
# Sensitivity statistics
# =============================================================================

def calculate_regression_equation(
    manipulation_df: pd.DataFrame,
    trackers: list[str] | tuple[str, ...],
    reference: str,
) -> dict[str, tuple[float, float]]:

    r_df = (
        manipulation_df
        .pivot(
            index=[
                "participant_code",
                "trial_name",
                "manipulation",
            ],
            columns="tracker",
            values="difference",
        )
        .reset_index()
    )

    manipulations = (
        r_df[
            "manipulation"
        ]
        .dropna()
        .unique()
    )

    regression_dict = {}

    for tracker in trackers:

        for manipulation in manipulations:

            queried = r_df[
                r_df["manipulation"]
                == manipulation
            ][
                [
                    reference,
                    tracker,
                ]
            ].dropna()

            x = (
                queried[
                    reference
                ]
                .to_numpy()
            )

            y = (
                queried[
                    tracker
                ]
                .to_numpy()
            )

            m, b = np.polyfit(
                x,
                y,
                1,
            )

            regression_dict[
                f"{tracker}_{manipulation}"
            ] = (
                m,
                b,
            )

    return regression_dict


def calculate_pearsons_r(
    manipulation_df: pd.DataFrame,
    trackers: list[str] | tuple[str, ...],
    reference: str,
) -> dict[str, float]:

    r_df = (
        manipulation_df
        .pivot(
            index=[
                "participant_code",
                "trial_name",
                "manipulation",
            ],
            columns="tracker",
            values="difference",
        )
        .reset_index()
    )

    manipulations = (
        r_df[
            "manipulation"
        ]
        .dropna()
        .unique()
    )

    r_value_dict = {}

    for tracker in trackers:

        for manipulation in manipulations:

            queried = r_df[
                r_df["manipulation"]
                == manipulation
            ][
                [
                    reference,
                    tracker,
                ]
            ].dropna()

            r_value_dict[
                f"{tracker}_{manipulation}"
            ] = (
                queried[
                    tracker
                ].corr(
                    queried[
                        reference
                    ]
                )
            )

    return r_value_dict


# =============================================================================
# Plot helpers
# =============================================================================

def limits_zoom_positive(
    sub: pd.DataFrame,
    cols,
    *,
    neg_buffer_frac: float = 0.15,
    margin_frac: float = 0.08,
    min_neg_buffer: float = 0.02,
):

    vals = (
        sub[
            cols
        ]
        .to_numpy()
        .astype(float)
        .ravel()
    )

    vals = vals[
        np.isfinite(
            vals
        )
    ]

    if vals.size == 0:
        return (
            -1,
            1,
        )

    vmin = float(
        vals.min()
    )

    vmax = float(
        vals.max()
    )

    span = max(
        vmax - vmin,
        1e-9,
    )

    pos_scale = max(
        abs(vmax),
        1e-9,
    )

    neg_buffer = max(
        min_neg_buffer,
        neg_buffer_frac
        * pos_scale,
    )

    upper = (
        vmax
        + margin_frac
        * span
    )

    lower = min(
        vmin
        - margin_frac
        * span,
        -neg_buffer,
    )

    lower = min(
        lower,
        0.0,
    )

    upper = max(
        upper,
        0.0,
    )

    return (
        lower,
        upper,
    )


# =============================================================================
# Primary MediaPipe figure
# =============================================================================

def plot_mediapipe_sensitivity(
    manipulation_df: pd.DataFrame,
    pearson_r_dict: dict,
    regression_dict: dict,
    cfg: PlotConfig,
    height: int = 400,
    width: int = 1000,
) -> go.Figure:

    tracker = "mediapipe"
    reference = (
        cfg.reference_system
    )

    plotting_df = (
        manipulation_df
        .pivot_table(
            index=[
                "participant_code",
                "trial_name",
                "manipulation",
            ],
            columns="tracker",
            values="difference",
            aggfunc="first",
        )
        .reset_index()
    )

    manip_order = list(
        cfg.plot_order_and_titles.keys()
    )

    fig = make_subplots(
        rows=1,
        cols=len(
            manip_order
        ),
        subplot_titles=tuple(
            cfg.plot_order_and_titles.values()
        ),
        shared_xaxes=False,
        shared_yaxes=False,
    )

    for col, manipulation in enumerate(
        manip_order,
        start=1,
    ):

        sub = plotting_df[
            plotting_df[
                "manipulation"
            ]
            == manipulation
        ]

        lower, upper = (
            limits_zoom_positive(
                sub,
                [
                    reference,
                    tracker,
                ],
            )
        )

        xaxis_ref = (
            "x"
            if col == 1
            else f"x{col}"
        )

        yaxis_ref = (
            "y"
            if col == 1
            else f"y{col}"
        )

        x_domain = (
            "x domain"
            if col == 1
            else f"x{col} domain"
        )

        y_domain = (
            "y domain"
            if col == 1
            else f"y{col} domain"
        )

        key = (
            f"{tracker}_"
            f"{manipulation}"
        )

        m, b = (
            regression_dict[
                key
            ]
        )

        r = (
            pearson_r_dict[
                key
            ]
        )

        r2 = r**2

        # Regression line
        x_line = np.array(
            [
                lower,
                upper,
            ]
        )

        y_line = (
            m * x_line
            + b
        )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    color="red",
                    width=1.5,
                ),
                showlegend=False,
                opacity=0.7,
            ),
            row=1,
            col=col,
        )

        # Statistics
        fig.add_annotation(
            x=0.95,
            y=0.05,
            xref=x_domain,
            yref=y_domain,
            text=(
                f"<i>r²</i> = "
                f"{r2:.2f}<br>"
                f"slope = {m:.2f}"
            ),
            showarrow=False,
            font=dict(
                size=12
            ),
            xanchor="right",
            yanchor="bottom",
        )

        # Zero reference lines
        fig.add_shape(
            type="line",
            x0=lower,
            x1=upper,
            y0=0,
            y1=0,
            line=(
                cfg.zero_reference_line_style
            ),
            xref=xaxis_ref,
            yref=yaxis_ref,
        )

        fig.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=lower,
            y1=upper,
            line=(
                cfg.zero_reference_line_style
            ),
            xref=xaxis_ref,
            yref=yaxis_ref,
        )

        # Identity line
        fig.add_trace(
            go.Scatter(
                x=[
                    lower,
                    upper,
                ],
                y=[
                    lower,
                    upper,
                ],
                mode="lines",
                line=dict(
                    color="black",
                    dash="dash",
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

        # Data
        fig.add_trace(
            go.Scatter(
                x=sub[
                    reference
                ],
                y=sub[
                    tracker
                ],
                mode="markers",
                marker=dict(
                    size=9,
                    opacity=0.7,
                    color=(
                        cfg.tracker_styles[
                            tracker
                        ]["color"]
                    ),
                    symbol=(
                        cfg.tracker_styles[
                            tracker
                        ]["symbol"]
                    ),
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        fig.update_xaxes(
            title_text=(
                cfg.x_axis_title
            ),
            title_font=(
                cfg.axis_title_font
            ),
            tickfont=(
                cfg.axis_tickfont
            ),
            scaleanchor=yaxis_ref,
            scaleratio=1,
            row=1,
            col=col,
        )

        fig.update_yaxes(
            title_text=(
                f"<b>{cfg.display_name(tracker)}"
                "<br>ΔCOM path length (mm)</b>"
                if col == 1
                else ""
            ),
            title_font=(
                cfg.axis_title_font
            ),
            tickfont=(
                cfg.axis_tickfont
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        height=height,
        width=width,
        template="simple_white",
        font=dict(
            family="Arial",
            size=14,
        ),
        margin=dict(
            l=120,
            r=80,
            t=60,
            b=80,
        ),
    )

    fig.update_annotations(
        font_size=(
            cfg.subplot_title_font[
                "size"
            ]
        )
    )

    return fig


# =============================================================================
# All-trackers supplementary figure
# =============================================================================

def plot_all_trackers_sensitivity(
    manipulation_df: pd.DataFrame,
    pearson_r_dict: dict,
    regression_dict: dict,
    cfg: PlotConfig,
    trackers: list[str] | None = None,
    row_height: int = 400,
    width: int = 1000,
) -> go.Figure:

    if trackers is None:
        trackers = list(
            cfg.freemocap_trackers
        )

    reference = (
        cfg.reference_system
    )

    manip_order = list(
        cfg.plot_order_and_titles.keys()
    )

    n_trackers = len(
        trackers
    )

    n_manips = len(
        manip_order
    )

    subplot_titles = []

    for i, _ in enumerate(
        trackers
    ):

        for manip_title in (
            cfg.plot_order_and_titles.values()
        ):

            if i == 0:
                subplot_titles.append(
                    manip_title
                )
            else:
                subplot_titles.append(
                    ""
                )

    fig = make_subplots(
        rows=n_trackers,
        cols=n_manips,
        subplot_titles=(
            subplot_titles
        ),
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    plotting_df = (
        manipulation_df
        .pivot_table(
            index=[
                "participant_code",
                "trial_name",
                "manipulation",
            ],
            columns="tracker",
            values="difference",
            aggfunc="first",
        )
        .reset_index()
    )

    for i, tracker in enumerate(
        trackers
    ):

        row = i + 1

        for j, manipulation in enumerate(
            manip_order
        ):

            col = j + 1

            sub = plotting_df[
                plotting_df[
                    "manipulation"
                ]
                == manipulation
            ]

            lower, upper = (
                limits_zoom_positive(
                    sub,
                    [
                        reference,
                        tracker,
                    ],
                )
            )

            ax_idx = (
                (row - 1)
                * n_manips
                + col
            )

            xaxis_ref = (
                "x"
                if ax_idx == 1
                else f"x{ax_idx}"
            )

            yaxis_ref = (
                "y"
                if ax_idx == 1
                else f"y{ax_idx}"
            )

            x_domain = (
                "x domain"
                if ax_idx == 1
                else f"x{ax_idx} domain"
            )

            y_domain = (
                "y domain"
                if ax_idx == 1
                else f"y{ax_idx} domain"
            )

            key = (
                f"{tracker}_"
                f"{manipulation}"
            )

            m, b = (
                regression_dict[
                    key
                ]
            )

            r = (
                pearson_r_dict[
                    key
                ]
            )

            r2 = r**2

            # Regression
            x_line = np.array(
                [
                    lower,
                    upper,
                ]
            )

            y_line = (
                m * x_line
                + b
            )

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(
                        color="red",
                        width=1.5,
                    ),
                    showlegend=False,
                    opacity=0.7,
                ),
                row=row,
                col=col,
            )

            # Statistics
            fig.add_annotation(
                x=0.95,
                y=0.05,
                xref=x_domain,
                yref=y_domain,
                text=(
                    f"<i>r²</i> = "
                    f"{r2:.2f}<br>"
                    f"slope = {m:.2f}"
                ),
                showarrow=False,
                font=dict(
                    size=12
                ),
                xanchor="right",
                yanchor="bottom",
            )

            # Zero lines
            fig.add_shape(
                type="line",
                x0=lower,
                x1=upper,
                y0=0,
                y1=0,
                line=(
                    cfg.zero_reference_line_style
                ),
                xref=xaxis_ref,
                yref=yaxis_ref,
            )

            fig.add_shape(
                type="line",
                x0=0,
                x1=0,
                y0=lower,
                y1=upper,
                line=(
                    cfg.zero_reference_line_style
                ),
                xref=xaxis_ref,
                yref=yaxis_ref,
            )

            # Identity
            fig.add_trace(
                go.Scatter(
                    x=[
                        lower,
                        upper,
                    ],
                    y=[
                        lower,
                        upper,
                    ],
                    mode="lines",
                    line=dict(
                        color="black",
                        dash="dash",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            # Data
            style = (
                cfg.tracker_styles.get(
                    tracker,
                    {
                        "color": "#1f77b4",
                        "symbol": "circle",
                    },
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=sub[
                        reference
                    ],
                    y=sub[
                        tracker
                    ],
                    mode="markers",
                    marker=dict(
                        size=9,
                        opacity=0.7,
                        color=(
                            style["color"]
                        ),
                        symbol=(
                            style["symbol"]
                        ),
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(
                title_text=(
                    cfg.x_axis_title
                    if row
                    == n_trackers
                    else ""
                ),
                title_font=(
                    cfg.axis_title_font
                ),
                tickfont=(
                    cfg.axis_tickfont
                ),
                scaleanchor=(
                    yaxis_ref
                ),
                scaleratio=1,
                row=row,
                col=col,
            )

            fig.update_yaxes(
                title_text=(
                    f"<b>{cfg.display_name(tracker)}"
                    "<br>ΔCOM path length (mm)</b>"
                    if col == 1
                    else ""
                ),
                title_font=(
                    cfg.axis_title_font
                ),
                tickfont=(
                    cfg.axis_tickfont
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=(
            row_height
            * n_trackers
        ),
        width=width,
        template="simple_white",
        font=dict(
            family="Arial",
            size=14,
        ),
        margin=dict(
            l=120,
            r=80,
            t=60,
            b=80,
        ),
    )

    fig.update_annotations(
        font_size=(
            cfg.subplot_title_font[
                "size"
            ]
        )
    )

    return fig


# =============================================================================
# Typst table
# =============================================================================

def generate_sensitivity_table_typst(
    pearson_r_dict: dict,
    regression_dict: dict,
    cfg: PlotConfig,
    trackers: list[str] | None = None,
) -> str:

    if trackers is None:
        trackers = list(
            cfg.freemocap_trackers
        )

    manip_order = list(
        cfg.plot_order_and_titles.keys()
    )

    manip_labels = {
        "eyes_on_solid": (
            "Visual \\ Perturbation"
        ),
        "foam_with_open": (
            "Proprioceptive \\ Perturbation"
        ),
        "hardest_vs_easiest": (
            "Visual + Proprioceptive "
            "\\ Perturbation"
        ),
    }

    n_manips = len(
        manip_order
    )

    lines = [
        "#let path-length-sensitivity = {",
        "  set text(size: 9pt)",
        "  table(",
        (
            f'    columns: '
            f'(1.2fr, '
            f'{", ".join(["1.5fr"] * n_manips)}),'
        ),
        (
            f'    align: '
            f'(left, '
            f'{", ".join(["center"] * n_manips)}),'
        ),
        "    stroke: none,",
        "    table.hline(stroke: 1pt),",
        "    table.header(",
        "      [*System*],",
    ]

    for manipulation in manip_order:
        lines.append(
            f"      "
            f"[*{manip_labels[manipulation]}*],"
        )

    lines.extend(
        [
            "    ),",
            "    table.hline(stroke: 0.5pt),",
        ]
    )

    for tracker in trackers:

        lines.append(
            f"    "
            f"[{cfg.display_name(tracker)}],"
        )

        for manipulation in manip_order:

            key = (
                f"{tracker}_"
                f"{manipulation}"
            )

            slope, _ = (
                regression_dict[
                    key
                ]
            )

            r_value = (
                pearson_r_dict[
                    key
                ]
            )

            r_squared = (
                r_value**2
            )

            lines.append(
                f"    "
                f"[slope = {slope:.2f}, "
                f"_r_#super[2] = "
                f"{r_squared:.2f}],"
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

    return (
        "\n".join(
            lines
        )
        + "\n"
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    cfg = PlotConfig()

 
    figures_path = output_dir("figures", "balance")

    tables_path = output_dir("tables", "balance")

    path_to_database = DATABASE_PATH


    # -------------------------------------------------------------------------
    # Load balance_metrics.csv
    # -------------------------------------------------------------------------

    db_df = query_df(
        path_to_db=(
            path_to_database
        ),
        trackers=(
            cfg.all_trackers
        ),
    )

    if db_df.empty:
        raise RuntimeError(
            "No balance_metrics.csv artifacts "
            "were found in validation.db."
        )

    balance_data = (
        parse_db_dataframe(
            db_df
        )
    )

    balance_data = (
        add_condition_metadata(
            balance_data
        )
    )

    # -------------------------------------------------------------------------
    # Compute the three perturbations used in the paper
    # -------------------------------------------------------------------------

    eyes_on_solid_effect = (
        manipulation_eyes_effect(
            balance_data,
            "solid",
        )
    )

    foam_with_eyes_open = (
        manipulation_foam_effect(
            balance_data,
            eyes="open",
        )
    )

    hardest_vs_easiest = (
        manipulation_hardest_easiest(
            balance_data
        )
    )

    manipulation_df = pd.concat(
        [
            eyes_on_solid_effect,
            foam_with_eyes_open,
            hardest_vs_easiest,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    r_value_dict = (
        calculate_pearsons_r(
            manipulation_df=(
                manipulation_df
            ),
            trackers=(
                cfg.freemocap_trackers
            ),
            reference=(
                cfg.reference_system
            ),
        )
    )

    slope_intercept_dict = (
        calculate_regression_equation(
            manipulation_df=(
                manipulation_df
            ),
            trackers=(
                cfg.freemocap_trackers
            ),
            reference=(
                cfg.reference_system
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Print values for parity checking
    # -------------------------------------------------------------------------

    print(
        "\nSensitivity statistics:"
    )

    for tracker in (
        cfg.freemocap_trackers
    ):

        print(
            f"\n{cfg.display_name(tracker)}"
        )

        for manipulation in (
            cfg.plot_order_and_titles
        ):

            key = (
                f"{tracker}_"
                f"{manipulation}"
            )

            slope, intercept = (
                slope_intercept_dict[
                    key
                ]
            )

            r = (
                r_value_dict[
                    key
                ]
            )

            print(
                f"  {manipulation}: "
                f"slope={slope:.4f}, "
                f"intercept={intercept:.4f}, "
                f"r={r:.4f}, "
                f"r²={r**2:.4f}"
            )

    # -------------------------------------------------------------------------
    # Supplementary figure
    # -------------------------------------------------------------------------

    fig_all = (
        plot_all_trackers_sensitivity(
            manipulation_df=(
                manipulation_df
            ),
            pearson_r_dict=(
                r_value_dict
            ),
            regression_dict=(
                slope_intercept_dict
            ),
            cfg=cfg,
        )
    )

    show_figure(fig_all)

    figure_path = (
        figures_path
        / "sensitivity_all_trackers.svg"
    )

    fig_all.write_image(
        figure_path,
        scale=3,
    )

    # -------------------------------------------------------------------------
    # Table
    # -------------------------------------------------------------------------

    typst_table = (
        generate_sensitivity_table_typst(
            pearson_r_dict=(
                r_value_dict
            ),
            regression_dict=(
                slope_intercept_dict
            ),
            cfg=cfg,
        )
    )

    table_path = (
        tables_path
        / "path_length_sensitivity_table.typ"
    )

    table_path.write_text(
        typst_table,
        encoding="utf-8",
    )

    print(
        f"\nFigure written to: "
        f"{figure_path}"
    )

    print(
        f"Table written to: "
        f"{table_path}"
    )