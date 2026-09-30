import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from validation.paths import DATABASE_PATH, output_dir


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

    tracker_colors: dict = field(
        default_factory=lambda: {
            "mediapipe": "#0072B2",
            "rtmpose": "#D55E00",
            "vitpose": "#006D43",
            "qualisys": "black",
        }
    )

    plot_height: int = 450
    plot_width: int = 1200

    subplot_title_font: dict = field(
        default_factory=lambda: dict(size=20)
    )

    # Machine-readable condition names used in balance_metrics.csv
    condition_order: tuple[str, ...] = (
        "eyes_open_solid",
        "eyes_closed_solid",
        "eyes_open_foam",
        "eyes_closed_foam",
    )

    condition_display_names: dict = field(
        default_factory=lambda: {
            "eyes_open_solid": "Eyes Open/Solid Ground",
            "eyes_closed_solid": "Eyes Closed/Solid Ground",
            "eyes_open_foam": "Eyes Open/Foam",
            "eyes_closed_foam": "Eyes Closed/Foam",
        }
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

    def condition_display_name(
        self,
        condition: str,
    ) -> str:
        return self.condition_display_names.get(
            condition,
            condition,
        )


# =============================================================================
# Database loading
# =============================================================================

def query_df(
    path_to_db: Path,
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
    """
    Load the canonical balance_metrics.csv artifact.

    Only the condition and path-length result are required for the
    agreement analyses.

    Expected columns:
        condition
        label
        start_frame
        end_frame
        path_length_mm
        ellipse_area_mm2
        mean_2d_velocity_mm_s
    """

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

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    keep_columns = [
        "condition",
        "path_length_mm",
    ]

    if "label" in df.columns:
        keep_columns.append(
            "label"
        )

    out = df[
        keep_columns
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


def parse_database_df(
    df: pd.DataFrame,
    cfg: PlotConfig,
) -> pd.DataFrame:

    dfs = []

    for _, row in df.iterrows():

        sub = load_balance_metrics_csv(
            row["path"]
        )

        sub["participant_code"] = (
            row["participant_code"]
        )

        sub["trial_name"] = (
            row["trial_name"]
        )

        sub["tracker"] = (
            row["tracker"]
        )

        dfs.append(
            sub
        )

    if not dfs:
        raise RuntimeError(
            "No balance_metrics.csv artifacts were found. "
            "Run the balance pipeline and rebuild validation.db first."
        )

    combined_df = pd.concat(
        dfs,
        ignore_index=True,
    )

    combined_df["condition"] = pd.Categorical(
        combined_df["condition"],
        categories=cfg.condition_order,
        ordered=True,
    )

    # If the CSV did not contain a human-readable label for some reason,
    # derive it from the known condition key.
    if "label" not in combined_df.columns:
        combined_df["label"] = (
            combined_df["condition"]
            .astype(str)
            .map(
                cfg.condition_display_names
            )
        )

    return combined_df


# =============================================================================
# ICC
# =============================================================================

def calculate_ICC(
    path_length_df: pd.DataFrame,
    trackers: list[str],
    reference: str,
) -> pd.DataFrame:

    path_length_df["target"] = (
        path_length_df[
            [
                "trial_name",
                "condition",
            ]
        ]
        .astype(str)
        .agg(
            "|".join,
            axis=1,
        )
    )

    icc_rows = []

    for tracker in trackers:

        sub_df = path_length_df[
            path_length_df["tracker"].isin(
                [
                    reference,
                    tracker,
                ]
            )
        ]

        icc_overall = pg.intraclass_corr(
            data=sub_df,
            targets="target",
            raters="tracker",
            ratings="path_length",
        )

        grouped = sub_df.groupby(
            "condition",
            observed=True,
        )

        for condition, group in grouped:

            icc = pg.intraclass_corr(
                data=group,
                targets="target",
                raters="tracker",
                ratings="path_length",
            )

            row = (
                icc
                .query(
                    "Type == 'ICC(A,1)'"
                )
                .iloc[0]
            )

            icc_rows.append(
                {
                    "tracker": tracker,
                    "condition": condition,
                    "ICC": row["ICC"],
                    "CI95%": row["CI95"],
                }
            )

        overall_row = {
            "tracker": tracker,
            "condition": "overall",
            "ICC": (
                icc_overall
                .query(
                    "Type == 'ICC(A,1)'"
                )
                .iloc[0]["ICC"]
            ),
            "CI95%": (
                icc_overall
                .query(
                    "Type == 'ICC(A,1)'"
                )
                .iloc[0]["CI95%"]
                if "CI95%" in icc_overall.columns
                else icc_overall
                .query(
                    "Type == 'ICC(A,1)'"
                )
                .iloc[0]["CI95"]
            ),
        }

        icc_rows.append(
            overall_row
        )

    return pd.DataFrame(
        icc_rows
    )


# =============================================================================
# Bland-Altman
# =============================================================================

def get_bland_altman_stats(
    differences: pd.Series,
) -> dict[str, float]:

    mean = np.mean(
        differences
    )

    std = np.std(
        differences,
        ddof=1,
    )

    loa_upper = (
        mean
        + 1.96 * std
    )

    loa_lower = (
        mean
        - 1.96 * std
    )

    return {
        "mean": mean,
        "std": std,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
    }


def calculate_bland_altman(
    path_length_df: pd.DataFrame,
    trackers: list[str],
    reference: str,
):

    path_length_wide = (
        path_length_df
        .pivot(
            index=[
                "condition",
                "participant_code",
                "trial_name",
            ],
            columns="tracker",
            values="path_length",
        )
        .reset_index()
    )

    rows = []
    overall_altmans = {}

    for tracker in trackers:

        overall_ba = path_length_wide[
            [
                "condition",
                "participant_code",
                "trial_name",
                tracker,
                reference,
            ]
        ].copy()

        overall_ba = overall_ba.dropna(
            subset=[
                tracker,
                reference,
            ]
        )

        overall_ba["ba_difference"] = (
            overall_ba[tracker]
            - overall_ba[reference]
        )

        overall_ba["ba_mean"] = (
            (
                overall_ba[reference]
                + overall_ba[tracker]
            )
            / 2
        )

        overall_ba_stats = (
            get_bland_altman_stats(
                overall_ba[
                    "ba_difference"
                ]
            )
        )

        overall_ba_stats[
            "condition"
        ] = "overall"

        overall_ba_stats[
            "tracker"
        ] = tracker

        rows.append(
            overall_ba_stats
        )

        for condition, group in (
            overall_ba.groupby(
                "condition",
                observed=True,
            )
        ):

            stats = (
                get_bland_altman_stats(
                    group[
                        "ba_difference"
                    ]
                )
            )

            stats[
                "condition"
            ] = condition

            stats[
                "tracker"
            ] = tracker

            rows.append(
                stats
            )

        overall_altmans[
            tracker
        ] = overall_ba

    ba_stats = pd.DataFrame(
        rows
    )

    return (
        overall_altmans,
        ba_stats,
    )


# =============================================================================
# Regression
# =============================================================================

def calculate_regression_equation(
    path_length_df: pd.DataFrame,
    trackers: list[str],
    reference: str,
) -> pd.DataFrame:

    regression_df = (
        path_length_df
        .pivot_table(
            index=[
                "condition",
                "participant_code",
                "trial_name",
                "target",
            ],
            columns="tracker",
            values="path_length",
        )
        .reset_index()
    )

    rows = []

    for tracker in trackers:

        sub_df = regression_df[
            [
                reference,
                tracker,
            ]
        ].dropna()

        x = (
            sub_df[
                reference
            ]
            .to_numpy()
        )

        y = (
            sub_df[
                tracker
            ]
            .to_numpy()
        )

        m, b = np.polyfit(
            x,
            y,
            1,
        )

        rows.append(
            {
                "tracker": tracker,
                "slope": m,
                "intercept": b,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Primary MediaPipe agreement figure
# =============================================================================

def plot_mediapipe_agreement_ba(
    path_length_df: pd.DataFrame,
    ba_plot_df: pd.DataFrame,
    ba_stats: pd.DataFrame,
    icc: float,
    slope: float,
    intercept: float,
    cfg: PlotConfig,
    color_by_condition: bool = True,
    height: int = 450,
    width: int = 1050,
) -> go.Figure:

    tracker = "mediapipe"
    reference = cfg.reference_system

    # -------------------------------------------------------------------------
    # Agreement data
    # -------------------------------------------------------------------------

    agree = (
        path_length_df[
            path_length_df[
                "tracker"
            ].isin(
                [
                    reference,
                    tracker,
                ]
            )
        ]
        .pivot_table(
            index=[
                "participant_code",
                "trial_name",
                "condition",
            ],
            columns="tracker",
            values="path_length",
            aggfunc="first",
        )
        .dropna(
            subset=[
                reference,
                tracker,
            ]
        )
        .reset_index()
    )

    all_vals = np.concatenate(
        [
            agree[
                reference
            ].to_numpy(),
            agree[
                tracker
            ].to_numpy(),
        ]
    )

    vmin = float(
        np.nanmin(
            all_vals
        )
    )

    vmax = float(
        np.nanmax(
            all_vals
        )
    )

    pad = (
        0.05
        * (vmax - vmin)
    )

    agree_range = [
        vmin - pad,
        vmax + pad,
    ]

    # -------------------------------------------------------------------------
    # Bland-Altman stats
    # -------------------------------------------------------------------------

    overall_row = (
        ba_stats.loc[
            (
                ba_stats[
                    "condition"
                ]
                == "overall"
            )
            & (
                ba_stats[
                    "tracker"
                ]
                == tracker
            )
        ]
        .iloc[0]
    )

    bias = float(
        overall_row[
            "mean"
        ]
    )

    loa_upper = float(
        overall_row[
            "loa_upper"
        ]
    )

    loa_lower = float(
        overall_row[
            "loa_lower"
        ]
    )

    x_min = float(
        ba_plot_df[
            "ba_mean"
        ].min()
    )

    x_max = float(
        ba_plot_df[
            "ba_mean"
        ].max()
    )

    x_pad = (
        0.08
        * (x_max - x_min)
    )

    x0 = x_min - x_pad
    x1 = x_max + x_pad

    m = max(
        abs(loa_upper),
        abs(loa_lower),
    )

    y_range = [
        -m * 1.15,
        m * 1.15,
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
    )

    # -------------------------------------------------------------------------
    # Left: agreement
    # -------------------------------------------------------------------------

    hover_conditions = (
        agree["condition"]
        .astype(str)
        .map(
            cfg.condition_display_names
        )
        .fillna(
            agree["condition"]
            .astype(str)
        )
    )

    customdata = np.column_stack(
        [
            agree[
                "participant_code"
            ].astype(str),
            agree[
                "trial_name"
            ].astype(str),
            hover_conditions,
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=agree[
                reference
            ],
            y=agree[
                tracker
            ],
            mode="markers",
            marker=dict(
                size=8,
                opacity=0.75,
            ),
            showlegend=False,
            hovertemplate=(
                "Participant: %{customdata[0]}<br>"
                "Trial: %{customdata[1]}<br>"
                "Condition: %{customdata[2]}<br>"
                "Reference: %{x:.3f}<br>"
                "MediaPipe: %{y:.3f}"
                "<extra></extra>"
            ),
            customdata=customdata,
        ),
        row=1,
        col=1,
    )

    # Identity line
    fig.add_trace(
        go.Scatter(
            x=agree_range,
            y=agree_range,
            mode="lines",
            line=dict(
                dash="dash",
                color="black",
            ),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Regression line
    x_line = np.array(
        agree_range
    )

    y_line = (
        slope * x_line
        + intercept
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
        col=1,
    )

    fig.update_xaxes(
        title_text=(
            f"<b>{cfg.display_name(reference)} "
            "path length (mm)</b>"
        ),
        range=agree_range,
        row=1,
        col=1,
        title_font=cfg.axis_title_font,
        tickfont=cfg.axis_tickfont,
    )

    fig.update_yaxes(
        title_text=(
            f"<b>{cfg.display_name(tracker)} "
            "path length (mm)</b>"
        ),
        range=agree_range,
        row=1,
        col=1,
        title_font=cfg.axis_title_font,
        tickfont=cfg.axis_tickfont,
    )

    # -------------------------------------------------------------------------
    # Right: Bland-Altman
    # -------------------------------------------------------------------------

    if color_by_condition:

        conds = list(
            cfg.condition_order
        )

        colors = sample_colorscale(
            "Viridis",
            [
                0.15
                + 0.6
                * (
                    i
                    / (
                        len(conds)
                        - 1
                    )
                )
                for i in range(
                    len(conds)
                )
            ],
        )

        color_map = dict(
            zip(
                conds,
                colors,
            )
        )

        for cond in conds:

            sub = ba_plot_df[
                ba_plot_df[
                    "condition"
                ]
                == cond
            ]

            if sub.empty:
                continue

            display_condition = (
                cfg.condition_display_name(
                    cond
                )
            )

            ba_customdata = np.column_stack(
                [
                    sub[
                        "participant_code"
                    ].astype(str),
                    np.repeat(
                        display_condition,
                        len(sub),
                    ),
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=sub[
                        "ba_mean"
                    ],
                    y=sub[
                        "ba_difference"
                    ],
                    mode="markers",
                    name=display_condition,
                    marker=dict(
                        size=9,
                        opacity=0.8,
                        color=color_map[
                            cond
                        ],
                    ),
                    hovertemplate=(
                        "Participant: %{customdata[0]}<br>"
                        "Condition: %{customdata[1]}<br>"
                        "Mean: %{x:.3f}<br>"
                        "Difference: %{y:.3f}"
                        "<extra></extra>"
                    ),
                    customdata=ba_customdata,
                ),
                row=1,
                col=2,
            )

    else:

        fig.add_trace(
            go.Scatter(
                x=ba_plot_df[
                    "ba_mean"
                ],
                y=ba_plot_df[
                    "ba_difference"
                ],
                mode="markers",
                marker=dict(
                    size=9,
                    opacity=0.8,
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Bias
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=bias,
        y1=bias,
        xref="x2",
        yref="y2",
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
    )

    # Limits of agreement
    for y in (
        loa_upper,
        loa_lower,
    ):

        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=y,
            y1=y,
            xref="x2",
            yref="y2",
            line=dict(
                color="black",
                width=1.5,
                dash="dot",
            ),
        )

    fig.update_xaxes(
        title_text=(
            "<b>Mean of systems (mm)</b>"
        ),
        row=1,
        col=2,
        title_font=cfg.axis_title_font,
        tickfont=cfg.axis_tickfont,
    )

    fig.update_yaxes(
        title_text=(
            "<b>Difference between systems (mm)</b>"
        ),
        range=y_range,
        row=1,
        col=2,
        title_font=cfg.axis_title_font,
        tickfont=cfg.axis_tickfont,
    )

    fig.update_layout(
        template="simple_white",
        height=height,
        width=width,
        margin=dict(
            t=90,
            b=70,
            l=80,
            r=40,
        ),
        legend_title_text=(
            "Condition"
            if color_by_condition
            else None
        ),
    )

    return fig


# =============================================================================
# Supplementary all-trackers agreement figure
# =============================================================================

def plot_all_trackers_agreement_ba(
    path_length_df: pd.DataFrame,
    overall_altmans: dict[str, pd.DataFrame],
    ba_stats: pd.DataFrame,
    icc_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    cfg: PlotConfig,
    trackers: list[str] | None = None,
    color_by_condition: bool = True,
    row_height: int = 400,
    width: int = 1100,
) -> go.Figure:

    if trackers is None:
        trackers = list(
            cfg.freemocap_trackers
        )

    reference = (
        cfg.reference_system
    )

    n_trackers = len(
        trackers
    )

    subplot_titles = []

    for tracker in trackers:

        subplot_titles.extend(
            [
                (
                    f"{cfg.display_name(tracker)} "
                    "– Agreement"
                ),
                (
                    f"{cfg.display_name(tracker)} "
                    "– Bland-Altman"
                ),
            ]
        )

    fig = make_subplots(
        rows=n_trackers,
        cols=2,
        horizontal_spacing=0.15,
        vertical_spacing=0.12,
        subplot_titles=subplot_titles,
    )

    legend_shown = False

    for i, tracker in enumerate(
        trackers
    ):

        row = i + 1

        # ---------------------------------------------------------------------
        # Tracker statistics
        # ---------------------------------------------------------------------

        tracker_icc = (
            icc_df.loc[
                (
                    icc_df[
                        "tracker"
                    ]
                    == tracker
                )
                & (
                    icc_df[
                        "condition"
                    ]
                    == "overall"
                ),
                "ICC",
            ]
            .item()
        )

        tracker_reg = (
            regression_df.loc[
                regression_df[
                    "tracker"
                ]
                == tracker
            ]
            .iloc[0]
        )

        slope = tracker_reg[
            "slope"
        ]

        intercept = tracker_reg[
            "intercept"
        ]

        ba_plot_df = (
            overall_altmans[
                tracker
            ]
        )

        tracker_ba_stats = (
            ba_stats.loc[
                (
                    ba_stats[
                        "tracker"
                    ]
                    == tracker
                )
                & (
                    ba_stats[
                        "condition"
                    ]
                    == "overall"
                )
            ]
            .iloc[0]
        )

        bias = float(
            tracker_ba_stats[
                "mean"
            ]
        )

        loa_upper = float(
            tracker_ba_stats[
                "loa_upper"
            ]
        )

        loa_lower = float(
            tracker_ba_stats[
                "loa_lower"
            ]
        )

        # ---------------------------------------------------------------------
        # Agreement data
        # ---------------------------------------------------------------------

        agree = (
            path_length_df[
                path_length_df[
                    "tracker"
                ].isin(
                    [
                        reference,
                        tracker,
                    ]
                )
            ]
            .pivot_table(
                index=[
                    "participant_code",
                    "trial_name",
                    "condition",
                ],
                columns="tracker",
                values="path_length",
                aggfunc="first",
            )
            .dropna(
                subset=[
                    reference,
                    tracker,
                ]
            )
            .reset_index()
        )

        all_vals = np.concatenate(
            [
                agree[
                    reference
                ].to_numpy(),
                agree[
                    tracker
                ].to_numpy(),
            ]
        )

        vmin = float(
            np.nanmin(
                all_vals
            )
        )

        vmax = float(
            np.nanmax(
                all_vals
            )
        )

        pad = (
            0.05
            * (vmax - vmin)
        )

        agree_range = [
            vmin - pad,
            vmax + pad,
        ]

        # ---------------------------------------------------------------------
        # Plotly axis indexing
        # ---------------------------------------------------------------------

        ax_idx = (
            2 * (row - 1)
            + 1
        )

        ba_ax_idx = (
            2 * (row - 1)
            + 2
        )

        x_ref_agree = (
            "x"
            if ax_idx == 1
            else f"x{ax_idx}"
        )

        y_ref_agree = (
            "y"
            if ax_idx == 1
            else f"y{ax_idx}"
        )

        x_ref_ba = (
            f"x{ba_ax_idx}"
        )

        y_ref_ba = (
            f"y{ba_ax_idx}"
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

        # ---------------------------------------------------------------------
        # Left: agreement
        # ---------------------------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=agree[
                    reference
                ],
                y=agree[
                    tracker
                ],
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.75,
                    color=(
                        cfg.tracker_colors[
                            tracker
                        ]
                    ),
                ),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Identity line
        fig.add_trace(
            go.Scatter(
                x=agree_range,
                y=agree_range,
                mode="lines",
                line=dict(
                    dash="dash",
                    color="black",
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

        # Regression
        x_line = np.array(
            agree_range
        )

        y_line = (
            slope * x_line
            + intercept
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
            col=1,
        )

        fig.add_annotation(
            x=0.95,
            y=0.05,
            xref=x_domain,
            yref=y_domain,
            text=(
                f"ICC(2,1) = "
                f"{tracker_icc:.3f}"
                "<br>"
                f"slope = {slope:.2f}"
            ),
            showarrow=False,
            font=dict(
                size=12
            ),
            xanchor="right",
            yanchor="bottom",
        )

        fig.update_xaxes(
            title_text=(
                f"<b>{cfg.display_name(reference)} "
                "path length (mm)</b>"
            ),
            range=agree_range,
            row=row,
            col=1,
            title_font=(
                cfg.axis_title_font
            ),
            tickfont=(
                cfg.axis_tickfont
            ),
        )

        fig.update_yaxes(
            title_text=(
                f"<b>{cfg.display_name(tracker)} "
                "path length (mm)</b>"
            ),
            range=agree_range,
            row=row,
            col=1,
            title_font=(
                cfg.axis_title_font
            ),
            tickfont=(
                cfg.axis_tickfont
            ),
        )

        # ---------------------------------------------------------------------
        # Right: Bland-Altman
        # ---------------------------------------------------------------------

        x_min = float(
            ba_plot_df[
                "ba_mean"
            ].min()
        )

        x_max = float(
            ba_plot_df[
                "ba_mean"
            ].max()
        )

        x_pad = (
            0.08
            * (x_max - x_min)
        )

        x0 = x_min - x_pad
        x1 = x_max + x_pad

        m_loa = max(
            abs(loa_upper),
            abs(loa_lower),
        )

        y_range = [
            -m_loa * 1.15,
            m_loa * 1.15,
        ]

        show_legend_this_row = (
            color_by_condition
            and not legend_shown
        )

        if color_by_condition:

            conds = list(
                cfg.condition_order
            )

            colors = sample_colorscale(
                "Viridis",
                [
                    0.15
                    + 0.6
                    * (
                        j
                        / (
                            len(conds)
                            - 1
                        )
                    )
                    for j in range(
                        len(conds)
                    )
                ],
            )

            color_map = dict(
                zip(
                    conds,
                    colors,
                )
            )

            for cond in conds:

                sub = ba_plot_df[
                    ba_plot_df[
                        "condition"
                    ]
                    == cond
                ]

                if sub.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=sub[
                            "ba_mean"
                        ],
                        y=sub[
                            "ba_difference"
                        ],
                        mode="markers",
                        name=(
                            cfg.condition_display_name(
                                cond
                            )
                        ),
                        legendgroup=cond,
                        marker=dict(
                            size=9,
                            opacity=0.8,
                            color=(
                                color_map[
                                    cond
                                ]
                            ),
                        ),
                        showlegend=(
                            show_legend_this_row
                        ),
                    ),
                    row=row,
                    col=2,
                )

            legend_shown = True

        else:

            fig.add_trace(
                go.Scatter(
                    x=ba_plot_df[
                        "ba_mean"
                    ],
                    y=ba_plot_df[
                        "ba_difference"
                    ],
                    mode="markers",
                    marker=dict(
                        size=8,
                        opacity=0.75,
                        color=(
                            cfg.tracker_colors[
                                tracker
                            ]
                        ),
                    ),
                    showlegend=False,
                ),
                row=row,
                col=2,
            )

        # Bias
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=bias,
            y1=bias,
            xref=x_ref_ba,
            yref=y_ref_ba,
            line=dict(
                color="black",
                width=2,
                dash="dash",
            ),
        )

        # Limits of agreement
        for y in (
            loa_upper,
            loa_lower,
        ):

            fig.add_shape(
                type="line",
                x0=x0,
                x1=x1,
                y0=y,
                y1=y,
                xref=x_ref_ba,
                yref=y_ref_ba,
                line=dict(
                    color="black",
                    width=1.5,
                    dash="dot",
                ),
            )

        fig.update_xaxes(
            title_text=(
                "<b>Mean of systems (mm)</b>"
            ),
            row=row,
            col=2,
            title_font=(
                cfg.axis_title_font
            ),
            tickfont=(
                cfg.axis_tickfont
            ),
        )

        fig.update_yaxes(
            title_text=(
                "<b>Difference between systems (mm)</b>"
            ),
            range=y_range,
            row=row,
            col=2,
            title_font=(
                cfg.axis_title_font
            ),
            tickfont=(
                cfg.axis_tickfont
            ),
        )

    fig.update_layout(
        template="simple_white",
        height=(
            row_height
            * n_trackers
        ),
        width=width,
        margin=dict(
            t=90,
            b=70,
            l=80,
            r=40,
        ),
        legend_title_text=(
            "Condition"
            if color_by_condition
            else None
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

def generate_agreement_table_typst(
    icc_df: pd.DataFrame,
    ba_stats: pd.DataFrame,
    regression_df: pd.DataFrame,
    cfg: PlotConfig,
    trackers: list[str] | None = None,
) -> str:

    if trackers is None:
        trackers = list(
            cfg.freemocap_trackers
        )

    lines = [
        "#let path-length-agreement = {",
        "  set text(size: 9pt)",
        "  table(",
        "    columns: (1.2fr, 1.8fr, 0.8fr, 1.2fr, 1fr),",
        "    align: (left, center, center, center, center),",
        "    stroke: none,",
        "    table.hline(stroke: 1pt),",
        "    table.header(",
        "      [*System*],",
        "      [*ICC(2,1) (95% CI)*],",
        "      [*Bias (mm)*],",
        "      [*LoA (mm)*],",
        "      [*Slope*],",
        "    ),",
        "    table.hline(stroke: 0.5pt),",
    ]

    for tracker in trackers:

        icc_row = (
            icc_df.loc[
                (
                    icc_df[
                        "tracker"
                    ]
                    == tracker
                )
                & (
                    icc_df[
                        "condition"
                    ]
                    == "overall"
                )
            ]
            .iloc[0]
        )

        ba_row = (
            ba_stats.loc[
                (
                    ba_stats[
                        "tracker"
                    ]
                    == tracker
                )
                & (
                    ba_stats[
                        "condition"
                    ]
                    == "overall"
                )
            ]
            .iloc[0]
        )

        reg_row = (
            regression_df.loc[
                regression_df[
                    "tracker"
                ]
                == tracker
            ]
            .iloc[0]
        )

        ci = (
            icc_row[
                "CI95%"
            ]
        )

        icc_str = (
            f'{icc_row["ICC"]:.3f} '
            f'({ci[0]:.3f}, {ci[1]:.3f})'
        )

        bias_str = (
            f'{ba_row["mean"]:.2f}'
        )

        loa_str = (
            f'({ba_row["loa_lower"]:.2f}, '
            f'{ba_row["loa_upper"]:.2f})'
        )

        slope_str = (
            f'{reg_row["slope"]:.2f}'
        )

        lines.extend(
            [
                (
                    f"    "
                    f"[{cfg.display_name(tracker)}],"
                ),
                f"    [{icc_str}],",
                f"    [{bias_str}],",
                f"    [{loa_str}],",
                f"    [{slope_str}],",
                "    table.hline(stroke: 0.5pt),",
            ]
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

    figures_path.mkdir(
        exist_ok=True,
        parents=True,
    )

    tables_path.mkdir(
        exist_ok=True,
        parents=True,
    )

    path_to_db = DATABASE_PATH

    # -------------------------------------------------------------------------
    # Load canonical balance_metrics.csv artifacts
    # -------------------------------------------------------------------------

    db_df = query_df(
        path_to_db,
        cfg.all_trackers,
    )

    if db_df.empty:
        raise RuntimeError(
            "No balance_metrics.csv artifacts were found in validation.db. "
            "Run the balance pipeline and rebuild the database first."
        )

    path_length_df = (
        parse_database_df(
            db_df,
            cfg,
        )
    )

    print(
        "\nLoaded balance path-length data:"
    )

    print(
        path_length_df.groupby(
            "tracker"
        ).size()
    )

    # -------------------------------------------------------------------------
    # Agreement statistics
    # -------------------------------------------------------------------------

    icc_all_trackers = (
        calculate_ICC(
            path_length_df,
            trackers=list(
                cfg.freemocap_trackers
            ),
            reference=(
                cfg.reference_system
            ),
        )
    )

    (
        ba_plot_df_overall,
        ba_stats_overall,
    ) = calculate_bland_altman(
        path_length_df,
        trackers=list(
            cfg.freemocap_trackers
        ),
        reference=(
            cfg.reference_system
        ),
    )

    regression_all_trackers = (
        calculate_regression_equation(
            path_length_df=(
                path_length_df
            ),
            trackers=list(
                cfg.freemocap_trackers
            ),
            reference=(
                cfg.reference_system
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Print statistics for parity checking
    # -------------------------------------------------------------------------

    print(
        "\nOverall ICC:"
    )

    print(
        icc_all_trackers[
            icc_all_trackers[
                "condition"
            ]
            == "overall"
        ][
            [
                "tracker",
                "ICC",
                "CI95%",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOverall Bland-Altman:"
    )

    print(
        ba_stats_overall[
            ba_stats_overall[
                "condition"
            ]
            == "overall"
        ][
            [
                "tracker",
                "mean",
                "loa_lower",
                "loa_upper",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRegression:"
    )

    print(
        regression_all_trackers[
            [
                "tracker",
                "slope",
                "intercept",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Primary MediaPipe figure
    # -------------------------------------------------------------------------

    mediapipe_icc = (
        icc_all_trackers
        .query(
            "tracker == 'mediapipe' "
            "and condition == 'overall'"
        )[
            "ICC"
        ]
        .item()
    )

    mediapipe_regression = (
        regression_all_trackers
        .query(
            "tracker == 'mediapipe'"
        )
        .iloc[0]
    )

    mediapipe_ba_stats = (
        ba_stats_overall[
            ba_stats_overall[
                "tracker"
            ]
            == "mediapipe"
        ]
    )

    fig_mp = (
        plot_mediapipe_agreement_ba(
            path_length_df=(
                path_length_df
            ),
            ba_plot_df=(
                ba_plot_df_overall[
                    "mediapipe"
                ]
            ),
            ba_stats=(
                mediapipe_ba_stats
            ),
            icc=(
                mediapipe_icc
            ),
            slope=(
                mediapipe_regression[
                    "slope"
                ]
            ),
            intercept=(
                mediapipe_regression[
                    "intercept"
                ]
            ),
            cfg=cfg,
        )
    )

    # -------------------------------------------------------------------------
    # Supplementary all-trackers figure
    # -------------------------------------------------------------------------

    fig_all = (
        plot_all_trackers_agreement_ba(
            path_length_df=(
                path_length_df
            ),
            overall_altmans=(
                ba_plot_df_overall
            ),
            ba_stats=(
                ba_stats_overall
            ),
            icc_df=(
                icc_all_trackers
            ),
            regression_df=(
                regression_all_trackers
            ),
            cfg=cfg,
        )
    )

    fig_mp.show()
    fig_all.show()

    # -------------------------------------------------------------------------
    # Typst table
    # -------------------------------------------------------------------------

    typst_table = (
        generate_agreement_table_typst(
            icc_df=(
                icc_all_trackers
            ),
            ba_stats=(
                ba_stats_overall
            ),
            regression_df=(
                regression_all_trackers
            ),
            cfg=cfg,
        )
    )

    table_output_path = (
        tables_path
        / "path_length_agreement_table.typ"
    )

    table_output_path.write_text(
        typst_table,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Figure export
    # -------------------------------------------------------------------------

    figure_output_path = (
        figures_path
        / "agreement_all_trackers.svg"
    )

    fig_all.write_image(
        figure_output_path,
        scale=3,
    )

    # Primary MediaPipe-only figure, if desired:
    #
    # fig_mp.write_image(
    #     figures_path
    #     / "com_path_length_agreement_ba.svg",
    #     scale=3,
    # )

    print(
        f"\nAgreement table written to: "
        f"{table_output_path}"
    )

    print(
        f"Agreement figure written to: "
        f"{figure_output_path}"
    )