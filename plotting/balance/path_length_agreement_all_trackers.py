import pandas as pd
import pingouin as pg
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

@dataclass
class PlotConfig:
    reference_system: str = "qualisys"
    freemocap_trackers: tuple[str, ...] = (
        "mediapipe",
        "vitpose",
        "rtmpose",
    )

    tracker_colors: dict = field(default_factory=lambda: {
        "mediapipe": "#0072B2",
        "rtmpose": "#D55E00",
        "vitpose": "#006D43",
        "qualisys": "black",
    })
    plot_height: int = 450
    plot_width: int = 1200

    subplot_title_font: dict = field(default_factory=lambda: dict(size=20))
    condition_order: tuple[str, ...] = (
        "Eyes Open/Solid Ground",
        "Eyes Closed/Solid Ground",
        "Eyes Open/Foam",
        "Eyes Closed/Foam",
    )

    axis_title_font: dict = field(default_factory=lambda: dict(family="Arial", size=20))
    axis_tickfont: dict = field(default_factory=lambda: dict(size=16))

    tracker_display_names: dict = field(default_factory=lambda: {
        "mediapipe": "MediaPipe",
        "rtmpose": "RTMPose",
        "vitpose": "ViTPose",
        "qualisys": "Reference",
    })

    def __post_init__(self):
        self.all_trackers = list(self.freemocap_trackers) + [self.reference_system]

    def display_name(self, tracker: str) -> str:
        return self.tracker_display_names.get(tracker, tracker)

def query_df(path_to_db: Path, trackers: list[str]):
    placeholders = ",".join(["?"] * len(trackers))

    query = f"""
        SELECT
            t.participant_code,
            t.trial_name,
            a.path,
            a.condition,
            a.tracker
        FROM artifacts a
        JOIN trials t ON a.trial_id = t.id
        WHERE
            t.trial_type = "balance"
            AND a.category = "com_analysis"
            AND a.tracker IN ({placeholders})
            AND a.file_exists = 1
            AND a.component_name LIKE '%path_length_com%'
        ORDER BY
            t.trial_name, a.path;
            """

    conn = sqlite3.connect(path_to_db)
    return pd.read_sql_query(query, conn, params=trackers)


def parse_database_df(df: pd.DataFrame, cfg: PlotConfig) -> pd.DataFrame:
    dfs = []
    for _, row in df.iterrows():
        sub = load_path_length_json(row["path"])
        sub["participant_code"] = row["participant_code"]
        sub["trial_name"] = row["trial_name"]
        sub["tracker"] = row["tracker"]
        dfs.append(sub)

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df["condition"] = pd.Categorical(
        combined_df["condition"], categories=cfg.condition_order, ordered=True
    )

    return combined_df


def load_path_length_json(json_path: str) -> pd.DataFrame:
    """
    Loads COM path length JSON artifact into a tidy dataframe with:
      condition, path_length, frame_interval (if present)
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {json_path}")

    raw = pd.read_json(p)

    col_map = {
        "Frame Intervals": "frame_interval",
        "Frame Interval": "frame_interval",
        "Path Lengths:": "path_length",
        "Path Lengths": "path_length",
        "Path Length": "path_length",
    }
    raw = raw.rename(columns=col_map)

    # If conditions are stored as the index, move them to a column.
    raw = raw.reset_index().rename(columns={"index": "condition"})

    # Keep only what we need
    keep = [c for c in ["condition", "path_length", "frame_interval"] if c in raw.columns]
    out = raw[keep].copy()

    # Ensure numeric
    out["path_length"] = pd.to_numeric(out["path_length"], errors="coerce")

    return out


def calculate_ICC(path_length_df: pd.DataFrame, 
                  trackers: list[str],
                  reference: str) -> pd.DataFrame:
    path_length_df["target"] = (
        path_length_df[["trial_name", "condition"]].astype(str).agg("|".join, axis=1)
    )

    icc_rows = []
    for tracker in trackers:
        sub_df = path_length_df[path_length_df['tracker'].isin([reference, tracker])]

        icc_overall = pg.intraclass_corr(
            data=sub_df,
            targets="target",
            raters="tracker",
            ratings="path_length",
        )

        icc_overall.set_index("Type")

        grouped = sub_df.groupby("condition")


        for condition, group in grouped:
            icc = pg.intraclass_corr(
                data=group,
                targets="target",
                raters="tracker",
                ratings="path_length",
            )
            row = icc.query("Type == 'ICC(A,1)'").iloc[0]
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
            "ICC": icc_overall.query("Type == 'ICC(A,1)'").iloc[0]["ICC"],
            "CI95%": icc_overall.query("Type == 'ICC(A,1)'").iloc[0]["CI95"],
        }

        icc_rows.append(overall_row)
    return pd.DataFrame(icc_rows)




def get_bland_altman_stats(differences: pd.Series) -> dict[str, float]:
    mean = np.mean(differences)
    std = np.std(differences, ddof=1)
    loa_upper = mean + 1.96 * std
    loa_lower = mean - 1.96 * std

    return {"mean": mean, "std": std, "loa_upper": loa_upper, "loa_lower": loa_lower}


def calculate_bland_altman(path_length_df: pd.DataFrame, 
                           trackers: list[str],
                           reference: str):
    path_length_wide = path_length_df.pivot(
        index=["condition", "participant_code", "trial_name"],
        columns="tracker",
        values="path_length",
    ).reset_index()

    rows = []
    overall_altmans = {}
    for tracker in trackers: 
        overall_ba = path_length_wide[["condition", "participant_code", "trial_name", tracker, reference]].copy()
        overall_ba["ba_difference"] = overall_ba[tracker] - overall_ba[reference]
        overall_ba["ba_mean"] = (overall_ba[reference] + overall_ba[tracker]) / 2

        overall_ba_stats = get_bland_altman_stats(overall_ba["ba_difference"])
        overall_ba_stats["condition"] = "overall"
        overall_ba_stats["tracker"] = tracker
        rows.append(overall_ba_stats)

        for condition, group in overall_ba.groupby("condition"):
            stats = get_bland_altman_stats(group["ba_difference"])
            stats["condition"] = condition
            stats["tracker"] = tracker
            rows.append(stats)

        overall_altmans[tracker] = overall_ba

    ba_stats = pd.DataFrame(rows)
    return overall_altmans, ba_stats

def calculate_regression_equation(path_length_df: pd.DataFrame, trackers: list[str], reference: str) -> dict:
    
    regression_df = path_length_df.pivot_table(
        index = ["condition", "participant_code", "trial_name", "target"],
        columns = "tracker",
        values = "path_length"
    ).reset_index()

    rows = []
    for tracker in trackers:
        sub_df = regression_df[[reference, tracker]]

        x = sub_df[reference].to_numpy()
        y = sub_df[tracker].to_numpy()

        m,b = np.polyfit(x,y,1)

        row = {
            "tracker": tracker,
            "slope": m,
            "intercept": b,
        }

        rows.append(row)
    return pd.DataFrame(rows)

def plot_mediapipe_agreement_ba(
    path_length_df: pd.DataFrame,
    ba_plot_df: pd.DataFrame,        # overall_altmans["mediapipe"]
    ba_stats: pd.DataFrame,          # filtered to tracker == "mediapipe"
    icc: float,
    slope: float,
    intercept: float,
    cfg: PlotConfig,
    color_by_condition: bool = True,
    height: int = 450,
    width: int = 1050,
) -> go.Figure:
    """Primary figure: single-row MediaPipe agreement + Bland-Altman."""
    tracker = "mediapipe"
    reference = cfg.reference_system

    # --- Agreement data ---
    agree = (
        path_length_df[path_length_df["tracker"].isin([reference, tracker])]
        .pivot_table(
            index=["participant_code", "trial_name", "condition"],
            columns="tracker",
            values="path_length",
            aggfunc="first",
        )
        .dropna(subset=[reference, tracker])
        .reset_index()
    )

    all_vals = np.concatenate([agree[reference].to_numpy(), agree[tracker].to_numpy()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = 0.05 * (vmax - vmin)
    agree_range = [vmin - pad, vmax + pad]

    # --- BA stats ---
    overall_row = ba_stats.loc[
        (ba_stats["condition"] == "overall") & (ba_stats["tracker"] == tracker)
    ].iloc[0]
    bias = float(overall_row["mean"])
    loa_upper = float(overall_row["loa_upper"])
    loa_lower = float(overall_row["loa_lower"])

    x_min, x_max = float(ba_plot_df["ba_mean"].min()), float(ba_plot_df["ba_mean"].max())
    x_pad = 0.08 * (x_max - x_min)
    x0, x1 = x_min - x_pad, x_max + x_pad

    m = max(abs(loa_upper), abs(loa_lower))
    y_range = [-m * 1.15, m * 1.15]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15)

    # --- Left: Agreement ---
    # x = reference, y = tracker (convention: gold standard on x)
    fig.add_trace(
        go.Scatter(
            x=agree[reference],
            y=agree[tracker],
            mode="markers",
            marker=dict(size=8, opacity=0.75),
            showlegend=False,
            hovertemplate=(
                "Participant: %{customdata[0]}<br>"
                "Trial: %{customdata[1]}<br>"
                "Condition: %{customdata[2]}<br>"
                f"Qualisys: %{{x:.3f}}<br>"
                f"MediaPipe: %{{y:.3f}}<extra></extra>"
            ),
            customdata=agree[["participant_code", "trial_name", "condition"]].astype(str).values,
        ),
        row=1, col=1,
    )

    # Identity line
    fig.add_trace(
        go.Scatter(
            x=agree_range, y=agree_range,
            mode="lines", line=dict(dash="dash", color="black"),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Regression line: y = slope * x + intercept
    x_line = np.array(agree_range)
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line, y=y_line,
            mode="lines", line=dict(color="red", width=1.5),
            showlegend=False, opacity=0.7,
        ),
        row=1, col=1,
    )

    # fig.add_annotation(
    #     x=0.95, y=0.05,
    #     xref="x domain", yref="y domain",
    #     text=f"ICC(2,1) = {icc:.3f}<br>slope = {slope:.2f}",
    #     showarrow=False, font=dict(size=12),
    #     xanchor="right", yanchor="bottom",
    # )

    fig.update_xaxes(
        title_text=f"<b>{cfg.display_name(reference)} path length (mm)</b>",
        range=agree_range, row=1, col=1,
        title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont,
    )
    fig.update_yaxes(
        title_text=f"<b>{cfg.display_name(tracker)} path length (mm)</b>",
        range=agree_range, row=1, col=1,
        title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont,
    )

    # --- Right: Bland-Altman ---
    if color_by_condition:
        conds = list(cfg.condition_order)
        colors = sample_colorscale(
            "Viridis", [0.15 + 0.6 * (i / (len(conds) - 1)) for i in range(len(conds))]
        )
        color_map = dict(zip(conds, colors))
        for cond, sub in ba_plot_df.groupby("condition"):
            fig.add_trace(
                go.Scatter(
                    x=sub["ba_mean"], y=sub["ba_difference"],
                    mode="markers", name=str(cond),
                    marker=dict(size=9, opacity=0.8, color=color_map.get(cond)),
                    hovertemplate=(
                        "Participant: %{customdata[0]}<br>"
                        "Condition: %{customdata[1]}<br>"
                        "Mean: %{x:.3f}<br>"
                        "Diff: %{y:.3f}<extra></extra>"
                    ),
                    customdata=sub[["participant_code", "condition"]].astype(str).values,
                ),
                row=1, col=2,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=ba_plot_df["ba_mean"], y=ba_plot_df["ba_difference"],
                mode="markers", marker=dict(size=9, opacity=0.8), showlegend=False,
            ),
            row=1, col=2,
        )

    # Bias + LoA lines
    fig.add_shape(type="line", x0=x0, x1=x1, y0=bias, y1=bias,
                  xref="x2", yref="y2", line=dict(color="black", width=2, dash="dash"))
    for y in (loa_upper, loa_lower):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                      xref="x2", yref="y2", line=dict(color="black", width=1.5, dash="dot"))

    fig.update_xaxes(title_text="<b>Mean of systems (mm)</b>", row=1, col=2,
                     title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont)
    fig.update_yaxes(title_text="<b>Difference between systems (mm)</b>",
                     range=y_range, row=1, col=2,
                     title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont)

    fig.update_layout(
        template="simple_white", height=height, width=width,
        margin=dict(t=90, b=70, l=80, r=40),
        legend_title_text="Condition" if color_by_condition else None,
    )

    return fig


def plot_all_trackers_agreement_ba(
    path_length_df: pd.DataFrame,
    overall_altmans: dict[str, pd.DataFrame],  # from calculate_bland_altman
    ba_stats: pd.DataFrame,
    icc_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    cfg: PlotConfig,
    trackers: list[str] = None,
    color_by_condition: bool = True,
    row_height: int = 400,
    width: int = 1100,
) -> go.Figure:
    """Supplementary figure: one row per tracker, agreement + BA columns."""
    if trackers is None:
        trackers = list(cfg.freemocap_trackers)

    reference = cfg.reference_system
    n_trackers = len(trackers)

    subplot_titles = []
    for t in trackers:
        subplot_titles.extend([
            f"{cfg.display_name(t)} – Agreement",
            f"{cfg.display_name(t)} – Bland-Altman",
        ])

    fig = make_subplots(
        rows=n_trackers, cols=2,
        horizontal_spacing=0.15, vertical_spacing=0.12,
        subplot_titles=subplot_titles,
    )

    legend_shown = False

    for i, tracker in enumerate(trackers):
        row = i + 1

        # Get this tracker's stats
        tracker_icc = icc_df.loc[
            (icc_df["tracker"] == tracker) & (icc_df["condition"] == "overall"), "ICC"
        ].item()
        tracker_reg = regression_df.loc[regression_df["tracker"] == tracker].iloc[0]
        slope = tracker_reg["slope"]
        intercept = tracker_reg["intercept"]
        ba_plot_df = overall_altmans[tracker]

        tracker_ba_stats = ba_stats.loc[
            (ba_stats["tracker"] == tracker) & (ba_stats["condition"] == "overall")
        ].iloc[0]
        bias = float(tracker_ba_stats["mean"])
        loa_upper = float(tracker_ba_stats["loa_upper"])
        loa_lower = float(tracker_ba_stats["loa_lower"])

        # --- Agreement data ---
        agree = (
            path_length_df[path_length_df["tracker"].isin([reference, tracker])]
            .pivot_table(
                index=["participant_code", "trial_name", "condition"],
                columns="tracker",
                values="path_length",
                aggfunc="first",
            )
            .dropna(subset=[reference, tracker])
            .reset_index()
        )

        all_vals = np.concatenate([agree[reference].to_numpy(), agree[tracker].to_numpy()])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        pad = 0.05 * (vmax - vmin)
        agree_range = [vmin - pad, vmax + pad]

        # --- Plotly axis indexing ---
        ax_idx = 2 * (row - 1) + 1  # agreement panel
        ba_ax_idx = 2 * (row - 1) + 2  # BA panel
        x_ref_agree = "x" if ax_idx == 1 else f"x{ax_idx}"
        y_ref_agree = "y" if ax_idx == 1 else f"y{ax_idx}"
        x_ref_ba = f"x{ba_ax_idx}"
        y_ref_ba = f"y{ba_ax_idx}"
        x_domain = "x domain" if ax_idx == 1 else f"x{ax_idx} domain"
        y_domain = "y domain" if ax_idx == 1 else f"y{ax_idx} domain"

        # --- Left: Agreement (reference on x, tracker on y) ---
        fig.add_trace(
            go.Scatter(
                x=agree[reference], y=agree[tracker],
                mode="markers", marker=dict(
                    size=8,
                    opacity=0.75,
                    color=cfg.tracker_colors[tracker],
                ),
                showlegend=False,
            ),
            row=row, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=agree_range, y=agree_range,
                mode="lines", line=dict(dash="dash", color="black"),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=1,
        )

        x_line = np.array(agree_range)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode="lines", line=dict(color="red", width=1.5),
                showlegend=False, opacity=0.7,
            ),
            row=row, col=1,
        )

        fig.add_annotation(
            x=0.95, y=0.05,
            xref=x_domain, yref=y_domain,
            text=f"ICC(2,1) = {tracker_icc:.3f}<br>slope = {slope:.2f}",
            showarrow=False, font=dict(size=12),
            xanchor="right", yanchor="bottom",
        )

        fig.update_xaxes(
            title_text=f"<b>{cfg.display_name(reference)} path length (mm)</b>",
            range=agree_range, row=row, col=1,
            title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont,
        )
        fig.update_yaxes(
            title_text=f"<b>{cfg.display_name(tracker)} path length (mm)</b>",
            range=agree_range, row=row, col=1,
            title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont,
        )

        # --- Right: Bland-Altman ---
        x_min, x_max = float(ba_plot_df["ba_mean"].min()), float(ba_plot_df["ba_mean"].max())
        x_pad = 0.08 * (x_max - x_min)
        x0, x1 = x_min - x_pad, x_max + x_pad
        m_loa = max(abs(loa_upper), abs(loa_lower))
        y_range = [-m_loa * 1.15, m_loa * 1.15]

        show_legend_this_row = color_by_condition and not legend_shown

        if color_by_condition:
            conds = list(cfg.condition_order)
            colors = sample_colorscale(
                "Viridis", [0.15 + 0.6 * (j / (len(conds) - 1)) for j in range(len(conds))]
            )
            color_map = dict(zip(conds, colors))
            for cond, sub in ba_plot_df.groupby("condition"):
                fig.add_trace(
                    go.Scatter(
                        x=sub["ba_mean"], y=sub["ba_difference"],
                        mode="markers", name=str(cond),
                        legendgroup=str(cond),
                        marker=dict(size=9, opacity=0.8, color=color_map.get(cond)),
                        showlegend=show_legend_this_row,
                    ),
                    row=row, col=2,
                )
            legend_shown = True
        else:
            fig.add_trace(
                go.Scatter(
                    x=ba_plot_df["ba_mean"], y=ba_plot_df["ba_difference"],
                    mode="markers", 
                    marker=dict(
                    size=8,
                    opacity=0.75,
                    color=cfg.tracker_colors[tracker],
                ),
                    showlegend=False,
                ),
                row=row, col=2,
            )

        fig.add_shape(type="line", x0=x0, x1=x1, y0=bias, y1=bias,
                      xref=x_ref_ba, yref=y_ref_ba,
                      line=dict(color="black", width=2, dash="dash"))
        for y in (loa_upper, loa_lower):
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                          xref=x_ref_ba, yref=y_ref_ba,
                          line=dict(color="black", width=1.5, dash="dot"))

        fig.update_xaxes(title_text="<b>Mean of systems (mm)</b>", row=row, col=2,
                         title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont)
        fig.update_yaxes(title_text="<b>Difference between systems (mm)</b>",
                         range=y_range, row=row, col=2,
                         title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont)

    fig.update_layout(
        template="simple_white",
        height=row_height * n_trackers,
        width=width,
        margin=dict(t=90, b=70, l=80, r=40),
        legend_title_text="Condition" if color_by_condition else None,
    )
    fig.update_annotations(font_size=cfg.subplot_title_font["size"])

    return fig

def generate_agreement_table_typst(
    icc_df: pd.DataFrame,
    ba_stats: pd.DataFrame,
    regression_df: pd.DataFrame,
    cfg: PlotConfig,
    trackers: list[str] | None = None,
) -> str:
    """Generate an importable Typst agreement table."""

    if trackers is None:
        trackers = list(cfg.freemocap_trackers)

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
        icc_row = icc_df.loc[
            (icc_df["tracker"] == tracker)
            & (icc_df["condition"] == "overall")
        ].iloc[0]

        ba_row = ba_stats.loc[
            (ba_stats["tracker"] == tracker)
            & (ba_stats["condition"] == "overall")
        ].iloc[0]

        reg_row = regression_df.loc[
            regression_df["tracker"] == tracker
        ].iloc[0]

        ci = icc_row["CI95%"]

        icc_str = (
            f'{icc_row["ICC"]:.3f} '
            f'({ci[0]:.3f}, {ci[1]:.3f})'
        )
        bias_str = f'{ba_row["mean"]:.2f}'
        loa_str = (
            f'({ba_row["loa_lower"]:.2f}, '
            f'{ba_row["loa_upper"]:.2f})'
        )
        slope_str = f'{reg_row["slope"]:.2f}'

        lines.extend(
            [
                f"    [{cfg.display_name(tracker)}],",
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

    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    cfg = PlotConfig()
    root_path = Path(
        r"D:\validation_public_release_v1"
    )
    root_path.mkdir(exist_ok=True, parents=True)

    path_to_db = Path(r"validation.db")

    db_df = query_df(path_to_db, cfg.all_trackers)

    path_length_df = parse_database_df(db_df, cfg)
    icc_df = calculate_ICC(path_length_df, 
                           trackers = ["mediapipe"], 
                           reference = cfg.reference_system)

    icc_all_trackers = calculate_ICC(
        path_length_df,
        trackers=cfg.freemocap_trackers,
        reference=cfg.reference_system
    )

    ba_plot_df, ba_stats = calculate_bland_altman(path_length_df,
                                                  trackers = ["mediapipe"],
                                                reference = cfg.reference_system
                                                )
    
    ba_plot_df_overall, ba_stats_overall = calculate_bland_altman(path_length_df,
                                                  trackers = cfg.freemocap_trackers,
                                                reference = cfg.reference_system
                                                )

    regression_mediapipe = calculate_regression_equation(path_length_df=path_length_df, 
                                  trackers = ["mediapipe"],
                                reference = cfg.reference_system)
    
    regression_all_trackers = calculate_regression_equation(
        path_length_df=path_length_df,
        trackers = cfg.freemocap_trackers,
        reference=cfg.reference_system
    )

    # Primary figure
    fig_mp = plot_mediapipe_agreement_ba(
        path_length_df=path_length_df,
        ba_plot_df=ba_plot_df_overall["mediapipe"],
        ba_stats=ba_stats,
        icc=icc_all_trackers.query("tracker == 'mediapipe' and condition == 'overall'")["ICC"].item(),
        slope=regression_all_trackers.query("tracker == 'mediapipe'")["slope"].item(),
        intercept=regression_all_trackers.query("tracker == 'mediapipe'")["intercept"].item(),
        cfg=cfg,
    )

    # Supplementary figure
    fig_all = plot_all_trackers_agreement_ba(
        path_length_df=path_length_df,
        overall_altmans=ba_plot_df_overall,
        ba_stats=ba_stats_overall,
        icc_df=icc_all_trackers,
        regression_df=regression_all_trackers,
        cfg=cfg,
)
    
    fig_mp.show()
    fig_all.show()
    typst_table = generate_agreement_table_typst(
    icc_df=icc_all_trackers,
    ba_stats=ba_stats_overall,
    regression_df=regression_all_trackers,
    cfg=cfg,
)
    (root_path / "tables" /"path_length_agreement_table.typ").write_text(typst_table)


    # fig_mp.write_image(root_path / "figures" / "com_path_length_agreement_ba.svg", scale=3)
    fig_all.write_image(root_path / "figures" /"agreement_all_trackers.svg", scale=3)

    f = 2