"""
    Top row    A: Agreement (Reference vs MediaPipe, identity + regression)
               B: Bland-Altman (bias + 95% LoA, colored by condition)
    Bottom row C: Visual perturbation        (sensitivity)
               D: Proprioceptive perturbation (sensitivity)
               E: Visual + proprioceptive     (sensitivity)
"""

import sqlite3
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pingouin as pg

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class PlotConfig:
    reference_system: str = "qualisys"
    freemocap_trackers: tuple[str, ...] = ("mediapipe", "vitpose", "rtmpose")
    primary_tracker: str = "mediapipe"

    condition_order: tuple[str, ...] = (
        "Eyes Open/Solid Ground",
        "Eyes Closed/Solid Ground",
        "Eyes Open/Foam",
        "Eyes Closed/Foam",
    )

    # Row group-titles (top row spans A-B, bottom row spans C-E)
    group_titles: tuple[str, str] = ("System Agreement", "Sensitivity to Perturbation")
    group_title_font: dict = field(default_factory=lambda: dict(family="Arial", size=22, color="#222"))

    # Manipulation panels (bottom row)
    manip_order_and_titles: dict = field(default_factory=lambda: {
        "eyes_on_solid": "Visual Perturbation",
        "foam_with_open": "Proprioceptive Perturbation",
        "hardest_vs_easiest": "Visual + Proprioceptive",
    })

    tracker_display_names: dict = field(default_factory=lambda: {
        "mediapipe": "MediaPipe",
        "rtmpose": "RTMPose",
        "vitpose": "ViTPose",
        "qualisys": "Reference",
    })

    primary_color: str = "#1f77b4"
    zero_reference_line_style: dict = field(
        default_factory=lambda: dict(color="darkgrey", width=1.5, dash="dot")
    )

    axis_title_font: dict = field(default_factory=lambda: dict(family="Arial", size=22))
    axis_tickfont: dict = field(default_factory=lambda: dict(size=18))
    subplot_title_font: dict = field(default_factory=lambda: dict(size=22))
    annotation_font: dict = field(default_factory=lambda: dict(size=18))

    def __post_init__(self):
        self.all_trackers = list(self.freemocap_trackers) + [self.reference_system]

    def display_name(self, tracker: str) -> str:
        return self.tracker_display_names.get(tracker, tracker)


# --------------------------------------------------------------------------- #
# Data loading (single DB read, shared by both halves)
# --------------------------------------------------------------------------- #
def query_df(path_to_db, trackers: list[str]) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(trackers))
    query = f"""
        SELECT t.participant_code, t.trial_name, a.path, a.condition, a.tracker
        FROM artifacts a
        JOIN trials t ON a.trial_id = t.id
        WHERE t.trial_type = "balance"
          AND a.category = "com_analysis"
          AND a.tracker IN ({placeholders})
          AND a.file_exists = 1
          AND a.component_name LIKE '%path_length_com%'
        ORDER BY t.trial_name, a.path;
    """
    conn = sqlite3.connect(path_to_db)
    out = pd.read_sql_query(query, conn, params=trackers)
    conn.close()
    return out


def load_path_length_json(json_path: str) -> pd.DataFrame:
    raw = pd.read_json(json_path)
    raw = raw.rename(columns={
        "Frame Intervals": "frame_interval",
        "Frame Interval": "frame_interval",
        "Path Lengths:": "path_length",
        "Path Lengths": "path_length",
        "Path Length": "path_length",
    })
    raw = raw.reset_index().rename(columns={"index": "condition"})
    keep = [c for c in ["condition", "path_length", "frame_interval"] if c in raw.columns]
    out = raw[keep].copy()
    out["path_length"] = pd.to_numeric(out["path_length"], errors="coerce")
    return out


def parse_database_df(db_df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for _, row in db_df.iterrows():
        sub = load_path_length_json(row["path"])
        sub["participant_code"] = row["participant_code"]
        sub["trial_name"] = row["trial_name"]
        sub["tracker"] = row["tracker"]
        dfs.append(sub)
    return pd.concat(dfs, ignore_index=True)


# --------------------------------------------------------------------------- #
# Agreement stats  (from path_length_agreement.py)
# --------------------------------------------------------------------------- #
def calculate_ICC(path_length_df: pd.DataFrame, trackers, reference) -> pd.DataFrame:
    # NOTE: this adds the 'target' column to path_length_df in place; the
    # agreement regression below depends on it, so call this first.
    path_length_df["target"] = (
        path_length_df[["trial_name", "condition"]].astype(str).agg("|".join, axis=1)
    )

    icc_rows = []
    for tracker in trackers:
        sub_df = path_length_df[path_length_df["tracker"].isin([reference, tracker])]
        icc_overall = pg.intraclass_corr(
            data=sub_df, targets="target", raters="tracker", ratings="path_length"
        )
        for condition, group in sub_df.groupby("condition"):
            icc = pg.intraclass_corr(
                data=group, targets="target", raters="tracker", ratings="path_length"
            )
            row = icc.query("Type == 'ICC(A,1)'").iloc[0]
            icc_rows.append({"tracker": tracker, "condition": condition,
                             "ICC": row["ICC"], "CI95%": row["CI95"]})
        icc_rows.append({
            "tracker": tracker, "condition": "overall",
            "ICC": icc_overall.query("Type == 'ICC(A,1)'").iloc[0]["ICC"],
            "CI95%": icc_overall.query("Type == 'ICC(A,1)'").iloc[0]["CI95"],
        })
    return pd.DataFrame(icc_rows)


def get_bland_altman_stats(differences: pd.Series) -> dict[str, float]:
    mean = np.mean(differences)
    std = np.std(differences, ddof=1)
    return {"mean": mean, "std": std,
            "loa_upper": mean + 1.96 * std, "loa_lower": mean - 1.96 * std}


def calculate_bland_altman(path_length_df: pd.DataFrame, trackers, reference):
    wide = path_length_df.pivot(
        index=["condition", "participant_code", "trial_name"],
        columns="tracker", values="path_length",
    ).reset_index()

    rows = []
    overall_altmans = {}
    for tracker in trackers:
        ba = wide[["condition", "participant_code", "trial_name", tracker, reference]].copy()
        ba["ba_difference"] = ba[tracker] - ba[reference]
        ba["ba_mean"] = (ba[reference] + ba[tracker]) / 2

        stats = get_bland_altman_stats(ba["ba_difference"])
        stats.update({"condition": "overall", "tracker": tracker})
        rows.append(stats)

        for condition, group in ba.groupby("condition"):
            s = get_bland_altman_stats(group["ba_difference"])
            s.update({"condition": condition, "tracker": tracker})
            rows.append(s)

        overall_altmans[tracker] = ba

    return overall_altmans, pd.DataFrame(rows)


def calculate_agreement_regression(path_length_df: pd.DataFrame, trackers, reference) -> pd.DataFrame:
    reg_df = path_length_df.pivot_table(
        index=["condition", "participant_code", "trial_name", "target"],
        columns="tracker", values="path_length",
    ).reset_index()

    rows = []
    for tracker in trackers:
        sub = reg_df[[reference, tracker]].dropna()
        m, b = np.polyfit(sub[reference].to_numpy(), sub[tracker].to_numpy(), 1)
        rows.append({"tracker": tracker, "slope": m, "intercept": b})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity stats  (from path_length_manipulation_effects.py)
# --------------------------------------------------------------------------- #
def manipulation_eyes_effect(df: pd.DataFrame, surface: str) -> pd.DataFrame:
    surface_name = {"solid": "Solid Ground", "foam": "Foam"}[surface]
    sdf = df.query("surface == @surface_name").copy()
    wide = sdf.pivot_table(
        index=["participant_code", "trial_name", "tracker"],
        columns="eyes", values="path_length", aggfunc="first",
    ).reset_index()
    wide["difference"] = wide["Closed"] - wide["Open"]
    wide["manipulation"] = f"eyes_on_{surface}"
    return wide[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]


def manipulation_foam_effect(df: pd.DataFrame, eyes: str) -> pd.DataFrame:
    eyes_name = {"open": "Open", "closed": "Closed"}[eyes]
    edf = df.query("eyes == @eyes_name").copy()
    wide = edf.pivot_table(
        index=["participant_code", "trial_name", "tracker"],
        columns="surface", values="path_length", aggfunc="first",
    ).reset_index()
    wide["difference"] = wide["Foam"] - wide["Solid Ground"]
    wide["manipulation"] = f"foam_with_{eyes}"
    return wide[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]


def manipulation_hardest_easiest(df: pd.DataFrame) -> pd.DataFrame:
    easy = df.query("eyes == 'Open' & surface == 'Solid Ground'").copy()
    hard = df.query("eyes == 'Closed' & surface == 'Foam'").copy()
    contrast = easy.merge(hard, on=["participant_code", "trial_name", "tracker"],
                          suffixes=("_easy", "_hard"))
    contrast["difference"] = contrast["path_length_hard"] - contrast["path_length_easy"]
    contrast["manipulation"] = "hardest_vs_easiest"
    return contrast[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]


def calculate_pearsons_r(manipulation_df, trackers, reference) -> dict[str, float]:
    r_df = manipulation_df.pivot(
        index=["participant_code", "trial_name", "manipulation"],
        columns="tracker", values="difference",
    ).reset_index()
    out = {}
    for tracker in trackers:
        for manip in r_df["manipulation"].dropna().unique():
            q = r_df.query("manipulation == @manip")
            out[f"{tracker}_{manip}"] = q[tracker].corr(q[reference])
    return out


def calculate_sensitivity_regression(manipulation_df, trackers, reference) -> dict[str, tuple]:
    r_df = manipulation_df.pivot(
        index=["participant_code", "trial_name", "manipulation"],
        columns="tracker", values="difference",
    ).reset_index()
    out = {}
    for tracker in trackers:
        for manip in r_df["manipulation"].dropna().unique():
            q = r_df.query("manipulation == @manip")
            m, b = np.polyfit(q[reference].to_numpy(), q[tracker].to_numpy(), 1)
            out[f"{tracker}_{manip}"] = (m, b)
    return out


def limits_zoom_positive(sub, cols, *, neg_buffer_frac=0.15, margin_frac=0.08, min_neg_buffer=0.02):
    vals = sub[cols].to_numpy().astype(float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1, 1)
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(vmax - vmin, 1e-9)
    neg_buffer = max(min_neg_buffer, neg_buffer_frac * max(abs(vmax), 1e-9))
    upper = max(vmax + margin_frac * span, 0.0)
    lower = min(min(vmin - margin_frac * span, -neg_buffer), 0.0)
    return (lower, upper)


# --------------------------------------------------------------------------- #
# Combined figure
# --------------------------------------------------------------------------- #
def plot_mediapipe_combined(
    path_length_df: pd.DataFrame,
    manipulation_df: pd.DataFrame,
    *,
    ba_plot_df: pd.DataFrame,      # overall_altmans[primary]
    ba_stats: pd.DataFrame,
    icc: float,
    agree_slope: float,
    agree_intercept: float,
    sens_regression: dict,
    sens_pearson: dict,
    cfg: PlotConfig,
    show_stats: bool = True,
    show_group_titles: bool = True,
    width: int = 1300,
    height: int = 1000,
) -> go.Figure:
    tracker = cfg.primary_tracker
    reference = cfg.reference_system

    # 6-col grid: top row two half-width panels, bottom row three third-width panels
    specs = [
        [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
        [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
    ]
    manip_order = list(cfg.manip_order_and_titles.keys())
    subplot_titles = [
        "<b>A.</b>  Agreement",
        "<b>B.</b>  Bland\u2013Altman",
        f"<b>C.</b>  {cfg.manip_order_and_titles[manip_order[0]]}",
        f"<b>D.</b>  {cfg.manip_order_and_titles[manip_order[1]]}",
        f"<b>E.</b>  {cfg.manip_order_and_titles[manip_order[2]]}",
    ]

    vspace = 0.22  # wider gap leaves room for the row-2 group title
    fig = make_subplots(
        rows=2, cols=6, specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.07, vertical_spacing=vspace,
    )

    # panel -> (row, col, x-axis name, y-axis name)  [axes numbered row-major over anchors]
    P = {
        "agreement": dict(row=1, col=1, xax="x",  yax="y"),
        "ba":        dict(row=1, col=4, xax="x2", yax="y2"),
        manip_order[0]: dict(row=2, col=1, xax="x3", yax="y3"),
        manip_order[1]: dict(row=2, col=3, xax="x4", yax="y4"),
        manip_order[2]: dict(row=2, col=5, xax="x5", yax="y5"),
    }

    # ------------------------------------------------------------------ #
    # A: Agreement
    # ------------------------------------------------------------------ #
    agree = (
        path_length_df[path_length_df["tracker"].isin([reference, tracker])]
        .pivot_table(index=["participant_code", "trial_name", "condition"],
                     columns="tracker", values="path_length", aggfunc="first")
        .dropna(subset=[reference, tracker])
        .reset_index()
    )
    all_vals = np.concatenate([agree[reference].to_numpy(), agree[tracker].to_numpy()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = 0.05 * (vmax - vmin)
    agree_range = [vmin - pad, vmax + pad]

    fig.add_trace(go.Scatter(
        x=agree[reference], y=agree[tracker], mode="markers",
        marker=dict(size=8, opacity=0.75, color=cfg.primary_color),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=agree_range, y=agree_range, mode="lines",
        line=dict(dash="dash", color="black"), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.array(agree_range), y=agree_slope * np.array(agree_range) + agree_intercept,
        mode="lines", line=dict(color="red", width=1.5), opacity=0.7, showlegend=False,
    ), row=1, col=1)

    if show_stats:
        fig.add_annotation(
            x=0.95, y=0.05, xref=f"{P['agreement']['xax']} domain",
            yref=f"{P['agreement']['yax']} domain",
            text=f"ICC(2,1) = {icc:.3f}<br>slope = {agree_slope:.2f}",
            showarrow=False, font=cfg.annotation_font, xanchor="right", yanchor="bottom",
        )

    fig.update_xaxes(title_text=f"<b>{cfg.display_name(reference)} Path Length (mm)</b>",
                     range=agree_range, scaleanchor="y", scaleratio=1,
                     title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont, row=1, col=1)
    fig.update_yaxes(title_text=f"<b>{cfg.display_name(tracker)} Path Length (mm)</b>",
                     range=agree_range,
                     title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont, row=1, col=1)

    # ------------------------------------------------------------------ #
    # B: Bland-Altman (colored by condition, legend shown once)
    # ------------------------------------------------------------------ #
    ov = ba_stats.loc[(ba_stats["condition"] == "overall") & (ba_stats["tracker"] == tracker)].iloc[0]
    bias, loa_upper, loa_lower = float(ov["mean"]), float(ov["loa_upper"]), float(ov["loa_lower"])

    x_min, x_max = float(ba_plot_df["ba_mean"].min()), float(ba_plot_df["ba_mean"].max())
    x_pad = 0.08 * (x_max - x_min)
    x0, x1 = x_min - x_pad, x_max + x_pad
    m_loa = max(abs(loa_upper), abs(loa_lower))
    y_range_ba = [-m_loa * 1.15, m_loa * 1.15]

    conds = list(cfg.condition_order)
    cond_colors = sample_colorscale(
        "Viridis", [0.15 + 0.6 * (i / (len(conds) - 1)) for i in range(len(conds))]
    )
    color_map = dict(zip(conds, cond_colors))
    for cond in conds:
        sub = ba_plot_df[ba_plot_df["condition"] == cond]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["ba_mean"], y=sub["ba_difference"], mode="markers",
            name=str(cond), legendgroup=str(cond),
            marker=dict(size=9, opacity=0.8, color=color_map.get(cond)),
            showlegend=True,
        ), row=1, col=4)

    fig.add_shape(type="line", x0=x0, x1=x1, y0=bias, y1=bias,
                  line=dict(color="black", width=2, dash="dash"), row=1, col=4)
    for y in (loa_upper, loa_lower):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                      line=dict(color="black", width=1.5, dash="dot"), row=1, col=4)

    fig.update_xaxes(title_text="<b>Mean of Systems (mm)</b>",
                     title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont, row=1, col=4)
    fig.update_yaxes(title_text="<b>Difference (mm)</b>", range=y_range_ba,
                     title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont, row=1, col=4)

    # ------------------------------------------------------------------ #
    # C/D/E: Sensitivity per manipulation
    # ------------------------------------------------------------------ #
    sens_wide = manipulation_df.pivot_table(
        index=["participant_code", "trial_name", "manipulation"],
        columns="tracker", values="difference", aggfunc="first",
    ).reset_index()

    for idx, manip in enumerate(manip_order):
        p = P[manip]
        sub = sens_wide.query("manipulation == @manip")
        lower, upper = limits_zoom_positive(sub, [reference, tracker])

        m, b = sens_regression[f"{tracker}_{manip}"]
        r2 = sens_pearson[f"{tracker}_{manip}"] ** 2

        # regression line
        fig.add_trace(go.Scatter(
            x=np.array([lower, upper]), y=m * np.array([lower, upper]) + b,
            mode="lines", line=dict(color="red", width=1.5), opacity=0.7, showlegend=False,
        ), row=p["row"], col=p["col"])
        # zero reference cross
        fig.add_shape(type="line", x0=lower, x1=upper, y0=0, y1=0,
                      line=cfg.zero_reference_line_style, row=p["row"], col=p["col"])
        fig.add_shape(type="line", x0=0, x1=0, y0=lower, y1=upper,
                      line=cfg.zero_reference_line_style, row=p["row"], col=p["col"])
        # identity
        fig.add_trace(go.Scatter(
            x=[lower, upper], y=[lower, upper], mode="lines",
            line=dict(color="black", dash="dash"), showlegend=False, hoverinfo="skip",
        ), row=p["row"], col=p["col"])
        # data
        fig.add_trace(go.Scatter(
            x=sub[reference], y=sub[tracker], mode="markers",
            marker=dict(size=9, opacity=0.7, color=cfg.primary_color, symbol="circle"),
            showlegend=False,
        ), row=p["row"], col=p["col"])

        if show_stats:
            fig.add_annotation(
                x=0.95, y=0.05, xref=f"{p['xax']} domain", yref=f"{p['yax']} domain",
                text=f"<i>r\u00b2</i> = {r2:.2f}<br>slope = {m:.2f}",
                showarrow=False, font=cfg.annotation_font, xanchor="right", yanchor="bottom",
            )

        fig.update_xaxes(
            title_text="<b>Reference \u0394Path Length (mm)</b>",
            scaleanchor=p["yax"].replace("x", "y"), scaleratio=1,
            title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont,
            row=p["row"], col=p["col"],
        )
        fig.update_yaxes(
            title_text=(f"<b>{cfg.display_name(tracker)} \u0394Path Length (mm)</b>"
                        if idx == 0 else ""),
            title_font=cfg.axis_title_font, tickfont=cfg.axis_tickfont,
            row=p["row"], col=p["col"],
        )

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    fig.update_layout(
        template="simple_white", width=width, height=height,
        font=dict(family="Arial", size=18),
        margin=dict(l=90, r=40, t=120, b=120),
        legend=dict(title_text="Condition", orientation="h",
                    yanchor="top", y=-0.10, xanchor="center", x=0.5),
    )
    fig.update_annotations(font_size=cfg.subplot_title_font["size"])
    # keep stat annotations at their own size (update_annotations hits all, so reset)
    for ann in fig.layout.annotations:
        if ann.text and ("ICC" in ann.text or "r\u00b2" in ann.text or "r²" in ann.text):
            ann.font.size = cfg.annotation_font["size"]

    # Row group-titles: top row sits in the top margin; bottom row sits in the
    # widened inter-row gap, just above the C/D/E subplot titles. Both rows span
    # the full width, so they center at x=0.5. Added last so the font isn't reset.
    if show_group_titles:
        row2_top = (1 - vspace) / 2
        for text, y in [(cfg.group_titles[0], 1.065),
                        (cfg.group_titles[1], row2_top + 0.065)]:
            fig.add_annotation(
                text=f"<b>{text}</b>", xref="paper", yref="paper",
                x=0.5, y=y, xanchor="center", yanchor="bottom",
                showarrow=False, font=cfg.group_title_font,
            )

    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = PlotConfig()
    path_to_db = Path("validation.db")
    root_path = Path(
        r"D:\validation_public_release_v1\figures"
    )
    root_path.mkdir(exist_ok=True, parents=True)

    # ---- single DB read ----
    db_df = query_df(path_to_db, cfg.all_trackers)
    path_length_df = parse_database_df(db_df)

    # ---- agreement half (calculate_ICC must precede regression: it adds 'target') ----
    icc_df = calculate_ICC(path_length_df, trackers=[cfg.primary_tracker],
                           reference=cfg.reference_system)
    overall_altmans, ba_stats = calculate_bland_altman(
        path_length_df, trackers=[cfg.primary_tracker], reference=cfg.reference_system)
    agree_reg = calculate_agreement_regression(
        path_length_df, trackers=[cfg.primary_tracker], reference=cfg.reference_system)

    icc_val = icc_df.query("tracker == @cfg.primary_tracker and condition == 'overall'")["ICC"].item()
    agree_slope = agree_reg.query("tracker == @cfg.primary_tracker")["slope"].item()
    agree_intercept = agree_reg.query("tracker == @cfg.primary_tracker")["intercept"].item()

    # ---- sensitivity half ----
    cats = path_length_df["condition"].str.extract(
        r"Eyes\s+(Open|Closed)\s*/\s*(Solid Ground|Foam)")
    balance_data = path_length_df.copy()
    balance_data["eyes"] = cats[0]
    balance_data["surface"] = cats[1]

    manipulation_df = pd.concat([
        manipulation_eyes_effect(balance_data, "solid"),
        manipulation_foam_effect(balance_data, eyes="open"),
        manipulation_hardest_easiest(balance_data),
    ], ignore_index=True)

    sens_pearson = calculate_pearsons_r(
        manipulation_df, trackers=[cfg.primary_tracker], reference=cfg.reference_system)
    sens_regression = calculate_sensitivity_regression(
        manipulation_df, trackers=[cfg.primary_tracker], reference=cfg.reference_system)

    # ---- combined figure ----
    fig = plot_mediapipe_combined(
        path_length_df=path_length_df,
        manipulation_df=manipulation_df,
        ba_plot_df=overall_altmans[cfg.primary_tracker],
        ba_stats=ba_stats,
        icc=icc_val,
        agree_slope=agree_slope,
        agree_intercept=agree_intercept,
        sens_regression=sens_regression,
        sens_pearson=sens_pearson,
        cfg=cfg,
    )
    fig.show()
    fig.write_image(root_path / "com_agreement_and_sensitivity.svg", scale=3)
