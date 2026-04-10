"""Experiment B: do whole-dollar level touches react differently from random bars?

Pipeline:
    1. Load bars for ticker and date range
    2. Filter to a time window (morning window or full RTH)
    3. Generate a level grid (e.g. SPY $0.50 step with $5/$1 tiers)
    4. Detect touches session-by-session
    5. Measure forward reactions at touches
    6. Compute baseline forward-move distribution on all bars in the same window
    7. Return both for comparison + a summary table

This is a pure signal study: no options pricing, no strategy rules, no
discretion. Answers the question "are level touches statistically different
from random bars?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.levels.psychological import LevelGrid, generate_grid
from mdq.levels.touches import TouchConfig, detect_touches
from mdq.stats.baseline import BaselineResult, baseline_summary, compute_baseline
from mdq.stats.reactions import ReactionConfig, measure_reactions


@dataclass
class ExperimentBResult:
    ticker: str
    grid_step: float
    time_window: tuple[str, str]
    touches: pd.DataFrame
    reactions: pd.DataFrame
    baseline: BaselineResult
    touch_summary: pd.DataFrame
    baseline_summary: pd.DataFrame


def run_experiment_b(
    ticker: str,
    start: str,
    end: str,
    grid: LevelGrid,
    time_window: tuple[str, str] = ("09:30", "11:15"),
    touch_cfg: TouchConfig = TouchConfig(),
    reaction_cfg: ReactionConfig = ReactionConfig(),
) -> ExperimentBResult:
    """Run Experiment B for one ticker / grid / time-window combination."""
    # 1. Load bars
    bars = load_bars(ticker, start, end)

    # 2. Filter to window
    bars_win = filter_window(bars, time_window[0], time_window[1])
    bars_win = bars_win.sort_values(["session_date", "t"]).reset_index(drop=True)

    # 3+4. Detect touches using the grid
    touches = detect_touches(bars_win, grid, touch_cfg)

    # 5. Measure reactions
    reactions = measure_reactions(bars_win, touches, reaction_cfg)

    # 6. Baseline on ALL bars in the window
    baseline = compute_baseline(bars_win, reaction_cfg)

    # 7. Summaries
    # Touch summary: overall + by tier + by approach
    def _touch_hit_rate(df: pd.DataFrame) -> pd.Series:
        out: dict = {"n": len(df)}
        if df.empty:
            return pd.Series(out)
        for h in reaction_cfg.horizons:
            mfe = df[f"mfe_{h}"].to_numpy()
            mae = df[f"mae_{h}"].to_numpy()
            # Window-max hit rate: did favorable excursion exceed target while
            # max adverse excursion stayed below stop within the same horizon?
            ok = (mfe >= reaction_cfg.target_move) & (mae < reaction_cfg.stop_move)
            out[f"hit_{h}_min"] = ok.mean()
            out[f"mfe_{h}_mean"] = np.nanmean(mfe)
            out[f"mae_{h}_mean"] = np.nanmean(mae)
        # First-passage outcome mix
        vc = df["fp_outcome"].value_counts(dropna=False)
        total = len(df)
        out["fp_hit"] = vc.get("target", 0) / total
        out["fp_stop"] = vc.get("stop", 0) / total
        out["fp_timeout"] = vc.get("timeout", 0) / total
        out["fp_expectancy_usd"] = (
            vc.get("target", 0) * reaction_cfg.target_move
            - vc.get("stop", 0) * reaction_cfg.stop_move
        ) / total
        return pd.Series(out)

    overall = _touch_hit_rate(reactions).to_frame().T
    overall.insert(0, "group", "ALL")

    grouped_tier = (
        reactions.groupby("tier", dropna=False)
        .apply(_touch_hit_rate, include_groups=False)
        .reset_index()
    )
    grouped_tier.insert(0, "group", grouped_tier["tier"].apply(lambda x: f"tier_{x}"))
    grouped_tier = grouped_tier.drop(columns=["tier"])

    grouped_approach = (
        reactions.groupby("approach", dropna=False)
        .apply(_touch_hit_rate, include_groups=False)
        .reset_index()
    )
    grouped_approach.insert(0, "group", grouped_approach["approach"])
    grouped_approach = grouped_approach.drop(columns=["approach"])

    grouped_first = (
        reactions[reactions["touch_num"] == 1]
        .groupby("tier", dropna=False)
        .apply(_touch_hit_rate, include_groups=False)
        .reset_index()
    )
    grouped_first.insert(
        0, "group", grouped_first["tier"].apply(lambda x: f"first_touch_tier_{x}")
    )
    grouped_first = grouped_first.drop(columns=["tier"])

    touch_summary = pd.concat(
        [overall, grouped_tier, grouped_approach, grouped_first], ignore_index=True
    )

    bl_summary = baseline_summary(
        baseline,
        target_move=reaction_cfg.target_move,
        stop_move=reaction_cfg.stop_move,
    )

    return ExperimentBResult(
        ticker=ticker,
        grid_step=grid.step,
        time_window=time_window,
        touches=touches,
        reactions=reactions,
        baseline=baseline,
        touch_summary=touch_summary,
        baseline_summary=bl_summary,
    )
