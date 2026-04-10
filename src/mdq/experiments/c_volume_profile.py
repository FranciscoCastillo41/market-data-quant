"""Experiment C: do prior-session volume-profile levels (POC/VAH/VAL) react?

Same experimental skeleton as Experiment B but using volume-profile levels
in place of arbitrary whole-dollar grids. Also tests confluence (VP level
within $0.50 of a whole dollar) which is the most promising hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.levels.touches import TouchConfig
from mdq.levels.volume_profile import (
    add_confluence_flag,
    compute_all_profiles,
    detect_vp_touches,
)
from mdq.stats.reactions import ReactionConfig, measure_reactions


@dataclass
class ExperimentCResult:
    ticker: str
    time_window: tuple[str, str]
    profiles: pd.DataFrame
    touches: pd.DataFrame
    reactions: pd.DataFrame


def run_experiment_c(
    ticker: str,
    start: str,
    end: str,
    time_window: tuple[str, str] = ("09:30", "11:15"),
    bin_size: float = 0.05,
    whole_dollar_step: float = 10.0,
    touch_cfg: TouchConfig = TouchConfig(),
    reaction_cfg: ReactionConfig = ReactionConfig(),
) -> ExperimentCResult:
    """Run Experiment C for one ticker.

    Returns touches + reactions with a `confluence` flag attached.
    """
    bars = load_bars(ticker, start, end)

    # Compute volume profiles from FULL RTH bars (not just the window)
    profiles = compute_all_profiles(bars, bin_size=bin_size)

    # Detect touches inside the trading window
    bars_win = filter_window(bars, time_window[0], time_window[1])
    bars_win = bars_win.sort_values(["session_date", "t"]).reset_index(drop=True)

    touches = detect_vp_touches(bars_win, profiles, touch_cfg)
    touches = add_confluence_flag(touches, whole_dollar_step=whole_dollar_step)

    reactions = measure_reactions(bars_win, touches, reaction_cfg)

    return ExperimentCResult(
        ticker=ticker,
        time_window=time_window,
        profiles=profiles,
        touches=touches,
        reactions=reactions,
    )
