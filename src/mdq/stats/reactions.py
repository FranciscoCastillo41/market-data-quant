"""Reaction metrics around touches.

For a set of touch events and the underlying bar data, compute:
  - Max Favorable Excursion (MFE) in the fade direction over horizons
  - Max Adverse Excursion (MAE) in the break direction over horizons
  - Close at T+H offsets
  - First-passage outcome: did the fade target hit before the stop?

We compute everything on the bars **after** the touch bar (T+1 through T+H)
so the touch bar itself is treated as the signal, not part of the reaction.

Fast via numpy indexing on session-sorted arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReactionConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 15)
    """Forward bar horizons (in minutes) for MFE/MAE/close metrics."""

    target_move: float = 0.30
    """Favorable move in dollars for first-passage target."""

    stop_move: float = 0.25
    """Adverse move in dollars for first-passage stop."""

    first_passage_horizon: int = 15
    """Max bars to wait for the target/stop race."""


def measure_reactions(
    bars: pd.DataFrame,
    touches: pd.DataFrame,
    cfg: ReactionConfig = ReactionConfig(),
) -> pd.DataFrame:
    """Compute forward reaction metrics for each touch.

    Args:
        bars:    multi-session bars frame (same one used for touch detection)
        touches: output of detect_touches

    Returns a DataFrame with one row per touch, columns including:
        (all touch columns) +
        mfe_{h}, mae_{h}, close_{h}  for each h in horizons
        fp_outcome  in {'target','stop','timeout'}
        fp_bars     number of bars until outcome

    All MFE/MAE values are in dollars, oriented so positive = favorable.
    """
    if touches.empty:
        return touches.copy()

    # Build per-session bar arrays for fast indexing
    bars_sorted = bars.sort_values(["session_date", "t"]).reset_index(drop=True)
    # Map (session_date, t) -> bar index within session
    bars_sorted["bar_idx_global"] = bars_sorted.index

    # We'll need per-session highs/lows/closes
    session_groups = dict(list(bars_sorted.groupby("session_date", sort=True)))

    # Pre-extract numpy arrays per session
    session_arrays: dict = {}
    for sd, g in session_groups.items():
        session_arrays[sd] = {
            "h": g["h"].to_numpy(),
            "l": g["l"].to_numpy(),
            "c": g["c"].to_numpy(),
            "t": g["t"].to_numpy(),
        }

    horizons = cfg.horizons
    max_h = max(max(horizons), cfg.first_passage_horizon)

    # Output containers
    n = len(touches)
    mfe = {h: np.full(n, np.nan) for h in horizons}
    mae = {h: np.full(n, np.nan) for h in horizons}
    close_h = {h: np.full(n, np.nan) for h in horizons}
    fp_outcome = np.empty(n, dtype=object)
    fp_bars = np.full(n, -1, dtype=int)

    t_vals = touches["t"].to_numpy()
    sessions = touches["session_date"].to_numpy()
    approaches = touches["approach"].to_numpy()
    entries = touches["entry_close"].to_numpy()

    for i in range(n):
        sd = sessions[i]
        arr = session_arrays.get(sd)
        if arr is None:
            continue
        # Find the bar index within this session matching touches[i].t
        bar_ts = arr["t"]
        # np.searchsorted on sorted t values
        bi = int(np.searchsorted(bar_ts, t_vals[i]))
        if bi >= len(bar_ts) or bar_ts[bi] != t_vals[i]:
            continue  # shouldn't happen, but skip if it does

        entry = entries[i]
        approach = approaches[i]
        # fade direction: below touch → short (fade up into resistance)
        #                 above touch → long  (bounce off support)
        is_short = (approach == "from_below")
        is_long = (approach == "from_above")
        if not (is_short or is_long):
            continue  # 'at' case, skip

        start = bi + 1
        end_max = min(start + max_h, len(bar_ts))
        if start >= end_max:
            continue

        fwd_h = arr["h"][start:end_max]
        fwd_l = arr["l"][start:end_max]
        fwd_c = arr["c"][start:end_max]
        fwd_n = end_max - start

        for h in horizons:
            k = min(h, fwd_n)
            if k <= 0:
                continue
            if is_short:
                # favorable = price drops: MFE = entry - min(low)
                mfe[h][i] = entry - fwd_l[:k].min()
                mae[h][i] = fwd_h[:k].max() - entry
            else:
                mfe[h][i] = fwd_h[:k].max() - entry
                mae[h][i] = entry - fwd_l[:k].min()
            close_h[h][i] = fwd_c[k - 1] - entry if is_long else entry - fwd_c[k - 1]

        # First-passage race
        fp_n = min(cfg.first_passage_horizon, fwd_n)
        if fp_n > 0:
            if is_short:
                target_px = entry - cfg.target_move
                stop_px = entry + cfg.stop_move
                hit_target = fwd_l[:fp_n] <= target_px
                hit_stop = fwd_h[:fp_n] >= stop_px
            else:
                target_px = entry + cfg.target_move
                stop_px = entry - cfg.stop_move
                hit_target = fwd_h[:fp_n] >= target_px
                hit_stop = fwd_l[:fp_n] <= stop_px

            first_target = np.argmax(hit_target) if hit_target.any() else -1
            first_stop = np.argmax(hit_stop) if hit_stop.any() else -1

            if first_target == -1 and first_stop == -1:
                fp_outcome[i] = "timeout"
                fp_bars[i] = fp_n
            elif first_target == -1:
                fp_outcome[i] = "stop"
                fp_bars[i] = int(first_stop)
            elif first_stop == -1:
                fp_outcome[i] = "target"
                fp_bars[i] = int(first_target)
            else:
                # Both hit inside the horizon. If they hit on the same bar we
                # conservatively call it a stop (worst-case intra-bar ordering).
                if first_target < first_stop:
                    fp_outcome[i] = "target"
                    fp_bars[i] = int(first_target)
                else:
                    fp_outcome[i] = "stop"
                    fp_bars[i] = int(first_stop)

    out = touches.copy()
    for h in horizons:
        out[f"mfe_{h}"] = mfe[h]
        out[f"mae_{h}"] = mae[h]
        out[f"close_{h}"] = close_h[h]
    out["fp_outcome"] = fp_outcome
    out["fp_bars"] = fp_bars
    return out


def summarize_reactions(
    reactions: pd.DataFrame,
    cfg: ReactionConfig = ReactionConfig(),
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summary table of reaction outcomes, optionally grouped."""
    if reactions.empty:
        return pd.DataFrame()

    def _summary(df: pd.DataFrame) -> pd.Series:
        out: dict = {"n": len(df)}
        total = len(df)
        if total == 0:
            return pd.Series(out)
        # First-passage outcome mix
        vc = df["fp_outcome"].value_counts(dropna=False)
        out["hit_rate"] = vc.get("target", 0) / total
        out["stop_rate"] = vc.get("stop", 0) / total
        out["timeout_rate"] = vc.get("timeout", 0) / total
        # Expectancy in dollars at first-passage (target +$0.30 win, -$0.25 loss, else close after 15)
        # Simple proxy: use target/stop amounts for resolved trades, 0 for timeout
        win_pnl = vc.get("target", 0) * cfg.target_move
        loss_pnl = vc.get("stop", 0) * (-cfg.stop_move)
        out["expectancy_usd"] = (win_pnl + loss_pnl) / total
        # Forward move distributions
        for h in cfg.horizons:
            out[f"mfe_{h}_mean"] = df[f"mfe_{h}"].mean()
            out[f"mae_{h}_mean"] = df[f"mae_{h}"].mean()
            out[f"close_{h}_mean"] = df[f"close_{h}"].mean()
        return pd.Series(out)

    if group_cols:
        return (
            reactions.groupby(group_cols, dropna=False)
            .apply(_summary, include_groups=False)
            .reset_index()
        )
    return _summary(reactions).to_frame().T
