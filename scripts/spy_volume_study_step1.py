"""SPY Volume Deep Study — Step 1: Characterize the intraday volume structure.

Questions we're answering:
  1. When does volume peak during the day? (by minute-of-day)
  2. How does buy/sell pressure distribute across the session?
  3. How often does SPY touch yesterday's levels (POC/VAH/VAL/H/L)?
  4. What happens to volume AT those levels vs away from them?
  5. What's the average forward move from each level at each time-of-day?

No trades, no signals. Pure characterization. 818 sessions of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.levels.volume_profile import compute_all_profiles
from mdq.config import RESULTS_DIR

START = "2023-01-03"
END = "2026-04-09"
OUT = RESULTS_DIR / "spy_volume_study"
OUT.mkdir(parents=True, exist_ok=True)


def _buy_pressure(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return 0.5
    return (c - l) / rng


def main() -> int:
    print("=" * 100)
    print("SPY Volume Deep Study — Step 1: Structure Characterization")
    print(f"Data: {START} -> {END}")
    print("=" * 100)

    bars = load_bars("SPY", START, END)
    rth = filter_rth(bars).reset_index(drop=True)
    profiles = compute_all_profiles(bars, bin_size=0.05)
    profiles["session_date"] = pd.to_datetime(profiles["session_date"]).dt.date

    print(f"RTH bars: {len(rth):,}")
    print(f"Sessions: {rth['session_date'].nunique()}")
    print(f"Profiles: {len(profiles)}")

    rth = rth.copy()
    rth["session_date"] = pd.to_datetime(rth["session_date"]).dt.date
    rth["minute"] = rth["ts_et"].dt.hour * 60 + rth["ts_et"].dt.minute
    rth["time_str"] = rth["ts_et"].dt.strftime("%H:%M")
    rth["bp"] = rth.apply(lambda r: _buy_pressure(r["o"], r["h"], r["l"], r["c"]), axis=1)
    rth["bar_delta"] = (2 * rth["bp"] - 1) * rth["v"]

    # ================================================================
    # Q1: Volume by minute of day
    # ================================================================
    print("\n" + "#" * 100)
    print("# Q1: AVERAGE VOLUME BY TIME OF DAY")
    print("#" * 100)

    vol_by_min = rth.groupby("minute").agg(
        mean_vol=("v", "mean"),
        median_vol=("v", "median"),
        mean_bp=("bp", "mean"),
        mean_delta=("bar_delta", "mean"),
        n=("v", "size"),
    ).reset_index()
    vol_by_min["time"] = vol_by_min["minute"].apply(
        lambda m: f"{m // 60:02d}:{m % 60:02d}"
    )

    # Show every 15 min
    checkpoints = [570, 585, 600, 615, 630, 645, 660, 675, 690, 720, 750, 780,
                   810, 840, 870, 900, 930, 955]  # 9:30 through 15:55
    print(f"\n  {'time':>6}  {'avg_vol':>12}  {'med_vol':>12}  {'buy_pressure':>13}  {'avg_delta':>12}")
    print("  " + "-" * 62)
    for m in checkpoints:
        row = vol_by_min[vol_by_min["minute"] == m]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {r['time']:>6}  {r['mean_vol']:>12,.0f}  {r['median_vol']:>12,.0f}  "
              f"{r['mean_bp']:>12.3f}  {r['mean_delta']:>12,.0f}")

    # Peak volume
    peak = vol_by_min.loc[vol_by_min["mean_vol"].idxmax()]
    trough = vol_by_min.loc[(vol_by_min["minute"] >= 630) & (vol_by_min["minute"] <= 840),
                            "mean_vol"].idxmin()
    trough_row = vol_by_min.iloc[trough] if trough is not None else None
    print(f"\n  Peak volume: {peak['time']} ({peak['mean_vol']:,.0f} avg)")
    if trough_row is not None:
        print(f"  Trough (midday): {trough_row['time']} ({trough_row['mean_vol']:,.0f} avg)")
        print(f"  Peak/trough ratio: {peak['mean_vol'] / trough_row['mean_vol']:.1f}x")

    # ================================================================
    # Q2: Buy pressure by time of day
    # ================================================================
    print("\n" + "#" * 100)
    print("# Q2: BUY PRESSURE BY TIME OF DAY (0.5 = neutral, >0.5 = buyers, <0.5 = sellers)")
    print("#" * 100)

    # First 30 min, midday, last 30 min
    windows = [
        ("open_15min", 570, 585),
        ("open_30min", 570, 600),
        ("open_60min", 570, 630),
        ("midday", 690, 810),
        ("close_60min", 900, 960),
        ("close_30min", 930, 960),
        ("close_15min", 945, 960),
    ]
    for label, start_m, end_m in windows:
        sub = rth[(rth["minute"] >= start_m) & (rth["minute"] < end_m)]
        if sub.empty:
            continue
        print(f"  {label:>15}: bp={sub['bp'].mean():.4f}  "
              f"delta_avg={sub['bar_delta'].mean():>+10,.0f}  "
              f"vol_avg={sub['v'].mean():>12,.0f}")

    # ================================================================
    # Q3: How often does SPY touch yesterday's levels?
    # ================================================================
    print("\n" + "#" * 100)
    print("# Q3: TOUCH FREQUENCY — how often does SPY reach yesterday's levels?")
    print("#" * 100)

    level_names = ["poc", "vah", "val", "high", "low", "close"]
    touch_stats: list[dict] = []

    for sd in sorted(rth["session_date"].unique()):
        prior = profiles[profiles["session_date"] < sd]
        if prior.empty:
            continue
        p = prior.iloc[-1]
        day_bars = rth[rth["session_date"] == sd]
        if day_bars.empty:
            continue

        day_high = day_bars["h"].max()
        day_low = day_bars["l"].min()

        for lname in level_names:
            lvl = float(p[lname])
            touched = (day_low <= lvl + 0.05) and (day_high >= lvl - 0.05)
            # First touch bar
            first_touch_idx = None
            if touched:
                for i, (_, bar) in enumerate(day_bars.iterrows()):
                    if bar["l"] <= lvl + 0.05 and bar["h"] >= lvl - 0.05:
                        first_touch_idx = i
                        break
            touch_stats.append({
                "session_date": sd,
                "level_name": lname,
                "level": lvl,
                "touched": touched,
                "first_touch_bar": first_touch_idx,
                "first_touch_minute": (
                    day_bars.iloc[first_touch_idx]["minute"]
                    if first_touch_idx is not None else None
                ),
            })

    ts_df = pd.DataFrame(touch_stats)
    print(f"\n  {'level':>8}  {'touch_rate':>11}  {'avg_first_touch':>16}  {'med_first_touch':>16}")
    print("  " + "-" * 58)
    for lname in level_names:
        sub = ts_df[ts_df["level_name"] == lname]
        touch_rate = sub["touched"].mean()
        touched_sub = sub[sub["touched"]]
        if not touched_sub.empty:
            avg_ft = touched_sub["first_touch_minute"].mean()
            med_ft = touched_sub["first_touch_minute"].median()
            avg_time = f"{int(avg_ft) // 60:02d}:{int(avg_ft) % 60:02d}"
            med_time = f"{int(med_ft) // 60:02d}:{int(med_ft) % 60:02d}"
        else:
            avg_time = "—"
            med_time = "—"
        print(f"  {lname:>8}  {touch_rate:>10.1%}  {avg_time:>16}  {med_time:>16}")

    # ================================================================
    # Q4: Volume AT levels vs away from levels
    # ================================================================
    print("\n" + "#" * 100)
    print("# Q4: VOLUME AT LEVEL TOUCHES vs AWAY FROM LEVELS")
    print("#" * 100)

    # For each bar, check if it's touching any prior-day level
    rth["touching_level"] = False
    rth["level_name_touched"] = None

    for sd in sorted(rth["session_date"].unique()):
        prior = profiles[profiles["session_date"] < sd]
        if prior.empty:
            continue
        p = prior.iloc[-1]
        mask = rth["session_date"] == sd
        for lname in level_names:
            lvl = float(p[lname])
            touch_mask = mask & (rth["l"] <= lvl + 0.10) & (rth["h"] >= lvl - 0.10)
            rth.loc[touch_mask, "touching_level"] = True
            rth.loc[touch_mask & rth["level_name_touched"].isna(), "level_name_touched"] = lname

    at_level = rth[rth["touching_level"]]
    away = rth[~rth["touching_level"]]
    print(f"\n  bars touching a level:     {len(at_level):>8,} ({len(at_level)/len(rth):.1%})")
    print(f"  bars NOT touching a level: {len(away):>8,} ({len(away)/len(rth):.1%})")
    print(f"\n  avg volume AT level:    {at_level['v'].mean():>12,.0f}")
    print(f"  avg volume AWAY:        {away['v'].mean():>12,.0f}")
    print(f"  ratio:                  {at_level['v'].mean() / away['v'].mean():.2f}x")
    print(f"\n  avg buy_pressure AT:    {at_level['bp'].mean():.4f}")
    print(f"  avg buy_pressure AWAY:  {away['bp'].mean():.4f}")

    # By level
    print(f"\n  {'level':>8}  {'n_bars':>8}  {'avg_vol':>12}  {'vs_away':>8}  {'avg_bp':>8}")
    print("  " + "-" * 50)
    for lname in level_names:
        sub = at_level[at_level["level_name_touched"] == lname]
        if sub.empty:
            continue
        ratio = sub["v"].mean() / away["v"].mean()
        print(f"  {lname:>8}  {len(sub):>8,}  {sub['v'].mean():>12,.0f}  {ratio:>7.2f}x  {sub['bp'].mean():>8.4f}")

    # ================================================================
    # Q5: Forward moves from level touches by time-of-day
    # ================================================================
    print("\n" + "#" * 100)
    print("# Q5: FORWARD 15-MIN MOVE FROM LEVEL TOUCHES — by time bucket")
    print("#" * 100)

    # For each first-touch, compute forward 15-bar MFE (in fade direction)
    results_q5: list[dict] = []
    for sd in sorted(rth["session_date"].unique()):
        prior = profiles[profiles["session_date"] < sd]
        if prior.empty:
            continue
        p = prior.iloc[-1]
        day_bars = rth[rth["session_date"] == sd].reset_index(drop=True)
        if len(day_bars) < 20:
            continue

        for lname in level_names:
            lvl = float(p[lname])
            # Find first touch
            for i, bar in day_bars.iterrows():
                if bar["l"] <= lvl + 0.05 and bar["h"] >= lvl - 0.05:
                    # Forward 15 bars
                    end = min(i + 16, len(day_bars))
                    if end <= i + 1:
                        break
                    fwd = day_bars.iloc[i + 1:end]
                    entry = bar["c"]
                    # Approach
                    if i > 0:
                        prev_c = day_bars.iloc[i - 1]["c"]
                    else:
                        prev_c = bar["o"]
                    from_below = prev_c < lvl

                    if from_below:
                        # Resistance — fade = short
                        mfe = (entry - fwd["l"].min()) / entry * 100
                        mae = (fwd["h"].max() - entry) / entry * 100
                        close_ret = (entry - fwd.iloc[-1]["c"]) / entry * 100
                    else:
                        # Support — fade = long
                        mfe = (fwd["h"].max() - entry) / entry * 100
                        mae = (entry - fwd["l"].min()) / entry * 100
                        close_ret = (fwd.iloc[-1]["c"] - entry) / entry * 100

                    minute = int(bar["minute"])
                    if minute < 600:
                        time_bucket = "open_30m"
                    elif minute < 630:
                        time_bucket = "30_60m"
                    elif minute < 720:
                        time_bucket = "60_150m"
                    elif minute < 870:
                        time_bucket = "midday"
                    else:
                        time_bucket = "close_90m"

                    results_q5.append({
                        "level_name": lname,
                        "time_bucket": time_bucket,
                        "minute": minute,
                        "from_below": from_below,
                        "mfe_pct": mfe,
                        "mae_pct": mae,
                        "close_ret_pct": close_ret,
                        "bar_vol": bar["v"],
                        "bar_bp": bar["bp"],
                    })
                    break  # first touch only

    q5 = pd.DataFrame(results_q5)
    if not q5.empty:
        print(f"\n  First-touch events: {len(q5)}")

        # By level × time bucket
        print(f"\n  {'level':>8}  {'time':>12}  {'n':>5}  {'mfe%':>7}  {'mae%':>7}  {'ret%':>7}  {'mfe/mae':>8}")
        print("  " + "-" * 65)

        for lname in level_names:
            for tb in ["open_30m", "30_60m", "60_150m", "midday", "close_90m"]:
                sub = q5[(q5["level_name"] == lname) & (q5["time_bucket"] == tb)]
                if len(sub) < 20:
                    continue
                ratio = sub["mfe_pct"].mean() / sub["mae_pct"].mean() if sub["mae_pct"].mean() > 0 else 0
                print(f"  {lname:>8}  {tb:>12}  {len(sub):>5}  "
                      f"{sub['mfe_pct'].mean():>+6.3f}  {sub['mae_pct'].mean():>+6.3f}  "
                      f"{sub['close_ret_pct'].mean():>+6.3f}  {ratio:>8.2f}")

        # Overall by level (all time buckets)
        print(f"\n  OVERALL BY LEVEL (all times):")
        print(f"  {'level':>8}  {'n':>5}  {'mfe%':>7}  {'mae%':>7}  {'ret%':>7}  {'mfe/mae':>8}  {'vol_spike':>10}")
        print("  " + "-" * 60)
        for lname in level_names:
            sub = q5[q5["level_name"] == lname]
            if sub.empty:
                continue
            ratio = sub["mfe_pct"].mean() / sub["mae_pct"].mean() if sub["mae_pct"].mean() > 0 else 0
            avg_vol = sub["bar_vol"].mean()
            print(f"  {lname:>8}  {len(sub):>5}  "
                  f"{sub['mfe_pct'].mean():>+6.3f}  {sub['mae_pct'].mean():>+6.3f}  "
                  f"{sub['close_ret_pct'].mean():>+6.3f}  {ratio:>8.2f}  {avg_vol:>10,.0f}")

        # Best bucket
        grouped = q5.groupby(["level_name", "time_bucket"]).agg(
            n=("mfe_pct", "size"),
            mfe=("mfe_pct", "mean"),
            mae=("mae_pct", "mean"),
            ret=("close_ret_pct", "mean"),
        ).reset_index()
        grouped = grouped[grouped["n"] >= 20]
        grouped["edge"] = grouped["mfe"] - grouped["mae"]
        grouped = grouped.sort_values("edge", ascending=False)

        print(f"\n  TOP 10 LEVEL × TIME COMBOS BY MFE-MAE EDGE:")
        print(f"  {'level':>8}  {'time':>12}  {'n':>5}  {'mfe%':>7}  {'mae%':>7}  {'edge':>7}")
        print("  " + "-" * 55)
        for _, r in grouped.head(10).iterrows():
            print(f"  {r['level_name']:>8}  {r['time_bucket']:>12}  {int(r['n']):>5}  "
                  f"{r['mfe']:>+6.3f}  {r['mae']:>+6.3f}  {r['edge']:>+6.3f}")

    q5.to_csv(OUT / "step1_forward_moves.csv", index=False)
    vol_by_min.to_csv(OUT / "step1_volume_by_minute.csv", index=False)
    ts_df.to_csv(OUT / "step1_touch_frequency.csv", index=False)

    print(f"\nStep 1 complete. Results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
