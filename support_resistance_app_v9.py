
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Support & Resistance", layout="wide")


@dataclass
class LevelCluster:
    kind: str
    level: float
    count: int
    anchor_idx: int
    last_idx: int
    score: float
    members: List[float]


@dataclass
class PatternSignal:
    idx: int
    name: str
    bias: str
    probability: float
    reason: str


GREEN = "#00C853"
RED = "#FF3B30"
BROWN = "#A66A2F"
MA_PRICE_COLOR = "#5EA3DA"
MA_VOL_COLOR = "#2C2C2C"
CARD_BG = "#07111F"
CARD_BORDER = "#17304D"
BADGE_BG = "rgba(11,163,74,0.18)"
BADGE_TEXT = "#7CFF9D"
BG_COLOR = "#F2F2F2"
PATTERN_COLOR = "#F4B400"
BLACK = "#0D0D0D"

TIMEFRAME_MAP = {
    "1 Menit": "1m",
    "5 Menit": "5m",
    "15 Menit": "15m",
    "1 Jam": "60m",
    "4 Jam": "4h",
    "1 Hari": "1d",
}

DURATION_MAP = {
    "1 Hari": "1d",
    "3 Hari": "3d",
    "1 Minggu": "7d",
    "1 Bulan": "1mo",
    "3 Bulan": "3mo",
    "6 Bulan": "6mo",
}

INTRADAY_LIMITS = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "60m": "730d",
    "4h": "730d",
}

PERIOD_RANK = {
    "1d": 1,
    "3d": 2,
    "7d": 3,
    "1mo": 4,
    "3mo": 5,
    "6mo": 6,
    "60d": 5,
    "730d": 7,
}


def normalize_symbol(raw: str) -> str:
    symbol = raw.strip().upper().replace(" ", "")
    if not symbol:
        return ""
    if symbol in {"IHSG", "JKSE", "^JKSE"}:
        return "^JKSE"
    if "." not in symbol:
        return f"{symbol}.JK"
    return symbol


def display_symbol(raw: str) -> str:
    symbol = raw.strip().upper().replace(" ", "")
    if symbol in {"IHSG", "JKSE", "^JKSE"}:
        return "IHSG"
    return symbol


def label_timeframe(interval: str) -> str:
    return {
        "1m": "1 Menit",
        "5m": "5 Menit",
        "15m": "15 Menit",
        "60m": "1 Jam",
        "4h": "4 Jam",
        "1d": "1 Hari",
    }.get(interval, interval)


def label_period(period: str) -> str:
    return {
        "1d": "1 Hari",
        "3d": "3 Hari",
        "7d": "1 Minggu",
        "1mo": "1 Bulan",
        "3mo": "3 Bulan",
        "6mo": "6 Bulan",
        "60d": "60 Hari",
        "730d": "730 Hari",
    }.get(period, period)


def coerce_period_for_interval(interval: str, period: str) -> Tuple[str, Optional[str]]:
    if interval not in INTRADAY_LIMITS:
        return period, None
    limit = INTRADAY_LIMITS[interval]
    if PERIOD_RANK.get(period, 0) <= PERIOD_RANK.get(limit, 99):
        return period, None
    note = (
        f"Durasi {label_period(period)} terlalu panjang untuk TF {label_timeframe(interval)}. "
        f"Dipakai maksimum {label_period(limit)}."
    )
    return limit, note


@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol: str, interval: str, period: str) -> Tuple[pd.DataFrame, Optional[str], str]:
    effective_period, note = coerce_period_for_interval(interval, period)
    download_interval = "60m" if interval == "4h" else interval

    df = yf.download(
        symbol,
        period=effective_period,
        interval=download_interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        group_by="column",
        threads=False,
    )

    if df is None or df.empty:
        try:
            df = yf.Ticker(symbol).history(
                period=effective_period,
                interval=download_interval,
                auto_adjust=False,
                back_adjust=False,
                actions=False,
                prepost=False,
            )
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(), note, effective_period

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df.index = pd.to_datetime(df.index)

    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("Asia/Jakarta")
        df.index = df.index.tz_localize(None)

    df = df[~df.index.duplicated(keep="last")].sort_index()

    if interval == "4h":
        df = (
            df.resample("4H")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )

    df["MA10"] = df["Close"].rolling(10).mean()
    df["VMA20"] = df["Volume"].rolling(20).mean()
    return df, note, effective_period


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    hi = df["High"].to_numpy()
    lo = df["Low"].to_numpy()
    for i in range(left, len(df) - right):
        hi_slice = hi[i - left : i + right + 1]
        lo_slice = lo[i - left : i + right + 1]
        if hi[i] == np.max(hi_slice) and np.sum(hi_slice == hi[i]) == 1:
            highs.append(i)
        if lo[i] == np.min(lo_slice) and np.sum(lo_slice == lo[i]) == 1:
            lows.append(i)
    return highs, lows


def cluster_levels(prices: List[float], indices: List[int], kind: str, tolerance_pct: float = 0.02) -> List[LevelCluster]:
    clusters: List[dict] = []
    for price, idx in zip(prices, indices):
        matched = False
        for c in clusters:
            tol = max(c["level"] * tolerance_pct, 2.0)
            if abs(price - c["level"]) <= tol:
                c["members"].append(price)
                c["indices"].append(idx)
                c["level"] = float(np.mean(c["members"]))
                matched = True
                break
        if not matched:
            clusters.append({"members": [price], "indices": [idx], "level": float(price)})

    output: List[LevelCluster] = []
    max_idx = max(indices) if indices else 1
    for c in clusters:
        count = len(c["members"])
        anchor_idx = min(c["indices"])
        last_idx = max(c["indices"])
        recency_bonus = last_idx / max(1, max_idx)
        spread_penalty = np.std(c["members"]) / max(1.0, c["level"])
        score = count * 2.0 + recency_bonus - spread_penalty * 10.0
        output.append(
            LevelCluster(
                kind=kind,
                level=float(np.mean(c["members"])),
                count=count,
                anchor_idx=anchor_idx,
                last_idx=last_idx,
                score=score,
                members=c["members"],
            )
        )
    return output


def choose_key_levels(df: pd.DataFrame, high_idx: List[int], low_idx: List[int]) -> Dict[str, Optional[LevelCluster]]:
    last_close = float(df["Close"].iloc[-1])

    supports = cluster_levels(df["Low"].iloc[low_idx].tolist(), low_idx, "support") if low_idx else []
    resistances = cluster_levels(df["High"].iloc[high_idx].tolist(), high_idx, "resistance") if high_idx else []

    support_candidates = [c for c in supports if c.level <= last_close]
    resistance_candidates = [c for c in resistances if c.level >= last_close]

    nearest_support = max(support_candidates, key=lambda x: x.level, default=None)
    nearest_resistance = min(resistance_candidates, key=lambda x: x.level, default=None)
    strong_support = max(support_candidates, key=lambda x: (x.score, x.count, x.level), default=None)
    strong_resistance = max(resistance_candidates, key=lambda x: (x.score, x.count, -x.level), default=None)

    def dedupe(primary: Optional[LevelCluster], pool: List[LevelCluster], side: str) -> Optional[LevelCluster]:
        if primary is None:
            return None
        others = [x for x in pool if abs(x.level - primary.level) > max(2.0, primary.level * 0.005)]
        if not others:
            return primary
        if side == "support":
            return max(others, key=lambda x: (x.score, x.level))
        return max(others, key=lambda x: (x.score, -x.level))

    if strong_support and nearest_support and abs(strong_support.level - nearest_support.level) <= max(2.0, nearest_support.level * 0.005):
        strong_support = dedupe(nearest_support, support_candidates, "support")
    if strong_resistance and nearest_resistance and abs(strong_resistance.level - nearest_resistance.level) <= max(2.0, nearest_resistance.level * 0.005):
        strong_resistance = dedupe(nearest_resistance, resistance_candidates, "resistance")

    return {
        "support_near": nearest_support,
        "support_strong": strong_support,
        "resistance_near": nearest_resistance,
        "resistance_strong": strong_resistance,
    }


def detect_trend(df: pd.DataFrame) -> str:
    high_idx, low_idx = find_pivots(df, left=2, right=2)
    last_two_highs = [float(df["High"].iloc[i]) for i in high_idx[-2:]] if len(high_idx) >= 2 else []
    last_two_lows = [float(df["Low"].iloc[i]) for i in low_idx[-2:]] if len(low_idx) >= 2 else []

    if len(last_two_highs) == 2 and len(last_two_lows) == 2:
        hh = last_two_highs[-1] > last_two_highs[-2]
        hl = last_two_lows[-1] > last_two_lows[-2]
        lh = last_two_highs[-1] < last_two_highs[-2]
        ll = last_two_lows[-1] < last_two_lows[-2]
        if hh and hl:
            return "Uptrend"
        if lh and ll:
            return "Downtrend"

    ma_now = float(df["MA10"].dropna().iloc[-1]) if not df["MA10"].dropna().empty else float(df["Close"].iloc[-1])
    ma_prev = float(df["MA10"].dropna().iloc[-4]) if len(df["MA10"].dropna()) >= 4 else ma_now
    close_now = float(df["Close"].iloc[-1])

    if close_now > ma_now and ma_now >= ma_prev:
        return "Uptrend"
    if close_now < ma_now and ma_now <= ma_prev:
        return "Downtrend"
    return "Sideways"


def _is_high_volume(df: pd.DataFrame, i: int) -> bool:
    vma = df["VMA20"].iloc[i]
    vol = df["Volume"].iloc[i]
    if pd.isna(vma) or vma <= 0:
        return False
    return vol >= vma * 1.1


def _near_level(price: float, level: Optional[LevelCluster]) -> bool:
    if level is None:
        return False
    tol = max(2.0, level.level * 0.015)
    return abs(price - level.level) <= tol


def detect_pattern_signals(df: pd.DataFrame, levels: Dict[str, Optional[LevelCluster]], trend: str) -> List[PatternSignal]:
    signals: List[PatternSignal] = []
    start_idx = max(1, len(df) - 40)

    for i in range(start_idx, len(df)):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        po = float(df["Open"].iloc[i - 1])
        pc = float(df["Close"].iloc[i - 1])

        body = abs(c - o)
        rng = max(h - l, 1e-9)
        upper = h - max(o, c)
        lower = min(o, c) - l
        body_pct = body / rng

        near_support = _near_level(l, levels.get("support_near")) or _near_level(l, levels.get("support_strong"))
        near_resistance = _near_level(h, levels.get("resistance_near")) or _near_level(h, levels.get("resistance_strong"))
        high_vol = _is_high_volume(df, i)

        def make_prob(base: float, bias: str) -> float:
            prob = base
            if bias == "naik" and near_support:
                prob += 0.07
            if bias == "turun" and near_resistance:
                prob += 0.07
            if high_vol:
                prob += 0.03
            if bias == "naik" and trend == "Downtrend":
                prob += 0.03
            if bias == "turun" and trend == "Uptrend":
                prob += 0.03
            return min(max(prob, 0.50), 0.78)

        if pc < po and c > o and o <= pc and c >= po and body > 0:
            signals.append(PatternSignal(i, "Bullish Engulfing", "naik", make_prob(0.57, "naik"), "reversal bullish"))
            continue
        if pc > po and c < o and o >= pc and c <= po and body > 0:
            signals.append(PatternSignal(i, "Bearish Engulfing", "turun", make_prob(0.57, "turun"), "reversal bearish"))
            continue
        if body_pct <= 0.40 and lower >= body * 2.0 and upper <= max(body, rng * 0.15):
            signals.append(PatternSignal(i, "Hammer", "naik", make_prob(0.54, "naik"), "rejection bawah"))
            continue
        if body_pct <= 0.40 and upper >= body * 2.0 and lower <= max(body, rng * 0.15):
            signals.append(PatternSignal(i, "Shooting Star", "turun", make_prob(0.54, "turun"), "rejection atas"))
            continue

    chosen: List[PatternSignal] = []
    for sig in sorted(signals, key=lambda s: (s.idx, s.probability), reverse=True):
        if all(abs(sig.idx - x.idx) > 2 for x in chosen):
            chosen.append(sig)
        if len(chosen) >= 4:
            break
    return sorted(chosen, key=lambda s: s.idx)


def price_step_formatter(x: float) -> str:
    if x >= 1000:
        return f"{int(round(x, -1)):,}".replace(",", ".")
    if x >= 100:
        return f"{x:.0f}"
    return f"{x:.2f}"


def merge_line_items(levels: Dict[str, Optional[LevelCluster]]) -> List[Tuple[LevelCluster, str]]:
    items: List[Tuple[LevelCluster, str]] = []
    s_near = levels.get("support_near")
    s_strong = levels.get("support_strong")
    r_near = levels.get("resistance_near")
    r_strong = levels.get("resistance_strong")

    if s_near and s_strong and abs(s_near.level - s_strong.level) <= max(2.0, s_near.level * 0.012):
        merged = LevelCluster(
            kind="support",
            level=float(np.mean([s_near.level, s_strong.level])),
            count=s_near.count + s_strong.count,
            anchor_idx=min(s_near.anchor_idx, s_strong.anchor_idx),
            last_idx=max(s_near.last_idx, s_strong.last_idx),
            score=s_near.score + s_strong.score,
            members=s_near.members + s_strong.members,
        )
        items.append((merged, "SKSD"))
    else:
        if s_strong is not None:
            items.append((s_strong, "SK"))
        if s_near is not None:
            items.append((s_near, "SD"))

    if r_near and r_strong and abs(r_near.level - r_strong.level) <= max(2.0, r_near.level * 0.012):
        merged = LevelCluster(
            kind="resistance",
            level=float(np.mean([r_near.level, r_strong.level])),
            count=r_near.count + r_strong.count,
            anchor_idx=min(r_near.anchor_idx, r_strong.anchor_idx),
            last_idx=max(r_near.last_idx, r_strong.last_idx),
            score=r_near.score + r_strong.score,
            members=r_near.members + r_strong.members,
        )
        items.append((merged, "RKRD"))
    else:
        if r_near is not None:
            items.append((r_near, "RD"))
        if r_strong is not None:
            items.append((r_strong, "RK"))

    items.sort(key=lambda x: x[0].level)
    return items


def distribute_label_positions(levels: List[float], y_min: float, y_max: float, min_gap: float) -> List[float]:
    if not levels:
        return []
    placed = list(levels)
    placed[0] = max(placed[0], y_min)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    if placed[-1] > y_max:
        shift = placed[-1] - y_max
        placed = [x - shift for x in placed]
    if placed[0] < y_min:
        shift = y_min - placed[0]
        placed = [x + shift for x in placed]
    return [min(max(x, y_min), y_max) for x in placed]


def choose_tick_positions(index: pd.DatetimeIndex, interval: str, period: str) -> Tuple[List[int], List[str]]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return [], []

    positions: List[int] = []
    labels: List[str] = []
    unique_days = pd.Series(index.normalize()).drop_duplicates().tolist()
    intraday_show_datetime = PERIOD_RANK.get(period, 99) <= PERIOD_RANK.get("1mo", 4) and interval != "1d"

    def add_nearest(ts: pd.Timestamp, label: str):
        diffs = np.abs((index - ts).total_seconds())
        pos = int(np.argmin(diffs))
        if pos not in positions:
            positions.append(pos)
            labels.append(label)

    if intraday_show_datetime:
        if period == "1d" or len(unique_days) == 1:
            preferred = ["09:00", "10:00", "11:00", "12:00", "13:30", "14:00", "15:00", "16:00"]
            base_day = unique_days[-1]
            for t in preferred:
                add_nearest(pd.Timestamp(f"{base_day.date()} {t}"), f"{base_day.strftime('%d %b %Y')}<br>{t}")
        elif len(unique_days) <= 7:
            preferred = ["09:00", "11:00", "13:30", "15:00"]
            for day in unique_days:
                for t in preferred:
                    add_nearest(pd.Timestamp(f"{day.date()} {t}"), f"{day.strftime('%d %b %Y')}<br>{t}")
        else:
            step = max(1, len(index) // 9)
            for i in range(0, len(index), step):
                positions.append(i)
                labels.append(index[i].strftime("%d %b %Y<br>%H:%M"))
    else:
        if interval == "1d":
            step = max(1, len(index) // 7)
            for i in range(0, len(index), step):
                positions.append(i)
                labels.append(index[i].strftime("%d %b %Y"))
        else:
            step = max(1, len(index) // 8)
            for i in range(0, len(index), step):
                positions.append(i)
                labels.append(index[i].strftime("%d %b %Y"))

    if len(index) - 1 not in positions:
        positions.append(len(index) - 1)
        labels.append(index[-1].strftime("%d %b %Y<br>%H:%M") if intraday_show_datetime else index[-1].strftime("%d %b %Y"))

    ordered = sorted(zip(positions, labels), key=lambda x: x[0])
    dedup_pos, dedup_lbl = [], []
    for pos, lbl in ordered:
        if not dedup_pos or pos != dedup_pos[-1]:
            dedup_pos.append(pos)
            dedup_lbl.append(lbl)
    return dedup_pos, dedup_lbl


def add_sr_lines(fig: go.Figure, df: pd.DataFrame, levels: Dict[str, Optional[LevelCluster]], y_min: float, y_max: float):
    line_items = merge_line_items(levels)
    raw_levels = [float(cluster.level) for cluster, _ in line_items]
    price_range = max(y_max - y_min, 1.0)
    placed_levels = distribute_label_positions(raw_levels, y_min + price_range * 0.06, y_max - price_range * 0.06, max(price_range * 0.055, 8.0))

    last_x = len(df) - 1
    for (cluster, tag), label_y in zip(line_items, placed_levels):
        start_x = max(0, cluster.anchor_idx)
        fig.add_shape(
            type="line",
            x0=start_x,
            x1=last_x,
            y0=cluster.level,
            y1=cluster.level,
            xref="x",
            yref="y",
            line=dict(color=BROWN, width=1.6),
        )
        fig.add_annotation(
            x=0.985,
            y=label_y,
            xref="paper",
            yref="y",
            xanchor="right",
            yanchor="middle",
            text=f"{tag}: {price_step_formatter(cluster.level)}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor=BROWN,
            borderwidth=1,
            font=dict(size=11, color=BROWN),
        )


def add_ma_and_close_labels(fig: go.Figure, df: pd.DataFrame, y_min: float, y_max: float):
    last_close = float(df["Close"].iloc[-1])
    fig.add_hline(y=last_close, line_color=BLACK, line_width=1.1, line_dash="dash", row=1, col=1)

    price_range = max(y_max - y_min, 1.0)
    min_gap = max(price_range * 0.04, 8.0)

    ma_series = df["MA10"].dropna()
    ma_value = float(ma_series.iloc[-1]) if not ma_series.empty else None

    close_label_y = last_close
    ma_label_y = ma_value
    if ma_value is not None and abs(ma_value - last_close) < min_gap:
        if ma_value >= last_close:
            ma_label_y = min(ma_value + min_gap * 0.55, y_max - min_gap * 0.2)
        else:
            ma_label_y = max(ma_value - min_gap * 0.55, y_min + min_gap * 0.2)

    fig.add_annotation(
        x=0.985,
        y=close_label_y,
        xref="paper",
        yref="y",
        text=f"Close {price_step_formatter(last_close)}",
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        font=dict(size=11, color=BLACK),
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor=BLACK,
        borderwidth=1,
    )

    if ma_value is not None:
        fig.add_annotation(
            x=0.985,
            y=ma_label_y,
            xref="paper",
            yref="y",
            text=f"MA10 {price_step_formatter(ma_value)}",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=11, color=MA_PRICE_COLOR),
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor=MA_PRICE_COLOR,
            borderwidth=1,
        )


def add_pattern_markers(fig: go.Figure, df: pd.DataFrame, signals: List[PatternSignal], y_pad: float):
    if not signals:
        return
    xs, ys, texts, colors = [], [], [], []
    for sig in signals:
        i = sig.idx
        high = float(df["High"].iloc[i])
        xs.append(i)
        ys.append(high + y_pad)
        texts.append(f"{sig.name}<br>{df.index[i].strftime('%d %b %Y %H:%M') if hasattr(df.index[i], 'strftime') else ''}<br>{sig.probability * 100:.0f}% {sig.bias}")
        colors.append(GREEN if sig.bias == "naik" else RED)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=["◎" for _ in xs],
            textposition="middle center",
            textfont=dict(size=18, color=PATTERN_COLOR),
            marker=dict(size=10, color=colors, line=dict(color=PATTERN_COLOR, width=1.3)),
            hovertext=texts,
            hoverinfo="text",
            name="Pattern",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def build_chart(df: pd.DataFrame, interval: str, period: str) -> Tuple[go.Figure, Dict[str, Optional[LevelCluster]], str, List[PatternSignal]]:
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    levels = choose_key_levels(df, high_idx, low_idx)
    trend = detect_trend(df)
    signals = detect_pattern_signals(df, levels, trend)

    x_vals = np.arange(len(df))
    custom_dt = [ts.strftime("%d %b %Y %H:%M") if interval != "1d" else ts.strftime("%d %b %Y") for ts in df.index]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )

    hover_template = (
        "<b>%{customdata}</b><br>"
        "Open: %{open}<br>High: %{high}<br>Low: %{low}<br>Close: %{close}<extra></extra>"
    )

    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            customdata=custom_dt,
            increasing_line_color=GREEN,
            increasing_fillcolor=GREEN,
            decreasing_line_color=RED,
            decreasing_fillcolor=RED,
            whiskerwidth=0.5,
            hovertemplate=hover_template,
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["MA10"],
            mode="lines",
            line=dict(color=MA_PRICE_COLOR, width=2),
            hovertemplate="<b>%{text}</b><br>MA10: %{y}<extra></extra>",
            text=custom_dt,
            name="MA10",
        ),
        row=1,
        col=1,
    )

    bar_colors = [GREEN if c >= o else RED for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=df["Volume"],
            marker_color=bar_colors,
            width=0.85,
            opacity=0.9,
            customdata=custom_dt,
            hovertemplate="<b>%{customdata}</b><br>Volume: %{y}<extra></extra>",
            name="Volume",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["VMA20"],
            mode="lines",
            line=dict(color=MA_VOL_COLOR, width=1.8),
            text=custom_dt,
            hovertemplate="<b>%{text}</b><br>VMA20: %{y}<extra></extra>",
            name="VMA20",
        ),
        row=2,
        col=1,
    )

    low_price = float(df["Low"].min())
    high_price = float(df["High"].max())
    price_range = max(high_price - low_price, max(float(df["Close"].iloc[-1]) * 0.08, 20.0))
    y_pad = price_range * 0.11
    y_min = low_price - y_pad
    y_max = high_price + y_pad

    add_sr_lines(fig, df, levels, y_min, y_max)
    add_ma_and_close_labels(fig, df, y_min, y_max)
    add_pattern_markers(fig, df, signals, price_range * 0.03)

    tickvals, ticktext = choose_tick_positions(df.index, interval, period)

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-28,
        showgrid=True,
        gridcolor="#D3D3D3",
        rangeslider_visible=False,
        showspikes=True,
        spikecolor="#555555",
        spikethickness=1,
        tickfont=dict(color="#111111", size=11),
        linecolor="#111111",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#D3D3D3",
        side="right",
        row=1,
        col=1,
        title_text="Price",
        tickfont=dict(color="#111111", size=11),
        title_font=dict(color="#111111", size=12),
        linecolor="#111111",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#D3D3D3",
        side="right",
        row=2,
        col=1,
        title_text="Volume",
        tickfont=dict(color="#111111", size=11),
        title_font=dict(color="#111111", size=12),
        linecolor="#111111",
    )
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)

    last_n_map = {
        "1m": 90,
        "5m": 100,
        "15m": 110,
        "60m": 100,
        "4h": 90,
        "1d": 100,
    }
    window = last_n_map.get(interval, 100)
    if len(df) > window:
        x_range = [len(df) - window, len(df) - 1 + 4]
    else:
        x_range = [-0.5, len(df) - 1 + 4]

    fig.update_xaxes(range=x_range, row=1, col=1)
    fig.update_xaxes(range=x_range, row=2, col=1)

    fig.update_layout(
        dragmode="pan",
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor="white",
        margin=dict(l=6, r=10, t=8, b=8),
        height=840,
        font=dict(color="#111111", size=12),
        bargap=0.04,
    )

    return fig, levels, trend, signals

def inject_css():
    st.markdown(
        f"""
        <style>
        .main-title {{
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }}
        .sr-card {{
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            background: {CARD_BG};
            padding: 14px 16px;
            min-height: 132px;
        }}
        .sr-touch-top {{
            color: #C9B089;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .sr-card-title {{
            color: white;
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .sr-card-value {{
            color: white;
            font-size: 32px;
            font-weight: 800;
            line-height: 1.0;
            margin-bottom: 14px;
        }}
        .sr-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: {BADGE_BG};
            color: {BADGE_TEXT};
            border: 1px solid rgba(11,163,74,0.34);
            font-size: 12px;
            font-weight: 700;
        }}
        .info-box {{
            border: 1px solid #D9D9D9;
            background: #FFFFFF;
            border-radius: 14px;
            padding: 16px 18px;
            margin-top: 8px;
            margin-bottom: 8px;
            text-align: center;
        }}
        .info-title {{
            font-size: 30px;
            font-weight: 900;
            color: #101010;
            line-height: 1.1;
            margin-bottom: 4px;
        }}
        .info-sub {{
            font-size: 18px;
            font-weight: 800;
            color: #101010;
            margin-bottom: 8px;
        }}
        .trend-pill {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 800;
            margin-right: 8px;
        }}
        .trend-up {{ background: rgba(11,163,74,0.12); color: #0A8B3E; }}
        .trend-down {{ background: rgba(220,38,38,0.10); color: #C62828; }}
        .trend-side {{ background: rgba(110,110,110,0.10); color: #555555; }}
        .signal-box {{
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            background: {CARD_BG};
            padding: 14px 16px;
            margin-bottom: 10px;
        }}
        .signal-title {{
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 8px;
            color: white;
        }}
        .signal-item {{
            font-size: 14px;
            margin-bottom: 6px;
            color: #D9E3F0;
        }}
        .stPlotlyChart, .stPlotlyChart > div {{
            width: 100% !important;
        }}
        @media (max-width: 768px) {{
            .main-title {{ font-size: 1.5rem; }}
            .sr-card {{ padding: 12px 12px; min-height: 112px; }}
            .sr-card-title {{ font-size: 14px; }}
            .sr-card-value {{ font-size: 26px; }}
            .info-box {{ padding: 12px 12px; }}
            .info-title {{ font-size: 20px; }}
            .info-sub {{ font-size: 14px; }}
            .signal-title {{ font-size: 15px; }}
            .signal-item {{ font-size: 13px; line-height: 1.45; }}
        }}
        .signal-up {{ color: #49F07C; font-weight: 800; }}
        .signal-down {{ color: #FF6868; font-weight: 800; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(col, title: str, cluster: Optional[LevelCluster]):
    touches = cluster.count if cluster is not None else 0
    value = f"{cluster.level:.2f}" if cluster is not None else "-"
    with col:
        st.markdown(
            f"""
            <div class="sr-card">
                <div class="sr-touch-top">touches {touches}</div>
                <div class="sr-card-title">{title}</div>
                <div class="sr-card-value">{value}</div>
                <div class="sr-badge">touches {touches}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header(display_code_value: str, data: pd.DataFrame, tf_label: str, duration_label: str, trend: str):
    last_close = float(data["Close"].iloc[-1])
    trend_class = "trend-up" if trend == "Uptrend" else "trend-down" if trend == "Downtrend" else "trend-side"
    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">SUPPORT &amp; RESISTANCE | {display_code_value}</div>
            <div class="info-sub">TF: {tf_label} | Durasi: {duration_label} | Last Close: {last_close:.2f}</div>
            <span class="trend-pill {trend_class}">Trend: {trend}</span>
            <span class="trend-pill trend-side">Time Zone: WIB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pattern_summary(signals: List[PatternSignal]):
    if not signals:
        st.markdown(
            """
            <div class="signal-box">
                <div class="signal-title">Sinyal candle penting</div>
                <div class="signal-item">Belum ada pola penting yang cukup jelas di area terbaru.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for sig in signals[::-1]:
        bias_class = "signal-up" if sig.bias == "naik" else "signal-down"
        rows.append(
            f'<div class="signal-item"><b>{sig.name}</b> • '
            f'<span class="{bias_class}">potensi {sig.bias} {sig.probability * 100:.0f}%</span> '
            f'• {sig.reason} <span style="color:#8FA5C2;">(heuristik)</span></div>'
        )

    st.markdown(
        f"""
        <div class="signal-box">
            <div class="signal-title">Sinyal candle penting</div>
            {''.join(rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css()

if "symbol_input" not in st.session_state:
    st.session_state.symbol_input = "IHSG"
if "tf_select" not in st.session_state:
    st.session_state.tf_select = "1 Hari"
if "durasi_select" not in st.session_state:
    st.session_state.durasi_select = "6 Bulan"

left_title, right_btn = st.columns([0.86, 0.14])
with left_title:
    st.markdown('<div class="main-title">Support & Resistance</div>', unsafe_allow_html=True)
with right_btn:
    if st.button("IHSG", use_container_width=True):
        st.session_state.symbol_input = "IHSG"
        st.rerun()

ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1, 1])
with ctrl1:
    st.text_input("Masukkan kode saham IDX", key="symbol_input", placeholder="Contoh: BBCA, TLKM, BMRI, DEWA")
with ctrl2:
    st.selectbox("TF", options=list(TIMEFRAME_MAP.keys()), key="tf_select", index=list(TIMEFRAME_MAP.keys()).index("1 Hari"))
with ctrl3:
    st.selectbox("Durasi", options=list(DURATION_MAP.keys()), key="durasi_select", index=list(DURATION_MAP.keys()).index("6 Bulan"))

raw_code = st.session_state.symbol_input.strip()
if raw_code:
    symbol = normalize_symbol(raw_code)
    shown_code = display_symbol(raw_code)
    interval = TIMEFRAME_MAP[st.session_state.tf_select]
    requested_period = DURATION_MAP[st.session_state.durasi_select]

    with st.spinner(f"Mengambil data {symbol}..."):
        data, note, effective_period = load_data(symbol, interval, requested_period)

    if note:
        st.warning(note)

    if data.empty:
        if symbol == "DEWA.JK":
            st.error("Yahoo Finance sedang tidak mengembalikan data untuk DEWA.JK. Coba saham lain dulu atau ulangi lagi nanti.")
        else:
            st.error(f"Data tidak tersedia untuk {symbol}.")
    elif len(data) < 20:
        st.error(f"Data terlalu sedikit untuk dianalisis. Rows: {len(data)}")
    else:
        fig, levels, trend, signals = build_chart(data, interval, effective_period)

        c1, c2, c3, c4 = st.columns(4)
        render_card(c1, "Support terdekat", levels.get("support_near"))
        render_card(c2, "Support kuat", levels.get("support_strong"))
        render_card(c3, "Resistance terdekat", levels.get("resistance_near"))
        render_card(c4, "Resistance kuat", levels.get("resistance_strong"))

        render_header(
            display_code_value=shown_code,
            data=data,
            tf_label=st.session_state.tf_select,
            duration_label=label_period(effective_period),
            trend=trend,
        )
        render_pattern_summary(signals)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
                "doubleClick": "reset+autosize",
                "modeBarButtonsToRemove": [
                    "select2d", "lasso2d", "drawline", "drawopenpath", "eraseshape",
                    "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"
                ],
            },
        )

        with st.expander("Keterangan"):
            st.write(
                "- Chart sekarang interaktif: bisa zoom, pan, autoscale, dan fullscreen dari toolbar kanan atas.\n"
                "- Semua jam intraday sudah dikonversi ke WIB.\n"
                "- MA10 ada di chart harga dan nilainya muncul di kanan.\n"
                "- Garis close terakhir hitam putus-putus dan ada label Close di kanan.\n"
                "- MA20 ada di panel volume.\n"
                "- Kalau support / resistance terlalu dekat, labelnya digabung jadi SKSD atau RKRD.\n"
                "- Probabilitas pola candle bersifat heuristik, bukan kepastian."
            )
else:
    st.info("Masukkan kode saham dulu untuk menampilkan chart.")
