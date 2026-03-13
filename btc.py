from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import html
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="BTC/IDR Analysis",
    layout="wide",
)

# =========================
# DATA CLASSES
# =========================
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


@dataclass
class FibLevel:
    ratio: float
    ratio_label: str
    price: float
    short_note: str
    description: str


@dataclass
class SignalCard:
    title: str
    value: str
    note: str


# =========================
# CONSTANTS
# =========================
SYMBOL = "BTC-IDR"

GREEN = "#00C853"
RED = "#FF3B30"
BROWN = "#A66A2F"
FIB_COLOR = "#1E88E5"
MA_COLORS = {
    10: "#5EA3DA",
    20: "#9C6ADE",
    50: "#FF9F1C",
    100: "#00B8D9",
    200: "#F15BB5",
}
MA_VOL_COLOR = "#3C3C3C"
CARD_BG = "#07111F"
CARD_BORDER = "#17304D"
BADGE_BG = "rgba(11,163,74,0.18)"
BADGE_TEXT = "#7CFF9D"
BG_COLOR = "#F2F2F2"
PATTERN_COLOR = "#F4B400"
BLACK = "#0D0D0D"
MACD_COLOR = "#1565C0"
SIGNAL_COLOR = "#F57C00"
RSI_COLOR = "#6A1B9A"
LIGHT_TEXT = "#DCE7F3"

MODE_OPTIONS = ["Support & Resistance", "Fibonacci", "MACD & RSI"]
MA_WINDOWS = [10, 20, 50, 100, 200]

TIMEFRAME_MAP = {
    "1 Jam": "60m",
    "2 Jam": "2h",
    "4 Jam": "4h",
    "1 Hari": "1d",
    "1 Minggu": "1wk",
}

DURATION_MAP = {
    "1 Bulan": "1mo",
    "3 Bulan": "3mo",
    "6 Bulan": "6mo",
    "1 Tahun": "1y",
    "2 Tahun": "2y",
}

INTRADAY_LIMITS = {
    "60m": "2y",
    "2h": "2y",
    "4h": "2y",
}

PERIOD_RANK = {
    "1mo": 1,
    "3mo": 2,
    "6mo": 3,
    "1y": 4,
    "2y": 5,
    "730d": 5,
}

FIB_LEVEL_SPECS = [
    {
        "ratio": 0.236,
        "ratio_label": "23,6% (0.236)",
        "short_note": "Koreksi ringan",
        "description": "Koreksi ringan, sering muncul saat tren utama masih kuat.",
    },
    {
        "ratio": 0.382,
        "ratio_label": "38,2% (0.382)",
        "short_note": "Koreksi umum",
        "description": "Zona koreksi umum dan area support/resistance awal.",
    },
    {
        "ratio": 0.500,
        "ratio_label": "50% (0.500)",
        "short_note": "Level psikologis",
        "description": "Bukan rasio Fibonacci asli, tapi sering jadi area pantulan.",
    },
    {
        "ratio": 0.618,
        "ratio_label": "61,8% (0.618)",
        "short_note": "Golden Ratio",
        "description": "Level Fibonacci paling penting yang sering dipantau trader.",
    },
    {
        "ratio": 0.786,
        "ratio_label": "78,6% (0.786)",
        "short_note": "Koreksi dalam",
        "description": "Koreksi dalam, sering jadi batas akhir sebelum reversal besar.",
    },
]


# =========================
# HELPERS
# =========================
def label_timeframe(interval: str) -> str:
    return {
        "60m": "1 Jam",
        "2h": "2 Jam",
        "4h": "4 Jam",
        "1d": "1 Hari",
        "1wk": "1 Minggu",
    }.get(interval, interval)


def label_period(period: str) -> str:
    return {
        "1mo": "1 Bulan",
        "3mo": "3 Bulan",
        "6mo": "6 Bulan",
        "1y": "1 Tahun",
        "2y": "2 Tahun",
        "730d": "2 Tahun",
    }.get(period, period)


def is_intraday_interval(interval: str) -> bool:
    return interval in {"60m", "2h", "4h"}


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


def value_formatter(x: float) -> str:
    if x >= 1_000_000_000:
        return f"{x:,.0f}"
    if x >= 1000:
        return f"{x:,.0f}"
    return f"{x:.2f}"


# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol: str, interval: str, period: str) -> Tuple[pd.DataFrame, Optional[str], str]:
    effective_period, note = coerce_period_for_interval(interval, period)
    download_interval = "60m" if interval in {"2h", "4h"} else interval

    df = yf.download(
        symbol,
        period=effective_period,
        interval=download_interval,
        auto_adjust=False,
        progress=False,
        prepost=True,
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
                prepost=True,
            )
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(), note, effective_period

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame(), note, effective_period

    df = df[required_cols].dropna().copy()
    df.index = pd.to_datetime(df.index)

    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("Asia/Jakarta")
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

    df = df[~df.index.duplicated(keep="last")].sort_index()

    if interval in {"2h", "4h"}:
        freq = "2H" if interval == "2h" else "4H"
        df = (
            df.resample(freq)
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna()
        )

    if df.empty:
        return pd.DataFrame(), note, effective_period

    for ma in MA_WINDOWS:
        df[f"MA{ma}"] = df["Close"].rolling(ma).mean()
    df["VMA20"] = df["Volume"].rolling(20).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    return df, note, effective_period


# =========================
# MARKET STRUCTURE
# =========================
def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    hi = df["High"].to_numpy()
    lo = df["Low"].to_numpy()

    if len(df) < left + right + 1:
        return highs, lows

    for i in range(left, len(df) - right):
        hi_slice = hi[i - left:i + right + 1]
        lo_slice = lo[i - left:i + right + 1]

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

    ma_series = df["MA20"].dropna() if "MA20" in df.columns else pd.Series(dtype=float)
    ma_now = float(ma_series.iloc[-1]) if not ma_series.empty else float(df["Close"].iloc[-1])
    ma_prev = float(ma_series.iloc[-4]) if len(ma_series) >= 4 else ma_now
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
    if len(df) < 2:
        return []

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


# =========================
# CHART HELPERS
# =========================
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
    intraday_show_datetime = PERIOD_RANK.get(period, 99) <= PERIOD_RANK.get("1mo", 1) and is_intraday_interval(interval)

    def add_nearest(ts: pd.Timestamp, label: str):
        diffs = np.abs((index - ts).total_seconds())
        pos = int(np.argmin(diffs))
        if pos not in positions:
            positions.append(pos)
            labels.append(label)

    if intraday_show_datetime:
        if len(unique_days) == 1:
            preferred = ["09:00", "11:00", "13:00", "15:00", "19:00", "23:00"]
            base_day = unique_days[-1]
            for t in preferred:
                add_nearest(pd.Timestamp(f"{base_day.date()} {t}"), f"{base_day.strftime('%d %b %Y')}<br>{t}")
        elif len(unique_days) <= 7:
            preferred = ["00:00", "08:00", "16:00"]
            for day in unique_days:
                for t in preferred:
                    add_nearest(pd.Timestamp(f"{day.date()} {t}"), f"{day.strftime('%d %b %Y')}<br>{t}")
        else:
            step = max(1, len(index) // 9)
            for i in range(0, len(index), step):
                positions.append(i)
                labels.append(index[i].strftime("%d %b %Y<br>%H:%M"))
    else:
        step = max(1, len(index) // 7)
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


def x_range_for_chart(df: pd.DataFrame, interval: str) -> List[float]:
    last_n_map = {
        "60m": 100,
        "2h": 110,
        "4h": 100,
        "1d": 100,
        "1wk": 80,
    }
    window = last_n_map.get(interval, 100)
    right_pad = 7.2

    if len(df) > window:
        return [len(df) - window, len(df) - 1 + right_pad]
    return [-0.5, len(df) - 1 + right_pad]


def base_price_volume_figure(df: pd.DataFrame, interval: str, period: str) -> Tuple[go.Figure, float, float]:
    x_vals = np.arange(len(df))
    custom_dt = [
        ts.strftime("%d %b %Y %H:%M") if is_intraday_interval(interval) else ts.strftime("%d %b %Y")
        for ts in df.index
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
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
            hovertemplate="<b>%{customdata}</b><br>Open: %{open}<br>High: %{high}<br>Low: %{low}<br>Close: %{close}<extra></extra>",
            name="Price",
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
    y_pad = price_range * 0.12
    y_min = low_price - y_pad
    y_max = high_price + y_pad

    tickvals, ticktext = choose_tick_positions(df.index, interval, period)
    xr = x_range_for_chart(df, interval)

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
        tickfont=dict(color="#111111", size=12),
        linecolor="#111111",
        zeroline=False,
    )
    fig.update_xaxes(range=xr, row=1, col=1)
    fig.update_xaxes(range=xr, row=2, col=1)

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#D3D3D3",
        side="right",
        row=1,
        col=1,
        title_text="Price",
        tickfont=dict(color="#111111", size=13),
        title_font=dict(color="#111111", size=13),
        linecolor="#111111",
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#D3D3D3",
        side="right",
        row=2,
        col=1,
        title_text="Volume",
        tickfont=dict(color="#111111", size=12),
        title_font=dict(color="#111111", size=12),
        linecolor="#111111",
        automargin=True,
    )
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)

    fig.update_layout(
        dragmode="pan",
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor="white",
        margin=dict(l=8, r=18, t=8, b=8),
        height=840,
        font=dict(color="#111111", size=13),
        bargap=0.04,
    )

    return fig, y_min, y_max


def add_price_annotations(fig: go.Figure, df: pd.DataFrame, items: List[dict], y_min: float, y_max: float):
    if not items:
        return

    price_range = max(y_max - y_min, 1.0)
    min_gap = max(price_range * 0.055, 12.0)
    grouped: Dict[float, List[dict]] = {}

    for item in items:
        x_val = float(item.get("x_override", len(df) - 1 + 3.2))
        grouped.setdefault(x_val, []).append(item)

    for x_val, group_items in grouped.items():
        sorted_items = sorted(group_items, key=lambda x: x["y"])
        values = [x["y"] for x in sorted_items]
        positions = distribute_label_positions(
            values,
            y_min + price_range * 0.07,
            y_max - price_range * 0.07,
            min_gap,
        )

        for item, label_y in zip(sorted_items, positions):
            fig.add_annotation(
                x=x_val,
                y=label_y,
                xref="x",
                yref="y",
                text=item["text"],
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=item.get("font_size", 14), color=item["font_color"]),
                bgcolor=item.get("bgcolor", "rgba(255,255,255,0.98)"),
                bordercolor=item["border_color"],
                borderwidth=1.4,
                borderpad=4,
                align="left",
            )


def add_sr_lines_and_labels(fig: go.Figure, df: pd.DataFrame, levels: Dict[str, Optional[LevelCluster]], y_min: float, y_max: float):
    line_items = merge_line_items(levels)
    labels: List[dict] = []
    last_x = len(df) - 1
    rail_left = len(df) - 1 + 1.95
    rail_right = len(df) - 1 + 4.35

    tag_x_map = {
        "SK": rail_left,
        "SD": rail_right,
        "SKSD": rail_right,
        "RD": rail_left,
        "RK": rail_right,
        "RKRD": rail_right,
    }

    for cluster, tag in line_items:
        start_x = max(0, cluster.anchor_idx)
        fig.add_shape(
            type="line",
            x0=start_x,
            x1=last_x,
            y0=cluster.level,
            y1=cluster.level,
            xref="x",
            yref="y",
            line=dict(color=BROWN, width=1.7),
        )
        labels.append(
            {
                "y": cluster.level,
                "text": f"{tag}: {value_formatter(cluster.level)}",
                "font_color": BROWN,
                "border_color": BROWN,
                "font_size": 14,
                "x_override": tag_x_map.get(tag, rail_right),
            }
        )

    add_price_annotations(fig, df, labels, y_min, y_max)


def add_ma_and_close_labels(fig: go.Figure, df: pd.DataFrame, selected_mas: List[int], y_min: float, y_max: float):
    labels: List[dict] = []
    last_close = float(df["Close"].iloc[-1])
    fig.add_hline(y=last_close, line_color=BLACK, line_width=1.1, line_dash="dash", row=1, col=1)

    ma_rail = len(df) - 1 + 3.05
    for ma in sorted(selected_mas):
        series = df[f"MA{ma}"].dropna()
        if series.empty:
            continue
        value = float(series.iloc[-1])
        color = MA_COLORS.get(ma, "#5EA3DA")
        labels.append(
            {
                "y": value,
                "text": f"MA{ma} {value_formatter(value)}",
                "font_color": color,
                "border_color": color,
                "font_size": 14,
                "x_override": ma_rail,
            }
        )

    add_price_annotations(fig, df, labels, y_min, y_max)


def add_pattern_markers(fig: go.Figure, df: pd.DataFrame, signals: List[PatternSignal], y_pad: float):
    if not signals:
        return

    xs, ys, texts, colors = [], [], [], []
    for sig in signals:
        i = sig.idx
        high = float(df["High"].iloc[i])
        xs.append(i)
        ys.append(high + y_pad)
        ts = df.index[i].strftime("%d %b %Y %H:%M") if hasattr(df.index[i], "strftime") else ""
        texts.append(f"{sig.name}<br>{ts}<br>{sig.probability * 100:.0f}% {sig.bias}")
        colors.append(GREEN if sig.bias == "naik" else RED)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=["◎" for _ in xs],
            textposition="middle center",
            textfont=dict(size=20, color=PATTERN_COLOR),
            marker=dict(size=11, color=colors, line=dict(color=PATTERN_COLOR, width=1.4)),
            hovertext=texts,
            hoverinfo="text",
            name="Pattern",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


# =========================
# FIBONACCI
# =========================
def calculate_fibonacci_context(df: pd.DataFrame) -> Dict[str, object]:
    low_idx = int(df["Low"].to_numpy().argmin())
    high_idx = int(df["High"].to_numpy().argmax())
    swing_low = float(df["Low"].iloc[low_idx])
    swing_high = float(df["High"].iloc[high_idx])
    price_range = max(swing_high - swing_low, 1e-9)

    direction = "up" if low_idx < high_idx else "down"
    direction_label = "Swing Low → High" if direction == "up" else "Swing High → Low"

    levels: List[FibLevel] = []
    for spec in FIB_LEVEL_SPECS:
        ratio = float(spec["ratio"])
        if direction == "up":
            price = swing_high - price_range * ratio
        else:
            price = swing_low + price_range * ratio

        levels.append(
            FibLevel(
                ratio=ratio,
                ratio_label=str(spec["ratio_label"]),
                price=float(price),
                short_note=str(spec["short_note"]),
                description=str(spec["description"]),
            )
        )

    return {
        "levels": levels,
        "direction": direction,
        "direction_label": direction_label,
        "swing_low": swing_low,
        "swing_high": swing_high,
        "low_idx": low_idx,
        "high_idx": high_idx,
    }


def add_fibonacci_lines(fig: go.Figure, df: pd.DataFrame, fib_context: Dict[str, object], y_min: float, y_max: float):
    levels: List[FibLevel] = sorted(fib_context["levels"], key=lambda x: x.price)
    labels: List[dict] = []

    start_x = int(min(fib_context["low_idx"], fib_context["high_idx"]))
    last_x = len(df) - 1
    fib_rail = len(df) - 1 + 3.35

    for level in levels:
        fig.add_shape(
            type="line",
            x0=start_x,
            x1=last_x,
            y0=level.price,
            y1=level.price,
            xref="x",
            yref="y",
            line=dict(color=FIB_COLOR, width=1.7),
        )
        labels.append(
            {
                "y": level.price,
                "text": f"{level.ratio_label}: {value_formatter(level.price)}",
                "font_color": FIB_COLOR,
                "border_color": FIB_COLOR,
                "font_size": 14,
                "x_override": fib_rail,
            }
        )

    last_close = float(df["Close"].iloc[-1])
    fig.add_hline(y=last_close, line_color=BLACK, line_width=1.1, line_dash="dash", row=1, col=1)
    add_price_annotations(fig, df, labels, y_min, y_max)


# =========================
# MACD / RSI
# =========================
def recent_cross(series_a: pd.Series, series_b: pd.Series, lookback: int = 5) -> Optional[Tuple[str, int]]:
    if len(series_a) < 2 or len(series_b) < 2:
        return None

    start = max(1, len(series_a) - lookback)
    for i in range(start, len(series_a)):
        prev_a = float(series_a.iloc[i - 1])
        prev_b = float(series_b.iloc[i - 1])
        curr_a = float(series_a.iloc[i])
        curr_b = float(series_b.iloc[i])

        if prev_a <= prev_b and curr_a > curr_b:
            return "bullish", len(series_a) - 1 - i
        if prev_a >= prev_b and curr_a < curr_b:
            return "bearish", len(series_a) - 1 - i

    return None


def extract_pivot_pairs(series: pd.Series, pivot_indices: List[int], max_points: int = 4) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for idx in pivot_indices[-max_points:]:
        if 0 <= idx < len(series):
            points.append((idx, float(series.iloc[idx])))
    return points


def detect_rsi_divergences(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    low_pairs = extract_pivot_pairs(df["Close"], low_idx, max_points=5)
    high_pairs = extract_pivot_pairs(df["Close"], high_idx, max_points=5)
    rsi = df["RSI"]

    bullish = None
    bearish = None

    if len(low_pairs) >= 2:
        (i1, p1), (i2, p2) = low_pairs[-2], low_pairs[-1]
        r1 = float(rsi.iloc[i1])
        r2 = float(rsi.iloc[i2])
        if p2 < p1 and r2 > r1 and max(i2 - i1, 1) <= max(len(df) // 3, 25):
            bullish = "Price lower low, RSI higher low"

    if len(high_pairs) >= 2:
        (i1, p1), (i2, p2) = high_pairs[-2], high_pairs[-1]
        r1 = float(rsi.iloc[i1])
        r2 = float(rsi.iloc[i2])
        if p2 > p1 and r2 < r1 and max(i2 - i1, 1) <= max(len(df) // 3, 25):
            bearish = "Price higher high, RSI lower high"

    return bullish, bearish


def build_macd_rsi_cards(df: pd.DataFrame) -> List[SignalCard]:
    cards: List[SignalCard] = []

    last_rsi = float(df["RSI"].iloc[-1])
    if last_rsi >= 70:
        cards.append(SignalCard("RSI Status", "Overbought", f"RSI {last_rsi:.2f} di atas 70"))
    elif last_rsi <= 30:
        cards.append(SignalCard("RSI Status", "Oversold", f"RSI {last_rsi:.2f} di bawah 30"))
    else:
        cards.append(SignalCard("RSI Status", "-", ""))

    cross = recent_cross(df["MACD"], df["MACD_SIGNAL"], lookback=5)
    if cross is None:
        cards.append(SignalCard("MACD Cross", "-", ""))
    else:
        direction, bars_ago = cross
        if direction == "bullish":
            note = "MACD naik memotong signal"
            cards.append(SignalCard("MACD Cross", "Golden Cross", note if bars_ago == 0 else f"{note} • {bars_ago} candle lalu"))
        else:
            note = "MACD turun memotong signal"
            cards.append(SignalCard("MACD Cross", "Bearish Cross", note if bars_ago == 0 else f"{note} • {bars_ago} candle lalu"))

    bull_div, bear_div = detect_rsi_divergences(df)
    cards.append(SignalCard("Bullish Divergence", bull_div or "-", "RSI memberi konfirmasi" if bull_div else ""))
    cards.append(SignalCard("Bearish Divergence", bear_div or "-", "RSI memberi konfirmasi" if bear_div else ""))
    return cards


def build_macd_rsi_chart(df: pd.DataFrame, interval: str, period: str) -> Tuple[go.Figure, str, List[SignalCard]]:
    x_vals = np.arange(len(df))
    custom_dt = [
        ts.strftime("%d %b %Y %H:%M") if is_intraday_interval(interval) else ts.strftime("%d %b %Y")
        for ts in df.index
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.25, 0.25],
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
            hovertemplate="<b>%{customdata}</b><br>Open: %{open}<br>High: %{high}<br>Low: %{low}<br>Close: %{close}<extra></extra>",
            name="Price",
        ),
        row=1,
        col=1,
    )

    hist_colors = [GREEN if h >= 0 else RED for h in df["MACD_HIST"]]
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=df["MACD_HIST"],
            marker_color=hist_colors,
            opacity=0.8,
            customdata=custom_dt,
            hovertemplate="<b>%{customdata}</b><br>MACD Hist: %{y:.3f}<extra></extra>",
            name="MACD Hist",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["MACD"],
            mode="lines",
            line=dict(color=MACD_COLOR, width=2.2),
            text=custom_dt,
            hovertemplate="<b>%{text}</b><br>MACD: %{y:.3f}<extra></extra>",
            name="MACD",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["MACD_SIGNAL"],
            mode="lines",
            line=dict(color=SIGNAL_COLOR, width=2.0),
            text=custom_dt,
            hovertemplate="<b>%{text}</b><br>Signal: %{y:.3f}<extra></extra>",
            name="Signal",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["RSI"],
            mode="lines",
            line=dict(color=RSI_COLOR, width=2.2),
            text=custom_dt,
            hovertemplate="<b>%{text}</b><br>RSI: %{y:.2f}<extra></extra>",
            name="RSI",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_color=RED, line_width=1.1, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_color=GREEN, line_width=1.1, line_dash="dash", row=3, col=1)

    low_price = float(df["Low"].min())
    high_price = float(df["High"].max())
    price_range = max(high_price - low_price, max(float(df["Close"].iloc[-1]) * 0.08, 20.0))
    y_pad = price_range * 0.10
    y_min = low_price - y_pad
    y_max = high_price + y_pad

    tickvals, ticktext = choose_tick_positions(df.index, interval, period)
    xr = x_range_for_chart(df, interval)

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-28,
        showgrid=True,
        gridcolor="#D3D3D3",
        rangeslider_visible=False,
        tickfont=dict(color="#111111", size=12),
        linecolor="#111111",
        zeroline=False,
    )

    for row in [1, 2, 3]:
        fig.update_xaxes(range=xr, row=row, col=1)

    fig.update_yaxes(
        range=[y_min, y_max],
        row=1,
        col=1,
        side="right",
        title_text="Price",
        tickfont=dict(size=13, color="#111111"),
        title_font=dict(size=13, color="#111111"),
        showgrid=True,
        gridcolor="#D3D3D3",
        linecolor="#111111",
    )
    fig.update_yaxes(
        row=2,
        col=1,
        side="right",
        title_text="MACD",
        tickfont=dict(size=12, color="#111111"),
        title_font=dict(size=12, color="#111111"),
        showgrid=True,
        gridcolor="#D3D3D3",
        linecolor="#111111",
        zeroline=True,
        zerolinecolor="#777777",
    )
    fig.update_yaxes(
        range=[0, 100],
        row=3,
        col=1,
        side="right",
        title_text="RSI",
        tickfont=dict(size=12, color="#111111"),
        title_font=dict(size=12, color="#111111"),
        showgrid=True,
        gridcolor="#D3D3D3",
        linecolor="#111111",
    )

    last_close = float(df["Close"].iloc[-1])
    fig.add_hline(y=last_close, line_color=BLACK, line_width=1.1, line_dash="dash", row=1, col=1)

    fig.update_layout(
        dragmode="pan",
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor="white",
        margin=dict(l=8, r=18, t=8, b=8),
        height=920,
        font=dict(color="#111111", size=13),
    )

    return fig, detect_trend(df), build_macd_rsi_cards(df)


# =========================
# BUILD CHARTS
# =========================
def build_sr_chart(df: pd.DataFrame, interval: str, period: str, selected_mas: List[int]) -> Tuple[go.Figure, Dict[str, Optional[LevelCluster]], str, List[PatternSignal]]:
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    levels = choose_key_levels(df, high_idx, low_idx)
    trend = detect_trend(df)
    signals = detect_pattern_signals(df, levels, trend)

    fig, y_min, y_max = base_price_volume_figure(df, interval, period)

    for ma in selected_mas:
        col_name = f"MA{ma}"
        color = MA_COLORS.get(ma, "#5EA3DA")
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(df)),
                y=df[col_name],
                mode="lines",
                line=dict(color=color, width=2.0),
                hovertemplate=f"<b>%{{text}}</b><br>{col_name}: %{{y:.2f}}<extra></extra>",
                text=[ts.strftime("%d %b %Y %H:%M") if is_intraday_interval(interval) else ts.strftime("%d %b %Y") for ts in df.index],
                name=col_name,
            ),
            row=1,
            col=1,
        )

    price_range = max(float(df["High"].max()) - float(df["Low"].min()), 1.0)
    add_sr_lines_and_labels(fig, df, levels, y_min, y_max)
    add_ma_and_close_labels(fig, df, selected_mas, y_min, y_max)
    add_pattern_markers(fig, df, signals, price_range * 0.03)

    return fig, levels, trend, signals


def build_fibonacci_chart(df: pd.DataFrame, interval: str, period: str) -> Tuple[go.Figure, Dict[str, object], str]:
    fib_context = calculate_fibonacci_context(df)
    trend = detect_trend(df)

    fig, y_min, y_max = base_price_volume_figure(df, interval, period)
    add_fibonacci_lines(fig, df, fib_context, y_min, y_max)

    return fig, fib_context, trend


# =========================
# UI HELPERS
# =========================
def assess_risk(levels: Dict[str, Optional[LevelCluster]], trend: str, last_close: float) -> Tuple[str, str]:
    support = levels.get("support_near") or levels.get("support_strong")
    resistance = levels.get("resistance_near") or levels.get("resistance_strong")

    reasons: List[str] = []
    score = 0

    if trend == "Downtrend":
        score += 2
        reasons.append("tren masih turun")
    elif trend == "Sideways":
        score += 1
        reasons.append("tren belum tegas")
    else:
        reasons.append("tren relatif mendukung")

    if support is not None:
        support_gap = max((last_close - support.level) / max(last_close, 1e-9), 0.0)
    else:
        support_gap = 0.05
        score += 1
        reasons.append("support dekat belum jelas")

    if resistance is not None:
        resistance_gap = max((resistance.level - last_close) / max(last_close, 1e-9), 0.0)
    else:
        resistance_gap = 0.05
        reasons.append("resistance dekat belum jelas")

    if resistance_gap < 0.03:
        score += 1
        reasons.append("upside ke resistance cukup sempit")
    if support_gap < 0.02:
        score += 1
        reasons.append("jarak ke support tipis")
    if resistance_gap > support_gap + 0.03 and trend == "Uptrend":
        score -= 1
        reasons.append("ruang ke resistance masih cukup lega")

    risk = "High Risk" if score >= 3 else "Low Risk"
    reason = "; ".join(dict.fromkeys(reasons))
    return risk, reason


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #020814 0%, #06111f 100%);
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
            font-size: 31px;
            font-weight: 800;
            line-height: 1.0;
            margin-bottom: 12px;
            word-break: break-word;
        }}
        .sr-card-note {{
            color: #D9E3F0;
            font-size: 13px;
            line-height: 1.35;
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
            margin-bottom: 10px;
        }}
        .info-box {{
            border: 1px solid {CARD_BORDER};
            background: {CARD_BG};
            border-radius: 14px;
            padding: 16px 18px;
            margin-top: 8px;
            margin-bottom: 8px;
            text-align: center;
        }}
        .info-title {{
            font-size: 24px;
            font-weight: 900;
            color: white;
            line-height: 1.1;
            margin-bottom: 4px;
        }}
        .info-sub {{
            font-size: 18px;
            font-weight: 800;
            color: {LIGHT_TEXT};
            margin-bottom: 8px;
        }}
        .info-note {{
            font-size: 13px;
            font-weight: 600;
            color: #BFD3E8;
            margin-top: 8px;
        }}
        .info-pill-row {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin-top: 2px;
        }}
        .trend-pill {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 800;
            margin-right: 8px;
            margin-bottom: 4px;
        }}
        .trend-up {{ background: rgba(11,163,74,0.15); color: #49F07C; }}
        .trend-down {{ background: rgba(220,38,38,0.15); color: #FF7D7D; }}
        .trend-side {{ background: rgba(255,255,255,0.10); color: #D5DCE5; }}
        .risk-high {{ background: rgba(220,38,38,0.15); color: #FF7D7D; }}
        .risk-low {{ background: rgba(11,163,74,0.15); color: #49F07C; }}
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
        .signal-up {{ color: #49F07C; font-weight: 800; }}
        .signal-down {{ color: #FF6868; font-weight: 800; }}
        .toolbar-title {{
            font-size: 14px;
            font-weight: 800;
            color: #D9E3F0;
            margin-bottom: 4px;
        }}
        .stPlotlyChart, .stPlotlyChart > div {{
            width: 100% !important;
        }}
        div[data-testid="stCheckbox"] label p {{
            font-size: 0.92rem;
            color: #D9E3F0;
        }}
        label, .stSelectbox label {{
            color: #D9E3F0 !important;
        }}
        @media (max-width: 768px) {{
            .sr-card {{ padding: 10px 10px; min-height: 110px; }}
            .sr-touch-top {{ font-size: 10px; margin-bottom: 4px; }}
            .sr-card-title {{ font-size: 12px; margin-bottom: 5px; }}
            .sr-card-value {{ font-size: 20px; margin-bottom: 8px; }}
            .sr-card-note {{ font-size: 11px; line-height: 1.28; }}
            .sr-badge {{ font-size: 10px; padding: 3px 8px; }}
            .info-box {{ padding: 12px 12px; }}
            .info-title {{ font-size: 18px; }}
            .info-sub {{ font-size: 14px; }}
            .info-note {{ font-size: 11px; }}
            .trend-pill {{ font-size: 11px; padding: 5px 9px; margin-right: 5px; }}
            .signal-title {{ font-size: 14px; }}
            .signal-item {{ font-size: 12px; line-height: 1.4; }}
            div[data-testid="stCheckbox"] label p {{ font-size: 0.80rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sr_card(col, title: str, cluster: Optional[LevelCluster]):
    touches = cluster.count if cluster is not None else 0
    value = value_formatter(cluster.level) if cluster is not None else "-"
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


def render_fib_card(col, level: FibLevel):
    with col:
        st.markdown(
            f"""
            <div class="sr-card">
                <div class="sr-badge">{level.short_note}</div>
                <div class="sr-card-title">{level.ratio_label}</div>
                <div class="sr-card-value">{value_formatter(level.price)}</div>
                <div class="sr-card-note">{level.description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_signal_card(col, card: SignalCard):
    value_html = card.value if card.value not in {"", None} else "&nbsp;"
    note_html = card.note if card.note not in {"", None} else "&nbsp;"
    with col:
        st.markdown(
            f"""
            <div class="sr-card">
                <div class="sr-card-title">{card.title}</div>
                <div class="sr-card-value">{value_html}</div>
                <div class="sr-card-note">{note_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header(
    mode: str,
    display_code_value: str,
    data: pd.DataFrame,
    tf_label: str,
    duration_label: str,
    trend: str,
    extra_pill: Optional[str] = None,
    risk_label: Optional[str] = None,
    risk_reason: Optional[str] = None,
):
    last_close = float(data["Close"].iloc[-1])
    trend_class = "trend-up" if trend == "Uptrend" else "trend-down" if trend == "Downtrend" else "trend-side"

    parts = [
        '<div class="info-box">',
        f'<div class="info-title">{html.escape(mode.upper())} | {html.escape(display_code_value)}</div>',
        f'<div class="info-sub">TF: {html.escape(tf_label)} | Durasi: {html.escape(duration_label)} | Last Close: {value_formatter(last_close)}</div>',
    ]

    pill_parts = []
    if trend:
        pill_parts.append(f'<span class="trend-pill {trend_class}">Trend: {html.escape(trend)}</span>')
    if risk_label:
        risk_class = "risk-high" if "high" in risk_label.lower() else "risk-low"
        pill_parts.append(f'<span class="trend-pill {risk_class}">Risk: {html.escape(risk_label)}</span>')
    if extra_pill:
        pill_parts.append(f'<span class="trend-pill trend-side">{html.escape(extra_pill)}</span>')

    if pill_parts:
        parts.append('<div class="info-pill-row">' + ''.join(pill_parts) + '</div>')

    if risk_reason:
        parts.append(f'<div class="info-note">Alasan: {html.escape(risk_reason)}</div>')

    parts.append('</div>')
    st.markdown(textwrap.dedent(''.join(parts)), unsafe_allow_html=True)


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


def make_ma_toolbar() -> List[int]:
    st.markdown('<div class="toolbar-title">Checklist MA</div>', unsafe_allow_html=True)
    cols = st.columns(len(MA_WINDOWS))
    selected: List[int] = []

    for col, ma in zip(cols, MA_WINDOWS):
        with col:
            checked = st.checkbox(f"MA {ma}", key=f"ma_{ma}_on")
            if checked:
                selected.append(ma)

    return selected


# =========================
# SESSION STATE DEFAULTS
# =========================
inject_css()

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Support & Resistance"
if "tf_select" not in st.session_state:
    st.session_state.tf_select = "1 Hari"
if "durasi_select" not in st.session_state:
    st.session_state.durasi_select = "1 Tahun"

for ma in MA_WINDOWS:
    key = f"ma_{ma}_on"
    if key not in st.session_state:
        st.session_state[key] = ma == 10


# =========================
# TOP CONTROLS
# =========================
left_title, right_btn = st.columns([0.78, 0.22])

with left_title:
    st.selectbox(
        "Mode Analisis",
        options=MODE_OPTIONS,
        key="analysis_mode",
        label_visibility="collapsed",
    )

with right_btn:
    st.markdown(
        """
        <div style="
            border:1px solid #0d4f93;
            background:#031427;
            color:white;
            border-radius:14px;
            padding:14px 12px;
            text-align:center;
            font-weight:800;
            margin-top:2px;
        ">
            BTC/IDR
        </div>
        """,
        unsafe_allow_html=True,
    )

ctrl1, ctrl2 = st.columns([1, 1])
with ctrl1:
    st.selectbox(
        "TF",
        options=list(TIMEFRAME_MAP.keys()),
        key="tf_select",
    )

with ctrl2:
    st.selectbox(
        "Durasi",
        options=list(DURATION_MAP.keys()),
        key="durasi_select",
    )


# =========================
# MAIN
# =========================
symbol = SYMBOL
shown_code = "BTC/IDR"
interval = TIMEFRAME_MAP[st.session_state.tf_select]
requested_period = DURATION_MAP[st.session_state.durasi_select]
mode = st.session_state.analysis_mode

with st.spinner(f"Mengambil data {symbol}..."):
    data, note, effective_period = load_data(symbol, interval, requested_period)

if note:
    st.warning(note)

if data.empty:
    st.error("Data tidak tersedia untuk BTC-IDR dari Yahoo Finance.")
elif len(data) < 30:
    st.error(f"Data terlalu sedikit untuk dianalisis. Rows: {len(data)}")
else:
    if mode == "Support & Resistance":
        fig_selected_mas = [ma for ma in MA_WINDOWS if st.session_state.get(f"ma_{ma}_on", False)]
        fig, levels, trend, signals = build_sr_chart(data, interval, effective_period, fig_selected_mas)
        risk_label, risk_reason = assess_risk(levels, trend, float(data["Close"].iloc[-1]))

        c1, c2, c3, c4 = st.columns(4)
        render_sr_card(c1, "Support terdekat", levels.get("support_near"))
        render_sr_card(c2, "Support kuat", levels.get("support_strong"))
        render_sr_card(c3, "Resistance terdekat", levels.get("resistance_near"))
        render_sr_card(c4, "Resistance kuat", levels.get("resistance_strong"))

        render_header(
            mode=mode,
            display_code_value=shown_code,
            data=data,
            tf_label=st.session_state.tf_select,
            duration_label=label_period(effective_period),
            trend=trend,
            risk_label=risk_label,
            risk_reason=risk_reason,
        )
        render_pattern_summary(signals)
        make_ma_toolbar()

    elif mode == "Fibonacci":
        fig, fib_context, trend = build_fibonacci_chart(data, interval, effective_period)
        fib_cols = st.columns(5)
        for col, fib_level in zip(fib_cols, fib_context["levels"]):
            render_fib_card(col, fib_level)

        render_header(
            mode=mode,
            display_code_value=shown_code,
            data=data,
            tf_label=st.session_state.tf_select,
            duration_label=label_period(effective_period),
            trend=trend,
            extra_pill=fib_context["direction_label"],
        )

    else:
        fig, trend, macd_cards = build_macd_rsi_chart(data, interval, effective_period)
        signal_cols = st.columns(4)
        for col, card in zip(signal_cols, macd_cards):
            render_signal_card(col, card)

        render_header(
            mode=mode,
            display_code_value=shown_code,
            data=data,
            tf_label=st.session_state.tf_select,
            duration_label=label_period(effective_period),
            trend=trend,
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "doubleClick": "reset+autosize",
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d",
                "drawline",
                "drawopenpath",
                "eraseshape",
                "toggleSpikelines",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
            ],
        },
    )

    with st.expander("Keterangan"):
        if mode == "Support & Resistance":
            selected = [f"MA{ma}" for ma in MA_WINDOWS if st.session_state.get(f"ma_{ma}_on", False)]
            ma_note = ", ".join(selected) if selected else "tidak ada MA harga yang dipilih"
            st.write(
                "- Mode Support & Resistance menampilkan garis SR, volume, VMA20, garis close, dan checklist MA harga.\n"
                f"- MA aktif saat ini: {ma_note}.\n"
                "- Label harga dipindah ke area kosong di kanan supaya tidak menutupi candle.\n"
                "- Probabilitas pola candle bersifat heuristik, bukan kepastian."
            )
        elif mode == "Fibonacci":
            fib_lines = [f"- {level.ratio_label}: {level.description}" for level in fib_context["levels"]]
            st.write(
                "- Mode Fibonacci menampilkan garis Fibonacci retracement, garis close terakhir, volume, dan VMA20.\n"
                "- Tidak menampilkan SK, SD, RK, RD, MA harga, atau sinyal candle.\n"
                "- Anchor Fibonacci diambil dari swing high dan swing low pada durasi yang dipilih.\n"
                + "\n".join(fib_lines)
            )
        else:
            st.write(
                "- Mode MACD & RSI menampilkan candle price, panel MACD, dan panel RSI.\n"
                "- Kartu di atas menyorot status RSI, cross MACD, serta divergence bullish/bearish bila terdeteksi.\n"
                "- Divergence dihitung dari pivot harga terbaru dan pembacaan RSI, jadi sifatnya indikatif."
            )
