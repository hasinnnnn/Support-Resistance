from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from email.utils import parsedate_to_datetime
import html
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Support & Resistance / Fibonacci / MACD & RSI / Buy Sell / News & Sentimen / Backtest", layout="wide")


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
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


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
    footer: str = ""
    tone: str = "neutral"
    sort_idx: int = 0


@dataclass
class NewsItem:
    title: str
    summary: str
    link: str
    source: str
    published: str
    thumbnail: str = ""
    sentiment: str = "neutral"
    sort_key: float = 0.0


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
MA_VOL_COLOR = "#2C2C2C"
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

MODE_OPTIONS = ["Support & Resistance", "Fibonacci", "MACD & RSI", "Sinyal Buy & Sell", "News & Sentimen", "Backtest Strategy"]
MA_WINDOWS = [10, 20, 50, 100, 200]


TIMEFRAME_MAP = {
    "5 Menit": "5m",
    "15 Menit": "15m",
    "30 Menit": "30m",
    "1 Jam": "60m",
    "2 Jam": "2h",
    "4 Jam": "4h",
    "1 Hari": "1d",
    "1 Minggu": "1wk",
}


DURATION_MAP = {
    "1 Hari": "1d",
    "3 Hari": "3d",
    "1 Minggu": "1wk",
    "2 Minggu": "2wk",
    "1 Bulan": "1mo",
    "3 Bulan": "3mo",
    "6 Bulan": "6mo",
    "1 Tahun": "1y",
    "2 Tahun": "2y",
}

ISSI_LIQUID_UNIVERSE = [
    "ADRO.JK", "AKRA.JK", "ANTM.JK", "BRIS.JK", "CPIN.JK", "EXCL.JK", "GOTO.JK",
    "ICBP.JK", "INDF.JK", "INTP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "KLBF.JK",
    "MAPI.JK", "MDKA.JK", "MEDC.JK", "MIKA.JK", "PGAS.JK", "PTBA.JK", "SIDO.JK",
    "SMGR.JK", "TINS.JK", "TLKM.JK", "UNTR.JK"
]

SCREENER_PRESETS = ["Breakout MA 200", "BPJS", "BSJP"]


INTRADAY_LIMITS = {
    "5m": "1mo",
    "15m": "1mo",
    "30m": "1mo",
    "60m": "2y",
    "2h": "2y",
    "4h": "2y",
}


PERIOD_RANK = {
    "1d": 1,
    "3d": 2,
    "1wk": 3,
    "2wk": 4,
    "1mo": 5,
    "3mo": 6,
    "6mo": 7,
    "1y": 8,
    "2y": 9,
    "730d": 9,
}

FIB_LEVEL_SPECS = [
    {
        "ratio": 0.236,
        "ratio_label": "23,6% (0.236)",
        "short_note": "Koreksi ringan",
        "description": "Menandakan koreksi ringan, sering muncul saat tren utama sangat kuat.",
    },
    {
        "ratio": 0.382,
        "ratio_label": "38,2% (0.382)",
        "short_note": "Koreksi umum",
        "description": "Zona koreksi umum dan area support/resistance awal yang sering dipantau trader.",
    },
    {
        "ratio": 0.500,
        "ratio_label": "50% (0.500)",
        "short_note": "Level psikologis",
        "description": "Bukan rasio Fibonacci asli, tetapi level penting tempat harga sering berbalik.",
    },
    {
        "ratio": 0.618,
        "ratio_label": "61,8% (0.618)",
        "short_note": "Golden Ratio",
        "description": "Level terpenting; probabilitas rebound atau reversal biasanya paling diperhatikan di area ini.",
    },
    {
        "ratio": 0.786,
        "ratio_label": "78,6% (0.786)",
        "short_note": "Koreksi dalam",
        "description": "Menandakan koreksi dalam dan sering jadi batas akhir sebelum pembalikan arah besar.",
    },
]


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
        "5m": "5 Menit",
        "15m": "15 Menit",
        "30m": "30 Menit",
        "60m": "1 Jam",
        "2h": "2 Jam",
        "4h": "4 Jam",
        "1d": "1 Hari",
        "1wk": "1 Minggu",
    }.get(interval, interval)



def label_period(period: str) -> str:
    return {
        "1d": "1 Hari",
        "3d": "3 Hari",
        "1wk": "1 Minggu",
        "2wk": "2 Minggu",
        "1mo": "1 Bulan",
        "3mo": "3 Bulan",
        "6mo": "6 Bulan",
        "1y": "1 Tahun",
        "2y": "2 Tahun",
        "730d": "2 Tahun",
    }.get(period, period)


def is_intraday_interval(interval: str) -> bool:
    return interval in {"5m", "15m", "30m", "60m", "2h", "4h"}


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
    download_interval = "60m" if interval in {"2h", "4h"} else interval

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

    if interval in {"2h", "4h"}:
        freq = "2H" if interval == "2h" else "4H"
        df = (
            df.resample(freq)
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )

    for ma in MA_WINDOWS:
        df[f"MA{ma}"] = df["Close"].rolling(ma).mean()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
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

    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    stoch_rsi = (df["RSI"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    df["STOCH_RSI_K"] = stoch_rsi.rolling(3).mean().fillna(50)
    df["STOCH_RSI_D"] = df["STOCH_RSI_K"].rolling(3).mean().fillna(50)

    return df, note, effective_period


# ----------------------------
# Market structure helpers
# ----------------------------

def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    hi = df["High"].to_numpy()
    lo = df["Low"].to_numpy()
    if len(df) < left + right + 1:
        return highs, lows
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



def detect_pattern_signals(df: pd.DataFrame, levels: Dict[str, Optional[LevelCluster]], trend: str, lookback: int = 40) -> List[PatternSignal]:
    if len(df) < 2:
        return []

    signals: List[PatternSignal] = []
    start_idx = max(1, len(df) - lookback)

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
            return min(max(prob, 0.50), 0.82)

        if pc < po and c > o and o <= pc and c >= po and body > 0:
            signals.append(PatternSignal(i, "Bullish Engulfing", "naik", make_prob(0.57, "naik"), "reversal bullish", i - 1, i))
            continue
        if pc > po and c < o and o >= pc and c <= po and body > 0:
            signals.append(PatternSignal(i, "Bearish Engulfing", "turun", make_prob(0.57, "turun"), "reversal bearish", i - 1, i))
            continue
        if body_pct <= 0.40 and lower >= body * 2.0 and upper <= max(body, rng * 0.15):
            signals.append(PatternSignal(i, "Hammer", "naik", make_prob(0.54, "naik"), "rejection bawah", i, i))
            continue
        if body_pct <= 0.40 and upper >= body * 2.0 and lower <= max(body, rng * 0.15):
            signals.append(PatternSignal(i, "Shooting Star", "turun", make_prob(0.54, "turun"), "rejection atas", i, i))
            continue
        prev_body = abs(pc - po)
        if prev_body / max(float(df["High"].iloc[i-1]) - float(df["Low"].iloc[i-1]), 1e-9) <= 0.12:
            if c > max(o, pc) and high_vol:
                signals.append(PatternSignal(i, "Morning Doji Star", "naik", make_prob(0.56, "naik"), "doji diikuti konfirmasi bullish", i - 1, i))
                continue
            if c < min(o, pc) and high_vol:
                signals.append(PatternSignal(i, "Evening Doji Star", "turun", make_prob(0.56, "turun"), "doji diikuti konfirmasi bearish", i - 1, i))
                continue

    chosen: List[PatternSignal] = []
    for sig in sorted(signals, key=lambda s: (s.idx, s.probability), reverse=True):
        if all(abs(sig.idx - x.idx) > 2 for x in chosen):
            chosen.append(sig)
        if len(chosen) >= 6:
            break
    return sorted(chosen, key=lambda s: s.idx)


# ----------------------------
# Chart helpers
# ----------------------------

def value_formatter(x: float) -> str:
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
    intraday_show_datetime = PERIOD_RANK.get(period, 99) <= PERIOD_RANK.get("1mo", 1) and is_intraday_interval(interval)

    def add_nearest(ts: pd.Timestamp, label: str):
        diffs = np.abs((index - ts).total_seconds())
        pos = int(np.argmin(diffs))
        if pos not in positions:
            positions.append(pos)
            labels.append(label)

    if intraday_show_datetime:
        if len(unique_days) == 1:
            preferred = ["09:00", "11:00", "13:00", "15:00"]
            base_day = unique_days[-1]
            for t in preferred:
                add_nearest(pd.Timestamp(f"{base_day.date()} {t}"), f"{base_day.strftime('%d %b %Y')}<br>{t}")
        elif len(unique_days) <= 7:
            preferred = ["09:00", "13:00", "15:00"]
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


def label_x_position(df: pd.DataFrame) -> float:
    return len(df) - 1 + 3.2


def x_range_for_chart(df: pd.DataFrame, interval: str) -> List[float]:
    last_n_map = {
        "5m": 120,
        "15m": 120,
        "30m": 110,
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


def locked_axis_bounds(low: float, high: float, floor_zero: bool = False) -> Tuple[float, float]:
    low = float(low)
    high = float(high)
    if floor_zero:
        low = max(0.0, low)
    if not np.isfinite(low) or not np.isfinite(high):
        return (0.0, 1.0)
    if high <= low:
        span = max(abs(high) * 0.01, 1e-6)
        high = high + span
        low = max(0.0, low - span) if floor_zero else low - span
    return low, high


def build_empty_chart(message: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor="white",
        margin=dict(l=8, r=18, t=8, b=8),
        height=560,
        showlegend=False,
        dragmode="pan",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(color="#111111", size=13),
    )
    if message:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=message,
            showarrow=False,
            font=dict(size=15, color="#666666"),
        )
    return fig


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
    y_min, y_max = locked_axis_bounds(low_price, high_price, floor_zero=True)
    _, volume_max = locked_axis_bounds(0.0, float(df["Volume"].max()), floor_zero=True)

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
        range=[y_min, y_max],
        minallowed=0,
        maxallowed=y_max,
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
        range=[0, volume_max],
        minallowed=0,
        maxallowed=volume_max,
    )

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


def add_price_annotations(
    fig: go.Figure,
    df: pd.DataFrame,
    items: List[dict],
    y_min: float,
    y_max: float,
):
    if not items:
        return

    price_range = max(y_max - y_min, 1.0)
    min_gap = max(price_range * 0.055, 12.0)
    grouped: Dict[float, List[dict]] = {}

    for item in items:
        x_val = float(item.get("x_override", label_x_position(df)))
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


def make_ma_toolbar(show_title: bool = False):
    if show_title:
        st.markdown('<div class="toolbar-title">Moving Average</div>', unsafe_allow_html=True)
    cols = st.columns(len(MA_WINDOWS))
    selected: List[int] = []
    for col, ma in zip(cols, MA_WINDOWS):
        with col:
            checked = st.checkbox(f"MA {ma}", key=f"ma_{ma}_on")
            if checked:
                selected.append(ma)
    return selected


# ----------------------------
# MACD / RSI helpers
# ----------------------------

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
            bullish = f"Price lower low, RSI higher low"

    if len(high_pairs) >= 2:
        (i1, p1), (i2, p2) = high_pairs[-2], high_pairs[-1]
        r1 = float(rsi.iloc[i1])
        r2 = float(rsi.iloc[i2])
        if p2 > p1 and r2 < r1 and max(i2 - i1, 1) <= max(len(df) // 3, 25):
            bearish = f"Price higher high, RSI lower high"

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
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.44, 0.22, 0.17, 0.17])

    fig.add_trace(go.Candlestick(x=x_vals, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], customdata=custom_dt,
        increasing_line_color=GREEN, increasing_fillcolor=GREEN, decreasing_line_color=RED, decreasing_fillcolor=RED,
        hovertemplate="<b>%{customdata}</b><br>Open: %{open}<br>High: %{high}<br>Low: %{low}<br>Close: %{close}<extra></extra>", name="Price"), row=1, col=1)

    hist_colors = [GREEN if h >= 0 else RED for h in df["MACD_HIST"]]
    fig.add_trace(go.Bar(x=x_vals, y=df["MACD_HIST"], marker_color=hist_colors, opacity=0.82, customdata=custom_dt, hovertemplate="<b>%{customdata}</b><br>MACD Hist: %{y:.3f}<extra></extra>", name="MACD Hist"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df["MACD"], mode="lines", line=dict(color=MACD_COLOR, width=2.2), text=custom_dt, hovertemplate="<b>%{text}</b><br>MACD: %{y:.3f}<extra></extra>", name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df["MACD_SIGNAL"], mode="lines", line=dict(color=SIGNAL_COLOR, width=2.0), text=custom_dt, hovertemplate="<b>%{text}</b><br>Signal: %{y:.3f}<extra></extra>", name="Signal"), row=2, col=1)

    fig.add_trace(go.Scatter(x=x_vals, y=df["RSI"], mode="lines", line=dict(color=RSI_COLOR, width=2.2), text=custom_dt, hovertemplate="<b>%{text}</b><br>RSI: %{y:.2f}<extra></extra>", name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_color=RED, line_width=1.1, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_color=GREEN, line_width=1.1, line_dash="dash", row=3, col=1)

    fig.add_trace(go.Scatter(x=x_vals, y=df["STOCH_RSI_K"], mode="lines", line=dict(color="#0057B8", width=2.0), text=custom_dt, hovertemplate="<b>%{text}</b><br>Stoch RSI %K: %{y:.2f}<extra></extra>", name="Stoch RSI %K"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df["STOCH_RSI_D"], mode="lines", line=dict(color="#8C52FF", width=1.9), text=custom_dt, hovertemplate="<b>%{text}</b><br>Stoch RSI %D: %{y:.2f}<extra></extra>", name="Stoch RSI %D"), row=4, col=1)
    fig.add_hline(y=80, line_color=RED, line_width=1.0, line_dash="dash", row=4, col=1)
    fig.add_hline(y=20, line_color=GREEN, line_width=1.0, line_dash="dash", row=4, col=1)

    low_price = float(df["Low"].min())
    high_price = float(df["High"].max())
    y_min, y_max = locked_axis_bounds(low_price, high_price, floor_zero=True)

    macd_min = float(np.nanmin([df["MACD"].min(), df["MACD_SIGNAL"].min(), df["MACD_HIST"].min()]))
    macd_max = float(np.nanmax([df["MACD"].max(), df["MACD_SIGNAL"].max(), df["MACD_HIST"].max()]))
    macd_low, macd_high = locked_axis_bounds(macd_min, macd_max, floor_zero=False)

    tickvals, ticktext = choose_tick_positions(df.index, interval, period)
    xr = x_range_for_chart(df, interval)
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=-28, showgrid=True, gridcolor="#D3D3D3", rangeslider_visible=False, tickfont=dict(color="#111111", size=12), linecolor="#111111", zeroline=False)
    for row in [1, 2, 3, 4]:
        fig.update_xaxes(range=xr, row=row, col=1)

    fig.update_yaxes(range=[y_min, y_max], minallowed=0, maxallowed=y_max, row=1, col=1, side="right", title_text="Price", tickfont=dict(size=13, color="#111111"), title_font=dict(size=13, color="#111111"), showgrid=True, gridcolor="#D3D3D3", linecolor="#111111")
    fig.update_yaxes(range=[macd_low, macd_high], minallowed=macd_low, maxallowed=macd_high, row=2, col=1, side="right", title_text="MACD", tickfont=dict(size=12, color="#111111"), title_font=dict(size=12, color="#111111"), showgrid=True, gridcolor="#D3D3D3", linecolor="#111111", zeroline=True, zerolinecolor="#777777")
    fig.update_yaxes(range=[0, 100], minallowed=0, maxallowed=100, row=3, col=1, side="right", title_text="RSI", tickfont=dict(size=12, color="#111111"), title_font=dict(size=12, color="#111111"), showgrid=True, gridcolor="#D3D3D3", linecolor="#111111")
    fig.update_yaxes(range=[0, 100], minallowed=0, maxallowed=100, row=4, col=1, side="right", title_text="Stoch RSI", tickfont=dict(size=12, color="#111111"), title_font=dict(size=12, color="#111111"), showgrid=True, gridcolor="#D3D3D3", linecolor="#111111")

    last_close = float(df["Close"].iloc[-1])
    fig.add_hline(y=last_close, line_color=BLACK, line_width=1.1, line_dash="dash", row=1, col=1)

    fig.update_layout(dragmode="pan", hovermode="x unified", showlegend=False, plot_bgcolor=BG_COLOR, paper_bgcolor="white", margin=dict(l=8, r=18, t=8, b=8), height=1080, font=dict(color="#111111", size=13))
    return fig, detect_trend(df), build_macd_rsi_cards(df)


# ----------------------------
# SR / Fib chart builders# ----------------------------
# SR / Fib chart builders
# ----------------------------

def build_sr_chart(df: pd.DataFrame, interval: str, period: str, selected_mas: List[int]) -> Tuple[go.Figure, Dict[str, Optional[LevelCluster]], str, List[PatternSignal]]:
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    levels = choose_key_levels(df, high_idx, low_idx)
    trend = detect_trend(df)
    signals = detect_pattern_signals(df, levels, trend, lookback=50)

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



# ----------------------------
# Extended signal builders
# ----------------------------


def format_idx_time(df: pd.DataFrame, idx: int, interval: str) -> str:
    ts = df.index[idx]
    return ts.strftime("%d %b %Y %H:%M") if is_intraday_interval(interval) else ts.strftime("%d %b %Y")


def format_idx_range(df: pd.DataFrame, start_idx: Optional[int], end_idx: Optional[int], interval: str) -> str:
    if start_idx is None and end_idx is None:
        return ""
    if start_idx is None:
        start_idx = end_idx
    if end_idx is None:
        end_idx = start_idx
    start_txt = format_idx_time(df, int(start_idx), interval)
    end_txt = format_idx_time(df, int(end_idx), interval)
    if start_txt == end_txt:
        return start_txt
    return f"{start_txt} s/d {end_txt}"


def detect_series_divergences(df: pd.DataFrame, oscillator_col: str) -> Tuple[Optional[dict], Optional[dict]]:
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    low_pairs = extract_pivot_pairs(df["Close"], low_idx, max_points=5)
    high_pairs = extract_pivot_pairs(df["Close"], high_idx, max_points=5)
    osc = df[oscillator_col]
    bullish = None
    bearish = None

    if len(low_pairs) >= 2:
        (i1, p1), (i2, p2) = low_pairs[-2], low_pairs[-1]
        o1 = float(osc.iloc[i1]); o2 = float(osc.iloc[i2])
        if p2 < p1 and o2 > o1 and 0 < i2 - i1 <= max(len(df) // 3, 25):
            bullish = {"idx": i2, "note": f"Price lower low, {oscillator_col} higher low"}

    if len(high_pairs) >= 2:
        (i1, p1), (i2, p2) = high_pairs[-2], high_pairs[-1]
        o1 = float(osc.iloc[i1]); o2 = float(osc.iloc[i2])
        if p2 > p1 and o2 < o1 and 0 < i2 - i1 <= max(len(df) // 3, 25):
            bearish = {"idx": i2, "note": f"Price higher high, {oscillator_col} lower high"}

    return bullish, bearish



def detect_structure_patterns(df: pd.DataFrame, trend: str, lookback: int = 90) -> List[PatternSignal]:
    patterns: List[PatternSignal] = []
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    high_idx = [i for i in high_idx if i >= max(0, len(df) - lookback)]
    low_idx = [i for i in low_idx if i >= max(0, len(df) - lookback)]

    if len(low_idx) >= 2:
        i1, i2 = low_idx[-2], low_idx[-1]
        p1 = float(df["Low"].iloc[i1]); p2 = float(df["Low"].iloc[i2])
        if p1 > 0 and abs(p2 - p1) / p1 <= 0.035 and i2 - i1 <= max(len(df) // 2, 35):
            patterns.append(PatternSignal(i2, "Double Bottom", "naik", 0.58, "dua low relatif sejajar; tekanan jual mulai tertahan", i1, i2))

    if len(high_idx) >= 2:
        i1, i2 = high_idx[-2], high_idx[-1]
        p1 = float(df["High"].iloc[i1]); p2 = float(df["High"].iloc[i2])
        if p1 > 0 and abs(p2 - p1) / p1 <= 0.035 and i2 - i1 <= max(len(df) // 2, 35):
            patterns.append(PatternSignal(i2, "Double Top", "turun", 0.58, "dua high relatif sejajar; tekanan beli mulai melemah", i1, i2))

    if len(low_idx) >= 3:
        a, b, c = low_idx[-3], low_idx[-2], low_idx[-1]
        pa, pb, pc = float(df["Low"].iloc[a]), float(df["Low"].iloc[b]), float(df["Low"].iloc[c])
        if abs(pa - pb) / max(pa, 1e-9) <= 0.04 and abs(pb - pc) / max(pb, 1e-9) <= 0.04:
            patterns.append(PatternSignal(c, "Triple Bottom", "naik", 0.61, "tiga low berdekatan; area demand terlihat bertahan", a, c))

    if len(high_idx) >= 3:
        a, b, c = high_idx[-3], high_idx[-2], high_idx[-1]
        pa, pb, pc = float(df["High"].iloc[a]), float(df["High"].iloc[b]), float(df["High"].iloc[c])
        if abs(pa - pb) / max(pa, 1e-9) <= 0.04 and abs(pb - pc) / max(pb, 1e-9) <= 0.04:
            patterns.append(PatternSignal(c, "Triple Top", "turun", 0.61, "tiga high berdekatan; tekanan jual mulai dominan", a, c))

    if len(high_idx) >= 3:
        a, b, c = high_idx[-3], high_idx[-2], high_idx[-1]
        ha, hb, hc = float(df["High"].iloc[a]), float(df["High"].iloc[b]), float(df["High"].iloc[c])
        if hb > ha * 1.02 and hb > hc * 1.02 and abs(ha - hc) / max(ha, 1e-9) <= 0.05:
            valley1 = float(df["Low"].iloc[a:b+1].min())
            valley2 = float(df["Low"].iloc[b:c+1].min())
            neckline = max(valley1, valley2)
            last_close = float(df["Close"].iloc[-1])
            if last_close <= neckline * 1.02:
                patterns.append(PatternSignal(c, "Head & Shoulders", "turun", 0.63, "bahu kiri, kepala, dan bahu kanan terbentuk; neckline mulai diuji", a, c))

    if len(low_idx) >= 3:
        a, b, c = low_idx[-3], low_idx[-2], low_idx[-1]
        la, lb, lc = float(df["Low"].iloc[a]), float(df["Low"].iloc[b]), float(df["Low"].iloc[c])
        if lb < la * 0.98 and lb < lc * 0.98 and abs(la - lc) / max(la, 1e-9) <= 0.05:
            crest1 = float(df["High"].iloc[a:b+1].max())
            crest2 = float(df["High"].iloc[b:c+1].max())
            neckline = min(crest1, crest2)
            last_close = float(df["Close"].iloc[-1])
            if last_close >= neckline * 0.98:
                patterns.append(PatternSignal(c, "Inverse Head & Shoulders", "naik", 0.63, "inverse head & shoulders terbentuk; neckline mulai diuji", a, c))

    window = min(len(df), lookback)
    if window >= 35:
        sub = df.iloc[-window:]
        seg = max(8, window // 4)
        left = sub.iloc[:seg]
        middle = sub.iloc[seg: window - seg]
        right = sub.iloc[window - seg:]
        if not left.empty and not middle.empty and not right.empty:
            left_high = float(left["High"].max())
            middle_low = float(middle["Low"].min())
            right_high = float(right["High"].max())
            depth = max(left_high - middle_low, 0.0)
            right_low = float(right["Low"].min())
            right_close = float(sub["Close"].iloc[-1])
            if left_high > 0 and depth / left_high >= 0.07 and abs(right_high - left_high) / left_high <= 0.06:
                handle_depth = max(right_high - right_low, 0.0)
                if depth > 0 and handle_depth / depth <= 0.45 and right_close >= right_high * 0.96:
                    start_idx = len(df) - window
                    patterns.append(PatternSignal(len(df) - 1, "Cup & Handle", "naik", 0.60, "rounded base dan handle dangkal mendekati breakout", start_idx, len(df) - 1))

    dedup = []
    seen = set()
    for sig in sorted(patterns, key=lambda s: (s.idx, s.probability), reverse=True):
        key = (sig.name, sig.start_idx, sig.end_idx)
        if key not in seen:
            seen.add(key)
            dedup.append(sig)
    return sorted(dedup[:6], key=lambda s: s.idx)


def compute_trade_metrics(side: str, entry: float, levels: Dict[str, Optional[LevelCluster]], trend: str) -> Tuple[str, str]:
    support = levels.get("support_near") or levels.get("support_strong")
    resistance = levels.get("resistance_near") or levels.get("resistance_strong")
    if side == "Buy":
        stop = support.level * 0.992 if support is not None and support.level < entry else entry * 0.96
        target = resistance.level if resistance is not None and resistance.level > entry else entry * 1.08
        rr = max((target - entry) / max(entry - stop, 1e-9), 0.0)
        risk_score = 0
        if trend == "Downtrend":
            risk_score += 2
        elif trend == "Sideways":
            risk_score += 1
        if rr < 1.3:
            risk_score += 1
        if support is not None and (entry - support.level) / max(entry, 1e-9) <= 0.02:
            risk_score -= 1
    else:
        stop = resistance.level * 1.008 if resistance is not None and resistance.level > entry else entry * 1.04
        target = support.level if support is not None and support.level < entry else entry * 0.92
        rr = max((entry - target) / max(stop - entry, 1e-9), 0.0)
        risk_score = 0
        if trend == "Uptrend":
            risk_score += 2
        elif trend == "Sideways":
            risk_score += 1
        if rr < 1.3:
            risk_score += 1
        if resistance is not None and (resistance.level - entry) / max(entry, 1e-9) <= 0.02:
            risk_score -= 1
    risk = "High Risk" if risk_score >= 2 else "Low Risk"
    return risk, f"{rr:.2f}R"


def add_price_event_labels(fig: go.Figure, df: pd.DataFrame, events: List[dict], y_span: float):
    if not events:
        return
    xs, ys, texts, colors, positions = [], [], [], [], []
    offset = max(y_span * 0.035, 5.0)
    for ev in events:
        i = int(ev["idx"])
        is_buy = ev["side"] == "Buy"
        base_y = float(df["Low"].iloc[i]) - offset if is_buy else float(df["High"].iloc[i]) + offset
        xs.append(i)
        ys.append(base_y)
        texts.append(ev["label"])
        colors.append(GREEN if is_buy else RED)
        positions.append("bottom center" if is_buy else "top center")

    for x, y, txt, color, pos in zip(xs, ys, texts, colors, positions):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[txt],
                textposition=pos,
                marker=dict(size=8, color=color, line=dict(color="white", width=1)),
                textfont=dict(size=13, color=color),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )


def add_panel_event_labels(fig: go.Figure, events: List[dict], row: int):
    for ev in events:
        pos = "top center" if ev.get("top", True) else "bottom center"
        fig.add_trace(
            go.Scatter(
                x=[ev["idx"]],
                y=[ev["y"]],
                mode="markers+text",
                text=[ev["label"]],
                textposition=pos,
                marker=dict(size=7, color=ev["color"], line=dict(color="white", width=1)),
                textfont=dict(size=12, color=ev["color"]),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )


def build_sr_chart_v2(df: pd.DataFrame, interval: str, period: str, selected_mas: List[int]) -> Tuple[go.Figure, Dict[str, Optional[LevelCluster]], str, List[PatternSignal], List[PatternSignal]]:
    fig, levels, trend, signals = build_sr_chart(df, interval, period, selected_mas)
    structures = detect_structure_patterns(df, trend, lookback=90)
    return fig, levels, trend, signals, structures



def build_macd_rsi_events_v2(df: pd.DataFrame, interval: str) -> Tuple[List[SignalCard], List[dict], List[dict], List[dict]]:
    cards: List[SignalCard] = []
    macd_labels: List[dict] = []
    rsi_labels: List[dict] = []
    stoch_labels: List[dict] = []
    events: List[dict] = []
    start = max(2, len(df) - 60)

    for i in range(start, len(df)):
        prev_rsi = float(df["RSI"].iloc[i - 1]); curr_rsi = float(df["RSI"].iloc[i])
        if prev_rsi > 30 and curr_rsi <= 30:
            events.append({"idx": i, "title": "Oversold", "value": "RSI Oversold", "note": f"RSI turun ke {curr_rsi:.1f} dan masuk area oversold", "footer": format_idx_time(df, i, interval), "type": "rsi", "label": "Oversold", "y": curr_rsi, "color": RED, "top": False, "tone": "negative"})
        if prev_rsi < 70 and curr_rsi >= 70:
            events.append({"idx": i, "title": "Overbought", "value": "RSI Overbought", "note": f"RSI naik ke {curr_rsi:.1f} dan masuk area overbought", "footer": format_idx_time(df, i, interval), "type": "rsi", "label": "Overbought", "y": curr_rsi, "color": GREEN, "top": True, "tone": "positive"})
        if prev_rsi <= 30 and curr_rsi > 30:
            events.append({"idx": i, "title": "RSI Rebound", "value": "Bullish Rebound", "note": f"RSI keluar dari oversold menuju {curr_rsi:.1f}", "footer": format_idx_time(df, i, interval), "type": "rsi", "label": "RSI Rebound", "y": curr_rsi, "color": GREEN, "top": False, "tone": "positive"})
        if prev_rsi >= 70 and curr_rsi < 70:
            events.append({"idx": i, "title": "RSI Cooling", "value": "Bearish Cooling", "note": f"RSI turun keluar dari overbought ke {curr_rsi:.1f}", "footer": format_idx_time(df, i, interval), "type": "rsi", "label": "RSI Cooling", "y": curr_rsi, "color": RED, "top": True, "tone": "negative"})

        prev_macd = float(df["MACD"].iloc[i - 1]); prev_sig = float(df["MACD_SIGNAL"].iloc[i - 1])
        curr_macd = float(df["MACD"].iloc[i]); curr_sig = float(df["MACD_SIGNAL"].iloc[i])
        if prev_macd <= prev_sig and curr_macd > curr_sig:
            events.append({"idx": i, "title": "Bullish Cross", "value": "Golden Cross", "note": "MACD memotong signal ke atas", "footer": format_idx_time(df, i, interval), "type": "macd", "label": "Golden Cross", "y": curr_macd, "color": GREEN, "top": False, "tone": "positive"})
        if prev_macd >= prev_sig and curr_macd < curr_sig:
            events.append({"idx": i, "title": "Bearish Cross", "value": "Bearish Cross", "note": "MACD memotong signal ke bawah", "footer": format_idx_time(df, i, interval), "type": "macd", "label": "Bearish Cross", "y": curr_macd, "color": RED, "top": True, "tone": "negative"})

        prev_k = float(df["STOCH_RSI_K"].iloc[i - 1]); curr_k = float(df["STOCH_RSI_K"].iloc[i])
        prev_d = float(df["STOCH_RSI_D"].iloc[i - 1]); curr_d = float(df["STOCH_RSI_D"].iloc[i])
        if prev_k <= prev_d and curr_k > curr_d and curr_k <= 25:
            events.append({"idx": i, "title": "Stoch Rebound", "value": "Bullish Stoch RSI", "note": f"%K memotong %D ke atas di area rendah ({curr_k:.1f})", "footer": format_idx_time(df, i, interval), "type": "stoch", "label": "Stoch Up", "y": curr_k, "color": GREEN, "top": False, "tone": "positive"})
        if prev_k >= prev_d and curr_k < curr_d and curr_k >= 75:
            events.append({"idx": i, "title": "Stoch Cooling", "value": "Bearish Stoch RSI", "note": f"%K memotong %D ke bawah di area tinggi ({curr_k:.1f})", "footer": format_idx_time(df, i, interval), "type": "stoch", "label": "Stoch Down", "y": curr_k, "color": RED, "top": True, "tone": "negative"})

    bull_rsi, bear_rsi = detect_series_divergences(df, "RSI")
    bull_macd, bear_macd = detect_series_divergences(df, "MACD")
    for item, title, value, typ, color, top, tone in [
        (bull_rsi, "Bullish Divergence", "RSI Divergence", "rsi", GREEN, False, "positive"),
        (bear_rsi, "Bearish Divergence", "RSI Divergence", "rsi", RED, True, "negative"),
        (bull_macd, "Bullish Divergence", "MACD Divergence", "macd", GREEN, False, "positive"),
        (bear_macd, "Bearish Divergence", "MACD Divergence", "macd", RED, True, "negative"),
    ]:
        if item is not None:
            idx = int(item["idx"])
            if idx >= start:
                y_val = float(df["RSI"].iloc[idx]) if typ == "rsi" else float(df["MACD"].iloc[idx])
                events.append({"idx": idx, "title": title, "value": value, "note": item["note"], "footer": format_idx_time(df, idx, interval), "type": typ, "label": title, "y": y_val, "color": color, "top": top, "tone": tone})

    uniq = []
    seen = set()
    for ev in sorted(events, key=lambda x: x["idx"], reverse=True):
        key = (ev["idx"], ev["title"], ev["value"])
        if key not in seen:
            seen.add(key)
            uniq.append(ev)
    uniq = sorted(uniq[:4], key=lambda x: x["idx"], reverse=True)

    for ev in uniq:
        cards.append(SignalCard(ev["title"], ev["value"], ev["note"], ev["footer"], ev["tone"], ev["idx"]))
        if ev["type"] == "macd":
            macd_labels.append(ev)
        elif ev["type"] == "stoch":
            stoch_labels.append(ev)
        else:
            rsi_labels.append(ev)
    return cards, macd_labels, rsi_labels, stoch_labels


def build_macd_rsi_chart_v2(df: pd.DataFrame, interval: str, period: str) -> Tuple[go.Figure, str, List[SignalCard]]:
    fig, trend, _ = build_macd_rsi_chart(df, interval, period)
    cards, macd_labels, rsi_labels, stoch_labels = build_macd_rsi_events_v2(df, interval)
    add_panel_event_labels(fig, macd_labels, row=2)
    add_panel_event_labels(fig, rsi_labels, row=3)
    add_panel_event_labels(fig, stoch_labels, row=4)
    return fig, trend, cards



def build_trade_setups_list(side: str, entry: float, levels: Dict[str, Optional[LevelCluster]], fib_context: Dict[str, object], signal_title: str) -> List[dict]:
    support = levels.get("support_near") or levels.get("support_strong")
    resistance = levels.get("resistance_near") or levels.get("resistance_strong")
    fib_prices = sorted([float(f.price) for f in fib_context["levels"]])
    fib_above = [p for p in fib_prices if p > entry]
    fib_below = [p for p in fib_prices if p < entry]

    def _pack(category: str, duration: str, stop: float, target: float):
        if side == "Buy":
            stop_pct = (entry - stop) / max(entry, 1e-9) * 100
            take_pct = (target - entry) / max(entry, 1e-9) * 100
            rr = max((target - entry) / max(entry - stop, 1e-9), 0.0)
        else:
            stop_pct = (stop - entry) / max(entry, 1e-9) * 100
            take_pct = (entry - target) / max(entry, 1e-9) * 100
            rr = max((entry - target) / max(stop - entry, 1e-9), 0.0)
        return {
            "category": category,
            "duration": duration,
            "entry": entry,
            "stop_loss": stop,
            "stop_pct": stop_pct,
            "take_profit": target,
            "take_pct": take_pct,
            "risk_reward": rr,
            "signal": signal_title,
            "side": side,
        }

    setups = []
    if side == "Buy":
        stop_1 = support.level * 0.997 if support is not None and support.level < entry else entry * 0.985
        stop_2 = support.level * 0.994 if support is not None and support.level < entry else entry * 0.970
        stop_3 = support.level * 0.990 if support is not None and support.level < entry else entry * 0.940
        tgt_1 = fib_above[0] if fib_above else (resistance.level if resistance is not None and resistance.level > entry else entry * 1.03)
        tgt_2 = fib_above[1] if len(fib_above) > 1 else (resistance.level if resistance is not None and resistance.level > entry else entry * 1.07)
        tgt_3 = fib_above[-1] if fib_above else (resistance.level * 1.02 if resistance is not None and resistance.level > entry else entry * 1.12)
    else:
        stop_1 = resistance.level * 1.003 if resistance is not None and resistance.level > entry else entry * 1.015
        stop_2 = resistance.level * 1.006 if resistance is not None and resistance.level > entry else entry * 1.030
        stop_3 = resistance.level * 1.010 if resistance is not None and resistance.level > entry else entry * 1.060
        tgt_1 = fib_below[-1] if fib_below else (support.level if support is not None and support.level < entry else entry * 0.97)
        tgt_2 = fib_below[-2] if len(fib_below) > 1 else (support.level if support is not None and support.level < entry else entry * 0.93)
        tgt_3 = fib_below[0] if fib_below else (support.level * 0.98 if support is not None and support.level < entry else entry * 0.88)

    setups.append(_pack("Intraday", "0-1 hari", stop_1, tgt_1))
    setups.append(_pack("Swing Pendek", "3-10 hari", stop_2, tgt_2))
    setups.append(_pack("Swing Panjang", "2-8 minggu", stop_3, tgt_3))
    return setups


def build_buy_sell_panels_figure(df: pd.DataFrame, interval: str, period: str, selected_filters: Dict[str, bool]):
    include_macd = selected_filters.get("macd", False)
    include_rsi = selected_filters.get("rsi", False)
    include_stoch = selected_filters.get("stoch", False)

    row_heights = [0.52, 0.18]
    row_map = {"price": 1, "volume": 2}
    next_row = 3
    if include_macd:
        row_map["macd"] = next_row
        row_heights.append(0.16)
        next_row += 1
    if include_rsi:
        row_map["rsi"] = next_row
        row_heights.append(0.14)
        next_row += 1
    if include_stoch:
        row_map["stoch"] = next_row
        row_heights.append(0.14)
        next_row += 1

    fig = make_subplots(rows=len(row_heights), cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

    x_vals = np.arange(len(df))
    custom_dt = [
        ts.strftime("%d %b %Y %H:%M") if is_intraday_interval(interval) else ts.strftime("%d %b %Y")
        for ts in df.index
    ]

    fig.add_trace(
        go.Candlestick(
            x=x_vals, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], customdata=custom_dt,
            increasing_line_color=GREEN, increasing_fillcolor=GREEN, decreasing_line_color=RED, decreasing_fillcolor=RED,
            whiskerwidth=0.5,
            hovertemplate="<b>%{customdata}</b><br>Open: %{open}<br>High: %{high}<br>Low: %{low}<br>Close: %{close}<extra></extra>",
            name="Price",
        ),
        row=1, col=1
    )
    if selected_filters.get("ma", False):
        fig.add_trace(
            go.Scatter(x=x_vals, y=df["EMA10"], mode="lines", line=dict(color=MA_COLORS[10], width=2.1), text=custom_dt,
                       hovertemplate="<b>%{text}</b><br>EMA10: %{y:.2f}<extra></extra>", name="EMA10"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_vals, y=df["EMA20"], mode="lines", line=dict(color=MA_COLORS[20], width=2.1), text=custom_dt,
                       hovertemplate="<b>%{text}</b><br>EMA20: %{y:.2f}<extra></extra>", name="EMA20"),
            row=1, col=1
        )
    bar_colors = [GREEN if c >= o else RED for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(x=x_vals, y=df["Volume"], marker_color=bar_colors, width=0.85, opacity=0.9, customdata=custom_dt,
               hovertemplate="<b>%{customdata}</b><br>Volume: %{y}<extra></extra>", name="Volume"),
        row=row_map["volume"], col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=df["VMA20"], mode="lines", line=dict(color=MA_VOL_COLOR, width=1.8), text=custom_dt,
                   hovertemplate="<b>%{text}</b><br>VMA20: %{y}<extra></extra>", name="VMA20"),
        row=row_map["volume"], col=1
    )

    if include_macd:
        hist_colors = [GREEN if h >= 0 else RED for h in df["MACD_HIST"]]
        fig.add_trace(go.Bar(x=x_vals, y=df["MACD_HIST"], marker_color=hist_colors, opacity=0.82, customdata=custom_dt,
                             hovertemplate="<b>%{customdata}</b><br>MACD Hist: %{y:.3f}<extra></extra>", name="MACD Hist"),
                      row=row_map["macd"], col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=df["MACD"], mode="lines", line=dict(color=MACD_COLOR, width=2.2), text=custom_dt,
                                 hovertemplate="<b>%{text}</b><br>MACD: %{y:.3f}<extra></extra>", name="MACD"),
                      row=row_map["macd"], col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=df["MACD_SIGNAL"], mode="lines", line=dict(color=SIGNAL_COLOR, width=2.0), text=custom_dt,
                                 hovertemplate="<b>%{text}</b><br>Signal: %{y:.3f}<extra></extra>", name="Signal"),
                      row=row_map["macd"], col=1)

    if include_rsi:
        fig.add_trace(go.Scatter(x=x_vals, y=df["RSI"], mode="lines", line=dict(color=RSI_COLOR, width=2.2), text=custom_dt,
                                 hovertemplate="<b>%{text}</b><br>RSI: %{y:.2f}<extra></extra>", name="RSI"),
                      row=row_map["rsi"], col=1)
        fig.add_hline(y=70, line_color=RED, line_width=1.1, line_dash="dash", row=row_map["rsi"], col=1)
        fig.add_hline(y=30, line_color=GREEN, line_width=1.1, line_dash="dash", row=row_map["rsi"], col=1)

    if include_stoch:
        fig.add_trace(go.Scatter(x=x_vals, y=df["STOCH_RSI_K"], mode="lines", line=dict(color="#0057B8", width=2.0), text=custom_dt,
                                 hovertemplate="<b>%{text}</b><br>Stoch RSI %K: %{y:.2f}<extra></extra>", name="Stoch RSI %K"),
                      row=row_map["stoch"], col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=df["STOCH_RSI_D"], mode="lines", line=dict(color="#8C52FF", width=1.9), text=custom_dt,
                                 hovertemplate="<b>%{text}</b><br>Stoch RSI %D: %{y:.2f}<extra></extra>", name="Stoch RSI %D"),
                      row=row_map["stoch"], col=1)
        fig.add_hline(y=80, line_color=RED, line_width=1.0, line_dash="dash", row=row_map["stoch"], col=1)
        fig.add_hline(y=20, line_color=GREEN, line_width=1.0, line_dash="dash", row=row_map["stoch"], col=1)

    low_price = float(df["Low"].min())
    high_price = float(df["High"].max())
    y_min, y_max = locked_axis_bounds(low_price, high_price, floor_zero=True)
    _, volume_max = locked_axis_bounds(0.0, float(df["Volume"].max()), floor_zero=True)

    tickvals, ticktext = choose_tick_positions(df.index, interval, period)
    xr = x_range_for_chart(df, interval)
    fig.update_xaxes(
        tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=-28, showgrid=True, gridcolor="#D3D3D3",
        rangeslider_visible=False, showspikes=True, spikecolor="#555555", spikethickness=1,
        tickfont=dict(color="#111111", size=12), linecolor="#111111", zeroline=False
    )
    for row in range(1, len(row_heights)+1):
        fig.update_xaxes(range=xr, row=row, col=1)

    fig.update_yaxes(range=[y_min, y_max], minallowed=0, maxallowed=y_max, row=row_map["price"], col=1, side="right",
                     title_text="Price", tickfont=dict(size=13, color="#111111"), title_font=dict(size=13, color="#111111"),
                     showgrid=True, gridcolor="#D3D3D3", linecolor="#111111")
    fig.update_yaxes(range=[0, volume_max], minallowed=0, maxallowed=volume_max, row=row_map["volume"], col=1, side="right",
                     title_text="Volume", tickfont=dict(size=12, color="#111111"), title_font=dict(size=12, color="#111111"),
                     showgrid=True, gridcolor="#D3D3D3", linecolor="#111111")

    if include_macd:
        macd_min = float(np.nanmin([df["MACD"].min(), df["MACD_SIGNAL"].min(), df["MACD_HIST"].min()]))
        macd_max = float(np.nanmax([df["MACD"].max(), df["MACD_SIGNAL"].max(), df["MACD_HIST"].max()]))
        macd_low, macd_high = locked_axis_bounds(macd_min, macd_max, floor_zero=False)
        fig.update_yaxes(range=[macd_low, macd_high], minallowed=macd_low, maxallowed=macd_high,
                         row=row_map["macd"], col=1, side="right", title_text="MACD",
                         tickfont=dict(size=12, color="#111111"), title_font=dict(size=12, color="#111111"),
                         showgrid=True, gridcolor="#D3D3D3", linecolor="#111111", zeroline=True, zerolinecolor="#777777")
    if include_rsi:
        fig.update_yaxes(range=[0, 100], minallowed=0, maxallowed=100,
                         row=row_map["rsi"], col=1, side="right", title_text="RSI",
                         tickfont=dict(size=12, color="#111111"), title_font=dict(size=12, color="#111111"),
                         showgrid=True, gridcolor="#D3D3D3", linecolor="#111111")
    if include_stoch:
        fig.update_yaxes(range=[0, 100], minallowed=0, maxallowed=100,
                         row=row_map["stoch"], col=1, side="right", title_text="Stoch RSI",
                         tickfont=dict(size=12, color="#111111"), title_font=dict(size=12, color="#111111"),
                         showgrid=True, gridcolor="#D3D3D3", linecolor="#111111")

    fig.add_hline(y=float(df["Close"].iloc[-1]), line_color=BLACK, line_width=1.2, line_dash="dash", row=1, col=1)
    fig.update_layout(dragmode="pan", hovermode="x unified", showlegend=False, plot_bgcolor=BG_COLOR, paper_bgcolor="white",
                      margin=dict(l=8, r=18, t=8, b=8), height=880 + 170 * max(0, len(row_heights)-2), font=dict(color="#111111", size=13))
    return fig, row_map, y_min, y_max


def build_buy_sell_cards_and_chart(df: pd.DataFrame, interval: str, period: str, selected_filters: Dict[str, bool]) -> Tuple[go.Figure, str, List[SignalCard], List[dict]]:
    if not any(bool(v) for v in selected_filters.values()):
        return build_empty_chart("Pilih checklist strategi untuk menampilkan chart dan sinyal."), detect_trend(df), [], []
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    levels = choose_key_levels(df, high_idx, low_idx)
    trend = detect_trend(df)
    fib_context = calculate_fibonacci_context(df)
    candle_signals = [s for s in detect_pattern_signals(df, levels, trend, lookback=50) if s.idx >= max(0, len(df) - 50)]
    structures = [s for s in detect_structure_patterns(df, trend, lookback=70) if s.idx >= max(0, len(df) - 50)]
    fig, row_map, y_min, y_max = build_buy_sell_panels_figure(df, interval, period, selected_filters)
    y_span = y_max - y_min
    if selected_filters.get("fibo", False):
        add_fibonacci_lines(fig, df, fib_context, y_min, y_max)

    collected: List[dict] = []
    seen = set()
    macd_events, rsi_events, stoch_events = [], [], []
    full_width_lines: List[dict] = []
    ma_cross_events: List[dict] = []

    def push_event(idx: int, side: str, title: str, reason: str, label: str, category: str,
                   footer: Optional[str] = None, panel: Optional[str] = None, panel_y: Optional[float] = None,
                   line_price: Optional[float] = None, line_color: Optional[str] = None, line_label: Optional[str] = None,
                   marker_symbol: Optional[str] = None):
        if idx < max(0, len(df) - 50):
            return
        if category in selected_filters and not selected_filters.get(category, True):
            return
        entry = float(df["Close"].iloc[idx])
        risk, rr = compute_trade_metrics(side, entry, levels, trend)
        key = (idx, side, title, footer or "", category)
        if key in seen:
            return
        seen.add(key)
        tone = "positive" if side == "Buy" else "negative"
        card_note = f"{reason} • Risk: {risk} • Risk/Reward: {rr}"
        collected.append({
            "idx": idx,
            "side": side,
            "entry": entry,
            "title": title,
            "card": SignalCard(title, side, card_note, footer or format_idx_time(df, idx, interval), tone, idx),
            "label": {"idx": idx, "side": side, "label": label},
        })
        if panel and panel in row_map and panel_y is not None:
            panel_ev = {"idx": idx, "label": label, "y": panel_y, "color": GREEN if side == "Buy" else RED, "top": side != "Buy"}
            if panel == "macd":
                macd_events.append(panel_ev)
            elif panel == "rsi":
                rsi_events.append(panel_ev)
            elif panel == "stoch":
                stoch_events.append(panel_ev)
        if line_price is not None:
            full_width_lines.append({"price": float(line_price), "color": line_color or (GREEN if side == "Buy" else RED), "label": line_label or f"{float(line_price):.2f}"})
        if marker_symbol:
            ma_cross_events.append({"idx": idx, "price": entry, "symbol": marker_symbol, "color": GREEN if side == "Buy" else RED, "text": label})

    start = max(1, len(df) - 50)
    last_idx = len(df) - 1
    last_close = float(df["Close"].iloc[-1])
    support = levels.get("support_near") or levels.get("support_strong")
    resistance = levels.get("resistance_near") or levels.get("resistance_strong")
    if support is not None and abs(last_close - support.level) / max(last_close, 1e-9) <= 0.02:
        push_event(last_idx, "Buy", "Near Support", "harga sedang dekat area support penting", "BUY Support", "candle")
    if resistance is not None and abs(resistance.level - last_close) / max(last_close, 1e-9) <= 0.02:
        push_event(last_idx, "Sell", "Near Resistance", "harga sedang dekat area resistance penting", "SELL Resist", "candle")

    for sig in candle_signals:
        reason = f"{sig.name} • potensi {sig.bias} {sig.probability * 100:.0f}% • {sig.reason}"
        push_event(sig.idx, "Buy" if sig.bias == "naik" else "Sell", sig.name, reason,
                   f"{'BUY' if sig.bias == 'naik' else 'SELL'} {sig.name}", "candle",
                   format_idx_range(df, sig.start_idx, sig.end_idx, interval))

    for sig in structures:
        reason = f"{sig.name} • potensi {sig.bias} {sig.probability * 100:.0f}% • {sig.reason}"
        push_event(sig.idx, "Buy" if sig.bias == "naik" else "Sell", sig.name, reason,
                   f"{'BUY' if sig.bias == 'naik' else 'SELL'} Pattern", "pattern",
                   format_idx_range(df, sig.start_idx, sig.end_idx, interval))

    rolling_high = df["High"].rolling(20).max().shift(1)
    rolling_low = df["Low"].rolling(20).min().shift(1)

    for i in range(start, len(df)):
        prev_short = float(df["MA10"].iloc[i - 1]); prev_long = float(df["MA20"].iloc[i - 1])
        curr_short = float(df["MA10"].iloc[i]); curr_long = float(df["MA20"].iloc[i])
        if not np.isnan([prev_short, prev_long, curr_short, curr_long]).any():
            if prev_short <= prev_long and curr_short > curr_long:
                push_event(i, "Buy", "MA Cross", "EMA10 memotong EMA20 ke atas", "BUY EMA10/20", "ma", marker_symbol="cross")
            if prev_short >= prev_long and curr_short < curr_long:
                push_event(i, "Sell", "MA Cross", "EMA10 memotong EMA20 ke bawah", "SELL EMA10/20", "ma", marker_symbol="cross")

        prev_rsi = float(df["RSI"].iloc[i - 1]); curr_rsi = float(df["RSI"].iloc[i])
        if prev_rsi > 30 and curr_rsi <= 30:
            push_event(i, "Buy", "RSI Oversold", f"RSI masuk area oversold di {curr_rsi:.1f}", "BUY Oversold", "rsi",
                       panel="rsi", panel_y=curr_rsi)
        if prev_rsi < 70 and curr_rsi >= 70:
            push_event(i, "Sell", "RSI Overbought", f"RSI masuk area overbought di {curr_rsi:.1f}", "SELL Overbought", "rsi",
                       panel="rsi", panel_y=curr_rsi)

        prev_macd = float(df["MACD"].iloc[i - 1]); prev_sig = float(df["MACD_SIGNAL"].iloc[i - 1])
        curr_macd = float(df["MACD"].iloc[i]); curr_sig = float(df["MACD_SIGNAL"].iloc[i])
        if prev_macd <= prev_sig and curr_macd > curr_sig:
            push_event(i, "Buy", "MACD Cross", "MACD memotong signal ke atas", "BUY MACD", "macd",
                       panel="macd", panel_y=curr_macd)
        if prev_macd >= prev_sig and curr_macd < curr_sig:
            push_event(i, "Sell", "MACD Cross", "MACD memotong signal ke bawah", "SELL MACD", "macd",
                       panel="macd", panel_y=curr_macd)

        prev_k = float(df["STOCH_RSI_K"].iloc[i - 1]); curr_k = float(df["STOCH_RSI_K"].iloc[i])
        prev_d = float(df["STOCH_RSI_D"].iloc[i - 1]); curr_d = float(df["STOCH_RSI_D"].iloc[i])
        if prev_k <= prev_d and curr_k > curr_d and curr_k <= 25:
            push_event(i, "Buy", "Stochastic Rebound", f"Stoch RSI bullish cross di area rendah ({curr_k:.1f})", "BUY Stoch", "stoch",
                       panel="stoch", panel_y=curr_k)
        if prev_k >= prev_d and curr_k < curr_d and curr_k >= 75:
            push_event(i, "Sell", "Stochastic Cooling", f"Stoch RSI bearish cross di area tinggi ({curr_k:.1f})", "SELL Stoch", "stoch",
                       panel="stoch", panel_y=curr_k)

        candle_close = float(df["Close"].iloc[i])
        candle_open = float(df["Open"].iloc[i])
        for fib in fib_context["levels"]:
            distance = abs(candle_close - fib.price) / max(candle_close, 1e-9)
            if distance <= 0.012:
                if candle_close >= candle_open:
                    push_event(i, "Buy", f"Near Fibo {fib.ratio_label}", f"harga memantul dekat area Fibonacci {fib.short_note.lower()}",
                               f"BUY Fibo {fib.ratio_label.split()[0]}", "fibo")
                else:
                    push_event(i, "Sell", f"Near Fibo {fib.ratio_label}", f"harga tertahan dekat area Fibonacci {fib.short_note.lower()}",
                               f"SELL Fibo {fib.ratio_label.split()[0]}", "fibo")
                break

        if not pd.isna(rolling_high.iloc[i]) and float(df["Close"].iloc[i]) > float(rolling_high.iloc[i]) and float(df["Volume"].iloc[i]) >= float(df["VMA20"].iloc[i]):
            push_event(i, "Buy", "Breakout", "close menembus high 20 candle dengan volume di atas rata-rata", "BUY Breakout", "breakout",
                       line_price=float(rolling_high.iloc[i]), line_color=GREEN, line_label=f"Breakout {float(rolling_high.iloc[i]):.2f}")
        if not pd.isna(rolling_low.iloc[i]) and float(df["Close"].iloc[i]) < float(rolling_low.iloc[i]) and float(df["Volume"].iloc[i]) >= float(df["VMA20"].iloc[i]):
            push_event(i, "Sell", "Breakdown", "close menembus low 20 candle dengan volume tinggi", "SELL Breakdown", "breakdown",
                       line_price=float(rolling_low.iloc[i]), line_color=RED, line_label=f"Breakdown {float(rolling_low.iloc[i]):.2f}")

    bull_rsi, bear_rsi = detect_series_divergences(df, "RSI")
    bull_macd, bear_macd = detect_series_divergences(df, "MACD")
    if bull_rsi is not None:
        push_event(int(bull_rsi["idx"]), "Buy", "Bullish Divergence RSI", bull_rsi["note"], "BUY RSI Div", "rsi",
                   panel="rsi", panel_y=float(df["RSI"].iloc[int(bull_rsi["idx"])]))
    if bear_rsi is not None:
        push_event(int(bear_rsi["idx"]), "Sell", "Bearish Divergence RSI", bear_rsi["note"], "SELL RSI Div", "rsi",
                   panel="rsi", panel_y=float(df["RSI"].iloc[int(bear_rsi["idx"])]))
    if bull_macd is not None:
        push_event(int(bull_macd["idx"]), "Buy", "Bullish Divergence MACD", bull_macd["note"], "BUY MACD Div", "macd",
                   panel="macd", panel_y=float(df["MACD"].iloc[int(bull_macd["idx"])]))
    if bear_macd is not None:
        push_event(int(bear_macd["idx"]), "Sell", "Bearish Divergence MACD", bear_macd["note"], "SELL MACD Div", "macd",
                   panel="macd", panel_y=float(df["MACD"].iloc[int(bear_macd["idx"])]))

    collected = sorted(collected, key=lambda x: x["idx"])
    cards = [item["card"] for item in collected]
    seen_lines = set()
    for line in full_width_lines:
        key = (round(line["price"], 6), line["color"])
        if key in seen_lines:
            continue
        seen_lines.add(key)
        fig.add_shape(type="line", x0=0, x1=len(df) - 1, y0=line["price"], y1=line["price"], xref="x", yref="y", line=dict(color=line["color"], width=1.8, dash="solid"))
        fig.add_annotation(x=len(df) - 1, y=line["price"], xref="x", yref="y", text=html.escape(line.get("label", f"{line['price']:.2f}")),
                           showarrow=False, xanchor="left", yanchor="middle", xshift=10,
                           font=dict(size=11, color=line["color"]), bgcolor="rgba(255,255,255,0.92)",
                           bordercolor=line["color"], borderwidth=1)
    add_price_event_labels(fig, df, [item["label"] for item in collected], y_span)
    if ma_cross_events:
        fig.add_trace(go.Scatter(
            x=[ev["idx"] for ev in ma_cross_events],
            y=[ev["price"] for ev in ma_cross_events],
            mode="markers+text",
            marker=dict(symbol="cross", size=14, color=[ev["color"] for ev in ma_cross_events], line=dict(width=2, color=[ev["color"] for ev in ma_cross_events])),
            text=["+" for _ in ma_cross_events],
            textposition="middle center",
            textfont=dict(size=14, color="white"),
            hovertext=[ev["text"] for ev in ma_cross_events],
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
            name="EMA Cross"
        ), row=1, col=1)
    if "macd" in row_map:
        add_panel_event_labels(fig, macd_events, row=row_map["macd"])
    if "rsi" in row_map:
        add_panel_event_labels(fig, rsi_events, row=row_map["rsi"])
    if "stoch" in row_map:
        add_panel_event_labels(fig, stoch_events, row=row_map["stoch"])

    latest = collected[-1] if collected else {"side": "Buy" if trend != "Downtrend" else "Sell", "entry": last_close, "title": trend or "Setup"}
    side = latest["side"]
    entry = float(latest.get("entry", last_close))
    setups = build_trade_setups_list(side, entry, levels, fib_context, latest.get("title", trend or "Setup"))
    return fig, trend, cards, setups




@st.cache_data(ttl=3600, show_spinner=False)
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", {}) or {}
        mcap = fi.get("market_cap") or fi.get("marketCap")
        if mcap is None:
            info = getattr(t, "info", {}) or {}
            mcap = info.get("marketCap", 0)
        return float(mcap or 0)
    except Exception:
        return 0.0


def parse_screener_universe(raw: str) -> List[str]:
    cleaned = raw.strip().upper()
    if not cleaned or cleaned in {"IHSG", "ISSI", "SYARIAH"}:
        return ISSI_LIQUID_UNIVERSE
    parts = re.split(r"[,\s;]+", cleaned)
    universe = []
    for p in parts:
        if not p:
            continue
        universe.append(normalize_symbol(p))
    return list(dict.fromkeys(universe))


@st.cache_data(ttl=1800, show_spinner=False)
def run_screener(preset: str, interval: str, period: str, raw_universe: str) -> Tuple[List[dict], List[str], str]:
    universe = parse_screener_universe(raw_universe)
    rows: List[dict] = []
    notes: List[str] = []
    eff_period = period
    for symbol in universe:
        df, note, eff = load_data(symbol, interval, period)
        eff_period = eff
        if note and note not in notes:
            notes.append(note)
        if df.empty or len(df) < 25:
            continue

        close = float(df["Close"].iloc[-1]); prev_close = float(df["Close"].iloc[-2])
        vol = float(df["Volume"].iloc[-1]); prev_vol = float(df["Volume"].iloc[-2]) if len(df) >= 2 else vol
        vol_ma20 = float(df["Volume"].tail(20).mean()) if len(df) >= 20 else float(df["Volume"].mean())
        value_now = close * vol
        value_ma20 = float((df["Close"].tail(20) * df["Volume"].tail(20)).mean()) if len(df) >= 20 else float((df["Close"] * df["Volume"]).mean())
        ma5 = float(df["Close"].rolling(5).mean().iloc[-1]) if len(df) >= 5 else float(df["Close"].mean())
        ma20 = float(df["MA20"].iloc[-1]) if not pd.isna(df["MA20"].iloc[-1]) else float(df["Close"].rolling(20).mean().iloc[-1])
        ma200 = float(df["MA200"].iloc[-1]) if not pd.isna(df["MA200"].iloc[-1]) else np.nan
        prev_ma200 = float(df["MA200"].iloc[-2]) if len(df) >= 2 and not pd.isna(df["MA200"].iloc[-2]) else np.nan
        ret1 = (close / max(prev_close, 1e-9) - 1.0) * 100.0
        mcap = get_market_cap(symbol)

        passed = False
        score = 0.0
        risk = "High Risk"
        why = ""

        if preset == "Breakout MA 200":
            if not np.isnan(ma200) and not np.isnan(prev_ma200):
                passed = (
                    close > max(ma200, 100)
                    and prev_close <= prev_ma200
                    and value_ma20 > 15_000_000_000
                    and vol_ma20 > 500_000
                )
                if passed:
                    vol_mult = vol / max(vol_ma20, 1.0)
                    score = min(100, 55 + ret1 * 4 + max(vol_mult - 1, 0) * 18)
                    risk = "Low Risk" if ret1 > 0 and vol_mult >= 1.2 else "High Risk"
                    why = "breakout di atas MA200 dengan likuiditas memadai"

        elif preset == "BPJS":
            passed = (
                close > 100
                and close >= prev_close
                and close >= ma20
                and ret1 > 0
                and value_ma20 > 30_000_000_000
                and vol_ma20 > 1_000_000
                and vol >= 1.5 * vol_ma20
                and mcap > 1_000_000_000_000
            )
            if passed:
                score = min(100, 60 + ret1 * 6 + (vol / max(vol_ma20, 1.0) - 1.5) * 12)
                risk = "Low Risk" if close >= ma20 * 1.01 and ret1 > 1 else "High Risk"
                why = "momentum pagi aktif: harga di atas MA20, return harian positif, volume meledak, market cap besar"

        elif preset == "BSJP":
            passed = (
                close >= 1.05 * prev_close
                and close >= ma5
                and vol >= 1.2 * prev_vol
                and value_now > 5_000_000_000
                and ret1 > 1.0
            )
            if passed:
                score = min(100, 58 + ret1 * 5 + (vol / max(prev_vol, 1.0) - 1.2) * 20)
                risk = "Low Risk" if close >= ma5 * 1.01 and ret1 > 2 else "High Risk"
                why = "closing strength kuat; filter frequency > 8000 belum bisa diterapkan dari data Yahoo Finance"

        if passed:
            rows.append({
                "symbol": display_symbol(symbol),
                "score": max(0.0, round(score, 1)),
                "risk": risk,
                "close": close,
                "ret1": ret1,
                "why": why,
            })
    rows = sorted(rows, key=lambda x: (x["score"], x["ret1"]), reverse=True)
    return rows, notes, eff_period


def render_screener_result_card(title: str, rows: List[dict], note: str = ""):
    if not rows:
        body = '<div class="signal-item">Belum ada saham yang lolos preset ini pada universe ISSI liquid saat ini.</div>'
    else:
        items = []
        for row in rows[:18]:
            risk_class = "signal-up" if row["risk"] == "Low Risk" else "signal-down"
            items.append(
                f'<div class="signal-item"><b>{row["symbol"]}</b> • Score <b>{row["score"]:.1f}</b> • '
                f'<span class="{risk_class}">{row["risk"]}</span> • Close {row["close"]:.2f} • '
                f'Return 1D {row["ret1"]:.2f}%<br><span style="color:#BFD3E8;">{html.escape(row["why"])}</span></div>'
            )
        body = "".join(items)

    st.markdown(
        f"""
        <div class="signal-box">
            <div class="signal-title">{html.escape(title)}</div>
            {body}
            {'<div class="info-note">' + html.escape(note) + '</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# UI helpers
# ----------------------------

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
        .sr-card {{
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            background: {CARD_BG};
            padding: 14px 16px;
            min-height: 132px;
        }}
        .sr-card.signal-card {{
            min-height: 176px;
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
            line-height: 1.38;
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
        .trend-up {{ background: rgba(11,163,74,0.20); color: #57FF8E; }}
        .trend-down {{ background: rgba(220,38,38,0.18); color: #FF8D8D; }}
        .trend-side {{ background: rgba(255,255,255,0.10); color: #D5DCE5; }}
        .risk-high {{ background: rgba(220,38,38,0.18); color: #FF8D8D; }}
        .risk-low {{ background: rgba(11,163,74,0.20); color: #57FF8E; }}
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
        .trade-setup-card {{
            border: 1px solid rgba(23,48,77,0.95);
            border-radius: 14px;
            background: #07111F;
            padding: 16px 18px;
            min-height: 172px;
            margin-bottom: 10px;
        }}
        .trade-setup-title {{
            color: white;
            font-size: 18px;
            font-weight: 900;
            margin-bottom: 10px;
        }}
        .trade-setup-sub {{
            color: #8EC5FF;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 8px;
        }}
        .trade-setup-row {{
            color: #DCE7F3;
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 4px;
        }}
        .metric-up {{ color:#57FF8E; font-weight:800; }}
        .metric-down {{ color:#FF8D8D; font-weight:800; }}
        .metric-neutral {{ color:#DCE7F3; font-weight:800; }}
        .news-card {{
            border: 1px solid rgba(23,48,77,0.95);
            border-radius: 14px;
            background: #07111F;
            padding: 14px 16px;
            min-height: 108px;
            margin-bottom: 12px;
        }}
        .news-card.positive {{ border-color: rgba(19,137,67,0.95); box-shadow: inset 0 0 0 1px rgba(52,211,102,0.12); }}
        .news-card.negative {{ border-color: rgba(170,34,34,0.96); box-shadow: inset 0 0 0 1px rgba(255,94,94,0.10); }}
        .news-card.neutral {{ border-color: rgba(23,48,77,0.95); }}
        .news-title a {{
            color: white;
            text-decoration: none;
            font-size: 16px;
            font-weight: 800;
            line-height: 1.35;
        }}
        .news-title a:hover {{ color: #8EC5FF; text-decoration: underline; }}
        .news-meta {{
            color: #BFD3E8;
            font-size: 13px;
            margin-top: 10px;
            line-height: 1.4;
        }}
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
        }}
        .export-align {{
            height: 26px;
        }}
        @media (max-width: 768px) {{
            .sr-card {{ padding: 10px 10px; min-height: 110px; }}
            .sr-card.signal-card {{ min-height: 154px; }}
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


def inject_symbol_input_script(current_value: str):
    js_value = current_value.replace("\\", "\\\\").replace("'", "\\'")
    components.html(
        f"""
        <script>
        const defaultValue = '{js_value}';
        function bindInput() {{
            const doc = window.parent.document;
            const inputs = Array.from(doc.querySelectorAll('input'));
            const input = inputs.find(el => el.getAttribute('aria-label') === 'Masukkan kode saham IDX');
            if (!input || input.dataset.boundFocusClear === '1') return;
            input.dataset.boundFocusClear = '1';
            input.addEventListener('focus', () => {{
                if (input.value === defaultValue) {{
                    input.dataset.prevValue = defaultValue;
                    input.value = '';
                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }});
            input.addEventListener('blur', () => {{
                if (!input.value.trim()) {{
                    const restore = input.dataset.prevValue || defaultValue;
                    input.value = restore;
                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }});
            input.addEventListener('keydown', (e) => {{
                if (e.key === 'Enter') {{
                    setTimeout(() => input.blur(), 20);
                }}
            }});
        }}
        const interval = setInterval(bindInput, 300);
        setTimeout(() => clearInterval(interval), 7000);
        bindInput();
        </script>
        """,
        height=0,
    )


def render_sr_card(col, title: str, cluster: Optional[LevelCluster]):
    touches = cluster.count if cluster is not None else 0
    value = value_formatter(cluster.level) if cluster is not None else "-"
    with col:
        st.markdown(
            f"""
            <div class="sr-card">
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
    footer_html = card.footer if getattr(card, "footer", "") not in {"", None} else "&nbsp;"
    tone = getattr(card, "tone", "neutral")
    if tone == "positive":
        title_color = "#49F07C"
        value_color = "#49F07C"
        border_style = "border:1px solid rgba(73,240,124,0.34); box-shadow: inset 0 0 0 1px rgba(73,240,124,0.08);"
    elif tone == "negative":
        title_color = "#FF7D7D"
        value_color = "#FF7D7D"
        border_style = "border:1px solid rgba(255,125,125,0.34); box-shadow: inset 0 0 0 1px rgba(255,125,125,0.08);"
    else:
        title_color = "white"
        value_color = "white"
        border_style = ""
    with col:
        st.markdown(
            f"""
            <div class="sr-card signal-card" style="{border_style}">
                <div class="sr-card-title" style="color:{title_color};">{card.title}</div>
                <div class="sr-card-value" style="color:{value_color};">{value_html}</div>
                <div class="sr-card-note">{note_html}</div>
                <div class="info-note" style="margin-top:10px;">{footer_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_card_grid(cards: List[SignalCard], per_row: int = 4):
    if not cards:
        cols = st.columns(per_row)
        render_signal_card(cols[0], SignalCard("Tidak ada sinyal", "-", "Belum ada sinyal yang cukup kuat.", ""))
        return
    for start in range(0, len(cards), per_row):
        row_cards = cards[start : start + per_row]
        cols = st.columns(per_row)
        for col, card in zip(cols, row_cards):
            render_signal_card(col, card)


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
        f'<div class="info-sub">TF: {html.escape(tf_label)} | Periode: {html.escape(duration_label)} | Last Close: {last_close:.2f}</div>',
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




def render_pattern_summary(df: pd.DataFrame, interval: str, signals: List[PatternSignal], structures: Optional[List[PatternSignal]] = None):
    candle_rows = []
    for sig in signals[::-1]:
        bias_class = "signal-up" if sig.bias == "naik" else "signal-down"
        when = format_idx_range(df, sig.start_idx, sig.end_idx, interval)
        candle_rows.append(
            f'<div class="signal-item"><b>{sig.name}</b> • {when} • '
            f'<span class="{bias_class}">potensi {sig.bias} {sig.probability * 100:.0f}%</span><br>'
            f'<span style="color:#BFD3E8;">{html.escape(sig.reason)}</span></div>'
        )

    if not candle_rows:
        candle_rows.append('<div class="signal-item">Belum ada sinyal candle penting yang cukup jelas di area terbaru.</div>')

    structure_rows = []
    for sig in (structures or []):
        bias_class = "signal-up" if sig.bias == "naik" else "signal-down"
        when = format_idx_range(df, sig.start_idx, sig.end_idx, interval)
        structure_rows.append(
            f'<div class="signal-item"><b>{sig.name}</b> • {when} • '
            f'<span class="{bias_class}">potensi {sig.bias} {sig.probability * 100:.0f}%</span><br>'
            f'<span style="color:#BFD3E8;">{html.escape(sig.reason)}</span></div>'
        )

    if not structure_rows:
        structure_rows.append('<div class="signal-item">Belum ada pola candle multi-candle yang cukup tegas.</div>')

    with st.expander("Sinyal candle penting", expanded=False):
        st.markdown(f'<div class="signal-box">{"".join(candle_rows)}</div>', unsafe_allow_html=True)
    with st.expander("Pola candle", expanded=False):
        st.markdown(f'<div class="signal-box">{"".join(structure_rows)}</div>', unsafe_allow_html=True)



def build_export_text(df: pd.DataFrame, interval: str) -> str:
    intraday = is_intraday_interval(interval)
    rows = ["Tanggal-Waktu	Open	High	Low	Close	Volume"]
    for ts, row in df.iterrows():
        dt_txt = ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d")
        rows.append(
            f"{dt_txt}	{float(row['Open']):.2f}	{float(row['High']):.2f}	{float(row['Low']):.2f}	{float(row['Close']):.2f}	{int(row['Volume'])}"
        )
    return "\n".join(rows)


def render_trade_setup_card(setups: List[dict]):
    if not setups:
        return
    cols = st.columns(min(3, max(1, len(setups))))
    for idx, setup in enumerate(setups):
        tone_color = "rgba(73,240,124,0.34)" if setup.get("side") == "Buy" else "rgba(255,125,125,0.34)"
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="trade-setup-card" style="border-color:{tone_color};">
                    <div class="trade-setup-title">TRADE SETUP</div>
                    <div class="trade-setup-sub">{html.escape(setup.get('category', 'Setup'))} • {html.escape(setup.get('duration', ''))}</div>
                    <div class="trade-setup-row"><b>Entry</b> : {setup['entry']:.2f}</div>
                    <div class="trade-setup-row"><b>Stop Loss</b> : {setup['stop_loss']:.2f} ({setup['stop_pct']:.2f}%)</div>
                    <div class="trade-setup-row"><b>Take Profit</b> : {setup['take_profit']:.2f} ({setup['take_pct']:.2f}%)</div>
                    <div class="trade-setup-row"><b>Risk Reward</b> : 1 : {setup['risk_reward']:.2f}</div>
                    <div class="trade-setup-row"><b>Signal</b> : {html.escape(str(setup['signal']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_buy_sell_strategy_controls():
    with st.container(border=True):
        st.markdown("**Strategi ditampilkan**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.checkbox("MA cross", key="bs_filter_ma")
            st.checkbox("MACD", key="bs_filter_macd")
            st.checkbox("RSI", key="bs_filter_rsi")
        with c2:
            st.checkbox("Fibonacci", key="bs_filter_fibo")
            st.checkbox("Candle", key="bs_filter_candle")
            st.checkbox("Candle Pattern", key="bs_filter_pattern")
        with c3:
            st.checkbox("Breakout", key="bs_filter_breakout")
            st.checkbox("Breakdown", key="bs_filter_breakdown")
            st.checkbox("Stochastic", key="bs_filter_stoch")


def _get_buy_sell_filters() -> Dict[str, bool]:
    return {
        "ma": st.session_state.get("bs_filter_ma", False),
        "macd": st.session_state.get("bs_filter_macd", False),
        "rsi": st.session_state.get("bs_filter_rsi", False),
        "fibo": st.session_state.get("bs_filter_fibo", False),
        "candle": st.session_state.get("bs_filter_candle", False),
        "pattern": st.session_state.get("bs_filter_pattern", False),
        "breakout": st.session_state.get("bs_filter_breakout", False),
        "breakdown": st.session_state.get("bs_filter_breakdown", False),
        "stoch": st.session_state.get("bs_filter_stoch", False),
    }


def _strip_html(raw: str) -> str:

    clean = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", html.unescape(clean)).strip()


def _extract_thumb(raw: str) -> str:
    m = re.search(r"src=[\'\"]([^\'\"]+)[\'\"]", raw or "", re.I)
    return m.group(1) if m else ""


def _news_sentiment(text: str) -> str:
    sample = (text or "").lower()
    positive_words = ["naik", "tumbuh", "laba", "surge", "gain", "beat", "bullish", "dividen", "upgrade", "rebound", "optimistic", "strong"]
    negative_words = ["turun", "rugi", "drop", "fall", "bearish", "downgrade", "cut", "miss", "loss", "lawsuit", "risk", "selloff", "warning"]
    pos = sum(w in sample for w in positive_words)
    neg = sum(w in sample for w in negative_words)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"



@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_sentiment(symbol: str, display_code_value: str) -> List[NewsItem]:
    items: List[NewsItem] = []
    seen = set()
    query = quote_plus(f'"{display_code_value}" saham OR "{display_code_value}" IDX OR "{display_code_value}" BEI')
    url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urlopen(req, timeout=15).read()
        root = ET.fromstring(raw)
        for node in root.findall(".//item"):
            title = (node.findtext("title", "") or "").strip()
            link = (node.findtext("link", "") or "").strip()
            desc = node.findtext("description", "") or ""
            source_node = node.find("source")
            source = (source_node.text or "Google News") if source_node is not None else "Google News"
            pub_raw = node.findtext("pubDate", "") or ""
            sort_key = 0.0
            try:
                dt = parsedate_to_datetime(pub_raw).astimezone()
                sort_key = dt.timestamp()
                published = dt.strftime("%d %b %Y %H:%M")
            except Exception:
                published = pub_raw
            summary = _strip_html(desc)
            if len(summary) > 220:
                summary = summary[:217].rstrip() + "..."
            key = (title, link)
            if not title or not link or key in seen:
                continue
            seen.add(key)
            items.append(NewsItem(
                title=title,
                summary=summary,
                link=link,
                source=source,
                published=published,
                thumbnail=_extract_thumb(desc),
                sentiment=_news_sentiment(title + " " + summary),
                sort_key=sort_key,
            ))
        items = sorted(items, key=lambda x: x.sort_key, reverse=True)
    except Exception:
        return []
    return items[:20]


def render_news_section(news_items: List[NewsItem]):
    if not news_items:
        st.markdown(
            '<div class="signal-box"><div class="signal-title">Belum ada berita terbaru yang berhasil diambil.</div></div>',
            unsafe_allow_html=True,
        )
        return
    cols = st.columns(2)
    for idx, item in enumerate(news_items):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="news-card {item.sentiment}">
                    <div class="news-title"><a href="{html.escape(item.link)}" target="_blank">{html.escape(item.title)}</a></div>
                    <div class="news-meta">{html.escape(item.source)} • {html.escape(item.published)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_news_header(display_code_value: str):
    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">News &amp; Sentimen | {html.escape(display_code_value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def run_backtest_strategy(df: pd.DataFrame, selected_rules: Dict[str, bool]) -> Tuple[List[str], dict, List[dict]]:
    trend = detect_trend(df)
    high_idx, low_idx = find_pivots(df, left=3, right=3)
    levels = choose_key_levels(df, high_idx, low_idx)
    fib_context = calculate_fibonacci_context(df)
    fib_prices = sorted([float(level.price) for level in fib_context["levels"]])
    rolling_res = df["High"].rolling(20).max().shift(1)
    rolling_sup = df["Low"].rolling(20).min().shift(1)
    bullish_patterns = {sig.idx: sig for sig in detect_pattern_signals(df, levels, trend, lookback=min(len(df), 220)) if sig.bias == "naik"}
    bullish_struct = {sig.idx: sig for sig in detect_structure_patterns(df, trend, lookback=min(len(df), 220)) if sig.bias == "naik"}
    bearish_patterns = {sig.idx: sig for sig in detect_pattern_signals(df, levels, trend, lookback=min(len(df), 220)) if sig.bias == "turun"}
    bearish_struct = {sig.idx: sig for sig in detect_structure_patterns(df, trend, lookback=min(len(df), 220)) if sig.bias == "turun"}

    label_map = {
        "rsi_oversold": "RSI oversold",
        "macd_cross": "MACD cross",
        "ma_cross": "MA cross",
        "breakout_resistance": "Breakout resistance",
        "near_support": "Beli dekat support",
        "bullish_pattern": "Pola / candle bullish",
        "fibo": "Fibonacci",
        **{f"breakout_ma_{ma}": f"Breakout MA {ma}" for ma in MA_WINDOWS},
    }
    active_lines = [f"☑ {label}" for key, label in label_map.items() if selected_rules.get(key, False)]

    trades: List[dict] = []
    equity_curve = [1.0]
    open_pos = None

    for i in range(35, len(df)):
        bullish, bearish = [], []
        if selected_rules.get("rsi_oversold", False) and float(df["RSI"].iloc[i]) <= 35:
            bullish.append("RSI oversold")
        if float(df["RSI"].iloc[i]) >= 68:
            bearish.append("RSI overbought")

        if selected_rules.get("macd_cross", False):
            if float(df["MACD"].iloc[i - 1]) <= float(df["MACD_SIGNAL"].iloc[i - 1]) and float(df["MACD"].iloc[i]) > float(df["MACD_SIGNAL"].iloc[i]):
                bullish.append("MACD cross")
            if float(df["MACD"].iloc[i - 1]) >= float(df["MACD_SIGNAL"].iloc[i - 1]) and float(df["MACD"].iloc[i]) < float(df["MACD_SIGNAL"].iloc[i]):
                bearish.append("Bearish MACD cross")

        if selected_rules.get("ma_cross", False):
            if float(df["MA10"].iloc[i - 1]) <= float(df["MA20"].iloc[i - 1]) and float(df["MA10"].iloc[i]) > float(df["MA20"].iloc[i]):
                bullish.append("MA cross")
            if float(df["MA10"].iloc[i - 1]) >= float(df["MA20"].iloc[i - 1]) and float(df["MA10"].iloc[i]) < float(df["MA20"].iloc[i]):
                bearish.append("Bearish MA cross")

        if selected_rules.get("breakout_resistance", False):
            if not pd.isna(rolling_res.iloc[i]) and float(df["Close"].iloc[i]) > float(rolling_res.iloc[i]) and float(df["Volume"].iloc[i]) >= float(df["VMA20"].iloc[i]):
                bullish.append("Breakout resistance")

        if selected_rules.get("near_support", False):
            support_val = float(rolling_sup.iloc[i]) if not pd.isna(rolling_sup.iloc[i]) else np.nan
            if not np.isnan(support_val) and abs(float(df["Close"].iloc[i]) - support_val) / max(float(df["Close"].iloc[i]), 1e-9) <= 0.02:
                bullish.append("Beli dekat support")

        if selected_rules.get("bullish_pattern", False):
            if i in bullish_patterns or i in bullish_struct:
                bullish.append("Pola / candle bullish")
            if i in bearish_patterns or i in bearish_struct:
                bearish.append("Pola / candle bearish")

        if selected_rules.get("fibo", False):
            close_i = float(df["Close"].iloc[i])
            open_i = float(df["Open"].iloc[i])
            low_i = float(df["Low"].iloc[i])
            high_i = float(df["High"].iloc[i])
            for fib_price in fib_prices:
                dist = abs(close_i - fib_price) / max(close_i, 1e-9)
                if dist <= 0.012:
                    if close_i >= open_i and low_i <= fib_price <= close_i:
                        bullish.append(f"Fibonacci bounce {fib_price:.2f}")
                    if close_i <= open_i and high_i >= fib_price >= close_i:
                        bearish.append(f"Fibonacci rejection {fib_price:.2f}")
                    break

        for ma in MA_WINDOWS:
            key = f"breakout_ma_{ma}"
            if selected_rules.get(key, False):
                ma_col = f"MA{ma}"
                if float(df["Close"].iloc[i - 1]) <= float(df[ma_col].iloc[i - 1]) and float(df["Close"].iloc[i]) > float(df[ma_col].iloc[i]):
                    bullish.append(f"Breakout MA {ma}")
                if float(df["Close"].iloc[i - 1]) >= float(df[ma_col].iloc[i - 1]) and float(df["Close"].iloc[i]) < float(df[ma_col].iloc[i]):
                    bearish.append(f"Breakdown MA {ma}")

        if open_pos is None and len(bullish) >= 1:
            entry = float(df["Close"].iloc[i])
            support_ref = float(rolling_sup.iloc[i]) if not pd.isna(rolling_sup.iloc[i]) else entry * 0.96
            stop = min(entry * 0.97, support_ref * 0.995)
            target = entry * 1.07
            if selected_rules.get("fibo", False) and fib_prices:
                lower_levels = [p for p in fib_prices if p < entry]
                upper_levels = [p for p in fib_prices if p > entry]
                if lower_levels:
                    stop = lower_levels[-1] * 0.995
                if upper_levels:
                    target = upper_levels[0]
            open_pos = {
                "entry_idx": i,
                "entry_price": entry,
                "stop": stop,
                "target": target,
                "reasons": bullish[:],
            }
            continue

        if open_pos is not None:
            low = float(df["Low"].iloc[i])
            high = float(df["High"].iloc[i])
            close = float(df["Close"].iloc[i])
            exit_price = None
            exit_reason = ""
            if low <= open_pos["stop"]:
                exit_price = open_pos["stop"]
                exit_reason = "Stop loss"
            elif high >= open_pos["target"]:
                exit_price = open_pos["target"]
                exit_reason = "Take profit"
            elif len(bearish) >= 1 or (i - open_pos["entry_idx"]) >= 12:
                exit_price = close
                exit_reason = bearish[0] if bearish else "Time exit"

            if exit_price is not None:
                ret_pct = (exit_price / open_pos["entry_price"] - 1.0) * 100.0
                trades.append({
                    "entry_idx": open_pos["entry_idx"],
                    "exit_idx": i,
                    "entry_price": open_pos["entry_price"],
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                    "entry_reason": ", ".join(open_pos["reasons"]),
                    "exit_reason": exit_reason,
                })
                equity_curve.append(equity_curve[-1] * (1 + ret_pct / 100.0))
                open_pos = None

    if not trades:
        stats = {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "total_trades": 0}
    else:
        profits = sum(t["return_pct"] for t in trades if t["return_pct"] > 0)
        losses = abs(sum(t["return_pct"] for t in trades if t["return_pct"] < 0))
        profit_factor = profits / losses if losses > 0 else profits
        eq = pd.Series(equity_curve)
        drawdown = (eq / eq.cummax() - 1.0).min() * 100.0
        stats = {
            "win_rate": 100.0 * sum(t["return_pct"] > 0 for t in trades) / len(trades),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(drawdown),
            "total_trades": len(trades),
        }
    return active_lines, stats, trades


def build_backtest_chart(df: pd.DataFrame, interval: str, period: str, trades: List[dict], selected_rules: Dict[str, bool]) -> go.Figure:

    filters = {
        "ma": bool(selected_rules.get("ma_cross", False) or any(selected_rules.get(f"breakout_ma_{ma}", False) for ma in MA_WINDOWS)),
        "macd": selected_rules.get("macd_cross", False),
        "rsi": selected_rules.get("rsi_oversold", False),
        "fibo": selected_rules.get("fibo", False),
        "candle": False,
        "pattern": False,
        "breakout": selected_rules.get("breakout_resistance", False),
        "breakdown": False,
        "stoch": selected_rules.get("stoch", False),
    }
    fig, row_map, y_min, y_max = build_buy_sell_panels_figure(df, interval, period, filters)
    x_vals = np.arange(len(df))
    custom_dt = [ts.strftime("%d %b %Y %H:%M") if is_intraday_interval(interval) else ts.strftime("%d %b %Y") for ts in df.index]

    if selected_rules.get("ma_cross", False):
        for col, name in [("EMA10", "EMA10"), ("EMA20", "EMA20")]:
            fig.add_trace(go.Scatter(x=x_vals, y=df[col], mode="lines", line=dict(color=MA_COLORS[10] if col=="EMA10" else MA_COLORS[20], width=2.1),
                                     text=custom_dt, hovertemplate=f"<b>%{{text}}</b><br>{name}: %{{y:.2f}}<extra></extra>", name=name), row=1, col=1)
    for ma in MA_WINDOWS:
        if selected_rules.get(f"breakout_ma_{ma}", False):
            fig.add_trace(go.Scatter(x=x_vals, y=df[f"MA{ma}"], mode="lines", line=dict(color=MA_COLORS.get(ma, "#5EA3DA"), width=1.9),
                                     text=custom_dt, hovertemplate=f"<b>%{{text}}</b><br>MA{ma}: %{{y:.2f}}<extra></extra>", name=f"MA{ma}"), row=1, col=1)
    if selected_rules.get("fibo", False):
        add_fibonacci_lines(fig, df, calculate_fibonacci_context(df), y_min, y_max)

    y_span = max(y_max - y_min, 1.0)
    events = []
    for tr in trades:
        events.append({"idx": int(tr["entry_idx"]), "side": "Buy", "label": "ENTRY"})
        events.append({"idx": int(tr["exit_idx"]), "side": "Sell", "label": "EXIT"})
        fig.add_shape(
            type="line",
            x0=int(tr["entry_idx"]), x1=int(tr["exit_idx"]),
            y0=float(tr["entry_price"]), y1=float(tr["exit_price"]),
            xref="x", yref="y",
            line=dict(color="#4AA3FF" if tr["return_pct"] >= 0 else "#FF7D7D", width=1.3, dash="dot"),
        )
    add_price_event_labels(fig, df, events, y_span)
    return fig



def render_backtest_controls():
    with st.container(border=True):
        st.markdown('**STRATEGY**')
        l1, l2, l3 = st.columns(3)
        with l1:
            st.checkbox('RSI oversold', key='bt_rsi_oversold')
            st.checkbox('MACD cross', key='bt_macd_cross')
            st.checkbox('Stochastic', key='bt_stoch')
        with l2:
            st.checkbox('MA cross', key='bt_ma_cross')
            st.checkbox('Breakout resistance', key='bt_breakout_resistance')
            st.checkbox('Beli dekat support', key='bt_near_support')
        with l3:
            st.checkbox('Pola / candle bullish', key='bt_bullish_pattern')
            st.checkbox('Fibonacci', key='bt_fibo')
            for ma in MA_WINDOWS:
                st.checkbox(f'Breakout MA {ma}', key=f'bt_breakout_ma_{ma}')
    return {
        'rsi_oversold': st.session_state.get('bt_rsi_oversold', True),
        'macd_cross': st.session_state.get('bt_macd_cross', True),
        'stoch': st.session_state.get('bt_stoch', False),
        'ma_cross': st.session_state.get('bt_ma_cross', True),
        'breakout_resistance': st.session_state.get('bt_breakout_resistance', True),
        'near_support': st.session_state.get('bt_near_support', True),
        'bullish_pattern': st.session_state.get('bt_bullish_pattern', True),
        'fibo': st.session_state.get('bt_fibo', False),
        **{f'breakout_ma_{ma}': st.session_state.get(f'bt_breakout_ma_{ma}', False) for ma in MA_WINDOWS},
    }


def render_backtest_result_card(stats: dict):
    win_class = "metric-up" if stats["win_rate"] >= 50 else "metric-down"
    pf_class = "metric-up" if stats["profit_factor"] >= 1 else "metric-down"
    dd_class = "metric-up" if stats["max_drawdown"] > -10 else "metric-down"
    trades_class = "metric-up" if stats["total_trades"] > 0 else "metric-neutral"
    st.markdown(
        f"""
        <div class="trade-setup-card">
            <div class="trade-setup-title">BACKTEST RESULT</div>
            <div class="trade-setup-row"><b>Win rate</b> : <span class="{win_class}">{stats['win_rate']:.1f}%</span></div>
            <div class="trade-setup-row"><b>Profit factor</b> : <span class="{pf_class}">{stats['profit_factor']:.2f}</span></div>
            <div class="trade-setup-row"><b>Max drawdown</b> : <span class="{dd_class}">{stats['max_drawdown']:.1f}%</span></div>
            <div class="trade-setup-row"><b>Total trades</b> : <span class="{trades_class}">{stats['total_trades']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css()
st.markdown(
    """
    <style>
    .big-note { color:#BFD3E8; font-size:13px; margin-top:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "symbol_input" not in st.session_state:
    st.session_state.symbol_input = "DEWA"
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

for key, default in {
    "bs_filter_ma": False,
    "bs_filter_macd": False,
    "bs_filter_rsi": False,
    "bs_filter_fibo": False,
    "bs_filter_candle": False,
    "bs_filter_pattern": False,
    "bs_filter_breakout": False,
    "bs_filter_breakdown": False,
    "bs_filter_stoch": False,
    "bt_rsi_oversold": True,
    "bt_macd_cross": True,
    "bt_ma_cross": True,
    "bt_breakout_resistance": True,
    "bt_near_support": True,
    "bt_bullish_pattern": True,
    "bt_fibo": False,
    "bt_stoch": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
for ma in MA_WINDOWS:
    bt_key = f"bt_breakout_ma_{ma}"
    if bt_key not in st.session_state:
        st.session_state[bt_key] = False

left_title, right_btn = st.columns([0.86, 0.14])
with left_title:
    st.selectbox("Mode Analisis", options=MODE_OPTIONS, key="analysis_mode", label_visibility="collapsed")
with right_btn:
    if st.button("IHSG", width="stretch"):
        st.session_state.symbol_input = "IHSG"
        st.rerun()

mode = st.session_state.analysis_mode

if mode == "News & Sentimen":
    ctrl1, spacer = st.columns([1.45, 1.55])
    with ctrl1:
        st.text_input("Masukkan kode saham IDX", key="symbol_input", placeholder="Contoh: BBCA, TLKM, BRIS")
        inject_symbol_input_script(st.session_state.symbol_input)
else:
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.45, 0.95, 0.95, 0.65])
    with ctrl1:
        st.text_input("Masukkan kode saham IDX", key="symbol_input", placeholder="Contoh: BBCA, TLKM, BRIS")
        inject_symbol_input_script(st.session_state.symbol_input)
    with ctrl2:
        st.selectbox("TF", options=list(TIMEFRAME_MAP.keys()), key="tf_select", index=list(TIMEFRAME_MAP.keys()).index("1 Hari"))
    with ctrl3:
        st.selectbox("Durasi", options=list(DURATION_MAP.keys()), key="durasi_select", index=list(DURATION_MAP.keys()).index("1 Tahun"))
    with ctrl4:
        st.markdown('<div class="export-align"></div>', unsafe_allow_html=True)
        export_placeholder = st.empty()

raw_code = st.session_state.symbol_input.strip()
interval = TIMEFRAME_MAP[st.session_state.tf_select]
requested_period = DURATION_MAP[st.session_state.durasi_select]

if not raw_code:
    st.info("Masukkan kode saham dulu untuk menampilkan chart.")
elif mode == "News & Sentimen":
    shown_code = display_symbol(raw_code)
    render_news_header(shown_code)
    render_news_section(fetch_news_sentiment(normalize_symbol(raw_code), shown_code))
else:
    symbol = normalize_symbol(raw_code)
    shown_code = display_symbol(raw_code)
    with st.spinner(f"Mengambil data {symbol}..."):
        data, note, effective_period = load_data(symbol, interval, requested_period)

    if note:
        st.warning(note)

    if data.empty:
        st.error(f"Data tidak tersedia untuk {symbol}.")
    elif len(data) < 3:
        st.error(f"Data terlalu sedikit untuk dianalisis. Rows: {len(data)}")
    else:
        if mode == "Support & Resistance":
            export_name = f"{shown_code}_{interval}_{effective_period}.txt".replace(" ", "_")
            export_placeholder.download_button("Export", data=build_export_text(data, interval), file_name=export_name, mime="text/plain", width="stretch")
        else:
            try:
                export_placeholder.empty()
            except Exception:
                pass


        fig = None
        fib_context = None
        trade_setups = []
        active_lines = []
        stats = None
        trades = []
        if mode == "Support & Resistance":
            selected_mas = [ma for ma in MA_WINDOWS if st.session_state.get(f"ma_{ma}_on", False)]
            fig, levels, trend, signals, structures = build_sr_chart_v2(data, interval, effective_period, selected_mas)
            risk_label, risk_reason = assess_risk(levels, trend, float(data["Close"].iloc[-1]))
            c1, c2, c3, c4 = st.columns(4)
            render_sr_card(c1, "Support terdekat", levels.get("support_near"))
            render_sr_card(c2, "Support kuat", levels.get("support_strong"))
            render_sr_card(c3, "Resistance terdekat", levels.get("resistance_near"))
            render_sr_card(c4, "Resistance kuat", levels.get("resistance_strong"))
            render_header(mode=mode, display_code_value=shown_code, data=data, tf_label=st.session_state.tf_select, duration_label=label_period(effective_period), trend=trend, risk_label=risk_label, risk_reason=risk_reason)
            render_pattern_summary(data, interval, signals, structures)
            with st.container(border=True):
                make_ma_toolbar(show_title=False)

        elif mode == "Fibonacci":
            fig, fib_context, trend = build_fibonacci_chart(data, interval, effective_period)
            fib_cols = st.columns(5)
            for col, fib_level in zip(fib_cols, fib_context["levels"]):
                render_fib_card(col, fib_level)
            render_header(mode=mode, display_code_value=shown_code, data=data, tf_label=st.session_state.tf_select, duration_label=label_period(effective_period), trend="")

        elif mode == "MACD & RSI":
            fig, trend, macd_cards = build_macd_rsi_chart_v2(data, interval, effective_period)
            render_card_grid(macd_cards)
            render_header(mode=mode, display_code_value=shown_code, data=data, tf_label=st.session_state.tf_select, duration_label=label_period(effective_period), trend=trend)

        elif mode == "Sinyal Buy & Sell":
            render_header(mode=mode, display_code_value=shown_code, data=data, tf_label=st.session_state.tf_select, duration_label=label_period(effective_period), trend=detect_trend(data))
            filters = _get_buy_sell_filters()
            fig, trend, bs_cards, trade_setups = build_buy_sell_cards_and_chart(data, interval, effective_period, filters)
            render_card_grid(bs_cards)
            render_buy_sell_strategy_controls()

        elif mode == "Backtest Strategy":
            render_header(mode=mode, display_code_value=shown_code, data=data, tf_label=st.session_state.tf_select, duration_label=label_period(effective_period), trend="")
            bt_left, bt_right = st.columns([1.35, 0.65])
            with bt_left:
                selected_rules = render_backtest_controls()
            active_lines, stats, trades = run_backtest_strategy(data, selected_rules)
            with bt_right:
                render_backtest_result_card(stats)
            fig = build_backtest_chart(data, interval, effective_period, trades, selected_rules)
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
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
            if mode == "Sinyal Buy & Sell":
                render_trade_setup_card(trade_setups)

            with st.expander("Keterangan"):
                if mode == "Support & Resistance":
                    selected = [f"MA{ma}" for ma in MA_WINDOWS if st.session_state.get(f"ma_{ma}_on", False)]
                    ma_note = ", ".join(selected) if selected else "tidak ada MA harga yang dipilih"
                    st.write(
                        f"- Mode Support & Resistance menampilkan garis SR, volume, VMA20, garis close, dan checklist MA harga.\n"
                        f"- MA aktif saat ini: {ma_note}.\n"
                        "- Kartu sinyal candle menampilkan tanggal/rentang waktu kejadian.\n"
                        "- Pola candle multi-candle memuat pola utama seperti double top/bottom, triple top/bottom, head & shoulders, inverse head & shoulders, dan cup & handle."
                    )
                elif mode == "Fibonacci":
                    fib_lines = [f"- {level.ratio_label}: {level.description}" for level in fib_context["levels"]]
                    st.write(
                        "- Mode Fibonacci hanya menampilkan garis Fibonacci retracement, garis close terakhir, volume, dan MA20 volume.\n"
                        "- Tidak ada SK, SD, RK, RD, SKSD, RKRD, MA harga, atau sinyal candle penting di mode ini.\n"
                        "- Anchor Fibonacci diambil dari swing high dan swing low tertinggi/terendah pada durasi yang dipilih.\n"
                        + "\n".join(fib_lines)
                    )
                elif mode == "MACD & RSI":
                    st.write(
                        "- Card diambil dari event teknikal terdekat.\n"
                        "- Judul hijau menunjukkan arah positif, judul merah menunjukkan arah negatif.\n"
                        "- Label event ditampilkan langsung di panel MACD, RSI, dan Stoch RSI."
                    )
                elif mode == "Sinyal Buy & Sell":
                    st.write(
                        "- Mode ini menampilkan chart harga, volume, dan garis close putus-putus.\n"
                        "- Sinyal hanya diambil dari 50 candle terakhir agar lebih mudah ditrack untuk backtest.\n"
                        "- Card diurutkan dari yang paling lama ke yang paling baru.\n"
                        "- Risk/Reward dihitung secara heuristik terhadap level support, resistance, dan Fibonacci terdekat.\n- Checklist strategi di bawah card mengatur indikator/panel yang tampil di chart."
                    )
                elif mode == "Backtest Strategy":
                    st.write(
                        "- Checklist strategy bisa diaktifkan atau dimatikan satu per satu.\n"
                        "- Label ENTRY dan EXIT ditampilkan di chart agar mudah melihat lokasi transaksi.\n"
                        "- Statistik backtest dihitung dari rule yang sedang aktif."
                    )
