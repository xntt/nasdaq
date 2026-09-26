"""
纳斯达克联动监控完整版
- QQQ 联动 / 权重贡献 / 时段统计 / 跟随假设
- 会话分裂：盘前 vs RTH、FADE、gap fill、开盘 regime
- 一周波段状态机：吸筹 / 拉升 / 派发 / 回撤
- 15 分钟自动刷新 + 手动刷新
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# =========================
# 配置
# =========================

BENCHMARK = "QQQ"
REFRESH_SECONDS = 15 * 60
MIN_DAYS = 25

DEFAULT_TICKERS = [
    "QQQ", "SPY", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "NFLX", "COST",
    "AMD", "MU", "TSM", "ASML", "ARM", "SMCI", "INTC", "QCOM", "LRCX", "AMAT", "KLAC", "WDC", "STX",
    "LLY", "NVO", "VRTX", "REGN", "MRK", "PFE", "AMGN", "ISRG",
    "XOM", "CVX", "USO", "GLD", "GDX",
    "SOXX", "SOXL", "SOXS", "LABU", "LABD", "TQQQ", "SQQQ",
    "HOOD", "COIN", "MSTR",
]

# 近似 QQQ/纳指权重（%），请按最新持仓自行调整
NDX_WEIGHT_PCT = {
    "AAPL": 9.0, "MSFT": 8.5, "NVDA": 8.0, "AMZN": 5.5, "META": 5.0,
    "GOOGL": 4.5, "AVGO": 4.5, "TSLA": 3.0, "COST": 2.5, "NFLX": 2.0,
    "AMD": 1.5, "QCOM": 1.2, "INTC": 0.8, "AMAT": 0.8,
}

TIME_BUCKETS = [
    ("夜盘凌晨 0-4", 0, 4),
    ("凌晨 4-8", 4, 8),
    ("早盘前 8-9", 8, 9),
    ("常规早盘 9-12", 9, 12),
    ("午盘 12-16", 12, 16),
    ("尾盘后 16-20", 16, 20),
    ("晚间 20-24", 20, 24),
]


# =========================
# 数据层
# =========================

@st.cache_data(ttl=600, show_spinner=False)
def download_history(tickers: List[str], period: str) -> pd.DataFrame:
    tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy() if "Close" in data.columns.get_level_values(0) else data.xs("Close", axis=1, level=0)
    else:
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]
    return close.dropna(how="all")


@st.cache_data(ttl=300, show_spinner=False)
def download_ohlcv_daily(tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
    """每票 Open/High/Low/Close，用于跳空与周状态"""
    out: Dict[str, pd.DataFrame] = {}
    tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                out[t] = df
        except Exception:
            continue
    return out


@st.cache_data(ttl=300, show_spinner=False)
def download_intraday(ticker: str, period: str = "10d", interval: str = "30m") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    col = "Datetime" if "Datetime" in df.columns else "Date"
    df.rename(columns={col: "dt"}, inplace=True)
    df["dt"] = pd.to_datetime(df["dt"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    else:
        df["dt"] = df["dt"].dt.tz_convert("America/New_York")
    return df.dropna(subset=["Close"])


# =========================
# 会话分裂 / FADE / 开盘 regime
# =========================

def analyze_session_split(intraday: pd.DataFrame, daily_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    盘前 / 开盘15m / RTH / FADE / gap fill / open_regime
    daily_row: 含 Open/Close 的日线最后一行（可选，增强跳空）
    """
    out = {
        "pre_pct": None,
        "open15_pct": None,
        "rth_pct": None,
        "post_pct": None,
        "gap_pct": None,
        "gap_fill_pct": None,
        "fade_flag": 0,
        "open_regime": "—",
        "session_split_score": None,
        "open_reverse": None,
        "pre_high": None,
        "pre_low": None,
    }
    if intraday is None or intraday.empty:
        return out

    df = intraday.copy().sort_values("dt")
    df["date"] = df["dt"].dt.date
    df["hm"] = df["dt"].dt.hour * 100 + df["dt"].dt.minute
    last_day = df["date"].max()
    day = df[df["date"] == last_day].sort_values("dt")
    if len(day) < 3:
        return out

    def seg_ret(mask) -> Optional[float]:
        seg = day[mask]
        if len(seg) < 2:
            return None
        a, b = float(seg["Close"].iloc[0]), float(seg["Close"].iloc[-1])
        return (b / a - 1) * 100 if a else None

    pre_mask = day["hm"] < 930
    rth_mask = (day["hm"] >= 930) & (day["hm"] <= 1600)
    post_mask = day["hm"] > 1600
    open15_mask = (day["hm"] >= 930) & (day["hm"] <= 945)

    pre = seg_ret(pre_mask)
    rth = seg_ret(rth_mask)
    post = seg_ret(post_mask)
    open15 = seg_ret(open15_mask)

    pre_day = day[pre_mask]
    pre_high = float(pre_day["High"].max()) if len(pre_day) and "High" in pre_day.columns else None
    pre_low = float(pre_day["Low"].min()) if len(pre_day) and "Low" in pre_day.columns else None
    # 若无 High/Low，用 Close 近似
    if pre_high is None and len(pre_day):
        pre_high = float(pre_day["Close"].max())
        pre_low = float(pre_day["Close"].min())

    out["pre_pct"] = pre
    out["rth_pct"] = rth
    out["post_pct"] = post
    out["open15_pct"] = open15
    out["pre_high"] = pre_high
    out["pre_low"] = pre_low

    # 跳空：优先日线 Open vs 昨收
    gap = None
    if daily_row is not None and len(daily_row) > 0:
        try:
            # daily 需外部传入 gap
            pass
        except Exception:
            pass

    # 用分钟线：RTH 首根 vs 昨收近似 —— 若只有当日数据则用 pre 末 vs RTH 初
    rth_day = day[rth_mask]
    if len(rth_day) >= 1 and len(pre_day) >= 1:
        pre_last = float(pre_day["Close"].iloc[-1])
        rth_first = float(rth_day["Close"].iloc[0])
        if pre_last:
            # 开盘相对盘前末的跳变
            open_vs_pre = (rth_first / pre_last - 1) * 100
        else:
            open_vs_pre = None
    else:
        open_vs_pre = None

    # gap：若有完整日线在外层算；这里用 pre 作为隔夜代理
    if pre is not None:
        gap = pre  # 简化：把盘前涨跌当作隔夜/盘前缺口代理
    out["gap_pct"] = gap

    # gap fill：RTH 往回吐掉盘前涨幅的比例
    gap_fill = None
    if pre is not None and rth is not None and abs(pre) > 0.05:
        # 盘前涨、盘中跌 → 回补为正
        if pre > 0 and rth < 0:
            gap_fill = min(100.0, abs(rth) / abs(pre) * 100)
        elif pre < 0 and rth > 0:
            gap_fill = min(100.0, abs(rth) / abs(pre) * 100)
        else:
            gap_fill = 0.0
    out["gap_fill_pct"] = gap_fill

    # fade_flag：盘前明显单边，RTH 反向
    fade = 0
    if pre is not None and rth is not None:
        if pre >= 0.4 and rth <= -0.15:
            fade = 1
        elif pre <= -0.4 and rth >= 0.15:
            fade = 1
    out["fade_flag"] = fade

    if pre is not None and rth is not None and abs(pre) > 0.15 and abs(rth) > 0.15:
        out["open_reverse"] = 1 if pre * rth < 0 else 0

    if pre is not None and rth is not None and pre * rth < 0:
        out["session_split_score"] = abs(pre) + abs(rth)
    elif pre is not None and rth is not None:
        out["session_split_score"] = abs(abs(pre) - abs(rth))

    # open_regime
    regime = "CHOP"
    if open15 is not None and pre is not None:
        if pre >= 0.3 and open15 >= 0.15:
            regime = "CONTINUATION"
        elif pre >= 0.3 and open15 <= -0.15:
            regime = "FADE"
        elif pre <= -0.3 and open15 <= -0.15:
            regime = "CONTINUATION"
        elif pre <= -0.3 and open15 >= 0.15:
            regime = "FADE"
        elif open15 is not None and abs(open15) < 0.12:
            regime = "CHOP"
    if fade == 1 and regime == "CHOP":
        regime = "FADE"
    out["open_regime"] = regime

    return out


def daily_gap_from_ohlcv(ohlcv: pd.DataFrame) -> Optional[float]:
    if ohlcv is None or len(ohlcv) < 2:
        return None
    prev_close = float(ohlcv["Close"].iloc[-2])
    today_open = float(ohlcv["Open"].iloc[-1])
    if prev_close == 0:
        return None
    return (today_open / prev_close - 1) * 100


# =========================
# 一周波段状态机
# =========================

def week_phase_label(close: pd.Series, rets: pd.Series) -> Dict[str, Any]:
    """
    UP_LEG / DISTRIBUTE / DOWN_LEG / ACCUMULATE
    """
    s = close.dropna().tail(30)
    r = rets.dropna().tail(30)
    if len(s) < 12 or len(r) < 10:
        return {"phase": "数据不足", "ret5": None, "alt5": None, "dd": None}

    ret5 = float(s.iloc[-1] / s.iloc[-6] - 1) * 100 if len(s) >= 6 else None
    r5 = r.tail(8)
    signs = np.sign(r5.values)
    alt5 = float(np.mean(signs[1:] * signs[:-1] < 0)) if len(signs) > 2 else None
    dd = float(s.iloc[-1] / s.tail(15).max() - 1) * 100

    # 近5日新高？
    near_high = s.iloc[-1] >= s.tail(10).max() * 0.998
    near_low = s.iloc[-1] <= s.tail(10).min() * 1.002

    phase = "ACCUMULATE"
    if ret5 is not None and ret5 >= 2.0 and near_high:
        phase = "UP_LEG"
    if ret5 is not None and ret5 >= 0.5 and alt5 is not None and alt5 >= 0.55 and dd is not None and dd > -4:
        phase = "DISTRIBUTE"
    if ret5 is not None and ret5 <= -2.0:
        phase = "DOWN_LEG"
    if ret5 is not None and ret5 > -1.5 and dd is not None and dd <= -6:
        phase = "ACCUMULATE"
    if ret5 is not None and ret5 >= 2.5:
        phase = "UP_LEG"

    return {"phase": phase, "ret5": ret5, "alt5": alt5, "dd": dd}


# =========================
# 跟随 / 规律
# =========================

def follow_metrics(rets: pd.Series, bench: pd.Series) -> Dict[str, Any]:
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    df.columns = ["r", "b"]
    if len(df) < MIN_DAYS:
        return {"corr": None, "beta": None, "same_dir_rate": None, "label": "数据不足", "roll_corr_std": None}
    corr = float(df["r"].corr(df["b"]))
    var_b = float(df["b"].var())
    beta = float(df["r"].cov(df["b"]) / var_b) if var_b > 0 else None
    same = float((np.sign(df["r"]) == np.sign(df["b"])).mean())
    roll = df["r"].rolling(5).corr(df["b"])
    roll_std = float(roll.std()) if roll.notna().sum() > 10 else None
    if corr >= 0.75 and same >= 0.65:
        label = "强跟随"
    elif corr >= 0.45 and same >= 0.55:
        label = "中等跟随"
    elif corr >= 0.2:
        label = "弱跟随"
    elif corr < 0:
        label = "常背离"
    else:
        label = "不明确"
    if roll_std is not None and roll_std > 0.35 and corr > 0.4:
        label += "（跟一段易断）"
    return {"corr": corr, "beta": beta, "same_dir_rate": same, "label": label, "roll_corr_std": roll_std}


def alternation_score(close: pd.Series, n: int = 40) -> Dict[str, Any]:
    r = close.pct_change().dropna().tail(n)
    if len(r) < 10:
        return {"alt_rate": None, "label": "不足"}
    s = np.sign(r.values)
    flips = float(np.mean(s[1:] * s[:-1] < 0))
    return {
        "alt_rate": flips,
        "label": "高低交替强" if flips >= 0.55 else ("趋势连贯" if flips <= 0.4 else "中性"),
    }


def local_top_gaps(close: pd.Series, lookback: int = 90) -> Dict[str, Any]:
    s = close.dropna().tail(lookback)
    if len(s) < 20:
        return {"median_gap": None, "tops": [], "note": "数据不足"}
    arr, dates = s.values, s.index
    tops = [dates[i] for i in range(3, len(arr) - 3) if arr[i] == max(arr[i - 3 : i + 4])]
    gaps = [(b - a).days for a, b in zip(tops[:-1], tops[1:])]
    med = float(np.median(gaps)) if gaps else None
    note = ""
    if med and 10 <= med <= 18:
        note = "局部顶间隔中位数约 2 周"
    elif med and 4 <= med <= 9:
        note = "局部顶较密（约 1 周波动）"
    elif med:
        note = f"局部顶中位间隔约 {med:.0f} 日"
    return {"median_gap": med, "tops": [str(t.date()) for t in tops[-6:]], "note": note}


def weekday_stats(close: pd.Series) -> pd.DataFrame:
    r = close.pct_change().dropna()
    idx = r.index
    wd = idx.tz_convert("America/New_York").dayofweek if getattr(idx, "tz", None) else idx.dayofweek
    tmp = pd.DataFrame({"ret": r.values, "wd": wd})
    g = tmp.groupby("wd")["ret"].agg(["mean", "count"])
    g = g.reindex(range(5))
    g.index = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    g["mean_pct"] = g["mean"] * 100
    return g


def tag_follow_reason(close: pd.DataFrame, ticker: str, bench: str, rets: pd.DataFrame, window: int = 10) -> Dict[str, Any]:
    if ticker not in close.columns or bench not in close.columns:
        return {"近窗同向率": None, "超额%": None, "回撤%": None, "标签": "—", "假设": "—"}
    r = rets[ticker].dropna().tail(window)
    b = rets[bench].reindex(r.index).dropna()
    r = r.reindex(b.index)
    if len(r) < max(5, window // 2):
        return {"近窗同向率": None, "超额%": None, "回撤%": None, "标签": "数据不足", "假设": "—"}
    same = float((np.sign(r) == np.sign(b)).mean())
    excess = float((1 + r).prod() - (1 + b).prod()) * 100
    px = close[ticker].dropna().tail(25)
    dd = float(px.iloc[-1] / px.max() - 1) * 100 if len(px) >= 10 else None

    if same >= 0.7 and excess >= 0:
        tag = "积极跟随偏强"
    elif same >= 0.7 and excess < 0:
        tag = "跟随但跑输"
    elif same <= 0.4 and excess > 1:
        tag = "不跟却偏强"
    elif same <= 0.4 and excess < -1:
        tag = "不跟且偏弱"
    else:
        tag = "弱相关摇摆"

    hyp = []
    if same >= 0.7 and dd is not None and dd <= -8:
        hyp.append("超跌后跟大盘修复概率偏高")
    if same >= 0.7 and excess > 3:
        hyp.append("高Beta/情绪共振")
    if same <= 0.45 and excess > 2 and (dd is None or dd > -5):
        hyp.append("可能滞涨补涨或个股逻辑独立")
    if same <= 0.45 and excess < -2:
        hyp.append("可能个股利空或已透支")
    if same >= 0.6 and dd is not None and dd > -3 and excess < 0:
        hyp.append("位置偏高仍跟跌，警惕滞涨转弱")
    if not hyp:
        hyp.append("规则未强命中，结合量能与新闻")
    return {"近窗同向率": same, "超额%": excess, "回撤%": dd, "标签": tag, "假设": "；".join(hyp)}


# =========================
# 时段 / 权重
# =========================

def qqq_move_timing_stats(intraday: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    if intraday is None or intraday.empty:
        return empty, empty
    df = intraday.copy().sort_values("dt")
    df["ret"] = df["Close"].pct_change()
    df["date"] = df["dt"].dt.date
    df["hour"] = df["dt"].dt.hour
    up_hours, down_hours, day_rows = [], [], []
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["ret"])
        if len(g) < 5:
            continue
        day_ret = (float(g["Close"].iloc[-1]) / float(g["Close"].iloc[0]) - 1) * 100
        up_idx, dn_idx = g["ret"].idxmax(), g["ret"].idxmin()
        up_h, dn_h = int(g.loc[up_idx, "hour"]), int(g.loc[dn_idx, "hour"])
        up_hours.append(up_h)
        down_hours.append(dn_h)
        day_rows.append({
            "日期": str(d),
            "日涨跌%": day_ret,
            "最强上攻ET时": up_h,
            "最强下杀ET时": dn_h,
            "最大单柱涨%": float(g.loc[up_idx, "ret"]) * 100,
            "最大单柱跌%": float(g.loc[dn_idx, "ret"]) * 100,
        })
    detail = pd.DataFrame(day_rows)

    def bucket_count(hours: List[int], name: str) -> pd.DataFrame:
        rows = []
        for label, a, b in TIME_BUCKETS:
            rows.append({"时段(美东)": label, "类型": name, "次数": sum(1 for h in hours if a <= h < b)})
        return pd.DataFrame(rows)

    summary = pd.concat(
        [bucket_count(up_hours, "日内最强上攻柱"), bucket_count(down_hours, "日内最强下杀柱")],
        ignore_index=True,
    )
    return summary, detail


def weight_contribution(close: pd.DataFrame, weights: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    avail = {k: v for k, v in weights.items() if k in close.columns}
    if not avail or BENCHMARK not in close.columns:
        return pd.DataFrame(), {}
    wsum = sum(avail.values())
    rets_1d = close.pct_change().iloc[-1]
    rows = []
    total_weighted = 0.0
    for t, w in avail.items():
        wn = w / wsum
        r = float(rets_1d.get(t, np.nan))
        if np.isnan(r):
            continue
        contrib = wn * r * 100
        total_weighted += contrib
        rows.append({"代码": t, "近似权重%": round(w, 2), "日涨跌%": r * 100, "近似贡献": contrib})
    df = pd.DataFrame(rows).sort_values("近似贡献", ascending=False)
    qqq_r = float(rets_1d.get(BENCHMARK, np.nan))
    qqq_pct = qqq_r * 100 if not np.isnan(qqq_r) else None
    if df.empty:
        return df, {}
    top, bottom = df.iloc[0], df.iloc[-1]
    up_n = int((df["日涨跌%"] > 0).sum())
    dn_n = int((df["日涨跌%"] < 0).sum())
    n = len(df)
    up_ratio = up_n / n if n else 0
    if up_ratio >= 0.7:
        regime = "权重多数同向向上 → 大盘易启动/延续偏多"
    elif up_ratio <= 0.3:
        regime = "权重多数同向向下 → 大盘易走弱"
    elif 0.4 <= up_ratio <= 0.6:
        regime = "权重涨跌接近各半 → 易横盘或轮动变盘前夜"
    else:
        regime = "权重分化中等 → 指数可能窄幅波动"
    if abs(top["近似贡献"]) >= abs(bottom["近似贡献"]):
        driver = f"上拉主导：{top['代码']}（{top['日涨跌%']:+.2f}% / 贡献 {top['近似贡献']:+.3f}）"
    else:
        driver = f"下拉主导：{bottom['代码']}（{bottom['日涨跌%']:+.2f}% / 贡献 {bottom['近似贡献']:+.3f}）"
    summary = {
        "QQQ日涨跌%": qqq_pct,
        "权重篮子加权涨跌%": total_weighted,
        "上涨权重家数": up_n,
        "下跌权重家数": dn_n,
        "上涨占比": up_ratio,
        "结构判断": regime,
        "主导标注": driver,
    }
    return df, summary


def multi_day_heavy_move(close: pd.DataFrame, weights: Dict[str, float], days: int = 8) -> pd.DataFrame:
    avail = [k for k in weights if k in close.columns]
    if len(avail) < 5:
        return pd.DataFrame()
    rets = close[avail].pct_change().dropna(how="all").tail(days)
    rows = []
    for dt, row in rets.iterrows():
        vals = row.dropna()
        if len(vals) < 5:
            continue
        up = float((vals > 0).mean())
        rows.append({
            "日期": str(dt.date()) if hasattr(dt, "date") else str(dt),
            "权重上涨占比": up,
            "解读": "共振偏多" if up >= 0.7 else ("共振偏空" if up <= 0.3 else "撕裂/轮动"),
        })
    return pd.DataFrame(rows)


def fmt(x, pct=False, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if pct or signed:
        return f"{x:+.2f}"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)

# =========================
# 模块：周期扫描 + 强者修复 + 规律被大盘打断 + 深跌基本面提示
# =========================

def find_swing_points(close: pd.Series, left: int = 7, right: int = 7):
    """局部顶/底：左右各 left/right 根内的极值"""
    s = close.dropna()
    if len(s) < left + right + 5:
        return [], []
    arr = s.values
    idx = s.index
    tops, bots = [], []
    for i in range(left, len(arr) - right):
        window = arr[i - left : i + right + 1]
        if arr[i] >= np.max(window) and arr[i] == window.max():
            tops.append((idx[i], float(arr[i])))
        if arr[i] <= np.min(window) and arr[i] == window.min():
            bots.append((idx[i], float(arr[i])))
    return tops, bots


def _gap_stats(points):
    """points: list[(ts, price)] -> 间隔天数统计"""
    if len(points) < 3:
        return None
    days = []
    for (a, _), (b, _) in zip(points[:-1], points[1:]):
        try:
            d = (pd.Timestamp(b) - pd.Timestamp(a)).days
            if d > 0:
                days.append(d)
        except Exception:
            continue
    if len(days) < 2:
        return None
    arr = np.array(days, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    med = float(np.median(arr))
    cv = std / mean if mean > 0 else None
    return {"mean": mean, "median": med, "std": std, "cv": cv, "gaps": days[-6:], "n": len(days)}


def cycle_score_from_stats(stt) -> float:
    if not stt or stt["cv"] is None:
        return 0.0
    # CV 越小分越高；样本越多略加分
    cv = stt["cv"]
    base = max(0.0, 1.0 - min(cv, 1.0))
    n_bonus = min(0.2, 0.03 * stt["n"])
    return round(100 * (0.8 * base + n_bonus), 1)


def label_cycle_bucket(median_days: Optional[float]) -> str:
    if median_days is None:
        return "—"
    m = median_days
    if 22 <= m <= 40:
        return "约月度"
    if 50 <= m <= 80:
        return "约季度"
    if 100 <= m <= 140:
        return "约4个月"
    if 10 <= m <= 20:
        return "约双周"
    return f"约{m:.0f}日"


def scan_ticker_cycle(close: pd.Series, left: int = 7, right: int = 7) -> Dict[str, Any]:
    tops, bots = find_swing_points(close, left=left, right=right)
    top_st = _gap_stats(tops)
    bot_st = _gap_stats(bots)
    score_t = cycle_score_from_stats(top_st) if top_st else 0.0
    score_b = cycle_score_from_stats(bot_st) if bot_st else 0.0
    score = max(score_t, score_b)

    last_top = str(pd.Timestamp(tops[-1][0]).date()) if tops else None
    last_bot = str(pd.Timestamp(bots[-1][0]).date()) if bots else None

    next_top_win = next_bot_win = None
    if top_st and tops:
        last = pd.Timestamp(tops[-1][0])
        med = top_st["median"]
        next_top_win = f"{(last + pd.Timedelta(days=med - 5)).date()} ~ {(last + pd.Timedelta(days=med + 5)).date()}"
    if bot_st and bots:
        last = pd.Timestamp(bots[-1][0])
        med = bot_st["median"]
        next_bot_win = f"{(last + pd.Timedelta(days=med - 5)).date()} ~ {(last + pd.Timedelta(days=med + 5)).date()}"

    return {
        "规律分": score,
        "底间隔中位": bot_st["median"] if bot_st else None,
        "顶间隔中位": top_st["median"] if top_st else None,
        "底CV": bot_st["cv"] if bot_st else None,
        "顶CV": top_st["cv"] if top_st else None,
        "周期标签": label_cycle_bucket(bot_st["median"] if bot_st else (top_st["median"] if top_st else None)),
        "最近底": last_bot,
        "最近顶": last_top,
        "下一次底窗口": next_bot_win,
        "下一次顶窗口": next_top_win,
        "底样本数": bot_st["n"] if bot_st else 0,
        "顶样本数": top_st["n"] if top_st else 0,
    }


def index_disrupts_cycle(
    stock_close: pd.Series,
    bench_close: pd.Series,
    cycle_info: Dict[str, Any],
    lookback: int = 15,
) -> Dict[str, Any]:
    """
    大盘曲线打乱个股原规律：
    - 个股处于「预测底/顶窗口」附近，但指数出现急跌/急涨主导
    - 或个股与指数近窗相关骤升且同向大波动（节奏被大盘拽走）
    """
    out = {"打乱": 0, "说明": "—"}
    if stock_close is None or bench_close is None:
        return out
    s = stock_close.dropna().tail(lookback + 5)
    b = bench_close.dropna().reindex(s.index).dropna()
    s = s.reindex(b.index)
    if len(s) < 8:
        return out

    s_ret = s.pct_change().dropna()
    b_ret = b.pct_change().reindex(s_ret.index).dropna()
    s_ret = s_ret.reindex(b_ret.index)
    corr = float(s_ret.tail(10).corr(b_ret.tail(10))) if len(s_ret) >= 10 else None
    b_move = float(b.iloc[-1] / b.iloc[-min(5, len(b))] - 1) * 100
    s_move = float(s.iloc[-1] / s.iloc[-min(5, len(s))] - 1) * 100

    in_bot_window = False
    in_top_window = False
    today = pd.Timestamp(s.index[-1]).normalize()
    for key, flag_name in [("下一次底窗口", "bot"), ("下一次顶窗口", "top")]:
        win = cycle_info.get(key)
        if not win or " ~ " not in str(win):
            continue
        try:
            a, c = str(win).split(" ~ ")
            a, c = pd.Timestamp(a.strip()), pd.Timestamp(c.strip())
            if a <= today <= c:
                if flag_name == "bot":
                    in_bot_window = True
                else:
                    in_top_window = True
        except Exception:
            pass

    reasons = []
    # 窗口内但走势被指数大波绑架
    if in_bot_window and b_move < -2.5 and s_move < -1.0:
        reasons.append("处于预期底窗口，但大盘急跌拖累，周期信号可能失效")
    if in_top_window and b_move > 2.5 and s_move > 1.0:
        reasons.append("处于预期顶窗口，但大盘急涨抬轿，见顶节奏可能延后")
    if corr is not None and corr > 0.85 and abs(b_move) > 3:
        reasons.append(f"近窗与大盘高度同向(corr={corr:.2f})，个股独立周期被指数主导")
    # 规律分尚可，但近5日个股波动几乎全是指数同向
    if cycle_info.get("规律分", 0) >= 55 and corr is not None and corr > 0.8 and abs(b_move) > 2:
        reasons.append("历史有规律，当前波段更像大盘驱动")

    if reasons:
        out["打乱"] = 1
        out["说明"] = "；".join(reasons)
    return out


def detect_qqq_pullback_events(bench_close: pd.Series, thresh: float = -0.02, max_events: int = 6) -> List[Dict[str, Any]]:
    """从近端找出 QQQ 短回撤事件：自滚动高点回撤超过 thresh"""
    s = bench_close.dropna().tail(120)
    if len(s) < 30:
        return []
    events = []
    i = 10
    while i < len(s) - 3:
        window = s.iloc[: i + 1]
        peak = window.max()
        peak_idx = window.idxmax()
        cur = s.iloc[i]
        dd = cur / peak - 1
        if dd <= thresh:
            # 事件低点：向后再找最多 8 日的更低
            j_end = min(len(s) - 1, i + 8)
            seg = s.iloc[i : j_end + 1]
            low_idx = seg.idxmin()
            low_px = float(seg.min())
            events.append({
                "peak_date": peak_idx,
                "low_date": low_idx,
                "peak_px": float(peak),
                "low_px": low_px,
                "dd": float(low_px / peak - 1),
            })
            i = list(s.index).index(low_idx) + 3
        else:
            i += 1
    return events[-max_events:]


def leader_bounce_metrics(
    stock_close: pd.Series,
    bench_close: pd.Series,
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """单次回撤事件上的承压与修复"""
    empty = {"承压比": None, "修复天数": None, "修复比": None, "标签": "—"}
    try:
        peak_d, low_d = event["peak_date"], event["low_date"]
        sc = stock_close.dropna()
        bc = bench_close.dropna()
        # 对齐
        if peak_d not in sc.index or low_d not in sc.index:
            # 用最近索引
            sc2 = sc.loc[(sc.index >= peak_d) & (sc.index <= low_d)]
            if sc2.empty:
                return empty
            s_peak = float(sc.loc[:peak_d].iloc[-1])
            s_low = float(sc2.min())
            s_low_date = sc2.idxmin()
        else:
            s_peak = float(sc.loc[peak_d])
            seg = sc.loc[peak_d:low_d]
            s_low = float(seg.min())
            s_low_date = seg.idxmin()

        b_dd = event["dd"]
        s_dd = s_low / s_peak - 1 if s_peak else None
        pressure = (s_dd / b_dd) if (s_dd is not None and b_dd and b_dd != 0) else None

        # 修复：从 s_low_date 起回到 s_peak 的天数
        after = sc.loc[s_low_date:]
        recover_days = None
        for k, (dt, px) in enumerate(after.items()):
            if float(px) >= s_peak * 0.995:
                recover_days = k
                break
        # 同学段指数反弹
        b_after = bc.loc[event["low_date"]:]
        b_low = event["low_px"]
        # 个股从低点反弹到目前或到恢复点
        if recover_days is not None and recover_days < len(after):
            s_rebound = float(after.iloc[recover_days]) / s_low - 1 if s_low else None
            b_px = float(b_after.iloc[min(recover_days, len(b_after) - 1)]) if len(b_after) else None
            b_rebound = (b_px / b_low - 1) if (b_px and b_low) else None
        else:
            s_rebound = float(after.iloc[-1]) / s_low - 1 if len(after) and s_low else None
            b_rebound = float(b_after.iloc[-1]) / b_low - 1 if len(b_after) and b_low else None
            recover_days = None if recover_days is None else recover_days

        repair_ratio = None
        if s_rebound is not None and b_rebound and abs(b_rebound) > 1e-6:
            repair_ratio = s_rebound / b_rebound

        tag = "—"
        if pressure is not None and repair_ratio is not None:
            if pressure <= 1.05 and repair_ratio >= 1.1 and (recover_days is not None and recover_days <= 8):
                tag = "LEADER_BOUNCE"
            elif pressure >= 1.25 or (recover_days is not None and recover_days > 12):
                tag = "LAGGARD"
            elif pressure <= 0.75:
                tag = "抗跌"
            else:
                tag = "中性"
        elif pressure is not None and pressure >= 1.3:
            tag = "LAGGARD"

        return {
            "承压比": pressure,
            "修复天数": recover_days,
            "修复比": repair_ratio,
            "标签": tag,
            "个股回撤%": s_dd * 100 if s_dd is not None else None,
            "指数回撤%": b_dd * 100 if b_dd is not None else None,
        }
    except Exception:
        return empty


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals_quick(ticker: str) -> Dict[str, Any]:
    """深跌后快速基本面查询（yfinance，字段可能缺失）"""
    info = {}
    try:
        t = yf.Ticker(ticker)
        inf = t.info or {}
        info = {
            "名称": inf.get("shortName") or inf.get("longName"),
            "行业": inf.get("industry"),
            "板块": inf.get("sector"),
            "市盈率TTM": inf.get("trailingPE"),
            "预期市盈率": inf.get("forwardPE"),
            "市净率": inf.get("priceToBook"),
            "利润率": inf.get("profitMargins"),
            "营收增长": inf.get("revenueGrowth"),
            "盈利增长": inf.get("earningsGrowth"),
            "负债权益": inf.get("debtToEquity"),
            "分析师目标价": inf.get("targetMeanPrice"),
            "建议": inf.get("recommendationKey"),
        }
    except Exception as e:
        info = {"错误": str(e)}
    return info


def deep_drawdown_flags(close: pd.Series, thresh: float = -0.15) -> Dict[str, Any]:
    """近 60 日相对高点回撤"""
    s = close.dropna().tail(60)
    if len(s) < 20:
        return {"深跌": 0, "回撤%": None}
    dd = float(s.iloc[-1] / s.max() - 1)
    return {"深跌": 1 if dd <= thresh else 0, "回撤%": dd * 100}
# =========================
# UI
# =========================

st.set_page_config(page_title="纳指联动完整监控", page_icon="📈", layout="wide")
st.title("📈 纳斯达克联动 · 会话分裂 · 周状态 · 权重贡献")
st.caption("盘前当情报｜开盘看 FADE/CONTINUATION｜一周状态约束当天｜15分钟自动刷新 + 手动刷新｜非投资建议")

with st.sidebar:
    st.header("观察池")
    base = st.text_area("列表（每行一个）", value="\n".join(DEFAULT_TICKERS), height=200)
    extra = st.text_input("添加（逗号分隔）", placeholder="SNDK, PLTR")
    remove = st.text_input("移除（逗号分隔）", "")
    period = st.selectbox("日线窗口", ["3mo", "6mo", "1y", "2y"], index=1)
    follow_window = st.slider("跟随窗口（日）", 5, 20, 10)
    fade_pre_th = st.number_input("FADE：盘前阈值%", value=0.40, step=0.05)
    alert_corr = st.slider("预警：相关低于", 0.0, 1.0, 0.25, 0.05)
    st.divider()
    manual = st.button("🔄 手动刷新", type="primary", use_container_width=True)
    auto = st.checkbox("约 15 分钟自动刷新", value=True)

tickers = [x.strip().upper() for x in base.splitlines() if x.strip()]
if extra:
    tickers += [x.strip().upper() for x in extra.split(",") if x.strip()]
if remove:
    rm = {x.strip().upper() for x in remove.split(",") if x.strip()}
    tickers = [t for t in tickers if t not in rm]
for k in NDX_WEIGHT_PCT:
    tickers.append(k)
tickers = sorted(set(tickers))
if BENCHMARK not in tickers:
    tickers = [BENCHMARK] + tickers

if "ndx_last_ts" not in st.session_state:
    st.session_state.ndx_last_ts = 0.0
need = manual or (time.time() - st.session_state.ndx_last_ts >= REFRESH_SECONDS)
if need:
    st.cache_data.clear()
    st.session_state.ndx_last_ts = time.time()

with st.spinner("拉取与计算中…"):
    close = download_history(tickers, period=period)
    if close.empty or BENCHMARK not in close.columns:
        st.error("无法获取 QQQ/行情")
        st.stop()

    rets = close.pct_change()
    bench_r = rets[BENCHMARK]
    ohlcv_map = download_ohlcv_daily(tickers[:50], period=period)  # 限流

    qqq_intra = download_intraday(BENCHMARK, period="15d", interval="30m")
    timing_summary, timing_detail = qqq_move_timing_stats(qqq_intra)
    q_alt = alternation_score(close[BENCHMARK])
    q_top = local_top_gaps(close[BENCHMARK])
    q_wd = weekday_stats(close[BENCHMARK])
    q_week = week_phase_label(close[BENCHMARK], rets[BENCHMARK])

    contrib_df, contrib_sum = weight_contribution(close, NDX_WEIGHT_PCT)
    heavy_days = multi_day_heavy_move(close, NDX_WEIGHT_PCT, days=8)

    # 优先拉分钟线的代码
    priority = set(list(NDX_WEIGHT_PCT.keys()) + [BENCHMARK, "SOXX", "TQQQ", "HOOD", "COIN", "MSTR", "MU", "NVDA", "AMD"])
    ordered = [BENCHMARK] + [t for t in close.columns if t in priority] + [t for t in close.columns if t not in priority]
    ordered = list(dict.fromkeys(ordered))

    rows = []
    for i, t in enumerate(ordered):
        if t not in close.columns or t not in rets.columns:
            continue
        fm = follow_metrics(rets[t], bench_r)
        alt = alternation_score(close[t])
        tops = local_top_gaps(close[t])
        tag = tag_follow_reason(close, t, BENCHMARK, rets, window=follow_window)
        wp = week_phase_label(close[t], rets[t])
        series = close[t].dropna()
if series.empty:
    continue
last = float(series.iloc[-1])
        chg1 = float(rets[t].dropna().iloc[-1]) * 100 if rets[t].notna().any() else None
for t in ordered:  # 或你的 for t in close.columns
    if t not in close.columns:
        continue
    if close[t].dropna().empty:
        continue

        sess = {
            "pre_pct": None, "open15_pct": None, "rth_pct": None, "post_pct": None,
            "gap_pct": None, "gap_fill_pct": None, "fade_flag": 0,
            "open_regime": "—", "session_split_score": None, "open_reverse": None,
        }
        if i < 40 or t in priority:
            try:
                intra = download_intraday(t, period="5d", interval="30m")
                sess = analyze_session_split(intra)
                # 日线跳空覆盖 gap
                if t in ohlcv_map:
                    g = daily_gap_from_ohlcv(ohlcv_map[t])
                    if g is not None:
                        sess["gap_pct"] = g
                # 侧栏阈值重判 fade
                pre, rth = sess.get("pre_pct"), sess.get("rth_pct")
                if pre is not None and rth is not None:
                    if pre >= fade_pre_th and rth <= -0.15:
                        sess["fade_flag"] = 1
                        if sess["open_regime"] == "CHOP":
                            sess["open_regime"] = "FADE"
                    elif pre <= -fade_pre_th and rth >= 0.15:
                        sess["fade_flag"] = 1
                        if sess["open_regime"] == "CHOP":
                            sess["open_regime"] = "FADE"
            except Exception:
                pass

        rows.append({
            "代码": t,
            "最新价": last,
            "日涨跌%": chg1,
            "周状态": wp["phase"],
            "周5日%": wp["ret5"],
            "开盘regime": sess["open_regime"],
            "FADE": sess["fade_flag"],
            "盘前%": sess["pre_pct"],
            "开盘15m%": sess["open15_pct"],
            "盘中RTH%": sess["rth_pct"],
            "盘后%": sess["post_pct"],
            "跳空gap%": sess["gap_pct"],
            "缺口回补%": sess["gap_fill_pct"],
            "会话分裂分": sess["session_split_score"],
            "跟随标签": fm["label"],
            "与QQQ相关": fm["corr"],
            "Beta": fm["beta"],
            "同向率": fm["same_dir_rate"],
            "近窗跟随": tag["标签"],
            "原因假设": tag["假设"],
            "近窗同向率": tag["近窗同向率"],
            "近窗超额%": tag["超额%"],
            "近窗回撤%": tag["回撤%"],
            "涨跌交替率": alt["alt_rate"],
            "顶间隔提示": tops["note"],
            "近端局部顶": ", ".join(tops.get("tops") or []),
        })

    table = pd.DataFrame(rows)

# ===== 展示 =====
ts = datetime.fromtimestamp(st.session_state.ndx_last_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.info(f"标的 {len(table)}｜基准 {BENCHMARK}｜QQQ周状态 **{q_week['phase']}**｜更新 {ts}")

# 策略摘要条
st.subheader("🎯 今日策略层（会话 + 周状态）")
s1, s2, s3, s4 = st.columns(4)
s1.metric("QQQ 周状态", q_week["phase"])
s2.metric("QQQ 5日%", fmt(q_week["ret5"], signed=True))
qqq_row = table[table["代码"] == BENCHMARK]
if not qqq_row.empty:
    s3.metric("QQQ regime", str(qqq_row.iloc[0]["开盘regime"]))
    s4.metric("QQQ FADE", int(qqq_row.iloc[0]["FADE"]))
else:
    s3.metric("QQQ regime", "—")
    s4.metric("QQQ FADE", "—")

tips = []
if q_week["phase"] == "DISTRIBUTE":
    tips.append("周状态=派发：提高「盘前冲高不追、开盘看 FADE」权重")
if q_week["phase"] == "UP_LEG":
    tips.append("周状态=拉升：FADE 需更严格确认；优先看 CONTINUATION")
if q_week["phase"] == "DOWN_LEG":
    tips.append("周状态=回撤：反弹盘前谨慎追多")
if q_week["phase"] == "ACCUMULATE":
    tips.append("周状态=吸筹：关注超跌跟随 + CONTINUATION")
if not qqq_row.empty and int(qqq_row.iloc[0]["FADE"]) == 1:
    tips.append("QQQ 触发 FADE：盘前方向与 RTH 反向，符合「开盘流动性回收」")
if tips:
    for t_ in tips:
        st.write("· " + t_)

st.divider()

# 权重
st.subheader("⚖️ 权重股对大盘影响")
if contrib_sum:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("QQQ日涨跌%", fmt(contrib_sum["QQQ日涨跌%"], signed=True))
    m2.metric("权重篮子加权%", f"{contrib_sum['权重篮子加权涨跌%']:+.3f}")
    m3.metric("权重上涨家数", str(contrib_sum["上涨权重家数"]))
    m4.metric("权重下跌家数", str(contrib_sum["下跌权重家数"]))
    st.success(contrib_sum["主导标注"])
    st.warning(contrib_sum["结构判断"])
    c_a, c_b = st.columns(2)
    with c_a:
        st.dataframe(contrib_df.head(8), use_container_width=True)
    with c_b:
        st.dataframe(contrib_df.tail(8).iloc[::-1], use_container_width=True)
    if not heavy_days.empty:
        st.write("近几日权重齐动")
        st.dataframe(heavy_days, use_container_width=True)
else:
    st.write("权重贡献暂不可用")

st.divider()

# 时段
st.subheader("🕒 动能释放时段（QQQ 30m）")
st.write(f"日线交替：{q_alt['label']}｜{q_top.get('note')}")
if not timing_summary.empty:
    fig_t = px.bar(timing_summary, x="时段(美东)", y="次数", color="类型", barmode="group")
    st.plotly_chart(fig_t, use_container_width=True)
    with st.expander("分日明细"):
        st.dataframe(timing_detail, use_container_width=True)
st.dataframe(q_wd[["mean_pct", "count"]], use_container_width=True)

st.divider()

# 预警
alerts = []
if contrib_sum:
    alerts.append(f"权重：{contrib_sum['结构判断']}｜{contrib_sum['主导标注']}")
alerts.append(f"QQQ 周状态={q_week['phase']}（5日 {fmt(q_week['ret5'], signed=True)}%）")
for _, r in table.iterrows():
    code = r["代码"]
    if r["FADE"] == 1:
        alerts.append(
            f"🔔 {code} FADE｜盘前 {fmt(r['盘前%'], signed=True)}% → RTH {fmt(r['盘中RTH%'], signed=True)}%｜regime={r['开盘regime']}"
        )
    if r["开盘regime"] == "CONTINUATION" and r["周状态"] in ("UP_LEG", "ACCUMULATE"):
        alerts.append(f"✅ {code} CONTINUATION + 周状态{r['周状态']} → 跟随条件较好")
    if r["开盘regime"] == "FADE" and r["周状态"] == "DISTRIBUTE":
        alerts.append(f"📅 {code} 派发周 + FADE → 偏收割窗口，慎追盘前")
    if code != BENCHMARK and r["与QQQ相关"] is not None and r["与QQQ相关"] < alert_corr:
        alerts.append(f"⚠ {code} 低相关 ({r['与QQQ相关']:.2f})｜{r['近窗跟随']}｜{r['原因假设']}")
    if r.get("顶间隔提示") and "2 周" in str(r["顶间隔提示"]):
        alerts.append(f"📆 {code} {r['顶间隔提示']}；{r['近端局部顶']}")

st.subheader("🚨 预警")
for a in alerts[:60]:
    st.write("· " + a)

st.divider()
st.subheader("📋 总表")
show = table.copy()
for col in ["与QQQ相关", "Beta", "同向率", "近窗同向率", "涨跌交替率", "会话分裂分"]:
    if col in show.columns:
        show[col] = show[col].apply(lambda x: fmt(x))
for col in ["日涨跌%", "周5日%", "盘前%", "开盘15m%", "盘中RTH%", "盘后%", "跳空gap%", "缺口回补%", "近窗超额%", "近窗回撤%"]:
    if col in show.columns:
        show[col] = show[col].apply(lambda x: fmt(x, signed=True))
st.dataframe(show, use_container_width=True, height=520)

st.subheader("🔀 会话分裂重点（FADE / 分裂分）")
focus_cols = ["代码", "周状态", "开盘regime", "FADE", "盘前%", "开盘15m%", "盘中RTH%", "缺口回补%", "会话分裂分", "近窗跟随"]
st.dataframe(show[focus_cols], use_container_width=True, height=360)

st.subheader("🏷️ 跟 vs 不跟假设")
st.dataframe(
    show[show["代码"] != BENCHMARK][["代码", "近窗跟随", "近窗同向率", "近窗超额%", "近窗回撤%", "原因假设", "跟随标签"]],
    use_container_width=True,
    height=320,
)

plot_df = table[table["代码"] != BENCHMARK].dropna(subset=["与QQQ相关"])
if not plot_df.empty:
    st.subheader("📊 与 QQQ 相关")
    fig = px.bar(
        plot_df.sort_values("与QQQ相关", ascending=True),
        x="与QQQ相关", y="代码", color="开盘regime", orientation="h",
        height=max(420, len(plot_df) * 16),
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("策略读法（已写入预警）")
st.markdown(
    "1. **盘前** = 情报（薄流动性/影子定价），默认不作为唯一入场依据。\n\n"
    "2. **开盘 regime**：`CONTINUATION` 确认盘前方向；`FADE` 为正式流动性打回；`CHOP` 等待。\n\n"
    "3. **周状态**：`DISTRIBUTE` + `FADE` → 偏收割、慎追盘前；"
    "`UP_LEG` + `CONTINUATION` → 趋势跟随更好；"
    "`ACCUMULATE` + 超跌跟随 → 观察修复。\n\n"
    "4. **权重撕裂 + 盘前单边** → 指数信号可能掺假，看贡献表。\n\n"
    "运行：`pip install -r requirements.txt` 然后 `streamlit run streamlit_app.py`"
)
# =========================
# 展示：周期扫描 / 强者修复 / 大盘打乱规律 / 深跌基本面
# =========================

st.divider()
st.subheader("📡 规律周期扫描（网红票/观察池）")

cycle_rows = []
disrupt_alerts = []
deep_list = []

for t in close.columns:
    if t == BENCHMARK:
        continue
    try:
        cyc = scan_ticker_cycle(close[t], left=7, right=7)
        dis = index_disrupts_cycle(close[t], close[BENCHMARK], cyc)
        dd = deep_drawdown_flags(close[t], thresh=-0.15)
        row = {"代码": t, **cyc, "规律被大盘打乱": dis["打乱"], "打乱说明": dis["说明"]}
        cycle_rows.append(row)
        if dis["打乱"] == 1:
            disrupt_alerts.append(f"⚠ {t} 规律可能被大盘打乱：{dis['说明']}")
        if dd["深跌"] == 1:
            deep_list.append((t, dd["回撤%"]))
    except Exception:
        continue

cycle_df = pd.DataFrame(cycle_rows)
if not cycle_df.empty:
    cycle_df = cycle_df.sort_values("规律分", ascending=False)
    show_c = cycle_df.copy()
    for col in ["底间隔中位", "顶间隔中位", "底CV", "顶CV"]:
        if col in show_c.columns:
            show_c[col] = show_c[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    st.dataframe(show_c, use_container_width=True, height=380)
    st.caption("规律分越高 = 顶/底间隔越整齐。下一次窗口为中位间隔 ±5 日，需人工确认，非自动买卖信号。")
else:
    st.write("周期扫描暂无结果")

if disrupt_alerts:
    st.subheader("🌪 大盘打乱原规律 — 提示")
    for a in disrupt_alerts[:40]:
        st.write("· " + a)
else:
    st.write("当前未检测到明显的「窗口内被指数绑架」样本。")

st.divider()
st.subheader("💪 强者恒强（回撤修复 LEADER_BOUNCE）")

events = detect_qqq_pullback_events(close[BENCHMARK], thresh=-0.02, max_events=5)
if not events:
    st.write("近端未识别到足够的 QQQ 回撤事件（可降低阈值或加长历史）。")
    leader_df = pd.DataFrame()
else:
    last_ev = events[-1]
    st.write(
        f"最近事件：高点 {pd.Timestamp(last_ev['peak_date']).date()} → "
        f"低点 {pd.Timestamp(last_ev['low_date']).date()}，QQQ回撤 {last_ev['dd']*100:.2f}%"
    )
    lb_rows = []
    for t in close.columns:
        if t == BENCHMARK:
            continue
        m = leader_bounce_metrics(close[t], close[BENCHMARK], last_ev)
        lb_rows.append({
            "代码": t,
            "标签": m["标签"],
            "承压比": m["承压比"],
            "个股回撤%": m["个股回撤%"],
            "指数回撤%": m["指数回撤%"],
            "修复天数": m["修复天数"],
            "修复比": m["修复比"],
        })
    leader_df = pd.DataFrame(lb_rows)
    if not leader_df.empty:
        leader_df = leader_df.sort_values(["标签", "修复比"], ascending=[True, False])
        show_l = leader_df.copy()
        for col in ["承压比", "修复比"]:
            show_l[col] = show_l[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        for col in ["个股回撤%", "指数回撤%"]:
            show_l[col] = show_l[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(show_l, use_container_width=True, height=360)
        leaders = leader_df[leader_df["标签"] == "LEADER_BOUNCE"]["代码"].tolist()
        if leaders:
            st.success("LEADER_BOUNCE：同一回撤中相对抗跌或修复更快 → " + ", ".join(leaders[:20]))

st.divider()
st.subheader("🔍 深跌提示 → 查询基本面")
st.caption("近60日相对高点回撤 ≤ -15% 的标的，建议核对业绩/负债/指引（数据来自 yfinance，仅供参考）。")

if deep_list:
    deep_list = sorted(deep_list, key=lambda x: x[1])
    st.write("深跌列表：" + ", ".join([f"{t}({v:.1f}%)" for t, v in deep_list[:30]]))
    pick = st.selectbox("选择深跌标的查看基本面", [t for t, _ in deep_list])
    if pick:
        fund = fetch_fundamentals_quick(pick)
        st.json(fund)
else:
    st.write("当前观察池无触发深跌阈值的标的。")
if auto:
    remain = max(0, REFRESH_SECONDS - int(time.time() - st.session_state.ndx_last_ts))
    st.caption(f"距自动刷新约 {remain // 60} 分 {remain % 60} 秒（保持页面打开）")
    time.sleep(min(30, max(1, remain)))
    st.rerun()
