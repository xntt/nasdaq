"""
纳斯达克联动 · 规律 · 权重贡献 · 时段统计（完整更新版）
- 15 分钟自动刷新 + 手动刷新
- QQQ 交替涨跌时段分布（美东）
- 个股跟随标签 + 原因假设
- 权重股对大盘贡献 / 齐动与撕裂
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    # 高权重科技（近似纳指核心）
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "NFLX", "COST",
    # 半导体 / 存储
    "AMD", "MU", "TSM", "ASML", "ARM", "SMCI", "INTC", "QCOM", "LRCX", "AMAT", "KLAC", "WDC", "STX",
    # 生医
    "LLY", "NVO", "VRTX", "REGN", "MRK", "PFE", "AMGN", "ISRG",
    # 能源贵金属
    "XOM", "CVX", "USO", "GLD", "GDX",
    # 热门 ETF / 杠杆
    "SOXX", "SOXL", "SOXS", "LABU", "LABD", "TQQQ", "SQQQ",
]

# 近似纳指/QQQ 权重（%）— 仅用于贡献拆解，需随成分变化自行改
# 合计不必精确 100；会按相对权重归一
NDX_WEIGHT_PCT = {
    "AAPL": 9.0,
    "MSFT": 8.5,
    "NVDA": 8.0,
    "AMZN": 5.5,
    "META": 5.0,
    "GOOGL": 4.5,  # 若同时有 GOOG 可再拆
    "AVGO": 4.5,
    "TSLA": 3.0,
    "COST": 2.5,
    "NFLX": 2.0,
    "AMD": 1.5,
    "QCOM": 1.2,
    "TMUS": 1.2,
    "CSCO": 1.1,
    "ADBE": 1.0,
    "PEP": 1.0,
    "LIN": 1.0,
    "INTU": 0.9,
    "INTC": 0.8,
    "AMAT": 0.8,
}

LEVERAGE_MAP = {
    "QQQ": {"bull2_3": ["TQQQ"], "bear2_3": ["SQQQ"]},
    "SOXX": {"bull2_3": ["SOXL"], "bear2_3": ["SOXS"]},
    "MU": {"bull2_3": ["SOXL"], "bear2_3": ["SOXS"], "note": "行业杠杆对照"},
    "NVDA": {"bull2_3": ["SOXL", "TQQQ"], "bear2_3": ["SOXS", "SQQQ"], "note": "高相关代理"},
    "AMD": {"bull2_3": ["SOXL"], "bear2_3": ["SOXS"], "note": "行业杠杆对照"},
    "WDC": {"bull2_3": ["SOXL"], "bear2_3": ["SOXS"], "note": "存储对照"},
    "STX": {"bull2_3": ["SOXL"], "bear2_3": ["SOXS"], "note": "存储对照"},
}

# 时段桶（美东小时）
TIME_BUCKETS = [
    ("夜盘凌晨 0-4", 0, 4),
    ("凌晨 4-8", 4, 8),
    ("早盘前 8-9:30", 8, 9),  # 粗分到整点；9:30 归入 RTH
    ("常规早盘 9-12", 9, 12),
    ("午盘 12-16", 12, 16),
    ("尾盘后 16-20", 16, 20),
    ("晚间 20-24", 20, 24),
]


# =========================
# 数据
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


def session_returns(intraday: pd.DataFrame) -> Dict[str, Optional[float]]:
    out = {"pre_pct": None, "rth_pct": None, "post_pct": None, "open_reverse": None}
    if intraday is None or intraday.empty:
        return out
    df = intraday.copy()
    df["date"] = df["dt"].dt.date
    df["hm"] = df["dt"].dt.hour * 100 + df["dt"].dt.minute
    last_day = df["date"].max()
    day = df[df["date"] == last_day].sort_values("dt")
    if len(day) < 3:
        return out

    def seg(mask):
        seg_df = day[mask]
        if len(seg_df) < 2:
            return None
        a, b = float(seg_df["Close"].iloc[0]), float(seg_df["Close"].iloc[-1])
        return (b / a - 1) * 100 if a else None

    pre = seg(day["hm"] < 930)
    rth = seg((day["hm"] >= 930) & (day["hm"] <= 1600))
    post = seg(day["hm"] > 1600)
    out["pre_pct"], out["rth_pct"], out["post_pct"] = pre, rth, post
    if pre is not None and rth is not None and abs(pre) > 0.15 and abs(rth) > 0.15:
        out["open_reverse"] = 1 if pre * rth < 0 else 0
    return out


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
        label = label + "（跟一段易断）"
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
        return {"gaps": [], "median_gap": None, "tops": [], "note": "数据不足"}
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
    return {"gaps": gaps[-8:], "median_gap": med, "tops": [str(t.date()) for t in tops[-6:]], "note": note}


def weekday_stats(close: pd.Series) -> pd.DataFrame:
    r = close.pct_change().dropna()
    idx = r.index
    wd = idx.tz_convert("America/New_York").dayofweek if getattr(idx, "tz", None) else idx.dayofweek
    tmp = pd.DataFrame({"ret": r.values, "wd": wd})
    g = tmp.groupby("wd")["ret"].agg(["mean", "count"])
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    g = g.reindex(range(5))
    g.index = names
    g["mean_pct"] = g["mean"] * 100
    return g


# =========================
# 新增：时段涨跌统计（单边交替时看“何时动”）
# =========================

def qqq_move_timing_stats(intraday: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    每个交易日找「涨幅最大的 30m 柱」和「跌幅最大的 30m 柱」的发生小时，
    汇总到时段桶。用于观察单边/交替行情里动能常在哪个时间点释放。
    """
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
        up_idx = g["ret"].idxmax()
        dn_idx = g["ret"].idxmin()
        up_h = int(g.loc[up_idx, "hour"])
        dn_h = int(g.loc[dn_idx, "hour"])
        up_hours.append(up_h)
        down_hours.append(dn_h)
        day_rows.append(
            {
                "日期": str(d),
                "日涨跌%": day_ret,
                "最大单柱上涨时段(ET小时)": up_h,
                "最大单柱下跌时段(ET小时)": dn_h,
                "最大单柱涨%": float(g.loc[up_idx, "ret"]) * 100,
                "最大单柱跌%": float(g.loc[dn_idx, "ret"]) * 100,
            }
        )

    detail = pd.DataFrame(day_rows)

    def bucket_count(hours: List[int], name: str) -> pd.DataFrame:
        rows = []
        for label, a, b in TIME_BUCKETS:
            c = sum(1 for h in hours if a <= h < b)
            rows.append({"时段(美东)": label, "类型": name, "次数": c})
        return pd.DataFrame(rows)

    summary = pd.concat(
        [bucket_count(up_hours, "日内最强上攻柱"), bucket_count(down_hours, "日内最强下杀柱")],
        ignore_index=True,
    )
    return summary, detail


# =========================
# 新增：跟随标签 + 原因假设（超跌/滞涨等）
# =========================

def tag_follow_reason(
    close: pd.DataFrame,
    ticker: str,
    bench: str,
    rets: pd.DataFrame,
    window: int = 10,
) -> Dict[str, Any]:
    if ticker not in close.columns or bench not in close.columns:
        return {"近窗同向率": None, "超额%": None, "回撤%": None, "标签": "—", "假设": "—"}

    r = rets[ticker].dropna().tail(window)
    b = rets[bench].reindex(r.index).dropna()
    r = r.reindex(b.index)
    if len(r) < max(5, window // 2):
        return {"近窗同向率": None, "超额%": None, "回撤%": None, "标签": "数据不足", "假设": "—"}

    same = float((np.sign(r) == np.sign(b)).mean())
    # 近窗累计超额
    excess = float((1 + r).prod() - (1 + b).prod()) * 100
    # 近 20 日回撤（相对前高）
    px = close[ticker].dropna().tail(25)
    dd = None
    if len(px) >= 10:
        dd = float(px.iloc[-1] / px.max() - 1) * 100

    # 标签
    if same >= 0.7 and excess >= 0:
        tag = "积极跟随偏强"
    elif same >= 0.7 and excess < 0:
        tag = "跟随但跑输"
    elif same <= 0.4 and excess > 1:
        tag = "不跟却偏强（独立/滞涨后补涨?）"
    elif same <= 0.4 and excess < -1:
        tag = "不跟且偏弱"
    else:
        tag = "弱相关摇摆"

    # 假设引擎
    hyp = []
    if same >= 0.7 and dd is not None and dd <= -8:
        hyp.append("超跌后跟大盘修复的概率偏高")
    if same >= 0.7 and excess > 3:
        hyp.append("高 Beta/情绪共振，跟涨放大")
    if same <= 0.45 and excess > 2 and (dd is None or dd > -5):
        hyp.append("可能滞涨后补涨或有个股逻辑独立走强")
    if same <= 0.45 and excess < -2:
        hyp.append("可能个股利空/资金撤离，或相对大盘已经透支")
    if same >= 0.6 and dd is not None and dd > -3 and excess < 0:
        hyp.append("位置偏高仍跟跌，警惕滞涨转弱")
    if not hyp:
        hyp.append("暂无清晰规则命中，需结合新闻与量能")

    return {
        "近窗同向率": same,
        "超额%": excess,
        "回撤%": dd,
        "标签": tag,
        "假设": "；".join(hyp),
    }


# =========================
# 新增：权重股贡献与齐动分析
# =========================

def weight_contribution(close: pd.DataFrame, weights: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    当日近似贡献 = 相对权重 * 个股日涨跌。
    注：不是官方指数点位，只用于解释「谁在拖动大盘」。
    """
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
        contrib = wn * r * 100  # 贡献百分点（相对权重篮子）
        total_weighted += contrib
        rows.append(
            {
                "代码": t,
                "近似权重%": round(w, 2),
                "日涨跌%": r * 100,
                "近似贡献": contrib,
            }
        )
    df = pd.DataFrame(rows).sort_values("近似贡献", ascending=False)
    qqq_r = float(rets_1d.get(BENCHMARK, np.nan))
    qqq_pct = qqq_r * 100 if not np.isnan(qqq_r) else None

    if df.empty:
        return df, {}

    top = df.iloc[0]
    bottom = df.iloc[-1]
    up_n = int((df["日涨跌%"] > 0).sum())
    dn_n = int((df["日涨跌%"] < 0).sum())
    n = len(df)
    up_ratio = up_n / n if n else 0

    if up_ratio >= 0.7:
        regime = "权重股多数同向向上 → 大盘易启动/延续偏多"
    elif up_ratio <= 0.3:
        regime = "权重股多数同向向下 → 大盘易走弱"
    elif 0.4 <= up_ratio <= 0.6:
        regime = "权重涨跌接近各半 → 易横盘或内部轮动变盘前夜"
    else:
        regime = "权重分化中等 → 指数可能窄幅波动"

    # 主导叙事
    if abs(top["近似贡献"]) >= abs(bottom["近似贡献"]):
        driver = f"今日权重篮子上拉主导：{top['代码']}（涨跌 {top['日涨跌%']:+.2f}% / 贡献 {top['近似贡献']:+.3f}）"
    else:
        driver = f"今日权重篮子下拉主导：{bottom['代码']}（涨跌 {bottom['日涨跌%']:+.2f}% / 贡献 {bottom['近似贡献']:+.3f}）"

    summary = {
        "QQQ日涨跌%": qqq_pct,
        "权重篮子加权涨跌%": total_weighted,
        "上涨权重家数": up_n,
        "下跌权重家数": dn_n,
        "上涨占比": up_ratio,
        "结构判断": regime,
        "主导标注": driver,
        "前3贡献": df.head(3)[["代码", "日涨跌%", "近似贡献"]].to_dict("records"),
        "后3拖累": df.tail(3)[["代码", "日涨跌%", "近似贡献"]].to_dict("records"),
    }
    return df, summary


def multi_day_heavy_move(close: pd.DataFrame, weights: Dict[str, float], days: int = 5) -> pd.DataFrame:
    """近几日权重股齐动得分：同向比例"""
    avail = [k for k in weights if k in close.columns]
    if len(avail) < 5:
        return pd.DataFrame()
    rets = close[avail].pct_change().dropna(how="all").tail(days)
    rows = []
    for dt, row in rets.iterrows():
        vals = row.dropna()
        if len(vals) < 5:
            continue
        up = (vals > 0).mean()
        rows.append(
            {
                "日期": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "权重上涨占比": float(up),
                "解读": (
                    "共振偏多" if up >= 0.7 else ("共振偏空" if up <= 0.3 else "撕裂/轮动")
                ),
            }
        )
    return pd.DataFrame(rows)


# =========================
# UI
# =========================

st.set_page_config(page_title="纳指联动规律监控", page_icon="📈", layout="wide")
st.title("📈 纳斯达克联动 · 时段 · 权重贡献 · 跟随假设")
st.caption("基准 QQQ｜15 分钟自动刷新 + 手动刷新｜统计为近似研究工具，非投资建议")

with st.sidebar:
    st.header("观察池")
    base = st.text_area("列表（每行一个）", value="\n".join(DEFAULT_TICKERS), height=200)
    extra = st.text_input("添加代码（逗号分隔）", placeholder="SNDK, PLTR")
    remove = st.text_input("移除代码（逗号分隔）", "")
    period = st.selectbox("日线窗口", ["3mo", "6mo", "1y", "2y"], index=1)
    follow_window = st.slider("跟随标签窗口（交易日）", 5, 20, 10)
    alert_corr = st.slider("预警：相关低于", 0.0, 1.0, 0.25, 0.05)
    alert_pre_gap = st.number_input("预警：|盘前-盘中|%", value=1.5, step=0.1)
    st.divider()
    manual = st.button("🔄 手动刷新", type="primary", use_container_width=True)
    auto = st.checkbox("约 15 分钟自动刷新", value=True)

tickers = [x.strip().upper() for x in base.splitlines() if x.strip()]
if extra:
    tickers += [x.strip().upper() for x in extra.split(",") if x.strip()]
if remove:
    rm = {x.strip().upper() for x in remove.split(",") if x.strip()}
    tickers = [t for t in tickers if t not in rm]
# 权重股强制进入下载列表
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
        st.error("无法获取 QQQ/行情，请重试。")
        st.stop()

    rets = close.pct_change()
    bench_r = rets[BENCHMARK]

    # QQQ 时段
    qqq_intra = download_intraday(BENCHMARK, period="15d", interval="30m")
    timing_summary, timing_detail = qqq_move_timing_stats(qqq_intra)
    q_alt = alternation_score(close[BENCHMARK])
    q_top = local_top_gaps(close[BENCHMARK])
    q_wd = weekday_stats(close[BENCHMARK])

    # 权重贡献
    contrib_df, contrib_sum = weight_contribution(close, NDX_WEIGHT_PCT)
    heavy_days = multi_day_heavy_move(close, NDX_WEIGHT_PCT, days=8)

    # 个股表
    rows = []
    for t in close.columns:
        if t not in rets.columns:
            continue
        fm = follow_metrics(rets[t], bench_r)
        alt = alternation_score(close[t])
        tops = local_top_gaps(close[t])
        tag = tag_follow_reason(close, t, BENCHMARK, rets, window=follow_window)
        last = float(close[t].dropna().iloc[-1])
        chg1 = float(rets[t].dropna().iloc[-1]) * 100 if rets[t].notna().any() else None

        sess = {"pre_pct": None, "rth_pct": None, "post_pct": None, "open_reverse": None}
        # 限流：只对前 35 个代码拉分钟线
        if list(close.columns).index(t) < 35 or t in NDX_WEIGHT_PCT or t == BENCHMARK:
            try:
                sess = session_returns(download_intraday(t, period="5d", interval="30m"))
            except Exception:
                pass

        rows.append(
            {
                "代码": t,
                "最新价": last,
                "日涨跌%": chg1,
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
                "交替标签": alt["label"],
                "局部顶中位间隔日": tops["median_gap"],
                "顶间隔提示": tops["note"],
                "近端局部顶": ", ".join(tops.get("tops") or []),
                "盘前%": sess["pre_pct"],
                "盘中%": sess["rth_pct"],
                "盘后%": sess["post_pct"],
                "开盘反转": sess["open_reverse"],
            }
        )

    table = pd.DataFrame(rows)

# ===== 展示 =====
ts = datetime.fromtimestamp(st.session_state.ndx_last_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.info(f"标的 {len(table)}｜基准 {BENCHMARK}｜最后更新 {ts}")

# --- 权重股影响（你强调的必须模块）---
st.subheader("⚖️ 权重股对大盘的影响（近似贡献）")
if contrib_sum:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("QQQ 日涨跌%", f"{contrib_sum['QQQ日涨跌%']:+.2f}%" if contrib_sum["QQQ日涨跌%"] is not None else "—")
    m2.metric("权重篮子加权%", f"{contrib_sum['权重篮子加权涨跌%']:+.3f}")
    m3.metric("权重上涨家数", f"{contrib_sum['上涨权重家数']}")
    m4.metric("权重下跌家数", f"{contrib_sum['下跌权重家数']}")
    st.success(contrib_sum["主导标注"])
    st.warning(contrib_sum["结构判断"])
    st.caption(
        "例：若 AAPL 大涨且贡献排名第一，可标注「指数新高/大涨由苹果权重拉动」。"
        "权重表为近似，请按最新 QQQ/NDX 权重自行改 NDX_WEIGHT_PCT。"
    )
    c_a, c_b = st.columns(2)
    with c_a:
        st.write("贡献 Top")
        st.dataframe(contrib_df.head(8), use_container_width=True)
    with c_b:
        st.write("拖累 Top")
        st.dataframe(contrib_df.tail(8).iloc[::-1], use_container_width=True)
    if not heavy_days.empty:
        st.write("近几日权重齐动")
        st.dataframe(heavy_days, use_container_width=True)
else:
    st.write("权重贡献计算失败（缺数据）")

st.divider()

# --- 时段统计 ---
st.subheader("🕒 单边/交替行情中的「何时在动」（QQQ 30m 近似）")
st.write(f"QQQ 日线涨跌交替：{q_alt['label']}（rate={q_alt.get('alt_rate')}）｜{q_top.get('note')}")
if not timing_summary.empty:
    fig_t = px.bar(
        timing_summary,
        x="时段(美东)",
        y="次数",
        color="类型",
        barmode="group",
        title="每日最强上攻柱 / 最强下杀柱 出现在哪些时段",
    )
    st.plotly_chart(fig_t, use_container_width=True)
    with st.expander("分日明细（最大单柱对应小时）"):
        st.dataframe(timing_detail, use_container_width=True)
else:
    st.write("分钟线不足，无法统计时段")

st.write("QQQ 星期效应（日均涨跌%）")
st.dataframe(q_wd[["mean_pct", "count"]], use_container_width=True)

st.divider()

# --- 预警 ---
alerts = []
if contrib_sum:
    alerts.append(f"权重结构：{contrib_sum['结构判断']}")
    alerts.append(f"主导：{contrib_sum['主导标注']}")
for _, r in table.iterrows():
    if r["代码"] == BENCHMARK:
        continue
    if r["与QQQ相关"] is not None and r["与QQQ相关"] < alert_corr:
        alerts.append(f"⚠ {r['代码']} 低相关 ({r['与QQQ相关']:.2f})｜{r['近窗跟随']}｜{r['原因假设']}")
    pre, rth = r["盘前%"], r["盘中%"]
    if pre is not None and rth is not None and abs(pre - rth) >= alert_pre_gap:
        alerts.append(f"🔔 {r['代码']} 盘前({pre:+.2f}%) vs 盘中({rth:+.2f}%)")
    if r["开盘反转"] == 1:
        alerts.append(f"🔄 {r['代码']} 开盘后近似反转")
    if r.get("顶间隔提示") and "2 周" in str(r["顶间隔提示"]):
        alerts.append(f"📅 {r['代码']} {r['顶间隔提示']}；{r['近端局部顶']}")

st.subheader("🚨 预警与标签")
for a in alerts[:50]:
    st.write("· " + a)

st.divider()
st.subheader("📋 总表（含近窗跟随假设）")
show = table.copy()
for col in ["与QQQ相关", "Beta", "同向率", "近窗同向率", "涨跌交替率"]:
    if col in show.columns:
        show[col] = show[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
for col in ["日涨跌%", "近窗超额%", "近窗回撤%", "盘前%", "盘中%", "盘后%"]:
    if col in show.columns:
        show[col] = show[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
st.dataframe(show, use_container_width=True, height=520)

st.subheader("🏷️ 反复行情：跟 vs 不跟")
tag_view = table[table["代码"] != BENCHMARK][
    ["代码", "近窗跟随", "近窗同向率", "近窗超额%", "近窗回撤%", "原因假设", "跟随标签"]
].copy()
st.dataframe(tag_view, use_container_width=True, height=360)

st.subheader("📊 与 QQQ 相关性")
plot_df = table[table["代码"] != BENCHMARK].dropna(subset=["与QQQ相关"])
if not plot_df.empty:
    # 恢复数值列用于作图
    plot_df = table[table["代码"] != BENCHMARK].dropna(subset=["与QQQ相关"])
    fig = px.bar(
        plot_df.sort_values("与QQQ相关", ascending=True),
        x="与QQQ相关",
        y="代码",
        color="近窗跟随",
        orientation="h",
        height=max(420, len(plot_df) * 16),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
### 本版如何对应你的需求
1. **单边仍一天涨一天跌 → 看时间点**  
   「时段」图统计每天最强上攻/下杀 30m 柱落在美东哪个时段（4点、8点、16点、20点等桶）。
2. **反复市跟/不跟打标签 + 假设**  
   「近窗跟随」「原因假设」列：超跌跟随、滞涨独立、高位跟跌等。
3. **权重股拖动大盘**  
   贡献表 + 主导标注（如苹果贡献第一）+ 齐动占比（共振/撕裂）。
4. **刷新**  
   侧边栏手动刷新；勾选后约每 15 分钟自动 `st.rerun()`。

### 使用注意
- 权重 `NDX_WEIGHT_PCT` 请按最新持仓改，否则贡献解释会偏。  
- 时段依赖 30m 数据，盘前/夜盘覆盖不完整时桶会偏 RTH。  
- 假设是规则引擎，不是因果证明。
"""
)

# 自动刷新
if auto:
    remain = max(0, REFRESH_SECONDS - int(time.time() - st.session_state.ndx_last_ts))
    st.caption(f"距自动刷新约 {remain // 60} 分 {remain % 60} 秒（保持页面打开）")
    time.sleep(min(30, max(1, remain)))
    st.rerun()
