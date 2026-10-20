"""
量化引擎 v5（不依賴網路與 LLM，可單獨測試）

  resolve_exit     回測與結算「共用」的單根日 K 出場判定（跳空、同日停損優先、盤中觸價）
  add_indicators   技術指標（RSI/ATR 為 Wilder 平滑）
  entry_signals    回測進場訊號（只用當日與過去資料，無未來函數）
  backtest         10 年回測 + bootstrap 信賴區間 + 買進持有基準
  simulate_plan    收盤價路徑蒙地卡羅（建倉點單一價；漂移預設 0，另算漂移×0.5 僅供顯示）
  kelly_fraction   凱利部位
  戰績 DB          全部分析皆入庫（含觀望），raw_json 快取、版本紀錄、可設定路徑

價格口徑：回測 / 模擬 / 結算一律使用「還原價」；畫面顯示與建倉點、停損價使用「原始價」。
兩者換算只發生在 evaluate_open（以還原因子 = 還原收盤 / 原始收盤 轉換價位），不可混用。
"""
import json
import logging
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

QUANT_VERSION = "5.0"
log = logging.getLogger("tw_dash.quant")

# ------------------------------------------------------------------ 常數
COST_STOCK = 0.00585     # 手續費 0.1425%×2 + 證交稅 0.3%（未計券商折讓，偏保守）
COST_ETF = 0.00385       # 手續費 0.1425%×2 + 證交稅 0.1%
HOLD_DAYS = 20           # 進場後最長持有交易日
ENTRY_WAIT = 10          # 等待建倉點成交的最長交易日
MAX_STOP_PCT = 8.0       # 停損離建倉點最大允許 %
MAX_STOP_ATR = 4.0       # 停損離建倉點最大允許 ATR 倍數
EV_MIN_R = 0.1           # 模擬期望值門檻（R）：低於此值不給買進
MIN_TRADES = 12          # 回測最少交易次數
BOOT_N = 2000            # bootstrap 次數
MIN_CLOSED_SAMPLES = 30  # 戰績已結算筆數低於此值 → 樣本不足
DEFAULT_DRIFT_SHRINK = 0.0   # 模擬漂移：預設不假設未來延續歷史漲勢
COMPARE_DRIFT_SHRINK = 0.5   # 另算一組漂移×0.5（僅顯示，不用於把關）


# ------------------------------------------------------------------ 代碼 / ETF / 成本 / 升降單位
_FULLWIDTH = str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
                           "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_code(raw: str, names_rev: dict | None = None) -> str:
    """
    統一股票代碼：去空白、全形轉半角、移除 .TW/.TWO/TPE: 前後綴、中文名稱轉代碼、
    不足 4 碼補 0（50 → 0050）。格式不符丟 ValueError。例：' ２３３０.TW ' → '2330'，'台積電' → '2330'
    """
    s = str(raw).strip().translate(_FULLWIDTH).upper()
    if names_rev and s in names_rev:
        return names_rev[s]
    s = re.sub(r"^(TPE:|TWSE:|TWO:)", "", s)
    s = re.sub(r"\.(TWO|TW)$", "", s)
    if s.isdigit() and len(s) < 4:
        s = s.zfill(4)
    if not re.fullmatch(r"\d{4,6}[A-Z]?", s):
        raise ValueError(f"無法辨識的股票代碼：{raw!r}（請輸入如 2330、0050、00878、006208、00937B）")
    return s


def is_etf(code: str, industry: str | None = None) -> bool:
    """ETF 判斷：FinMind 產業別為 ETF/ETN，或代碼以 00 開頭。"""
    return str(code).startswith("00") or (industry or "").strip().upper() in ("ETF", "ETN")


def cost_rt(code: str, industry: str | None = None) -> float:
    """來回交易成本：股票 0.585%（證交稅 0.3%）、ETF 0.385%（證交稅 0.1%）。"""
    return COST_ETF if is_etf(code, industry) else COST_STOCK


def tick_size(p: float, etf: bool = False) -> float:
    if etf:                                   # ETF：50 元以下 0.01、50 元以上 0.05
        return 0.01 if p < 50 else 0.05
    return (0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100
            else 0.5 if p < 500 else 1 if p < 1000 else 5)


def round_tick(p: float, etf: bool = False) -> float:
    t = tick_size(p, etf)
    return round(math.floor(p / t + 0.5) * t, 2)


# ------------------------------------------------------------------ 指標與進場訊號
def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """技術指標。RSI、ATR 皆為 Wilder 平滑（ewm alpha=1/14, adjust=False）。"""
    d = hist[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    c = d["Close"]
    for n in (5, 20, 60, 120):
        d[f"ma{n}"] = c.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(),
                    (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return d


def entry_signals(d: pd.DataFrame) -> pd.Series:
    """
    多頭拉回訊號：MA20>MA60、收盤>MA60、MA60 向上、收盤在 MA20 −3%~+2%、RSI 35~60。
    每個日期的訊號只用「當天與更早」的資料（rolling/ewm/shift 都是因果的），無未來函數。
    """
    c, ma20, ma60, rsi = d["Close"], d["ma20"], d["ma60"], d["rsi"]
    sig = ((ma20 > ma60) & (c > ma60) & (ma60 > ma60.shift(10))
           & (c >= 0.97 * ma20) & (c <= 1.02 * ma20) & (rsi >= 35) & (rsi <= 60))
    return sig.fillna(False)


def propose_levels(d: pd.DataFrame, etf: bool = False) -> dict:
    """
    機械式基準價位（原始價）：
      建倉點：回測 5 日線（現價已在 5 日線下就用現價）
      加碼點：月線離建倉點夠遠 → 「回測月線」；否則 → 「突破壓力」（20 日高點上方）
      停損：近 10 日低點 / 建倉−1ATR 取低再減 0.5ATR，限制在 8% 與 4ATR 之內；目標：2.5R / 4R
    """
    x = d.iloc[-1]
    price, atr = float(x["Close"]), float(x["atr"])
    ma5, ma20 = float(x["ma5"]), float(x["ma20"])
    entry = min(price, ma5)
    swing = float(d["Low"].tail(10).min())
    stop = min(swing, entry - atr) - 0.5 * atr
    stop = max(stop, entry * (1 - MAX_STOP_PCT / 100 * 0.9), entry - (MAX_STOP_ATR - 0.5) * atr)
    if ma20 < entry - 0.5 * atr:
        add, add_type = ma20, "回測月線"
    else:
        add, add_type = max(float(d["High"].tail(20).max()) * 1.003, price + 0.5 * atr), "突破壓力"
    risk = entry - stop
    out = {k: round_tick(v, etf) for k, v in {
        "entry_price": entry, "add_price": add, "stop_loss": stop,
        "target1": entry + 2.5 * risk, "target2": entry + 4 * risk}.items()}
    out["add_type"] = add_type
    out["entry_basis"] = "回測5日線" if entry < price else "現價附近"
    return out


# ------------------------------------------------------------------ 共用出場規則
def resolve_exit(o, h, l, stop, target, first_day: bool = False):
    """
    單根日 K 的出場判定（回測與結算共用，兩邊口徑一致）。回傳 (出場價, 原因) 或 None（續抱）。
      1. 跳空（first_day=False 才檢查；進場當天開盤即成交價，不適用）：
           開盤 ≤ 停損 → 以開盤價出場（gap_stop）；開盤 ≥ 目標 → 以開盤價出場（gap_target）
      2. 盤中：最低 ≤ 停損 → 以停損價成交（stop）——同日同時觸及停損與目標，停損優先；
               最高 ≥ 目標 → 以目標價成交（target）
    """
    if not first_day:
        if o <= stop:
            return float(o), "gap_stop"
        if o >= target:
            return float(o), "gap_target"
    if l <= stop:
        return float(stop), "stop"
    if h >= target:
        return float(target), "target"
    return None


# ------------------------------------------------------------------ 回測
def _stats(R: np.ndarray) -> dict:
    if len(R) == 0:
        return {"交易次數": 0}
    pos, neg = R[R > 0], R[R <= 0]
    cum = np.cumsum(R)
    return {
        "交易次數": int(len(R)),
        "勝率%": round(float((R > 0).mean() * 100), 1),
        "平均獲利R": round(float(pos.mean()), 2) if len(pos) else 0.0,
        "平均虧損R": round(float(neg.mean()), 2) if len(neg) else 0.0,
        "期望值R": round(float(R.mean()), 2),
        "獲利因子": round(float(pos.sum() / abs(neg.sum())), 2) if neg.sum() < 0 else 99.0,
        "最大連續回撤R": round(float((np.maximum.accumulate(cum) - cum).max()), 2),
    }


def _boot_ci(R: np.ndarray, n: int = BOOT_N, seed: int = 11):
    """對交易 R 值重抽樣，回傳期望值的 (5%, 95%) 分位數（90% 區間）；樣本 < 2 回傳 None。"""
    if len(R) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(R, size=(n, len(R)), replace=True).mean(axis=1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def _benchmark(d: pd.DataFrame, start: int) -> dict:
    """回測期間（第一個可交易日起）買進持有的年化報酬與最大回撤；僅供參考，不影響評級。"""
    c = d["Close"].iloc[start:]
    if len(c) < 30:
        return {}
    years = len(c) / 252
    cagr = (float(c.iloc[-1]) / float(c.iloc[0])) ** (1 / years) - 1
    mdd = float(((c / c.cummax()) - 1).min())
    return {"回測期間": f"{c.index[0]:%Y-%m} ~ {c.index[-1]:%Y-%m}（{years:.1f} 年）",
            "買進持有年化%": round(cagr * 100, 1), "買進持有最大回撤%": round(mdd * 100, 1)}


def backtest(d: pd.DataFrame, cost: float = COST_STOCK) -> dict:
    """
    多頭拉回策略（固定規則、無擬合參數；請傳入「還原價」指標 DataFrame）：
      進場：訊號日隔日開盤；停損：近10日低點/進場−1ATR 取低再減 0.5ATR（風險 > 10% 不進場）
      目標：2.5R；最長持有 20 日（到期以收盤價出場）；出場判定用 resolve_exit（與結算共用）
    edge_ok：交易次數 ≥ 12、期望值 90% 區間下緣（bootstrap 5% 分位）> 0、獲利因子 > 1.2、近 1 年期望值 > 0（≥3 筆時）
    """
    o, h, l, c = (d[k].values for k in ("Open", "High", "Low", "Close"))
    atr = d["atr"].values
    sig = entry_signals(d).values
    n = len(d)
    R, idx = [], []
    start = 125
    i = start
    while i < n - 2:
        if not sig[i]:
            i += 1
            continue
        entry = o[i + 1]
        stop = min(l[i - 9:i + 1].min(), entry - atr[i]) - 0.5 * atr[i]
        risk = entry - stop
        if not np.isfinite(risk) or risk <= 0 or risk / entry > 0.10:
            i += 1
            continue
        target = entry + 2.5 * risk
        last = min(i + HOLD_DAYS, n - 1)
        exit_px, j = c[last], last
        for j in range(i + 1, last + 1):
            res = resolve_exit(o[j], h[j], l[j], stop, target, first_day=(j == i + 1))
            if res:
                exit_px = res[0]
                break
        R.append((exit_px - entry) / risk - cost * entry / risk)
        idx.append(i)
        i = j + 1
    R, idx = np.array(R), np.array(idx)
    recent = R[idx >= n - 252] if len(R) else R
    allst, recst = _stats(R), _stats(recent)
    ci = _boot_ci(R)
    if ci:
        allst["期望值R_90%區間"] = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
    edge = bool(allst["交易次數"] >= MIN_TRADES and ci is not None and ci[0] > 0
                and allst["獲利因子"] > 1.2 and (recst["交易次數"] < 3 or recst["期望值R"] > 0))
    return {"全期間": allst, "近1年": recst, "edge_ok": edge,
            "ci5": ci[0] if ci else None, "ci95": ci[1] if ci else None,
            "基準比較": _benchmark(d, start)}


# ------------------------------------------------------------------ 機率模擬
def _shocks(close: pd.Series, n: int, seed: int):
    """濾波歷史模擬：近 3 年對數報酬除以 EWMA 波動 → 標準化殘差，重抽樣。回傳 (shocks, 目前波動, 歷史平均日報酬)。"""
    r = np.log(close.astype(float)).diff().dropna().values[-750:]
    lam = 0.94
    var = np.empty_like(r)
    var[0] = r[:20].var()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    z = r / np.sqrt(var)
    z = (z - z.mean()) / z.std()
    sig_now = float(np.sqrt(lam * var[-1] + (1 - lam) * r[-1] ** 2))
    T = ENTRY_WAIT + HOLD_DAYS + 1
    rng = np.random.default_rng(seed)
    return rng.choice(z, size=(n, T)), sig_now, float(r.mean())


def simulate_plan(close: pd.Series, price, entry, target, stop, cost: float = COST_STOCK,
                  n: int = 4000, seed: int = 7, drift_shrink: float = DEFAULT_DRIFT_SHRINK):
    """
    建倉點（單一價）計畫的蒙地卡羅。close 請傳「還原收盤價」（只用其報酬分佈），price/entry/target/stop 為原始價。
    ※ 只有收盤價路徑：以「收盤觸價」判斷停損/目標（收盤 ≤ 停損 → 以該收盤價出場；收盤 ≥ 目標 → 以目標價出場）。
      盤中先觸停損、收盤又拉回的路徑不會被計為停損，因此整體「略偏樂觀」（也會漏算盤中觸及目標的情形）；
      與回測/結算（resolve_exit，含盤中觸價與跳空）口徑不同，僅作機率的大致參考。
    ※ 漂移預設 0（不假設未來延續歷史漲勢）；同時以歷史平均漂移×0.5 另算一組，存在 ev_r_drift，只顯示、不把關。
    價位不合邏輯回傳 None。
    """
    risk = entry - stop
    if not (stop < entry < target and risk > 0):
        return None
    shocks, sig_now, mu_hist = _shocks(close, n, seed)
    cost_r = cost * entry / risk
    r_win = (target - entry) / risk - cost_r
    BIG = 10 ** 9

    def run(shrink: float) -> dict:
        steps = mu_hist * shrink + sig_now * shocks
        P = np.hstack([np.full((n, 1), float(price)), price * np.exp(np.cumsum(steps, axis=1))])
        filled = wins = losses = times = 0
        rs, loss_rs, days = [], [], []
        for p in P:
            hit = np.flatnonzero(p[:ENTRY_WAIT + 1] <= entry)
            if hit.size == 0:
                continue
            f = int(hit[0])
            filled += 1
            if p[f] <= stop:                           # 成交當天收盤就已跌破停損
                R = (p[f] - entry) / risk - cost_r
                losses += 1
                rs.append(R)
                loss_rs.append(R)
                continue
            seg = p[f + 1:f + 1 + HOLD_DAYS]
            s_hit, t_hit = np.flatnonzero(seg <= stop), np.flatnonzero(seg >= target)
            s = int(s_hit[0]) if s_hit.size else BIG
            t = int(t_hit[0]) if t_hit.size else BIG
            if s == BIG and t == BIG:
                rs.append((seg[-1] - entry) / risk - cost_r)
                times += 1
            elif s < t:
                R = (seg[s] - entry) / risk - cost_r
                rs.append(R)
                loss_rs.append(R)
                losses += 1
                days.append(s + 1)
            else:
                rs.append(r_win)
                wins += 1
                days.append(t + 1)
        return {"filled": filled, "wins": wins, "losses": losses, "times": times,
                "rs": rs, "loss_rs": loss_rs, "days": days}

    main, alt = run(drift_shrink), run(COMPARE_DRIFT_SHRINK)
    ev_alt = float(np.mean(alt["rs"])) if alt["rs"] else 0.0
    f = main["filled"]
    if f == 0:
        return {"成交機率%": 0.0, "先達標機率%": 0.0, "先停損機率%": 0.0, "期滿未決機率%": 0.0,
                "達標機率(排除期滿)%": 0.0, "期望值R": 0.0, "期望值R(漂移×0.5)": round(ev_alt, 2),
                "損益兩平勝率%": 0.0, "p_win": 0.0, "p_loss": 0.0, "p_dec": 0.0, "breakeven": 1.0,
                "ev_r": 0.0, "ev_r_drift": ev_alt, "r_win": r_win, "r_loss": -1 - cost_r}
    r_loss = float(np.mean(main["loss_rs"])) if main["loss_rs"] else -1 - cost_r
    breakeven = -r_loss / (r_win - r_loss)
    dec = main["wins"] + main["losses"]
    p_dec = main["wins"] / dec if dec else 0.0
    ev = float(np.mean(main["rs"]))
    return {
        "成交機率%": round(f / n * 100, 1),
        "先達標機率%": round(main["wins"] / f * 100, 1),
        "先停損機率%": round(main["losses"] / f * 100, 1),
        "期滿未決機率%": round(main["times"] / f * 100, 1),
        "達標機率(排除期滿)%": round(p_dec * 100, 1),
        "期望值R": round(ev, 2),
        "期望值R(漂移×0.5)": round(ev_alt, 2),
        "損益兩平勝率%": round(breakeven * 100, 1),
        "中位數達標天數": int(np.median(main["days"])) if main["days"] else None,
        # 以下給程式判斷用
        "p_win": main["wins"] / f, "p_loss": main["losses"] / f,
        "p_dec": p_dec,                      # 勝率口徑：排除「期滿未決」，與 breakeven 口徑一致
        "breakeven": breakeven, "ev_r": ev, "ev_r_drift": ev_alt,
        "r_win": r_win, "r_loss": r_loss,
    }


def kelly_fraction(p_win, p_loss, r_win, r_loss) -> float:
    decided = p_win + p_loss
    if decided <= 0:
        return 0.0
    p, b = p_win / decided, r_win / abs(r_loss)
    return max(p - (1 - p) / b, 0.0)


# ------------------------------------------------------------------ 籌碼特徵
def _streak(series) -> int:
    """連買(+)/連賣(−)天數：由最新一天往回數同方向的連續天數，持平為 0。"""
    vals = [float(v) for v in pd.Series(series).dropna()]
    if not vals or vals[-1] == 0:
        return 0
    sign = 1 if vals[-1] > 0 else -1
    n = 0
    for v in reversed(vals):
        if v * sign > 0:
            n += 1
        else:
            break
    return n * sign


def chips_features(inst, margin, vol5_lots=None, ret5_pct=None, is_etf: bool = False) -> dict:
    """
    把三大法人與融資資料整理成可直接判讀的變數，並依固定規則給出「籌碼訊號」。
    ETF 不適用（法人/融資訊號對 ETF 沒有個股的意義）：回傳「ETF 不適用」、分數 0。
    inst:   index=日期、欄位 外資/投信/自營商（單位：張，買超為正）
    margin: index=日期、欄位 融資餘額/融券餘額（單位：張）
    """
    if is_etf:
        return {"籌碼訊號": "ETF 不適用", "籌碼分數": 0}
    out = {}
    foreign5 = trust5 = inst5 = None
    if inst is not None and len(inst):
        inst = inst.sort_index().copy()
        for c in ("外資", "投信", "自營商"):
            if c not in inst:
                inst[c] = 0.0
        inst["三大法人"] = inst[["外資", "投信", "自營商"]].sum(axis=1)
        t5 = inst.tail(5).sum()
        foreign5, trust5, inst5 = int(t5["外資"]), int(t5["投信"]), int(t5["三大法人"])
        out.update({
            "資料日期": str(inst.index[-1]),
            "外資連買(+)連賣(−)天數": _streak(inst["外資"]),
            "投信連買(+)連賣(−)天數": _streak(inst["投信"]),
            "外資最新一日(張)": int(inst["外資"].iloc[-1]),
            "投信最新一日(張)": int(inst["投信"].iloc[-1]),
            "自營商最新一日(張)": int(inst["自營商"].iloc[-1]),
            "外資近5日合計(張)": foreign5, "投信近5日合計(張)": trust5, "三大法人近5日合計(張)": inst5,
        })
        if vol5_lots:
            out["三大法人近5日買超占成交量%"] = round(inst5 / (vol5_lots * 5) * 100, 1)
    pct5 = None
    if margin is not None and len(margin):
        m = margin.sort_index()
        mb, sb = m["融資餘額"].astype(float), m["融券餘額"].astype(float)
        out["融資餘額(張)"] = int(mb.iloc[-1])
        out["融券餘額(張)"] = int(sb.iloc[-1])
        if mb.iloc[-1] > 0:
            out["券資比%"] = round(float(sb.iloc[-1] / mb.iloc[-1] * 100), 1)
        if len(mb) > 5:
            chg = mb.iloc[-1] - mb.iloc[-6]
            out["融資近5日增減(張)"] = int(chg)
            if mb.iloc[-6] > 0:
                pct5 = float(chg / mb.iloc[-6] * 100)
                out["融資近5日增減%"] = round(pct5, 1)
    if inst5 is None:
        out["籌碼訊號"] = "資料不足"
        out["籌碼分數"] = 0
        return out
    chase = pct5 is not None and pct5 > 3
    if inst5 > 0 and not chase:
        label, score = "🟢 法人吃貨、籌碼沉澱", 2 if (pct5 is not None and pct5 < 0) else 1
    elif inst5 > 0 and chase:
        label, score = "🟡 法人與散戶同買（籌碼偏熱）", 0
    elif inst5 < 0 and chase:
        label, score = "🔴 散戶追高、法人倒貨（籌碼凌亂）", -2
    elif inst5 < 0:
        label, score = "🟡 法人賣、融資減（籌碼換手，觀察落底）", -1
    else:
        label, score = "⚪ 中性", 0
    notes = []
    if trust5 is not None and out.get("投信連買(+)連賣(−)天數", 0) >= 3:
        notes.append("投信連買認養中")
    if out.get("外資連買(+)連賣(−)天數", 0) >= 5:
        notes.append("外資連買 5 日以上")
    if chase and ret5_pct is not None and ret5_pct > 8:
        notes.append("股價急漲且融資暴增，追價風險高")
    out["籌碼訊號"] = label + ("（" + "、".join(notes) + "）" if notes else "")
    out["籌碼分數"] = score
    return out


# ------------------------------------------------------------------ JSON 序列化（結果快取）
def to_jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__df__": {"index": [str(i) for i in obj.index], "columns": [str(c) for c in obj.columns],
                           "data": to_jsonable(obj.values.tolist())}}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if obj is None or isinstance(obj, (int, str)):
        return obj
    return str(obj)


def from_jsonable(obj):
    if isinstance(obj, dict):
        if set(obj) == {"__df__"}:
            x = obj["__df__"]
            return pd.DataFrame(x["data"], index=x["index"], columns=x["columns"])
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj


# ------------------------------------------------------------------ 戰績 DB
_DB_PATH: Path | None = None
_NEW_COLS = {"rating_final": "TEXT", "model": "TEXT", "prompt_version": "TEXT", "app_version": "TEXT",
             "quant_version": "TEXT", "raw_json": "TEXT"}


def set_db_path(path) -> None:
    """設定 DB 路徑（例如 Streamlit secrets 的 DB_PATH）；傳 None/空字串 → 回到預設。"""
    global _DB_PATH
    _DB_PATH = Path(path).expanduser() if path else None


def db_path() -> Path:
    if _DB_PATH is not None:
        return _DB_PATH
    return Path(os.getenv("TW_DASH_DB") or Path(__file__).with_name("predictions.db"))


def is_ephemeral_fs() -> bool:
    """啟發式判斷 DB 是否放在重啟會清空的位置（Streamlit Cloud 的 /mount/src、/tmp、Heroku/Cloud Run/Render 的應用目錄）。"""
    p = db_path().resolve().as_posix()
    if p.startswith(("/mount/src", "/tmp/", "/var/tmp/")):
        return True
    on_paas = any(os.getenv(k) for k in ("DYNO", "K_SERVICE", "RENDER"))
    return bool(on_paas and p.startswith(Path(__file__).resolve().parent.as_posix()))


def _conn():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS preds(
        date TEXT, code TEXT, name TEXT, price REAL, entry_low REAL, entry_high REAL,
        target1 REAL, stop_loss REAL, rr REAL, p_win REAL, ev_r REAL,
        status TEXT DEFAULT 'open', resolved_date TEXT, outcome_r REAL,
        PRIMARY KEY(date, code))""")
    have = {r[1] for r in con.execute("PRAGMA table_info(preds)")}
    for col, typ in _NEW_COLS.items():      # 舊版資料庫：補欄位（舊紀錄 rating_final 為 NULL，視為「買進」）
        if col not in have:
            con.execute(f"ALTER TABLE preds ADD COLUMN {col} {typ}")
    return con


@contextmanager
def _db():
    con = _conn()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def save_result(r: dict, app_version: str, prompt_version: str) -> None:
    """所有分析結果（含觀望、賣出）都寫入；同 (日期, 代碼) 重新分析會覆蓋並重置結算狀態。賣出不結算（status='na'）。"""
    sim = r.get("sim") or {}
    status = "na" if r["rating_final"] == "賣出" else "open"
    e = r["entry_price"]
    with _db() as con:
        con.execute(
            """INSERT INTO preds(date,code,name,price,entry_low,entry_high,target1,stop_loss,rr,p_win,ev_r,
                                 status,rating_final,model,prompt_version,app_version,quant_version,raw_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(date,code) DO UPDATE SET
                 name=excluded.name, price=excluded.price, entry_low=excluded.entry_low,
                 entry_high=excluded.entry_high, target1=excluded.target1, stop_loss=excluded.stop_loss,
                 rr=excluded.rr, p_win=excluded.p_win, ev_r=excluded.ev_r, status=excluded.status,
                 resolved_date=NULL, outcome_r=NULL, rating_final=excluded.rating_final,
                 model=excluded.model, prompt_version=excluded.prompt_version,
                 app_version=excluded.app_version, quant_version=excluded.quant_version,
                 raw_json=excluded.raw_json""",
            (r["date"], r["code"], r["name"], r["price"], e, e, r["target1"], r["stop_loss"], r.get("rr"),
             sim.get("p_win"), sim.get("ev_r"), status, r["rating_final"], r.get("model") or r.get("_model"),
             prompt_version, app_version, QUANT_VERSION, json.dumps(to_jsonable(r), ensure_ascii=False)))


def load_result(code: str, date: str, app_version: str):
    """同一交易日同一檔已有同版本結果就直接讀取（不重打 Gemini）；沒有或版本不符回傳 None。"""
    with _db() as con:
        row = con.execute("SELECT raw_json, app_version FROM preds WHERE code=? AND date=?",
                          (code, date)).fetchone()
    if not row or not row[0] or row[1] != app_version:
        return None
    try:
        return from_jsonable(json.loads(row[0]))
    except Exception as e:
        log.warning("讀取快取結果失敗 %s %s：%s", code, date, e)
        return None


def resolve_plan(h: pd.DataFrame, entry, target, stop, cost: float = COST_STOCK):
    """
    用建議日之後的真實日線（還原價尺度）結算「建倉點單一價」計畫。回傳 (status, R, 結算日) 或 None（仍在進行）。
      成交：限價買單，最低 ≤ 建倉點才成交；開盤低於建倉點以開盤價成交，否則以建倉點成交；ENTRY_WAIT 日內沒成交 → unfilled
      成交日起的出場一律用 resolve_exit（同日停損優先、跳空以開盤價、盤中觸價以停損/目標價）；成交當天不檢查跳空
      成交當天開盤就跳空跌破停損 → 視為成交後停損（−1R）；持有滿 HOLD_DAYS → 以收盤價出場（timeout）
      R 以計畫風險（建倉點 − 停損）為單位，含交易成本。
    """
    risk = entry - stop
    cost_r = cost * entry / risk
    fill, f = None, None
    for k, (ts, row) in enumerate(h.iterrows()):
        day = ts.strftime("%Y-%m-%d")
        o, hi, lo, cl = (float(row[c]) for c in ("Open", "High", "Low", "Close"))
        if fill is None:
            if k >= ENTRY_WAIT:
                return "unfilled", 0.0, day
            if lo > entry:
                continue
            fill, f = min(o, entry), k
            if fill <= stop:
                return "loss", -1 - cost_r, day
            res = resolve_exit(o, hi, lo, stop, target, first_day=True)
        else:
            res = resolve_exit(o, hi, lo, stop, target, first_day=False)
        if res:
            status = "win" if res[1] in ("target", "gap_target") else "loss"
            return status, (res[0] - fill) / risk - cost_r, day
        if k - f >= HOLD_DAYS:
            return "timeout", (cl - fill) / risk - cost_r, day
    return None


def evaluate_open(fetch_hist) -> int:
    """
    結算所有進行中的紀錄（買進與「觀望的假想結果」）。
    fetch_hist(code) → 還原價日線（Open/High/Low/Close），並含 RawClose 欄（同日期的原始收盤價）。
    DB 內的價位是原始價：先用建議日當天的「還原因子 = 還原收盤 / 原始收盤」換算成還原價尺度，再與還原日線比較。
    """
    done = 0
    with _db() as con:
        rows = con.execute("SELECT date,code,entry_high,target1,stop_loss FROM preds WHERE status='open'").fetchall()
        for date, code, entry, t1, sl in rows:
            try:
                h = fetch_hist(code)
                days = h.index.strftime("%Y-%m-%d")
                before = h[days <= date]
                factor = 1.0
                if len(before) and "RawClose" in h:
                    last = before.iloc[-1]
                    if pd.notna(last["RawClose"]) and last["RawClose"] > 0:
                        factor = float(last["Close"]) / float(last["RawClose"])
                after = h[days > date]
            except Exception as e:
                log.warning("結算 %s %s 取價失敗：%s", code, date, e)
                continue
            res = resolve_plan(after, entry * factor, t1 * factor, sl * factor, cost_rt(code))
            if res:
                con.execute("UPDATE preds SET status=?, outcome_r=?, resolved_date=? WHERE date=? AND code=?",
                            (res[0], res[1], res[2], date, code))
                done += 1
    return done


def track_record():
    """回傳 (全部紀錄 DataFrame, 摘要 dict)。買進與觀望（假想結果）分開統計，驗證過濾是否有價值。"""
    with _db() as con:
        df = pd.read_sql_query("SELECT * FROM preds ORDER BY date DESC, code", con)
    overview = {"累計記錄": len(df), "進行中": 0, "未成交": 0, "已結算": 0}
    groups = {"買進": {"已結算筆數": 0, "樣本不足": True}, "觀望（假想：若照建倉點買進）": {"已結算筆數": 0, "樣本不足": True}}
    if df.empty:
        return df, {"總覽": overview, "分組": groups, "已結算": 0, "樣本不足": True}
    df["rating_eff"] = df["rating_final"].fillna("買進")
    closed = df[df["status"].isin(["win", "loss", "timeout"])]
    overview.update({"進行中": int((df["status"] == "open").sum()),
                     "未成交": int((df["status"] == "unfilled").sum()), "已結算": len(closed)})
    for key, label in (("買進", "買進"), ("觀望", "觀望（假想：若照建倉點買進）")):
        sub = closed[closed["rating_eff"] == key]
        g = {"已結算筆數": len(sub), "樣本不足": len(sub) < MIN_CLOSED_SAMPLES}
        if len(sub):
            g["實際勝率%"] = round(float((sub["status"] == "win").mean() * 100), 1)
            g["平均R"] = round(float(sub["outcome_r"].mean()), 2)
            if sub["p_win"].notna().any():
                g["模型預測勝率%"] = round(float(sub["p_win"].mean() * 100), 1)
            if sub["ev_r"].notna().any():
                g["模型預測期望值R"] = round(float(sub["ev_r"].mean()), 2)
        groups[label] = g
    buy, wait = groups["買進"], groups["觀望（假想：若照建倉點買進）"]
    if "平均R" in buy and "平均R" in wait:
        groups["買進"]["過濾效益(買進平均R−觀望假想平均R)"] = round(buy["平均R"] - wait["平均R"], 2)
    return df, {"總覽": overview, "分組": groups, "已結算": len(closed),
                "樣本不足": len(closed) < MIN_CLOSED_SAMPLES}
