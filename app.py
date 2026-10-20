"""
台股自動化決策儀表板 v5
-----------------------
資料：yfinance（原始價 3 年＋還原價 10 年、加權指數、VIX）、FinMind（股票資訊、法人、融資券、本益比、月營收、毛利率、除權息）、
      Google News RSS（法說/營收 1 則＋財經媒體優先，最新 3 則含內文摘要）
判斷：Gemini（五大面向）＋ quant.py（回測、機率模擬、凱利部位、戰績追蹤）

評級：買進 / 觀望 / 賣出。本工具「只做多」：賣出＝已持有者應減碼/出場，不是放空訊號。
「買進」需同時通過（任一不過 → 強制觀望）：
    預期盈虧比 ≥ 2:1｜停損不過遠｜不追高｜流動性｜資料完整度｜回測有優勢（bootstrap）｜建倉點可成交｜模擬期望值
價格口徑：回測/模擬/結算用「還原價」；畫面與建倉點、停損價用「原始價」。

執行：
    streamlit run app.py                 # 網頁（手機友善卡片）
    python app.py 2330 0050 00878        # 終端機
設定（擇一）：Streamlit Secrets 的 GEMINI_API_KEY（有設定就自動隱藏輸入框）／環境變數 GEMINI_API_KEY
選填：FINMIND_TOKEN、GEMINI_MODEL、APP_PASSWORD、DAILY_LIMIT、DB_PATH（見 .streamlit/secrets.toml.example）
"""
import os
import re
import sys
import json
import hmac
import html
import math
import time
import logging
import functools
import threading
import datetime as dt
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import requests
import pandas as pd
import yfinance as yf

import quant

APP_VERSION = "5.0"
PROMPT_VERSION = "p5-2026-10"
REQUIRED_QUANT_VERSION = "5.0"
if getattr(quant, "QUANT_VERSION", None) != REQUIRED_QUANT_VERSION:
    raise ImportError(
        f"quant.py 版本不符：app.py 需要 {REQUIRED_QUANT_VERSION}，目前是 "
        f"{getattr(quant, 'QUANT_VERSION', '舊版（未標示版本）')}。請把 app.py 與 quant.py 一起更新，"
        "並重新啟動（Streamlit Cloud：Manage app → Reboot app）。")

log = logging.getLogger("tw_dash")

# ------------------------------------------------------------------ 設定
DEFAULT_STOCKS = ["2330", "0050", "2454"]
# 2026-10 時點：gemini-1.5 全系列、gemini-2.0 已下架；gemini-2.5-pro/flash 於 2026-10-16 下架。
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite"]
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
TZ = ZoneInfo("Asia/Taipei")

# 快取與重試
TTL_FINMIND, TTL_YF, TTL_NEWS = 3600, 900, 1800     # 秒；FinMind / yfinance / 新聞分開
FINMIND_RETRIES, FINMIND_BACKOFF = 2, 1.0           # 重試 2 次，指數退避 1s、2s
WORKERS = 3
DEFAULT_DAILY_LIMIT = 10

# 風控過濾門檻
MIN_RR = 2.0
MAX_STOP_PCT = quant.MAX_STOP_PCT
MAX_STOP_ATR = quant.MAX_STOP_ATR
OVERHEAT_EXT20, OVERHEAT_RET5, OVERHEAT_RSI = 12.0, 10.0, 78.0
OVERHEAT_MIN_PULLBACK = 0.03
MIN_FILL_PROB = 25.0
MIN_TURNOVER_WAN = 5000.0        # 近 20 日平均成交金額下限：5,000 萬元
MAX_MISSING = 2                  # 資料四項中缺 2 項以上 → 觀望
TARGET_CAP_MULT = 1.02           # target1 不得高於近 60 日高點 ×1.02（除非有突破依據）
SIM_GATE_ENABLED = True          # False：模擬只顯示、不把關（見 quant_gate；回測 edge_ok 仍把關）
W_ENTRY, W_ADD = 0.6, 0.4        # 建倉 / 加碼 資金比例

NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2412": "中華電",
    "2308": "台達電", "2303": "聯電", "3711": "日月光投控", "2881": "富邦金",
    "2882": "國泰金", "2891": "中信金", "2382": "廣達", "2603": "長榮",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息",
}
NAMES_REV = {v: k for k, v in NAMES.items()}
PREFERRED_SITES = ("cnyes.com", "money.udn.com", "ctee.com.tw", "ltn.com.tw")   # 鉅亨網、經濟日報、工商時報、自由財經
RATINGS = ("買進", "觀望", "賣出")

SYSTEM_PROMPT = """你是精通台股交易的資深量化分析師與風控專家，目標是「賺多賠少、盈虧比高」。本工具只做多（不放空）。
從五大面向分析：基本面（月營收 YoY/累計 YoY、毛利率、現金流、負債）、估值面（本益比/股價淨值比百分位、殖利率）、
產業競爭力、技術與籌碼面（均線、外資/投信連買日數與買超、融資增減、支撐壓力）、總經與消息面（大盤、VIX、最新個股新聞與法說/營收消息）。
個股新聞附有標題與內文摘要；摘要為空代表只有標題，不得臆測內文。資料欄位若為空或在「資料缺漏」清單內，請對應面向寫「資料不足」。
標的類型為 ETF 時：基本面、估值、月營收、籌碼訊號一律「不適用」（籌碼訊號為「ETF 不適用」），不得據此判斷法人吃貨/散戶追高。

【評級定義】
- 買進：符合下列風控規則，可分批建倉。
- 觀望：不符合進場條件，等待。
- 賣出：僅代表「已持有者應減碼/出場」（偏空結構：例如跌破關鍵支撐、均線轉空、籌碼轉空），不是放空建議。
  賣出時 stop_loss 填「持有者出場參考價（跌破即全數出場）」，其餘價位欄位僅作參考。
  下列盈虧比等進場規則只適用於「買進」。

【嚴格風控規則（針對買進）】（違反即視為錯誤輸出）
1. 買進點拆成兩點：
   - 建倉點 entry_price：回測短均線的拉回價（例如 5 日線附近），必須 ≤ 現價；依據寫入 entry_basis（如「回測5日線」）。
   - 加碼點 add_price：二擇一，add_type 填「回測月線」（約 20 日線，低於建倉點）或「突破壓力」（高於現價，約 20 日高/關鍵壓力上方 0.3~2%）。加碼只在趨勢確認後進行。
2. 預期盈虧比 = (目標價1 − 建倉點) / (建倉點 − 強制停損價)，以建倉點計算（最保守）。
   輸出前必須代入實際數字檢算，寫入 expected_rr（數字，小數 2 位）與 rr_calc（算式字串，如 "(112-100)/(100-94)=2.00"）。
3. 若 expected_rr < 2，rating 必須是「觀望」，沒有例外；此時仍須給出建倉點、加碼點、強制停損價（供等待參考）。
4. 強制停損必須是明確技術位（跌破某支撐/均線），且離建倉點不得超過 8% 或 4 倍 ATR；太遠就改用更近的技術位或直接觀望，
   不可為了湊出盈虧比而把停損拉遠。
5. 目標價1 不得高於「近 60 日高點 ×1.02」，除非加碼類型為「突破壓力」（有突破依據）。
6. 若「技術面.過熱」為 true（剛大漲完），建倉點必須是明確拉回價（至少低於現價 3%），不可貼近現價追高，否則 rating=觀望。
7. 回測統計 edge_ok=false，或籌碼與技術面嚴重矛盾，rating 也必須是「觀望」。
8. 「籌碼(FinMind)」內含程式依規則算出的籌碼訊號與外資/投信連買日數、法人近5日買超、融資增減。個股請明確判斷是
   「法人吃貨」還是「散戶追高」，寫入 chips_technical；與程式訊號不同時須說明原因。
9. 價位須以現價、均線、支撐壓力、ATR 為依據，可參考「基準價位」微調，不得憑空捏造。資料沒有的數字不要編造。
   流動性不足、資料缺漏、除權息將至等由程式另外過濾，你不需處理，但請在 risks 中提及。
trend_bias 為純方向看法（偏多/中性/偏空），不受盈虧比規則限制。

只輸出 JSON（不要 markdown）：
{"rating":"買進"|"觀望"|"賣出","trend_bias":"偏多"|"中性"|"偏空","risk_level":"高"|"中"|"低",
 "entry_price":數字,"entry_basis":"如 回測5日線","add_price":數字,"add_type":"回測月線"|"突破壓力",
 "stop_loss":數字,"target1":數字,"target2":數字,"expected_rr":數字,"rr_calc":"算式字串",
 "position_pct":建議總部位佔總資金百分比(數字，觀望/賣出填0；建倉6成、加碼4成),
 "batches":"建倉與加碼的執行條件，一句話",
 "fundamentals_valuation":"一句話","industry":"一句話","chips_technical":"一句話（個股含法人吃貨/散戶追高判斷）",
 "news_macro":"一句話","reason":"評級理由一句話","risks":["風險1","風險2"]}"""


# ------------------------------------------------------------------ 執行環境 / 快取 / 並行
def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None   # 純 CLI 時不要印 missing ScriptRunContext
        except TypeError:                                                  # 舊版 Streamlit 沒有這個參數
            return get_script_run_ctx() is not None
    except Exception:
        return False


def _cache(ttl: int, lru: bool = False):
    """Streamlit 內用 st.cache_data(ttl)（例外不會被快取）；CLI 模式不快取（lru=True 時用 lru_cache，同樣不快取例外）。"""
    def deco(fn):
        if _in_streamlit():
            import streamlit as st
            return st.cache_data(ttl=ttl, show_spinner=False)(fn)
        return functools.lru_cache(maxsize=256)(fn) if lru else fn
    return deco


def _make_pool(n: int) -> ThreadPoolExecutor:
    """工作執行緒沿用 Streamlit 的 ScriptRunContext（否則 st.cache_data 在子執行緒會報 missing ScriptRunContext）。"""
    ctx = None
    if _in_streamlit():
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
        except Exception:
            ctx = None

    def init():
        if ctx is not None:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(threading.current_thread(), ctx)

    return ThreadPoolExecutor(max_workers=n, initializer=init)


def _safe(name: str, fn, gaps: list, default, *args, **kwargs):
    """呼叫抓取函式；失敗時記錄 warning、把原因放進 gaps、回傳預設值。"""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = f"{name}：{str(e)[:90]}"
        log.warning("資料抓取失敗 %s", msg)
        gaps.append(msg)
        return default


class Budget:
    """每個 session 每日 Gemini 分析次數上限（執行緒安全）。快取命中不扣額度。"""
    def __init__(self, n: int):
        self.n, self._lock = n, threading.Lock()

    def take(self):
        with self._lock:
            if self.n <= 0:
                raise RuntimeError("已達今日分析次數上限（DAILY_LIMIT）；快取結果仍可讀取")
            self.n -= 1


# ------------------------------------------------------------------ FinMind / 代碼解析
@_cache(TTL_FINMIND)
def _finmind(dataset: str, code=None, days: int = 30) -> pd.DataFrame:
    """FinMind API：HTTP 或 status 非 200 都視為失敗；失敗重試 FINMIND_RETRIES 次（指數退避）。"""
    params = {"dataset": dataset}
    if code:
        params["data_id"] = code
    if days:
        params["start_date"] = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    if os.getenv("FINMIND_TOKEN"):
        params["token"] = os.getenv("FINMIND_TOKEN")
    last = None
    for attempt in range(FINMIND_RETRIES + 1):
        try:
            r = requests.get(FINMIND_URL, params=params, timeout=25)
            r.raise_for_status()
            j = r.json()
            if j.get("status") not in (None, 200):
                raise RuntimeError(f"FinMind status {j.get('status')}: {j.get('msg')}")
            return pd.DataFrame(j.get("data", []))
        except Exception as e:
            last = e
            if attempt < FINMIND_RETRIES:
                time.sleep(FINMIND_BACKOFF * (2 ** attempt))
    raise last


@_cache(86400, lru=True)
def _load_stock_info() -> dict:
    """失敗會丟例外（例外不會被快取，下次呼叫會重試）；成功才快取。"""
    df = _finmind("TaiwanStockInfo", None, days=0)
    if df.empty:
        raise ValueError("TaiwanStockInfo 回傳空資料")
    return {str(r.stock_id): {"name": r.stock_name, "type": r.type, "industry": r.industry_category}
            for r in df.itertuples()}


def _stock_info() -> dict:
    try:
        return _load_stock_info()
    except Exception as e:
        log.warning("股票清單抓取失敗（將於下次重試）：%s", e)
        return {}


def _suffix_order(code: str) -> list:
    t = (_stock_info().get(code) or {}).get("type")
    if t == "tpex":
        return [".TWO", ".TW"]
    if t == "twse":
        return [".TW", ".TWO"]
    return [".TW", ".TWO"] if code.startswith(("0", "1", "2")) else [".TWO", ".TW"]


# ------------------------------------------------------------------ 價格（原始價 / 還原價）
@_cache(TTL_YF)
def _yf_history(symbol: str, period: str, adjust: bool) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, auto_adjust=adjust)


def fetch_price(raw: str) -> dict:
    """
    輸入 2330 / 0050 / 2330.TW / 台積電 皆可；自動補 .TW 或 .TWO。
    回傳 {"symbol", "raw"（原始價 3 年，畫面/技術指標用）, "adj"（還原價 10 年，回測/模擬/結算用）, "notes"}。
    """
    code = quant.normalize_code(raw, NAMES_REV)
    for suffix in _suffix_order(code):
        sym = code + suffix
        rawdf = _yf_history(sym, "3y", False)
        if rawdf.empty:
            continue
        rawdf = rawdf.dropna(subset=["Close"])
        tz = rawdf.index.tz
        now = pd.Timestamp.now(tz=tz) if tz else pd.Timestamp.now()
        if now - rawdf.index[-1] > pd.Timedelta(days=30):
            raise ValueError(f"{sym} 最近一筆資料是 {rawdf.index[-1]:%Y-%m-%d}，可能已下市或停牌")
        notes = []
        adj = _yf_history(sym, "10y", True)
        if adj.empty:
            adj = rawdf.copy()
            notes.append("還原價取得失敗，回測改用原始價（可能受除權息影響）")
        else:
            adj = adj.dropna(subset=["Close"])
        return {"symbol": sym, "raw": rawdf, "adj": adj, "notes": notes}
    raise ValueError(f"找不到 {code}（已試 .TW 與 .TWO），請確認代碼是否正確")


def drop_incomplete_bar(df: pd.DataFrame, now: dt.datetime | None = None):
    """盤中防呆：交易日 13:30 前，若最後一根是『今天』的 K 棒（尚未收盤）就丟棄。回傳 (df, 是否丟棄)。"""
    if df is None or df.empty:
        return df, False
    now = now or dt.datetime.now(TZ)
    if df.index[-1].date() == now.date() and now.weekday() < 5 and now.time() < dt.time(13, 30):
        return df.iloc[:-1], True
    return df, False


def _settlement_hist(code: str) -> pd.DataFrame:
    """給戰績結算用：還原價日線 + RawClose（同日原始收盤，用來換算還原因子）。"""
    pdx = fetch_price(code)
    raw, _ = drop_incomplete_bar(pdx["raw"])
    adj, _ = drop_incomplete_bar(pdx["adj"])
    h = adj[["Open", "High", "Low", "Close"]].copy()
    h["RawClose"] = raw["Close"].reindex(h.index)
    return h


# ------------------------------------------------------------------ 資料抓取
def compute_technicals(d: pd.DataFrame) -> dict:
    """d：原始價的指標 DataFrame（顯示與建倉/停損價都用原始價）。"""
    x, prev = d.iloc[-1], d.iloc[-2]
    r = lambda v: round(float(v), 2)
    c = d["Close"]
    last = float(x["Close"])
    ret5 = (last / float(c.iloc[-6]) - 1) * 100
    ext20 = (last / float(x["ma20"]) - 1) * 100
    hot = ext20 > OVERHEAT_EXT20 or ret5 > OVERHEAT_RET5 or float(x["rsi"]) > OVERHEAT_RSI
    c3 = c.tail(756)
    turnover = float((d["Close"] * d["Volume"]).tail(20).mean()) / 1e4
    return {
        "最近交易日": d.index[-1].strftime("%Y-%m-%d"),
        "收盤價": r(last), "漲跌幅%": r((last / float(prev["Close"]) - 1) * 100),
        "成交量(張)": int(x["Volume"] / 1000), "5日均量(張)": int(d["Volume"].tail(5).mean() / 1000),
        "20日均成交金額(萬元)": round(turnover),
        "MA5": r(x["ma5"]), "MA20": r(x["ma20"]), "MA60": r(x["ma60"]), "MA120": r(x["ma120"]),
        "RSI14": r(x["rsi"]), "ATR14": r(x["atr"]),
        "5日漲幅%": r(ret5), "距MA20%": r(ext20), "過熱": bool(hot),
        "20日支撐": r(d["Low"].tail(20).min()), "20日壓力": r(d["High"].tail(20).max()),
        "60日支撐": r(d["Low"].tail(60).min()), "60日壓力": r(d["High"].tail(60).max()),
        "近3年價位百分位%": r((last - c3.min()) / (c3.max() - c3.min()) * 100),
    }


def fetch_chips(code: str, tech: dict, gaps: list) -> dict:
    """三大法人（外資/投信連買日數、買超張數）＋融資增減 → quant.chips_features；單一資料集失敗會記入 gaps。"""
    inst = marg = None
    try:
        df = _finmind("TaiwanStockInstitutionalInvestorsBuySell", code, days=45)
        if df.empty:
            raise ValueError("無資料")
        df["net"] = (df["buy"] - df["sell"]) / 1000

        def grp(n):
            if n.startswith("Foreign"):
                return "外資"
            if n == "Investment_Trust":
                return "投信"
            return "自營商" if n.startswith("Dealer") else "其他"

        df["group"] = df["name"].map(grp)
        inst = df.groupby(["date", "group"])["net"].sum().unstack(fill_value=0).sort_index()
    except Exception as e:
        log.warning("三大法人抓取失敗 %s：%s", code, e)
        gaps.append(f"三大法人：{str(e)[:80]}")
    try:
        m = _finmind("TaiwanStockMarginPurchaseShortSale", code, days=45).sort_values("date")
        if m.empty:
            raise ValueError("無資料")
        marg = m.rename(columns={"MarginPurchaseTodayBalance": "融資餘額",
                                 "ShortSaleTodayBalance": "融券餘額"}).set_index("date")[["融資餘額", "融券餘額"]]
    except Exception as e:
        log.warning("融資券抓取失敗 %s：%s", code, e)
        gaps.append(f"融資券：{str(e)[:80]}")
    if inst is None and marg is None:
        return {}
    return quant.chips_features(inst, marg, tech.get("5日均量(張)"), tech.get("5日漲幅%"))


def _pct_rank(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    return round(float((s < s.iloc[-1]).mean() * 100)) if len(s) >= 30 else None


def fetch_valuation(code: str) -> dict:
    """本益比河流：目前 PER/PBR 在近 5 年的百分位（越高越貴）、殖利率 vs 5 年平均。失敗/無資料會丟例外。"""
    df = _finmind("TaiwanStockPER", code, days=365 * 5).sort_values("date")
    if df.empty:
        raise ValueError("無本益比資料")
    per = pd.to_numeric(df["PER"], errors="coerce")
    pbr = pd.to_numeric(df["PBR"], errors="coerce")
    dy = pd.to_numeric(df["dividend_yield"], errors="coerce")
    return {
        "本益比": round(float(per.iloc[-1]), 1), "本益比5年百分位%": _pct_rank(per),
        "本益比5年中位數": round(float(per[per > 0].median()), 1),
        "股價淨值比": round(float(pbr.iloc[-1]), 2), "股價淨值比5年百分位%": _pct_rank(pbr),
        "殖利率%": round(float(dy.iloc[-1]), 2), "殖利率5年平均%": round(float(dy.mean()), 2),
    }


def fetch_revenue(code: str) -> dict:
    """月營收：最新 YoY、近 6 月 YoY、MoM、累計 YoY、YoY 連續成長月數，以及下次公布（每月 10 日前）提醒。"""
    df = _finmind("TaiwanStockMonthRevenue", code, days=640).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("無月營收資料")
    rev = df["revenue"].astype(float)
    yoy, mom = rev.pct_change(12) * 100, rev.pct_change() * 100
    months = [f"{y}-{int(m):02d}" for y, m in zip(df["revenue_year"], df["revenue_month"])]
    ly, lm = int(df["revenue_year"].iloc[-1]), int(df["revenue_month"].iloc[-1])
    cur = df[df["revenue_year"] == ly]["revenue"].astype(float).sum()
    prev = df[(df["revenue_year"] == ly - 1) & (df["revenue_month"] <= lm)]["revenue"].astype(float).sum()
    streak = 0
    for v in reversed(yoy.dropna().tolist()):
        if v > 0:
            streak += 1
        else:
            break
    today = dt.datetime.now(TZ).date()
    exp = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    nxt = dt.date(today.year, today.month, 10) if today.day <= 10 else (
        dt.date(today.year + (today.month == 12), today.month % 12 + 1, 10))
    return {
        "最新營收年月": months[-1], "最新月營收(億)": round(float(rev.iloc[-1]) / 1e8, 1),
        "最新YoY%": round(float(yoy.iloc[-1]), 1), "最新MoM%": round(float(mom.iloc[-1]), 1),
        "累計營收YoY%": round(float((cur / prev - 1) * 100), 1) if prev > 0 else None,
        "YoY連續成長月數": streak, "近6月YoY%": [round(v, 1) for v in yoy.tail(6)],
        "近3月平均YoY%": round(float(yoy.tail(3).mean()), 1), "月份": months[-6:],
        "上月營收公布狀態": "已公布" if (ly, lm) >= exp else "尚未公布（每月10日前公布）",
        "距下次公布截止日(天)": (nxt - today).days,
    }


def fetch_margins_trend(code: str) -> dict:
    df = _finmind("TaiwanStockFinancialStatements", code, days=900)
    df = df[df["type"].isin(["Revenue", "GrossProfit", "OperatingIncome"])]
    p = df.pivot_table(index="date", columns="type", values="value").sort_index()
    return {"季別": list(p.index[-6:]),
            "近6季毛利率%": (p["GrossProfit"] / p["Revenue"] * 100).round(1).tail(6).tolist(),
            "近6季營益率%": (p["OperatingIncome"] / p["Revenue"] * 100).round(1).tail(6).tolist()}


def fetch_upcoming_ex_dividend(code: str, today: dt.date | None = None) -> list:
    """未來 5 個交易日內的除權息日（FinMind 股利政策表的現金/股票除息交易日）。失敗會丟例外。"""
    today = today or dt.datetime.now(TZ).date()
    df = _finmind("TaiwanStockDividend", code, days=400)
    dates = set()
    for col in ("CashExDividendTradingDate", "StockExDividendTradingDate"):
        if col in df.columns:
            for v in df[col].dropna().astype(str):
                try:
                    dates.add(dt.date.fromisoformat(v[:10]))
                except ValueError:
                    pass
    return [d.isoformat() for d in sorted(dates) if d >= today and int(np.busday_count(today, d)) <= 5]


def fetch_regime() -> dict:
    out = {"趨勢": "未知"}
    try:
        c = _yf_history("^TWII", "1y", False)["Close"].dropna()
        last, ma60, ma120 = float(c.iloc[-1]), float(c.tail(60).mean()), float(c.tail(120).mean())
        out = {"加權指數": round(last), "距MA60%": round((last / ma60 - 1) * 100, 1),
               "20日報酬%": round((last / float(c.iloc[-21]) - 1) * 100, 1),
               "趨勢": "多頭" if last > ma60 > ma120 else "空頭" if last < ma60 < ma120 else "盤整"}
    except Exception as e:
        log.warning("大盤趨勢抓取失敗：%s", e)
    try:
        out["VIX"] = round(float(_yf_history("^VIX", "5d", False)["Close"].dropna().iloc[-1]), 1)
    except Exception as e:
        log.warning("VIX 抓取失敗：%s", e)
    return out


# ------------------------------------------------------------------ 新聞（標題＋內文摘要）
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


@_cache(TTL_NEWS)
def _article_summary(link: str) -> str:
    """盡力取得原文摘要（解 Google News 轉址 → og:description / meta description / 前幾段）。失敗回傳空字串。"""
    try:
        from bs4 import BeautifulSoup
        url = link
        try:
            from googlenewsdecoder import gnewsdecoder
            res = gnewsdecoder(link)
            if res.get("status"):
                url = res["decoded_url"]
        except Exception:
            pass
        if "news.google.com" in url:
            return ""
        r = requests.get(url, timeout=8, headers=UA)
        if r.encoding in (None, "ISO-8859-1"):
            r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, "html.parser")
        for attrs in ({"property": "og:description"}, {"name": "description"}, {"name": "twitter:description"}):
            tag = soup.find("meta", attrs=attrs)
            if tag and len(tag.get("content", "").strip()) >= 30:
                return _clean(tag["content"])[:220]
        body = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return _clean(body)[:220]
    except Exception as e:
        log.info("新聞摘要抓取失敗 %s：%s", link[:60], e)
        return ""


def _uniq_sorted(items: list) -> list:
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["when"], reverse=True):
        if it["title"] not in seen:
            seen.add(it["title"])
            out.append(it)
    return out


@_cache(TTL_NEWS)
def _rss_items(query: str) -> list:
    url = (f"https://news.google.com/rss/search?q={quote(query)}"
           "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    r = requests.get(url, timeout=15, headers=UA)
    r.raise_for_status()
    items = []
    for it in ET.fromstring(r.content).iter("item"):
        try:
            when = parsedate_to_datetime(it.findtext("pubDate", ""))
        except Exception:
            when = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        items.append({"title": _clean(it.findtext("title", "")), "link": it.findtext("link", ""),
                      "source": _clean(it.findtext("source", "")), "when": when,
                      "rss_desc": _clean(it.findtext("description", ""))})
    return _uniq_sorted(items)


def _rss_safe(query: str) -> list:
    try:
        return _rss_items(query)
    except Exception as e:
        log.warning("Google News 抓取失敗（%s）：%s", query[:40], e)
        return []


def _finalize_news(items: list, with_summary: bool) -> list:
    if with_summary and items:
        with _make_pool(len(items)) as ex:
            sums = list(ex.map(lambda it: _article_summary(it["link"]), items))
    else:
        sums = [""] * len(items)
    out = []
    for it, s in zip(items, sums):
        src = "原文" if s else ""
        if not s and it["rss_desc"] and not it["rss_desc"].startswith(it["title"][:15]) \
                and len(it["rss_desc"]) > len(it["title"]) + 20:
            s, src = it["rss_desc"][:200], "RSS"
        out.append({"title": it["title"], "link": it["link"], "source": it["source"], "tag": it.get("tag", ""),
                    "date": it["when"].strftime("%Y-%m-%d") if it["when"].year > 1 else "",
                    "summary": s, "summary_src": src or "無"})
    return out


def fetch_news(query: str, n: int = 3, with_summary: bool = False) -> list:
    """單一查詢：Google News RSS 取最新 n 則（依發布時間排序）。"""
    return _finalize_news(_rss_safe(query)[:n], with_summary)


def fetch_stock_news(code: str, name: str, n: int = 3) -> list:
    """個股最新 n 則：① 1 則法說/營收/展望事件消息 ② 優先鉅亨網、經濟日報等財經媒體 ③ 不足用一般新聞補；依時間新到舊。"""
    site = " OR ".join(f"site:{s}" for s in PREFERRED_SITES)
    queries = {
        "event": f"{name} (法說會 OR 營收 OR 展望) when:30d",
        "pref": f"{name} {code} ({site}) when:14d",
        "gen": f"{code} {name} when:7d",
    }
    with _make_pool(3) as ex:
        futs = {k: ex.submit(_rss_safe, q) for k, q in queries.items()}
        res = {k: f.result() for k, f in futs.items()}
    picked, seen = [], set()

    def take(lst, k, tag=""):
        for it in lst:
            if k <= 0 or len(picked) >= n:
                break
            if it["title"] in seen:
                continue
            seen.add(it["title"])
            picked.append({**it, "tag": tag})
            k -= 1

    take(res["event"], 1, "法說/營收")
    take(res["pref"], n)
    take(res["gen"], n)
    picked.sort(key=lambda x: x["when"], reverse=True)
    return _finalize_news(picked[:n], True)


# ------------------------------------------------------------------ Gemini
def call_gemini(api_key: str, prompt: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json", temperature=0.2)

    def attempt(models, tried):
        last = None
        for m in models:
            if m in tried:
                continue
            tried.append(m)
            try:
                resp = client.models.generate_content(model=m, contents=prompt, config=config)
                data = json.loads(resp.text)
                data["_model"] = m
                return data, None
            except Exception as e:   # 404 下架 / 無權限 / JSON 解析失敗 → 換下一個
                last = f"{m}: {e}"
                log.warning("Gemini 模型失敗 %s", last[:200])
        return None, last

    tried = []
    data, err = attempt([GEMINI_MODEL] + FALLBACK_MODELS, tried)
    if data:
        return data
    try:   # 全部失敗：查詢這把金鑰目前可用的模型，pro 優先、版本新的優先
        bad = ("image", "tts", "audio", "live", "embedding", "robotics", "lite", "veo", "imagen", "learnlm")
        found = []
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            acts = getattr(m, "supported_actions", None) or ["generateContent"]
            if name.startswith("gemini-") and "generateContent" in acts and not any(b in name for b in bad):
                found.append(name)
        found.sort(key=lambda n: ("pro" in n, n), reverse=True)
        data, err2 = attempt(found, tried)
        if data:
            return data
        err = err2 or err
    except Exception as e:
        err = f"{err}；列出可用模型也失敗：{e}"
    raise RuntimeError(f"Gemini 呼叫失敗（已嘗試 {tried}）：{err}")


# ------------------------------------------------------------------ 風控關卡
def _clean_basis(s) -> str:
    """AI 自由文字欄位只保留中英數與少數符號、限制長度（會被放進卡片與複製指令）。"""
    return re.sub(r"[^\w\s（）()+\-/%.]", "", str(s or ""))[:20].strip()


def enforce_rules(r: dict, ctx: dict, base: dict) -> dict:
    """
    關卡 1（自動風控過濾），程式自己重算、不信任 AI 自報。
    【買進】以下任一不過 → 強制「觀望」：價位邏輯、預期盈虧比 <2:1、停損太遠、剛大漲還追高、流動性、資料完整度、AI 自評盈虧比。
    【賣出】= 已持有者減碼/出場（本工具只做多）：走獨立分支，只檢查價位邏輯，不套用多方盈虧比等規則，不會被降為觀望。
    目標價不得高於近 60 日高點×1.02（無突破依據時自動下修）；目標距建倉點超過 3×ATR×√20 → 加註「目標過遠」。
    ctx: price, atr, overheated, ext20, ret5, rsi, high60, turnover_wan, missing(list), etf；base: quant.propose_levels()。
    """
    flags = []
    price, atr, etf = ctx["price"], ctx.get("atr") or 0.0, ctx.get("etf", False)
    rt = lambda v: quant.round_tick(float(v), etf)
    rating = r.get("rating", "觀望")
    if rating not in RATINGS:
        rating = "觀望"

    ai_valid = False
    try:
        e, sl, t1 = rt(r["entry_price"]), rt(r["stop_loss"]), rt(r["target1"])
        t2 = rt(r.get("target2") or r["target1"])
        ai_valid = sl < e < t1 and e <= price * 1.005
    except (KeyError, TypeError, ValueError):
        pass
    if not ai_valid:   # 整組價位不合邏輯：換成系統基準價位，讓畫面上仍是合理數字
        e, sl, t1, t2 = base["entry_price"], base["stop_loss"], base["target1"], base["target2"]
        r["add_price"], r["add_type"], r["entry_basis"] = base["add_price"], base["add_type"], base["entry_basis"]
        if rating == "買進":
            rating = "觀望"
            flags.append("AI 價位邏輯不一致，已改用系統基準價位，並強制標註為觀望")
        elif rating == "觀望":
            flags.append("AI 價位邏輯不一致，已改用系統基準價位")

    add_type = "突破壓力" if "突破" in str(r.get("add_type", "")) else "回測月線"
    if rating != "賣出":
        high60 = ctx.get("high60")
        if high60:   # 目標價合理性
            cap = rt(high60 * TARGET_CAP_MULT)
            if t1 > cap and add_type != "突破壓力":
                flags.append(f"目標價 {t1:g} 高於近60日高點×{TARGET_CAP_MULT}（{cap:g}）且無突破依據，已下修至 {cap:g}")
                t1, t2 = cap, cap
        if atr and (t1 - e) > 3 * atr * math.sqrt(20):
            flags.append("目標過遠，20 日內難達成")
    t2 = max(t2, t1)

    try:   # 加碼點驗證（回測月線：停損 < 加碼 < 建倉；突破壓力：現價 ≤ 加碼 < 目標1）
        add = rt(r["add_price"])
        add_ok = (sl < add < e) if add_type == "回測月線" else (price <= add < t1)
    except (KeyError, TypeError, ValueError):
        add, add_ok = None, False
    if not add_ok:
        add, add_type = base["add_price"], base["add_type"]
        if ai_valid and rating != "賣出":
            flags.append("AI 加碼點不合邏輯，已改用系統基準加碼點")
    r["entry_basis"] = _clean_basis(r.get("entry_basis")) or base["entry_basis"]

    rr = (t1 - e) / (e - sl) if (e - sl > 0 and t1 - e > 0) else None
    avg_cost = W_ENTRY * e + W_ADD * add
    rr_blend = (t1 - avg_cost) / (avg_cost - sl) if avg_cost > sl and t1 > avg_cost else None
    stop_pct = (e - sl) / e * 100
    stop_atr = (e - sl) / atr if atr else None

    if rating == "買進":
        fails = []
        if rr is None or rr < MIN_RR:
            fails.append(f"預期盈虧比 {rr:.2f}:1 < 2:1，依規則強制標註為觀望" if rr
                         else "預期盈虧比不合格，依規則強制標註為觀望")
        if stop_pct > MAX_STOP_PCT or (stop_atr is not None and stop_atr > MAX_STOP_ATR):
            fails.append(f"停損離建倉點太遠（{stop_pct:.1f}% / {stop_atr:.1f} ATR，上限 {MAX_STOP_PCT:g}% / "
                         f"{MAX_STOP_ATR:g} ATR），風險過大")
        if ctx.get("overheated") and e > price * (1 - OVERHEAT_MIN_PULLBACK):
            fails.append(f"剛大漲完（距月線 {ctx.get('ext20', 0):+.1f}%、5日 {ctx.get('ret5', 0):+.1f}%、"
                         f"RSI {ctx.get('rsi', 0):.0f}），建倉點貼近現價等於追高，等拉回")
        tw = ctx.get("turnover_wan")
        if tw is not None and tw < MIN_TURNOVER_WAN:
            fails.append(f"流動性不足：近 20 日平均成交金額 {tw:,.0f} 萬元 < {MIN_TURNOVER_WAN:,.0f} 萬元")
        missing = ctx.get("missing") or []
        if len(missing) >= MAX_MISSING:
            fails.append(f"資料不完整（缺：{'、'.join(missing)}），強制觀望")
        try:   # AI 自評盈虧比 <2 卻給買進 → 也降級
            if float(r.get("expected_rr")) < MIN_RR:
                fails.append(f"AI 自評盈虧比 {float(r['expected_rr']):.2f} < 2:1")
        except (TypeError, ValueError):
            pass
        if fails:
            rating = "觀望"
            flags += [f if "觀望" in f else f + "，已強制標註為觀望" for f in fails]
        else:
            if stop_atr is not None and stop_atr < 1.0:
                flags.append(f"停損僅 {stop_atr:.1f} ATR，容易被正常波動洗出")
            if rr_blend is not None and rr_blend < MIN_RR:
                flags.append(f"加碼後平均成本的盈虧比 {rr_blend:.2f}:1 偏低，加碼量建議縮小")
    if rating != "賣出" and ai_valid:
        try:   # AI 自報值與程式重算對不上 → 提醒（以程式為準）
            ai_rr = float(r.get("expected_rr"))
            if rr is not None and abs(ai_rr - rr) > 0.3:
                flags.append(f"AI 自報盈虧比 {ai_rr:.2f} 與程式重算 {rr:.2f} 不一致，以程式為準")
        except (TypeError, ValueError):
            pass

    r.update({"entry_price": e, "stop_loss": sl, "target1": t1, "target2": t2, "add_price": add, "add_type": add_type,
              "rating_final": rating, "rr": rr, "rr_blend": rr_blend, "avg_cost": round(avg_cost, 2),
              "stop_pct": stop_pct, "stop_atr": stop_atr, "flags": flags, "levels_ok": True,
              "entry_low": e, "entry_high": e})   # entry_low/high：相容舊欄位（建倉點為單一價）
    return r


def quant_gate(r: dict, sim, bt: dict, regime: dict) -> dict:
    """
    關卡 2、3（只針對買進）：回測優勢（bootstrap）、建倉點可成交、模擬把關；通過才給部位。
    勝率口徑一致：用 p_dec = p_win/(p_win+p_loss)（排除期滿未決）與 breakeven 比較；期望值門檻 ev_r < EV_MIN_R 即降級。
    部位：四分之一凱利、上限 20%、建倉 6 成/加碼 4 成。
    """
    rating, flags, pos = r["rating_final"], r["flags"], 0.0
    if rating == "買進":
        if not bt["edge_ok"]:
            rating = "觀望"
            ci = bt["全期間"].get("期望值R_90%區間", "—")
            flags.append(f"回測無統計優勢（次數 ≥{quant.MIN_TRADES}、期望值 90% 區間下緣 >0、獲利因子 >1.2；"
                         f"目前 {bt['全期間'].get('交易次數', 0)} 筆、區間 {ci}），已降為觀望")
        elif sim is None:
            rating = "觀望"
            flags.append("無法模擬此價位計畫，已降為觀望")
        elif sim["成交機率%"] < MIN_FILL_PROB:
            rating = "觀望"
            flags.append(f"建倉點離現價太遠，10 日內成交機率僅 {sim['成交機率%']:.0f}%，已降為觀望")
        elif SIM_GATE_ENABLED:
            p_dec, be, ev = sim["p_dec"], sim["breakeven"], sim["ev_r"]
            if p_dec <= be or ev < quant.EV_MIN_R:
                rating = "觀望"
                flags.append(f"模擬達標機率（排除期滿）{p_dec*100:.0f}% vs 損益兩平 {be*100:.0f}%、"
                             f"期望值 {ev:+.2f}R（門檻 {quant.EV_MIN_R:g}R），已降為觀望")
    if rating == "買進":
        kelly = quant.kelly_fraction(sim["p_win"], sim["p_loss"], sim["r_win"], sim["r_loss"])
        pos = min(float(r.get("position_pct") or 0), 20.0, kelly * 25)   # 四分之一凱利，上限 20%
        if regime.get("趨勢") == "空頭":
            pos *= 0.5
            flags.append("大盤空頭，部位減半")
    r.update({"rating_final": rating, "position_final": round(pos, 1), "sim": sim,
              "pos_entry": round(pos * W_ENTRY, 1), "pos_add": round(pos * W_ADD, 1)})
    return r


def wait_condition(r: dict) -> str:
    """依 flags 自動產生「等待條件」（觀望/賣出卡片用）。"""
    g = _g
    if r["rating_final"] == "買進":
        return ""
    if r["rating_final"] == "賣出":
        return f"已持有者建議減碼；跌破 {g(r['stop_loss'])} 全數出場"
    fl = " ".join(r["flags"])
    e = g(r["entry_price"])
    conds = []
    if "盈虧比" in fl or "停損離" in fl:
        conds.append(f"等拉回至 {e} 且盈虧比≥2（停損≤{MAX_STOP_PCT:g}%）")
    if "剛大漲完" in fl:
        conds.append(f"等拉回至 {e}（低於現價 3% 以上）再評估")
    if "成交機率" in fl:
        conds.append("等股價靠近建倉點")
    if "回測無統計優勢" in fl:
        conds.append("等策略回測出現統計優勢")
    if "模擬達標" in fl:
        conds.append("等模擬期望值轉正")
    if "流動性" in fl:
        conds.append(f"等成交金額回到 {MIN_TURNOVER_WAN:,.0f} 萬元以上")
    if "資料不完整" in fl:
        conds.append("等資料補齊")
    if "價位邏輯" in fl:
        conds.append("等 AI 價位重新分析")
    if not conds:
        conds.append(f"等拉回至 {e} 且盈虧比≥2")
    out, seen = [], set()
    for c in conds:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "；".join(out[:2])


# ------------------------------------------------------------------ 主流程
def analyze_stock(raw_code: str, api_key: str, macro_news: list, regime: dict, *,
                  force: bool = False, budget: Budget | None = None) -> dict:
    code = quant.normalize_code(raw_code, NAMES_REV)
    pdx = fetch_price(code)
    raw, cut_raw = drop_incomplete_bar(pdx["raw"])
    adj, _ = drop_incomplete_bar(pdx["adj"])
    if len(raw) < 130:
        raise ValueError(f"{code} 歷史資料不足（{len(raw)} 筆）")
    date = raw.index[-1].strftime("%Y-%m-%d")

    if not force:   # 同一交易日同一檔已有同版本結果 → 直接讀取，不重打 Gemini
        cached = quant.load_result(code, date, APP_VERSION)
        if cached:
            cached["from_cache"] = True
            return cached

    gaps = list(pdx["notes"])
    info = _stock_info().get(code, {})
    industry = info.get("industry")
    etf = quant.is_etf(code, industry)
    name = info.get("name") or NAMES.get(code) or code
    cost = quant.cost_rt(code, industry)

    d_raw = quant.add_indicators(raw)                       # 原始價：顯示、技術面、建倉/停損價
    tech = compute_technicals(d_raw)
    price = tech["收盤價"]
    bt = quant.backtest(quant.add_indicators(adj), cost)     # 還原價：回測
    base = quant.propose_levels(d_raw, etf)

    news = fetch_stock_news(code, name, n=3)
    if etf:
        valuation = revenue = margins = {}
        chips = quant.chips_features(None, None, is_etf=True)
        missing = []
    else:
        valuation = _safe("估值", fetch_valuation, gaps, {}, code)
        revenue = _safe("月營收", fetch_revenue, gaps, {}, code)
        margins = _safe("毛利率", fetch_margins_trend, gaps, {}, code)
        chips = fetch_chips(code, tech, gaps)
        missing = [n for n, v in (("估值", valuation), ("月營收", revenue)) if not v]
        if not chips or chips.get("籌碼訊號") == "資料不足":
            missing.append("籌碼")
    ex_div = _safe("除權息", fetch_upcoming_ex_dividend, gaps, [], code)

    today = dt.datetime.now(TZ).date()
    events = []
    if ex_div:
        events.append(f"🔔 除權息 {ex_div[0]} 將生效（5 個交易日內）：股價會跳降，建倉/停損價請於除息後重新評估；"
                      "回測與結算已使用還原價")
    if 7 <= today.day <= 10:
        events.append("🔔 事件風險：月營收將於 10 日前公布，公布前後波動可能放大")

    payload = {
        "股票": f"{code} {name}", "標的類型": "ETF" if etf else "個股", "產業別": industry or "未知",
        "今日日期": today.isoformat(), "資料缺漏": gaps, "事件提醒": events,
        "技術面": tech, "籌碼(FinMind)": chips,
        "估值(FinMind)": valuation, "月營收": revenue, "毛利率趨勢": margins,
        "大盤環境": regime, "回測統計": {**bt, "說明": "固定規則多頭拉回策略（還原價、10年），R=風險單位，已含交易成本"},
        "基準價位": {"建倉點": base["entry_price"], "建倉依據": base["entry_basis"],
                  "加碼點": base["add_price"], "加碼類型": base["add_type"],
                  "強制停損價": base["stop_loss"], "目標價1": base["target1"], "目標價2": base["target2"]},
        "風控過濾門檻": {"最低預期盈虧比": MIN_RR, "停損上限%": MAX_STOP_PCT, "停損上限ATR倍數": MAX_STOP_ATR,
                    "過熱時建倉點至少低於現價%": OVERHEAT_MIN_PULLBACK * 100,
                    "最低20日均成交金額(萬元)": MIN_TURNOVER_WAN},
        "個股最新3則新聞": [{"標題": n["title"], "來源": n["source"], "日期": n["date"], "類型": n["tag"] or "一般",
                         "內文摘要": n["summary"] or "（無摘要，僅標題）"} for n in news],
        "總經新聞標題": [n["title"] for n in macro_news],
    }
    prompt = "請分析以下台股資料並輸出 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    if budget is not None:
        budget.take()
    ai = call_gemini(api_key, prompt)

    ctx = {"price": price, "atr": tech["ATR14"], "overheated": tech["過熱"], "ext20": tech["距MA20%"],
           "ret5": tech["5日漲幅%"], "rsi": tech["RSI14"], "high60": tech["60日壓力"],
           "turnover_wan": tech["20日均成交金額(萬元)"], "missing": missing, "etf": etf}
    r = enforce_rules(ai, ctx, base)
    sim = None
    if r["rating_final"] != "賣出":      # 賣出（已持有者減碼）不模擬建倉計畫
        sim = quant.simulate_plan(adj["Close"], price, r["entry_price"], r["target1"], r["stop_loss"], cost)
    r = quant_gate(r, sim, bt, regime)
    r["flags"] = r["flags"] + events      # 事件提醒放在降級原因之後；不強制觀望

    chart = d_raw[["Open", "High", "Low", "Close", "ma5", "ma20"]].tail(120).rename(
        columns={"ma5": "MA5", "ma20": "MA20"})
    chart.index = chart.index.strftime("%Y-%m-%d")
    r.update({"code": code, "name": name, "symbol": pdx["symbol"], "price": price, "etf": etf,
              "change": tech["漲跌幅%"], "news": news, "tech": tech, "bt": bt, "chart": chart,
              "date": date, "valuation": valuation, "revenue": revenue, "chips": chips,
              "overheated": tech["過熱"], "data_gaps": gaps, "missing": missing, "regime": regime,
              "used_prev_close": cut_raw, "from_cache": False,
              "app_version": APP_VERSION, "quant_version": quant.QUANT_VERSION,
              "prompt_version": PROMPT_VERSION, "model": ai.get("_model")})
    return r


def run_all(codes, api_key, progress=None, *, force: bool = False, budget: Budget | None = None,
            workers: int = WORKERS) -> list:
    """多檔並行分析（最多 workers 檔同時），結果順序與輸入一致；結果寫入戰績 DB（含觀望）。"""
    try:   # 先結算過去的預測，戰績才會是最新的
        quant.evaluate_open(_settlement_hist)
    except Exception as e:
        log.warning("戰績結算失敗：%s", e)
    regime = fetch_regime()
    macro = fetch_news("台股 外資 聯準會 總體經濟 when:3d", n=6)

    plan, seen = [], set()
    for raw in codes:
        try:
            code = quant.normalize_code(raw, NAMES_REV)
        except ValueError as e:
            plan.append((str(raw), None, str(e)))
            continue
        if code not in seen:
            seen.add(code)
            plan.append((code, code, None))

    def one(code):
        try:
            return analyze_stock(code, api_key, macro, regime, force=force, budget=budget)
        except Exception as e:
            log.warning("分析 %s 失敗：%s", code, e)
            return {"code": code, "error": str(e)}

    out = []
    with _make_pool(workers) as ex:
        futs = [(raw, c, err, ex.submit(one, c) if c else None) for raw, c, err in plan]
        for i, (raw, c, err, fut) in enumerate(futs):
            r = fut.result() if fut else {"code": raw, "error": err}
            if "error" not in r and not r.get("from_cache"):
                try:
                    quant.save_result(r, APP_VERSION, PROMPT_VERSION)
                except Exception as e:
                    log.warning("寫入戰績失敗 %s：%s", r.get("code"), e)
            out.append(r)
            if progress:
                progress((i + 1) / len(futs), raw)
    return out


RATING_ICON = {"買進": "🟦", "觀望": "🟨", "賣出": "🟫"}


def _g(v) -> str:
    """價格顯示：950.0 → 950，955.5 → 955.5"""
    return "—" if v is None else f"{float(v):g}"


def trade_command(r: dict) -> str:
    """一鍵複製用的交易指令（貼到筆記或券商 App 提醒自己）。"""
    head = f"【{r['code']} {r['name']}】"
    if r["rating_final"] == "賣出":
        return f"{head}賣出 {RATING_ICON['賣出']}（已持有者減碼，本工具不放空）｜出場參考價：{_g(r['stop_loss'])}"
    body = (f"建倉點：{_g(r['entry_price'])}（{r['entry_basis']}）｜加碼點：{_g(r['add_price'])}（{r['add_type']}）"
            f"｜停損價：{_g(r['stop_loss'])}｜目標價：{_g(r['target1'])}/{_g(r['target2'])}")
    rr = f"｜盈虧比 {r['rr']:.2f}:1" if r.get("rr") else ""
    if r["rating_final"] == "買進":
        return f"{head}買進 {RATING_ICON['買進']}｜{body}{rr}｜部位：建倉{r['pos_entry']:g}%＋加碼{r['pos_add']:g}%"
    wait = wait_condition(r)
    return (f"{head}觀望 {RATING_ICON['觀望']}（未達進場條件，僅供參考）｜{body}{rr}"
            + (f"｜等待：{wait}" if wait else ""))


def summary_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"股票代碼": r["code"], "當前評級": "⚠️ 錯誤", "現價": r["error"][:50]})
            continue
        s, bt = r.get("sim"), r["bt"]["全期間"]
        sell = r["rating_final"] == "賣出"
        rows.append({
            "股票代碼": f"{r['code']} {r['name']}",
            "現價": f"{r['price']:.2f}", "漲跌幅%": r["change"],
            "當前評級": f"{RATING_ICON[r['rating_final']]} {r['rating_final']}",
            "多空傾向": r.get("trend_bias", "—"),
            "建倉點 / 加碼點": "—" if sell else f"{_g(r['entry_price'])} / {_g(r['add_price'])}",
            "目標價": "—" if sell else f"{_g(r['target1'])} / {_g(r['target2'])}",
            "強制停損價": f"{_g(r['stop_loss'])} (-{r['stop_pct']:.1f}%)" if not sell else f"出場 {_g(r['stop_loss'])}",
            "預期盈虧比": f"{r['rr']:.2f}:1" if (r["rr"] and not sell) else "—",
            "籌碼訊號": (r.get("chips") or {}).get("籌碼訊號", "—"),
            "達標機率(模擬)": f"{s['先達標機率%']}%" if s else "—",
            "期望值(R)": f"{s['期望值R']:+.2f}" if s else "—",
            "回測勝率/期望值": f"{bt.get('勝率%', '—')}% / {bt.get('期望值R', '—')}R",
            "建議部位%": f"{r['pos_entry']:g}+{r['pos_add']:g}",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 介面
CARD_CSS = """
<style>
.dash-card{border-radius:18px;overflow:hidden;border:1px solid rgba(128,128,128,.35);margin:2px 0 6px 0}
.dash-head{padding:12px 16px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.dash-head .t{font-size:1.05rem;font-weight:700;line-height:1.25}
.dash-head .rating{font-size:1.6rem;font-weight:800;white-space:nowrap}
.dash-price{padding:8px 16px 0 16px;font-size:1.05rem}
.dash-price .up{color:#ef4444;font-weight:700}
.dash-price .down{color:#22c55e;font-weight:700}
.dash-price .flat{opacity:.7;font-weight:700}
.dash-price small{opacity:.75}
.dash-wait{margin:8px 14px 0 14px;padding:7px 10px;border-radius:10px;font-size:.82rem;
  background:rgba(245,158,11,.16);border:1px solid rgba(245,158,11,.5)}
.dash-body{padding:10px 14px 2px 14px;display:grid;grid-template-columns:1fr 1fr;gap:8px 14px}
.dash-body .l{font-size:.74rem;opacity:.7}
.dash-body .v{font-size:1.12rem;font-weight:700}
.dash-body .v small{font-size:.72rem;font-weight:500;opacity:.75}
.dash-foot{padding:4px 14px 10px 14px;font-size:.83rem;opacity:.9;line-height:1.4}
.dash-warn{margin:0 14px 8px 14px;padding:6px 10px;border-radius:10px;font-size:.78rem;
  background:rgba(127,29,29,.12);border:1px solid rgba(127,29,29,.4)}
</style>
"""
# 台灣習慣色：漲紅、跌綠。評級卡片色刻意避開紅/綠：買進＝藍綠、觀望＝琥珀、賣出＝深灰紅
CARD_COLORS = {"買進": ("#0f766e", "#ffffff"), "觀望": ("#f59e0b", "#1f2937"), "賣出": ("#7f1d1d", "#ffffff")}


def card_html(r: dict) -> str:
    """評級大卡片（手機兩欄格狀、字體放大）。價格漲跌：紅漲綠跌。"""
    e = html.escape
    bg, fg = CARD_COLORS[r["rating_final"]]
    chg = r["change"]
    cls = "up" if chg > 0 else "down" if chg < 0 else "flat"
    cell = lambda l, v, sub="": ('<div><div class="l">%s</div><div class="v">%s%s</div></div>'
                                 % (e(l), v, (' <small>%s</small>' % e(sub)) if sub else ""))
    if r["rating_final"] == "賣出":
        cells = [cell("建議", "減碼/出場", "已持有者"), cell("全數出場參考價", _g(r["stop_loss"])),
                 cell("新進場", "不建議"), cell("說明", "只做多", "不放空")]
    else:
        rr = f"{r['rr']:.2f}:1" if r.get("rr") else "—"
        pos = f"{r['pos_entry']:g}% + {r['pos_add']:g}%" if r["rating_final"] == "買進" else "0%"
        cells = [cell("建倉點", _g(r["entry_price"]), r["entry_basis"]), cell("加碼點", _g(r["add_price"]), r["add_type"]),
                 cell("強制停損價", _g(r["stop_loss"]), f"-{r['stop_pct']:.1f}%"),
                 cell("目標價", f"{_g(r['target1'])} / {_g(r['target2'])}"),
                 cell("預期盈虧比", rr), cell("建議部位", pos, "建倉+加碼")]
    wait = wait_condition(r) if r["rating_final"] != "買進" else ""
    warn = "".join(f"<div>⚠️ {e(f)}</div>" for f in r["flags"][:2])
    chips = (r.get("chips") or {}).get("籌碼訊號", "")
    foot = e(str(r.get("reason", ""))) + (("<br>" + e(chips)) if chips else "")
    prev = " ⏱前一交易日收盤" if r.get("used_prev_close") else ""
    return (
        '<div class="dash-card">'
        f'<div class="dash-head" style="background:{bg};color:{fg}">'
        f'<div class="t">{e(r["code"])} {e(r["name"])}</div>'
        f'<div class="rating">{RATING_ICON[r["rating_final"]]} {e(r["rating_final"])}</div></div>'
        f'<div class="dash-price"><b>{r["price"]:.2f}</b> <span class="{cls}">{chg:+.2f}%</span> '
        f'<small>｜{e(str(r.get("trend_bias", "")))}{e(prev)}</small></div>'
        + (f'<div class="dash-wait">⏳ 等待條件：{e(wait)}</div>' if wait else "")
        + f'<div class="dash-body">{"".join(cells)}</div>'
        + (f'<div class="dash-warn">{warn}</div>' if warn else "")
        + f'<div class="dash-foot">{foot}</div></div>')


def copy_button(text: str, label: str = "📋 一鍵複製交易指令"):
    """
    點一下複製到剪貼簿（clipboard API 被擋時退回 execCommand）。新版 Streamlit 用 st.iframe
    （components.v1.html 已公告移除），舊版自動退回 components.html。文字以 JSON 字串嵌入並跳脫所有「<」。
    """
    import streamlit as st
    t = json.dumps(text).replace("<", "\\u003c")
    lab = json.dumps(label).replace("<", "\\u003c")
    page = f"""
<body style="margin:0;background:transparent">
<button id="b" style="width:100%;box-sizing:border-box;padding:14px 10px;font-size:16px;font-weight:600;
 border-radius:12px;border:1px solid #9ca3af;background:#f3f4f6;color:#111827;cursor:pointer">{html.escape(label)}</button>
<script>
const t={t}, b=document.getElementById('b');
b.onclick=async()=>{{
  try{{await navigator.clipboard.writeText(t);}}
  catch(e){{const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();
    try{{document.execCommand('copy');}}catch(_){{}} a.remove();}}
  b.textContent='✅ 已複製';setTimeout(()=>b.textContent={lab},1600);
}};
</script></body>"""
    if hasattr(st, "iframe"):
        st.iframe(page, height=62)
    else:
        import streamlit.components.v1 as components
        components.html(page, height=62)


def make_figure(r: dict):
    """Plotly K 線（台灣習慣色：漲紅跌綠）＋ MA5/MA20 ＋ 建倉/加碼/停損/目標水平線。"""
    import plotly.graph_objects as go
    c = r["chart"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=list(c.index), open=c["Open"], high=c["High"], low=c["Low"], close=c["Close"], name="K線",
        increasing_line_color="#ef4444", increasing_fillcolor="#ef4444",
        decreasing_line_color="#22c55e", decreasing_fillcolor="#22c55e"))
    fig.add_trace(go.Scatter(x=list(c.index), y=c["MA5"], mode="lines", name="MA5",
                             line=dict(width=1.3, color="#f59e0b")))
    fig.add_trace(go.Scatter(x=list(c.index), y=c["MA20"], mode="lines", name="MA20",
                             line=dict(width=1.3, color="#3b82f6")))
    if r["rating_final"] == "賣出":
        lines = [("出場參考", r["stop_loss"], "#7f1d1d")]
    else:
        lines = [("建倉", r["entry_price"], "#0f766e"), ("加碼", r["add_price"], "#2563eb"),
                 ("停損", r["stop_loss"], "#b91c1c"), ("目標1", r["target1"], "#d97706")]
    for name, y, col in lines:
        fig.add_hline(y=y, line_dash="dash", line_color=col, line_width=1.2,
                      annotation_text=f"{name} {_g(y)}", annotation_position="top left",
                      annotation_font_color=col)
    fig.add_trace(go.Scatter(x=[c.index[-1]] * len(lines), y=[l[1] for l in lines], mode="markers",
                             marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))   # 讓縮放範圍涵蓋水平線
    fig.update_layout(height=380, margin=dict(l=8, r=8, t=28, b=8), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.1), xaxis=dict(type="category", nticks=8))
    return fig


def _secret(name: str, default: str = "") -> str:
    """讀 Streamlit Secrets；沒有 secrets 檔、沒這個鍵都安靜回傳預設值。"""
    try:
        import streamlit as st
        v = st.secrets.get(name, default)
        return str(v).strip() if v else default
    except Exception:
        return default


def _show_detail(st, r, STRETCH, track):
    # 1) 資料可信度
    a = r["bt"]["全期間"]
    trades = a.get("交易次數", 0)
    st.markdown("**🔎 資料可信度**")
    cred = {
        "回測交易次數": f"{trades}（門檻 {quant.MIN_TRADES}）" + ("" if trades >= quant.MIN_TRADES else " ⚠️ 偏少"),
        "期望值R 90%區間": a.get("期望值R_90%區間", "—"),
        "回測有優勢": "✅ 是" if r["bt"]["edge_ok"] else "❌ 否",
        "資料缺漏": "、".join(r.get("data_gaps") or []) or "無",
        "戰績已結算樣本": f"{track['已結算']} 筆" + ("（樣本不足，僅供參考）" if track["樣本不足"] else ""),
        "版本": f"app {r.get('app_version', '—')}｜quant {r.get('quant_version', '—')}｜"
              f"prompt {r.get('prompt_version', '—')}｜model {r.get('model', '—')}",
    }
    st.table(pd.Series(cred, name="值").astype(str))
    if r.get("from_cache"):
        st.info("📦 今日（同一交易日）已分析過，這是讀取的快取結果；勾選「強制重新分析」可重跑。")
    if r.get("used_prev_close"):
        st.info("⏱ 盤中（13:30 前）：已丟棄今日未收盤的 K 棒，使用前一交易日收盤分析。")
    for fl in r["flags"]:
        st.warning(fl)
    rr_txt = f"{r['rr']:.2f}:1" if r["rr"] else "—"
    blend = f"{r['rr_blend']:.2f}:1" if r.get("rr_blend") else "—"
    st.markdown(
        f"- **多空傾向**：{r.get('trend_bias', '—')}｜**風險等級**：{r.get('risk_level', '—')}\n"
        f"- **盈虧比**：AI 算式 `{r.get('rr_calc', '—')}`｜程式重算（建倉點）`{rr_txt}`｜加碼後平均成本 `{blend}`"
        f"（均價 {r['avg_cost']:g}）\n"
        f"- **停損距離**：{r['stop_pct']:.1f}%"
        + (f" / {r['stop_atr']:.1f} ATR" if r.get("stop_atr") else "")
        + f"｜**過熱**：{'是 🔥' if r.get('overheated') else '否'}"
        + f"｜**20日均成交金額**：{r['tech'].get('20日均成交金額(萬元)', 0):,.0f} 萬元\n"
        f"- **基本/估值**：{r.get('fundamentals_valuation', '')}\n"
        f"- **產業**：{r.get('industry', '')}\n"
        f"- **籌碼/技術**：{r.get('chips_technical', '')}\n"
        f"- **消息/總經**：{r.get('news_macro', '')}\n"
        f"- **分批執行**：{r.get('batches', '')}\n"
        f"- **⚠️ 風險**：" + "；".join(r.get("risks", [])))
    # 2) K 線圖
    try:
        st.plotly_chart(make_figure(r), key=f"chart_{r['code']}", **STRETCH)
    except ImportError:
        st.line_chart(r["chart"][["Close", "MA5", "MA20"]])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**機率模擬**（建倉點，30 交易日，收盤觸價）")
        if r.get("sim"):
            st.table(pd.Series({k: v for k, v in r["sim"].items()
                                if (k.endswith("%") or "R" in k or "天數" in k) and v is not None}, name="值").astype(str))
        else:
            st.caption("賣出評級不模擬")
    with c2:
        st.markdown("**策略回測**（還原價，全期間 / 近1年）")
        st.table(pd.DataFrame({"全期間": r["bt"]["全期間"], "近1年": r["bt"]["近1年"]}).astype(str))
        bm = r["bt"].get("基準比較")
        if bm:
            st.markdown("**基準比較**（買進持有，僅供參考）")
            st.table(pd.Series(bm, name="值").astype(str))
    with c3:
        st.markdown("**估值**")
        st.table(pd.Series(r["valuation"] or {"資料": "無／不適用"}, name="值").astype(str))
    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**籌碼（外資/投信/融資）**")
        chips = {k: v for k, v in (r.get("chips") or {}).items() if k != "籌碼分數"}
        st.table(pd.Series(chips or {"資料": "無"}, name="值").astype(str))
    with c5:
        st.markdown("**月營收**")
        rev = {k: v for k, v in (r.get("revenue") or {}).items() if k != "月份"}
        st.table(pd.Series(rev or {"資料": "無／不適用"}, name="值").astype(str))
    st.markdown("**最新 3 則新聞**")
    for n in r["news"]:
        tag = f"〔{n['tag']}〕" if n.get("tag") else ""
        st.markdown(f"**[{n['title']}]({n['link']})**  \n{tag}{n['source']} · {n['date']}")
        st.caption((n["summary"] or "（無法取得內文摘要，僅標題）") + f"　〔摘要來源：{n['summary_src']}〕")


def _style_summary(df: pd.DataFrame):
    """漲跌幅：紅漲綠跌。"""
    def color(v):
        if pd.isna(v) or v == 0:
            return ""
        return "color:#ef4444;font-weight:600" if v > 0 else "color:#22c55e;font-weight:600"
    sty = df.style.format({"漲跌幅%": "{:+.2f}%"}, na_rep="—") if "漲跌幅%" in df else df.style
    if "漲跌幅%" in df:
        sty = sty.map(color, subset=["漲跌幅%"]) if hasattr(sty, "map") else sty.applymap(color, subset=["漲跌幅%"])
    return sty


def run_streamlit():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    global GEMINI_MODEL

    # use_container_width 已被 Streamlit 移除/棄用，新版改用 width="stretch"；舊版仍用舊參數（相容兩者）
    _ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
    STRETCH = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}

    # 保險：同一次腳本執行裡若 UI 被呼叫第二次（例如被別的檔案 import / exec 又跑一遍），直接略過，
    # 否則同樣的元件會產生兩次 → StreamlitDuplicateElementId。（每次重新執行 ctx.cursors 都是新字典，拿來當「這一輪」的記號）
    run_token = getattr(get_script_run_ctx(), "cursors", None)
    if run_token is not None:
        if st.session_state.get("_dash_run_token") is run_token:
            return
        st.session_state["_dash_run_token"] = run_token

    st.set_page_config(page_title="台股決策儀表板", page_icon="📈", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("📈 台股決策儀表板")

    # Secrets → 環境變數 / 設定（有設定才套用）
    if _secret("FINMIND_TOKEN") and not os.getenv("FINMIND_TOKEN"):
        os.environ["FINMIND_TOKEN"] = _secret("FINMIND_TOKEN")
    if _secret("GEMINI_MODEL"):
        GEMINI_MODEL = _secret("GEMINI_MODEL")
    quant.set_db_path(_secret("DB_PATH") or None)

    # 選用：公開部署時用密碼擋住陌生人（hmac.compare_digest 防時間差攻擊）
    app_pw = _secret("APP_PASSWORD")
    if app_pw and not st.session_state.get("authed"):
        pw = st.text_input("存取密碼", type="password", key="ui_app_password")
        if pw and hmac.compare_digest(pw.encode("utf-8"), app_pw.encode("utf-8")):
            st.session_state["authed"] = True
            st.rerun()
        st.stop()

    try:
        daily_limit = int(_secret("DAILY_LIMIT") or DEFAULT_DAILY_LIMIT)
    except ValueError:
        daily_limit = DEFAULT_DAILY_LIMIT
    today = dt.datetime.now(TZ).date().isoformat()
    quota = st.session_state.get("_quota")
    if not quota or quota["date"] != today:
        quota = st.session_state["_quota"] = {"date": today, "used": 0}
    remaining = max(daily_limit - quota["used"], 0)

    backend_key = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    with st.expander("⚙️ 股票與分析設定", expanded=not st.session_state.get("results")):
        codes_txt = st.text_input("股票代碼（逗號分隔）", ",".join(DEFAULT_STOCKS), key="ui_codes_input",
                                  help="可輸入 2330、0050、00878、2330.TW 或中文名稱（常見股）；上市/上櫃後綴會自動判斷")
        if backend_key:
            key = backend_key
            st.success("🔑 已使用後台預設 API Key")
        else:
            key = st.text_input("Gemini API Key", type="password", key="ui_gemini_key",
                                help="想免輸入：在 Streamlit Secrets 設定 GEMINI_API_KEY，這個欄位就會自動隱藏")
        force = st.checkbox("強制重新分析（忽略今日快取）", key="ui_force",
                            help="同一交易日同一檔已有結果會直接讀取、不重打 Gemini；勾選才重新分析")
        go = st.button("🔄 開始 / 重新分析", type="primary", key="ui_run_button", **STRETCH)
        st.caption(f"今日剩餘 Gemini 分析額度：{remaining} / {daily_limit}（快取命中不扣額度；以 session 計）")
    st.caption("買進需通過：盈虧比≥2:1、停損不過遠、不追高、流動性、資料完整、回測有優勢、模擬期望值。"
               "本工具只做多，「賣出」＝已持有者減碼。台灣習慣色：漲紅、跌綠；"
               "評級卡片：買進＝藍綠、觀望＝琥珀、賣出＝深灰紅（刻意避開紅綠，以免與漲跌混淆）。僅供研究，非投資建議。")

    if go:
        if not key:
            st.error("請先輸入 Gemini API Key（或在 Secrets 設定 GEMINI_API_KEY）")
            st.stop()
        codes = [c.strip() for c in re.split(r"[,，、\s]+", codes_txt) if c.strip()]
        budget = Budget(remaining)
        bar = st.progress(0.0, text="分析中…")
        st.session_state["results"] = run_all(
            codes, key, lambda p, c: bar.progress(p, text=f"已完成 {c}"), force=force, budget=budget)
        quota["used"] += remaining - budget.n
        st.session_state["time"] = dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        bar.empty()

    try:
        _, track = quant.track_record()
    except Exception as e:
        log.warning("讀取戰績失敗：%s", e)
        track = {"總覽": {}, "分組": {}, "已結算": 0, "樣本不足": True}

    tab1, tab2, tab3 = st.tabs(["🃏 卡片", "📋 總表", "🎯 戰績"])
    results = st.session_state.get("results")
    ok = [r for r in (results or []) if "error" not in r]
    with tab1:
        if not results:
            st.info("上方設定好股票代碼，按「開始分析」。")
        else:
            reg = ok[0].get("regime", {}) if ok else {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{RATING_ICON['買進']} 買進", sum(r["rating_final"] == "買進" for r in ok))
            c2.metric(f"{RATING_ICON['觀望']} 觀望", sum(r["rating_final"] == "觀望" for r in ok))
            c3.metric(f"{RATING_ICON['賣出']} 賣出（減碼）", sum(r["rating_final"] == "賣出" for r in ok))
            c4.metric("大盤", reg.get("趨勢", "—"), f"VIX {reg.get('VIX', '—')}", delta_color="off")
            note = "｜⏱ 部分標的盤中，使用前一交易日收盤" if any(r.get("used_prev_close") for r in ok) else ""
            st.caption(f"更新 {st.session_state.get('time', '')}｜資料為最近交易日收盤{note}")
            for r in results:
                if "error" in r:
                    st.error(f"{r['code']}：{r['error']}")
            ncols = min(max(len(ok), 1), 3)     # 桌機最多 3 欄；手機 Streamlit 會自動改成單欄直排
            cols = st.columns(ncols)
            for i, r in enumerate(ok):
                with cols[i % ncols]:
                    st.markdown(card_html(r), unsafe_allow_html=True)
                    cmd = trade_command(r)
                    copy_button(cmd)
                    with st.expander("文字版指令"):
                        st.code(cmd, language=None)     # 備援：複製按鈕被瀏覽器擋住時，可用右上角複製圖示
            st.subheader("個股細節")
            for r in ok:
                with st.expander(f"{RATING_ICON[r['rating_final']]} {r['code']} {r['name']}（{r['symbol']}）｜"
                                 f"{r['rating_final']}｜{r.get('reason', '')}"):
                    _show_detail(st, r, STRETCH, track)
    with tab2:
        if results:
            st.dataframe(_style_summary(summary_table(results)), hide_index=True, **STRETCH)
        else:
            st.info("尚無分析結果")
    with tab3:
        if quant.is_ephemeral_fs():
            st.warning("⚠️ 戰績資料庫放在暫存檔案系統（例如 Streamlit Cloud），**重啟後戰績會遺失**。"
                       "請定期用下方「匯出 CSV」備份，或在 Secrets 設定 DB_PATH 指向持久化磁碟。")
        df, track = quant.track_record()
        st.write("每次分析**所有**標的都會存檔（含觀望）。「買進」用來驗證預測，「觀望」是假想結果"
                 "（若當初照建倉點買進會如何），兩者比較可看出過濾有沒有價值。")
        if track["樣本不足"]:
            st.info(f"已結算 {track['已結算']} 筆，不足 {quant.MIN_CLOSED_SAMPLES} 筆：樣本不足，僅供參考。")
        st.table(pd.Series(track["總覽"], name="值").astype(str))
        for label, g in track["分組"].items():
            st.markdown(f"**{label}**" + ("　（樣本不足，僅供參考）" if g.get("樣本不足") else ""))
            st.table(pd.Series({k: v for k, v in g.items() if k != "樣本不足"}, name="值").astype(str))
        if len(df):
            st.dataframe(df.drop(columns=["raw_json"], errors="ignore"), hide_index=True, **STRETCH)
            st.download_button("⬇️ 匯出戰績 CSV", key="ui_dl_csv", mime="text/csv", file_name="predictions.csv",
                               data=df.drop(columns=["raw_json"], errors="ignore").to_csv(index=False).encode("utf-8-sig"))


def run_cli(codes):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("請先設定環境變數 GEMINI_API_KEY")
    results = run_all(codes, key, lambda p, c: print(f"完成 {c}", file=sys.stderr))
    pd.set_option("display.width", 260)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n台股決策儀表板  {dt.date.today()}\n")
    print(summary_table(results).to_string(index=False))
    for r in results:
        if "error" in r:
            continue
        print(f"\n{trade_command(r)}")
        print(f"  {r.get('reason', '')}")
        for fl in r["flags"]:
            print("  ⚠", fl)
        print("  風險：" + "；".join(r.get("risks", [])))
        for n in r["news"]:
            print(f"  📰 {n['title']}\n     {n['summary'] or '（無摘要）'}")
    _, track = quant.track_record()
    print("\n戰績：", track["總覽"], track["分組"])
    if quant.is_ephemeral_fs():
        print("⚠️ 戰績 DB 位於暫存檔案系統，重啟後會遺失。", file=sys.stderr)


if _in_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    run_cli(sys.argv[1:] or DEFAULT_STOCKS)
