"""
台股自動化決策儀表板 v4
-----------------------
資料：yfinance（價量、加權指數、VIX）、FinMind（股票資訊、三大法人、融資券、本益比、月營收、毛利率）、
      Google News RSS（偏好鉅亨網/經濟日報等財經媒體，最新 3 則：標題＋內文摘要）
判斷：Gemini（五大面向）＋ quant.py（回測、機率模擬、凱利部位、戰績追蹤、籌碼特徵）
買進點：建倉點（回測 5 日線）＋ 加碼點（回測月線 或 突破壓力）
自動風控過濾（任一不過 → 強制「觀望 🟡」）：
    預期盈虧比 < 2:1｜停損離建倉點太遠｜剛大漲完還追高｜回測無優勢｜模擬期望值 ≤ 0｜建倉點太難成交

執行：
    streamlit run app.py                 # 網頁（手機友善卡片）
    python app.py 2330 0050 00878        # 終端機
設定（擇一）：Streamlit Secrets 的 GEMINI_API_KEY（有設定就自動隱藏輸入框）／環境變數 GEMINI_API_KEY
選填：FINMIND_TOKEN、GEMINI_MODEL、APP_PASSWORD（見 .streamlit/secrets.toml.example）
"""
import os
import re
import sys
import json
import html
import functools
import datetime as dt
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests
import pandas as pd
import yfinance as yf

import quant

# ------------------------------------------------------------------ 設定
DEFAULT_STOCKS = ["2330", "0050", "2454"]
# 2026-10 時點：gemini-1.5 全系列、gemini-2.0 已下架；gemini-2.5-pro/flash 於 2026-10-16 下架。
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite"]
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

# 風控過濾門檻
MIN_RR = 2.0                 # 預期盈虧比下限（以建倉點計算，最保守）
MAX_STOP_PCT = 0.08   # 直接定義 8% 停損，不依賴 quant 模組
MAX_STOP_ATR = quant.MAX_STOP_ATR   # 停損離建倉點最大 ATR 倍數
OVERHEAT_EXT20 = 12.0        # 收盤高於月線 12% 視為過熱
OVERHEAT_RET5 = 10.0         # 5 日漲幅 10% 視為過熱
OVERHEAT_RSI = 78.0
OVERHEAT_MIN_PULLBACK = 0.03  # 過熱時，建倉點至少要比現價低 3%，否則視為追高
MIN_FILL_PROB = 25.0         # 10 日內建倉點成交機率低於此值 → 觀望（價位離現價太遠）
W_ENTRY, W_ADD = 0.6, 0.4    # 建倉 / 加碼 資金比例

NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2412": "中華電",
    "2308": "台達電", "2303": "聯電", "3711": "日月光投控", "2881": "富邦金",
    "2882": "國泰金", "2891": "中信金", "2382": "廣達", "2603": "長榮",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息",
}
NAMES_REV = {v: k for k, v in NAMES.items()}
PREFERRED_SITES = ("cnyes.com", "money.udn.com", "ctee.com.tw", "ltn.com.tw")   # 鉅亨網、經濟日報、工商時報、自由財經

SYSTEM_PROMPT = """你是精通台股交易的資深量化分析師與風控專家，目標是「賺多賠少、盈虧比高」。
從五大面向分析：基本面（月營收 YoY/累計 YoY、毛利率、現金流、負債；ETF 則「不適用」）、估值面（本益比/股價淨值比百分位、殖利率）、
產業競爭力、技術與籌碼面（均線、外資/投信連買日數與買超、融資增減、支撐壓力）、總經與消息面（大盤、VIX、最新個股新聞與法說/營收消息）。
個股新聞附有標題與內文摘要；摘要為空代表只有標題，不得臆測內文。

【嚴格風控規則】（違反即視為錯誤輸出）
1. 買進點拆成兩點：
   - 建倉點 entry_price：回測短均線的拉回價（例如 5 日線附近），必須 ≤ 現價；依據寫入 entry_basis（如「回測5日線」）。
   - 加碼點 add_price：二擇一，add_type 填「回測月線」（約 20 日線，低於建倉點）或「突破壓力」（高於現價，約 20 日高/關鍵壓力上方 0.3~2%）。加碼只在趨勢確認後進行。
2. 預期盈虧比 = (目標價1 − 建倉點) / (建倉點 − 強制停損價)，以建倉點計算（最保守）。
   輸出前必須代入實際數字檢算，寫入 expected_rr（數字，小數 2 位）與 rr_calc（算式字串，如 "(112-100)/(100-94)=2.00"）。
3. 若 expected_rr < 2，rating 必須是「觀望」，沒有例外；此時仍須給出建倉點、加碼點、強制停損價（供等待參考）。
4. 強制停損必須是明確技術位（跌破某支撐/均線），且離建倉點不得超過 8% 或 4 倍 ATR；太遠就改用更近的技術位或直接觀望，
   不可為了湊出盈虧比而把停損拉遠。
5. 若「技術面.過熱」為 true（剛大漲完），建倉點必須是明確拉回價（至少低於現價 3%），不可貼近現價追高，否則 rating=觀望。
6. 回測統計 edge_ok=false，或籌碼與技術面嚴重矛盾，rating 也必須是「觀望」。
7. 「籌碼(FinMind)」內含程式依規則算出的籌碼訊號與外資/投信連買日數、法人近5日買超、融資增減。請明確判斷是
   「法人吃貨」還是「散戶追高」，並寫入 chips_technical；與程式訊號不同時須說明原因。
8. 價位須以現價、均線、支撐壓力、ATR 為依據，可參考「基準價位」微調，不得憑空捏造。資料沒有的數字不要編造。
rating 只能是「買進」「觀望」「賣出」。trend_bias 為純方向看法（偏多/中性/偏空），不受盈虧比規則限制。

只輸出 JSON（不要 markdown）：
{"rating":"買進"|"觀望"|"賣出","trend_bias":"偏多"|"中性"|"偏空","risk_level":"高"|"中"|"低",
 "entry_price":數字,"entry_basis":"如 回測5日線","add_price":數字,"add_type":"回測月線"|"突破壓力",
 "stop_loss":數字,"target1":數字,"target2":數字,"expected_rr":數字,"rr_calc":"算式字串",
 "position_pct":建議總部位佔總資金百分比(數字，觀望/賣出填0；建倉6成、加碼4成),
 "batches":"建倉與加碼的執行條件，一句話",
 "fundamentals_valuation":"一句話","industry":"一句話","chips_technical":"一句話（含法人吃貨/散戶追高判斷）",
 "news_macro":"一句話","reason":"評級理由一句話","risks":["風險1","風險2"]}"""


# ------------------------------------------------------------------ FinMind / 代碼解析
def _finmind(dataset: str, code=None, days: int = 30) -> pd.DataFrame:
    params = {"dataset": dataset}
    if code:
        params["data_id"] = code
    if days:
        params["start_date"] = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    if os.getenv("FINMIND_TOKEN"):
        params["token"] = os.getenv("FINMIND_TOKEN")
    r = requests.get(FINMIND_URL, params=params, timeout=25)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("data", []))


@functools.lru_cache(maxsize=1)
def _stock_info() -> dict:
    """FinMind 全市場股票清單：代碼 → 中文名稱、上市(twse)/上櫃(tpex)、產業別。失敗回空 dict。"""
    try:
        df = _finmind("TaiwanStockInfo", None, days=0)
        return {str(r.stock_id): {"name": r.stock_name, "type": r.type, "industry": r.industry_category}
                for r in df.itertuples()}
    except Exception:
        return {}


def _suffix_order(code: str) -> list:
    """優先用官方上市/上櫃分類決定 .TW / .TWO；查不到才用代碼開頭猜，猜錯會自動換另一個。"""
    t = (_stock_info().get(code) or {}).get("type")
    if t == "tpex":
        return [".TWO", ".TW"]
    if t == "twse":
        return [".TW", ".TWO"]
    return [".TW", ".TWO"] if code.startswith(("0", "1", "2")) else [".TWO", ".TW"]


def fetch_price(raw: str):
    """輸入 2330 / 0050 / 2330.TW / 台積電 皆可；自動補 .TW 或 .TWO。回傳 (Ticker, 3 年日線)。"""
    code = quant.normalize_code(raw, NAMES_REV)
    for suffix in _suffix_order(code):
        t = yf.Ticker(code + suffix)
        hist = t.history(period="3y", auto_adjust=False)
        if hist.empty:
            continue
        hist = hist.dropna(subset=["Close"])
        tz = hist.index.tz
        now = pd.Timestamp.now(tz=tz) if tz else pd.Timestamp.now()
        if now - hist.index[-1] > pd.Timedelta(days=30):
            raise ValueError(f"{code}{suffix} 最近一筆資料是 {hist.index[-1]:%Y-%m-%d}，可能已下市或停牌")
        return t, hist
    raise ValueError(f"找不到 {code}（已試 .TW 與 .TWO），請確認代碼是否正確")


# ------------------------------------------------------------------ 資料抓取
def compute_technicals(d: pd.DataFrame) -> dict:
    x, prev = d.iloc[-1], d.iloc[-2]
    r = lambda v: round(float(v), 2)
    c = d["Close"]
    last = float(x["Close"])
    ret5 = (last / float(c.iloc[-6]) - 1) * 100
    ext20 = (last / float(x["ma20"]) - 1) * 100
    hot = ext20 > OVERHEAT_EXT20 or ret5 > OVERHEAT_RET5 or float(x["rsi"]) > OVERHEAT_RSI
    return {
        "最近交易日": d.index[-1].strftime("%Y-%m-%d"),
        "收盤價": r(last), "漲跌幅%": r((last / float(prev["Close"]) - 1) * 100),
        "成交量(張)": int(x["Volume"] / 1000), "5日均量(張)": int(d["Volume"].tail(5).mean() / 1000),
        "MA5": r(x["ma5"]), "MA20": r(x["ma20"]), "MA60": r(x["ma60"]), "MA120": r(x["ma120"]),
        "RSI14": r(x["rsi"]), "ATR14": r(x["atr"]),
        "5日漲幅%": r(ret5), "距MA20%": r(ext20), "過熱": bool(hot),
        "20日支撐": r(d["Low"].tail(20).min()), "20日壓力": r(d["High"].tail(20).max()),
        "60日支撐": r(d["Low"].tail(60).min()), "60日壓力": r(d["High"].tail(60).max()),
        "近3年價位百分位%": r((last - c.min()) / (c.max() - c.min()) * 100),
    }


def fetch_chips(code: str, tech: dict) -> dict:
    """三大法人（外資/投信連買日數、買超張數）＋融資增減 → quant.chips_features 整理成變數與籌碼訊號。"""
    inst = marg = None
    try:
        df = _finmind("TaiwanStockInstitutionalInvestorsBuySell", code, days=45)
        if not df.empty:
            df["net"] = (df["buy"] - df["sell"]) / 1000

            def grp(n):
                if n.startswith("Foreign"):
                    return "外資"
                if n == "Investment_Trust":
                    return "投信"
                return "自營商" if n.startswith("Dealer") else "其他"

            df["group"] = df["name"].map(grp)
            inst = df.groupby(["date", "group"])["net"].sum().unstack(fill_value=0).sort_index()
    except Exception:
        pass
    try:
        m = _finmind("TaiwanStockMarginPurchaseShortSale", code, days=45).sort_values("date")
        if not m.empty:
            marg = m.rename(columns={"MarginPurchaseTodayBalance": "融資餘額",
                                     "ShortSaleTodayBalance": "融券餘額"}).set_index("date")[["融資餘額", "融券餘額"]]
    except Exception:
        pass
    if inst is None and marg is None:
        return {}
    return quant.chips_features(inst, marg, tech.get("5日均量(張)"), tech.get("5日漲幅%"))


def _pct_rank(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    return round(float((s < s.iloc[-1]).mean() * 100)) if len(s) >= 30 else None


def fetch_valuation(code: str) -> dict:
    """FinMind 本益比河流：目前 PER/PBR 在近 5 年的百分位（越高越貴）、殖利率 vs 5 年平均。"""
    try:
        df = _finmind("TaiwanStockPER", code, days=365 * 5).sort_values("date")
        per = pd.to_numeric(df["PER"], errors="coerce")
        pbr = pd.to_numeric(df["PBR"], errors="coerce")
        dy = pd.to_numeric(df["dividend_yield"], errors="coerce")
        return {
            "本益比": round(float(per.iloc[-1]), 1), "本益比5年百分位%": _pct_rank(per),
            "本益比5年中位數": round(float(per[per > 0].median()), 1),
            "股價淨值比": round(float(pbr.iloc[-1]), 2), "股價淨值比5年百分位%": _pct_rank(pbr),
            "殖利率%": round(float(dy.iloc[-1]), 2), "殖利率5年平均%": round(float(dy.mean()), 2),
        }
    except Exception:
        return {}


def fetch_revenue(code: str) -> dict:
    """月營收：最新 YoY、近 6 月 YoY、MoM、累計 YoY、YoY 連續成長月數，以及下次公布（每月 10 日前）提醒。"""
    try:
        df = _finmind("TaiwanStockMonthRevenue", code, days=640).sort_values("date").reset_index(drop=True)
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
        today = dt.date.today()
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
    except Exception:
        return {}


def fetch_margins_trend(code: str) -> dict:
    try:
        df = _finmind("TaiwanStockFinancialStatements", code, days=900)
        df = df[df["type"].isin(["Revenue", "GrossProfit", "OperatingIncome"])]
        p = df.pivot_table(index="date", columns="type", values="value").sort_index()
        return {"季別": list(p.index[-6:]),
                "近6季毛利率%": (p["GrossProfit"] / p["Revenue"] * 100).round(1).tail(6).tolist(),
                "近6季營益率%": (p["OperatingIncome"] / p["Revenue"] * 100).round(1).tail(6).tolist()}
    except Exception:
        return {}


def fetch_regime() -> dict:
    out = {"趨勢": "未知"}
    try:
        c = yf.Ticker("^TWII").history(period="1y")["Close"].dropna()
        last, ma60, ma120 = float(c.iloc[-1]), float(c.tail(60).mean()), float(c.tail(120).mean())
        out = {"加權指數": round(last), "距MA60%": round((last / ma60 - 1) * 100, 1),
               "20日報酬%": round((last / float(c.iloc[-21]) - 1) * 100, 1),
               "趨勢": "多頭" if last > ma60 > ma120 else "空頭" if last < ma60 < ma120 else "盤整"}
    except Exception:
        pass
    try:
        out["VIX"] = round(float(yf.Ticker("^VIX").history(period="5d")["Close"].dropna().iloc[-1]), 1)
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ 新聞（標題＋內文摘要）
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def _article_summary(link: str) -> str:
    """
    盡力取得原文摘要：解開 Google News 轉址 → 讀 og:description / meta description → 退而取前幾段文字。
    任何一步失敗回傳空字串（Google News 轉址有時解不開，或網站擋爬蟲）。
    """
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
    except Exception:
        return ""


def _rss_items(query: str) -> list:
    url = (f"https://news.google.com/rss/search?q={quote(query)}"
           "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    try:
        r = requests.get(url, timeout=15, headers=UA)
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
    except Exception:
        return []


def _uniq_sorted(items: list) -> list:
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["when"], reverse=True):
        if it["title"] not in seen:
            seen.add(it["title"])
            out.append(it)
    return out


def _finalize_news(items: list, with_summary: bool) -> list:
    if with_summary and items:
        with ThreadPoolExecutor(max_workers=len(items)) as ex:
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
    return _finalize_news(_rss_items(query)[:n], with_summary)


def fetch_stock_news(code: str, name: str, n: int = 3) -> list:
    """
    個股最新 n 則新聞：① 先取 1 則「法說會/營收/展望」事件消息 ② 其餘優先鉅亨網、經濟日報等財經媒體 ③ 不足再用一般新聞補。
    最後依時間新到舊排序，並抓內文摘要。
    """
    site = " OR ".join(f"site:{s}" for s in PREFERRED_SITES)
    queries = {
        "event": f"{name} (法說會 OR 營收 OR 展望) when:30d",
        "pref": f"{name} {code} ({site}) when:14d",
        "gen": f"{code} {name} when:7d",
    }
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {k: ex.submit(_rss_items, q) for k, q in queries.items()}
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
        return None, last

    tried = []
    data, err = attempt([GEMINI_MODEL] + FALLBACK_MODELS, tried)
    if data:
        return data
    # 全部失敗（例如又有模型下架）：查詢這把金鑰目前可用的模型，pro 優先、版本新的優先
    try:
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
def enforce_rules(r: dict, ctx: dict, base: dict) -> dict:
    """
    關卡 1（自動風控過濾）：程式自己重算，不信任 AI 自報。以下任一不過 → 強制「觀望」：
      ① 價位邏輯不合 ② 預期盈虧比 < 2:1 ③ 停損離建倉點太遠 ④ 剛大漲完卻貼近現價追高
    加碼點不合邏輯 → 改用系統基準加碼點；整組價位不合邏輯 → 改用系統基準價位（並強制觀望）。
    ctx: price, atr, overheated, ext20, ret5, rsi；base: quant.propose_levels() 的結果。
    """
    flags, rr, valid = [], None, False
    price, atr = ctx["price"], ctx.get("atr") or 0.0
    try:
        for k in ("entry_price", "stop_loss", "target1"):
            r[k] = quant.round_tick(float(r[k]))
        r["target2"] = quant.round_tick(float(r.get("target2") or r["target1"]))
        e, sl, t1 = r["entry_price"], r["stop_loss"], r["target1"]
        if e - sl > 0 and t1 - e > 0:
            rr = (t1 - e) / (e - sl)
        valid = sl < e < t1 and e <= price * 1.005
    except (KeyError, TypeError, ValueError):
        pass
    ai_levels = valid
    if not valid:   # 整組價位不合邏輯：換成系統基準價位，讓畫面上仍是合理數字
        for k in ("entry_price", "add_price", "add_type", "entry_basis", "stop_loss", "target1", "target2"):
            r[k] = base[k]
        e, sl, t1 = r["entry_price"], r["stop_loss"], r["target1"]
        rr = (t1 - e) / (e - sl)
        flags.append("AI 價位邏輯不一致，已改用系統基準價位，並強制標註為觀望")

    # 加碼點驗證（回測月線：停損 < 加碼 < 建倉；突破壓力：現價 ≤ 加碼 < 目標1）
    add_type = "突破壓力" if "突破" in str(r.get("add_type", "")) else "回測月線"
    try:
        add = quant.round_tick(float(r["add_price"]))
        add_ok = (sl < add < e) if add_type == "回測月線" else (price <= add < t1)
    except (KeyError, TypeError, ValueError):
        add, add_ok = None, False
    if not add_ok:
        add, add_type = base["add_price"], base["add_type"]
        if ai_levels:
            flags.append("AI 加碼點不合邏輯，已改用系統基準加碼點")
    r["add_price"], r["add_type"] = add, add_type
    basis = re.sub(r"[^\w\s（）()+\-/%.]", "", str(r.get("entry_basis") or ""))[:20].strip()
    r["entry_basis"] = basis or base["entry_basis"]

    avg_cost = W_ENTRY * e + W_ADD * add
    rr_blend = (t1 - avg_cost) / (avg_cost - sl) if avg_cost > sl and t1 > avg_cost else None
    stop_pct = (e - sl) / e * 100
    stop_atr = (e - sl) / atr if atr else None

    rating = r.get("rating", "觀望")
    if rating not in ("買進", "觀望", "賣出"):
        rating = "觀望"
    if rating != "觀望":
        if not ai_levels:
            rating = "觀望"
        elif rr is None or rr < MIN_RR:
            rating = "觀望"
            flags.append(f"預期盈虧比 {rr:.2f}:1 < 2:1，依規則強制標註為觀望" if rr
                         else "預期盈虧比不合格，依規則強制標註為觀望")
        elif stop_pct > MAX_STOP_PCT or (stop_atr is not None and stop_atr > MAX_STOP_ATR):
            rating = "觀望"
            flags.append(f"停損離建倉點太遠（{stop_pct:.1f}% / {stop_atr:.1f} ATR，上限 {MAX_STOP_PCT:.0f}% / "
                         f"{MAX_STOP_ATR:.0f} ATR），風險過大，已強制標註為觀望")
        elif ctx.get("overheated") and e > price * (1 - OVERHEAT_MIN_PULLBACK):
            rating = "觀望"
            flags.append(f"剛大漲完（距月線 {ctx.get('ext20', 0):+.1f}%、5日 {ctx.get('ret5', 0):+.1f}%、"
                         f"RSI {ctx.get('rsi', 0):.0f}），建倉點貼近現價等於追高，等拉回，已強制標註為觀望")
        else:
            if stop_atr is not None and stop_atr < 1.0:
                flags.append(f"停損僅 {stop_atr:.1f} ATR，容易被正常波動洗出")
            if rr_blend is not None and rr_blend < MIN_RR:
                flags.append(f"加碼後平均成本的盈虧比 {rr_blend:.2f}:1 偏低，加碼量建議縮小")
    try:   # AI 自報值與程式重算對不上 → 提醒；AI 自己都說 <2 卻給買進 → 也降級
        ai_rr = float(r.get("expected_rr"))
        if ai_levels and rr is not None and abs(ai_rr - rr) > 0.3:
            flags.append(f"AI 自報盈虧比 {ai_rr:.2f} 與程式重算 {rr:.2f} 不一致，以程式為準")
        if rating == "買進" and ai_rr < MIN_RR:
            rating = "觀望"
            flags.append(f"AI 自評盈虧比 {ai_rr:.2f} < 2:1，已強制標註為觀望")
    except (TypeError, ValueError):
        pass
    r.update({"rating_final": rating, "rr": rr, "rr_blend": rr_blend, "avg_cost": round(avg_cost, 2),
              "stop_pct": stop_pct, "stop_atr": stop_atr, "flags": flags, "levels_ok": True,
              "entry_low": e, "entry_high": e})   # entry_low/high：給模擬與戰績追蹤用（建倉點為單一價）
    return r


def quant_gate(r: dict, sim, bt: dict, regime: dict) -> dict:
    """關卡 2、3：回測優勢、建倉點可成交、模擬期望值；通過才給部位（四分之一凱利，建倉 6 成/加碼 4 成）。"""
    rating, flags, pos = r["rating_final"], r["flags"], 0.0
    if rating == "買進":
        if not bt["edge_ok"]:
            rating = "觀望"
            flags.append("回測無統計優勢（次數/期望值/獲利因子不足），已降為觀望")
        elif sim is None:
            rating = "觀望"
            flags.append("無法模擬此價位計畫，已降為觀望")
        elif sim["成交機率%"] < MIN_FILL_PROB:
            rating = "觀望"
            flags.append(f"建倉點離現價太遠，10 日內成交機率僅 {sim['成交機率%']:.0f}%，已降為觀望")
        elif sim["ev_r"] <= 0 or sim["p_win"] <= sim["breakeven"]:
            rating = "觀望"
            flags.append(f"模擬先達標機率 {sim['p_win']*100:.0f}% 未超過損益兩平 "
                         f"{sim['breakeven']*100:.0f}%（期望值 {sim['ev_r']:+.2f}R），已降為觀望")
    if rating == "買進":
        kelly = quant.kelly_fraction(sim["p_win"], sim["p_loss"], sim["r_win"], sim["r_loss"])
        pos = min(float(r.get("position_pct") or 0), 20.0, kelly * 25)   # 四分之一凱利，上限 20%
        if regime.get("趨勢") == "空頭":
            pos *= 0.5
            flags.append("大盤空頭，部位減半")
    r.update({"rating_final": rating, "position_final": round(pos, 1), "sim": sim,
              "pos_entry": round(pos * W_ENTRY, 1), "pos_add": round(pos * W_ADD, 1)})
    return r


# ------------------------------------------------------------------ 主流程
def analyze_stock(raw_code: str, api_key: str, macro_news: list, regime: dict) -> dict:
    code = quant.normalize_code(raw_code, NAMES_REV)
    ticker, hist = fetch_price(code)
    d = quant.add_indicators(hist)
    tech = compute_technicals(d)
    price = tech["收盤價"]
    info = _stock_info().get(code, {})
    name = info.get("name") or NAMES.get(code)
    if not name:
        try:
            name = ticker.info.get("shortName") or code
        except Exception:
            name = code

    bt = quant.backtest(d)
    base = quant.propose_levels(d)
    news = fetch_stock_news(code, name, n=3)
    chips = fetch_chips(code, tech)
    revenue = fetch_revenue(code)
    payload = {
        "股票": f"{code} {name}", "產業別": info.get("industry", "未知"),
        "今日日期": dt.date.today().isoformat(),
        "技術面": tech, "籌碼(FinMind)": chips,
        "估值(FinMind)": fetch_valuation(code), "月營收": revenue, "毛利率趨勢": fetch_margins_trend(code),
        "大盤環境": regime, "回測統計": {**bt, "說明": "固定規則多頭拉回策略，R=風險單位，已含交易成本"},
        "基準價位": {"建倉點": base["entry_price"], "建倉依據": base["entry_basis"],
                  "加碼點": base["add_price"], "加碼類型": base["add_type"],
                  "強制停損價": base["stop_loss"], "目標價1": base["target1"], "目標價2": base["target2"]},
        "風控過濾門檻": {"最低預期盈虧比": MIN_RR, "停損上限%": MAX_STOP_PCT, "停損上限ATR倍數": MAX_STOP_ATR,
                    "過熱時建倉點至少低於現價%": OVERHEAT_MIN_PULLBACK * 100},
        "個股最新3則新聞": [{"標題": n["title"], "來源": n["source"], "日期": n["date"], "類型": n["tag"] or "一般",
                         "內文摘要": n["summary"] or "（無摘要，僅標題）"} for n in news],
        "總經新聞標題": [n["title"] for n in macro_news],
    }
    prompt = "請分析以下台股資料並輸出 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    ctx = {"price": price, "atr": tech["ATR14"], "overheated": tech["過熱"],
           "ext20": tech["距MA20%"], "ret5": tech["5日漲幅%"], "rsi": tech["RSI14"]}
    r = enforce_rules(call_gemini(api_key, prompt), ctx, base)
    sim = quant.simulate_plan(hist["Close"], price, r["entry_price"], r["entry_price"], r["target1"], r["stop_loss"])
    r = quant_gate(r, sim, bt, regime)

    chart = d[["Close", "ma5", "ma20"]].tail(120).rename(columns={"Close": "收盤", "ma5": "MA5", "ma20": "MA20"})
    chart["建倉點"], chart["停損"], chart["目標1"] = r["entry_price"], r["stop_loss"], r["target1"]
    chart.index = chart.index.strftime("%m-%d")
    r.update({"code": code, "name": name, "symbol": ticker.ticker, "price": price,
              "change": tech["漲跌幅%"], "news": news, "tech": tech, "bt": bt, "chart": chart,
              "date": tech["最近交易日"], "valuation": payload["估值(FinMind)"], "revenue": revenue,
              "chips": chips, "overheated": tech["過熱"]})
    return r


def run_all(codes, api_key, progress=None) -> list:
    try:   # 先結算過去的預測，戰績才會是最新的
        quant.evaluate_open(lambda c: quant.add_indicators(fetch_price(c)[1]))
    except Exception:
        pass
    regime = fetch_regime()
    macro = fetch_news("台股 外資 聯準會 總體經濟 when:3d", n=6)
    out, seen = [], set()
    for i, raw in enumerate(codes):
        try:
            code = quant.normalize_code(raw, NAMES_REV)
            if code in seen:
                continue
            seen.add(code)
            r = analyze_stock(code, api_key, macro, regime)
            quant.log_prediction(r["date"], r)
            r["regime"] = regime
            out.append(r)
        except Exception as e:
            out.append({"code": str(raw), "error": str(e)})
        if progress:
            progress((i + 1) / len(codes), str(raw))
    return out


RATING_ICON = {"買進": "🟢", "觀望": "🟡", "賣出": "🔴"}


def _g(v) -> str:
    """價格顯示：950.0 → 950，955.5 → 955.5"""
    return "—" if v is None else f"{float(v):g}"


def trade_command(r: dict) -> str:
    """一鍵複製用的交易指令（貼到筆記或券商 App 提醒自己）。"""
    head = f"【{r['code']} {r['name']}】"
    body = (f"建倉點：{_g(r['entry_price'])}（{r['entry_basis']}）｜加碼點：{_g(r['add_price'])}（{r['add_type']}）"
            f"｜停損價：{_g(r['stop_loss'])}｜目標價：{_g(r['target1'])}/{_g(r['target2'])}")
    rr = f"｜盈虧比 {r['rr']:.2f}:1" if r.get("rr") else ""
    if r["rating_final"] == "買進":
        return f"{head}買進 🟢｜{body}{rr}｜部位：建倉{r['pos_entry']:g}%＋加碼{r['pos_add']:g}%"
    return f"{head}{r['rating_final']} {RATING_ICON[r['rating_final']]}（未達進場條件，僅供參考）｜{body}{rr}"


def summary_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"股票代碼": r["code"], "當前評級": "⚠️ 錯誤", "現價": r["error"][:50]})
            continue
        s, bt = r.get("sim"), r["bt"]["全期間"]
        rows.append({
            "股票代碼": f"{r['code']} {r['name']}",
            "現價": f"{r['price']:.2f} ({r['change']:+.2f}%)",
            "當前評級": f"{RATING_ICON[r['rating_final']]} {r['rating_final']}",
            "多空傾向": r.get("trend_bias", "—"),
            "建倉點 / 加碼點": f"{_g(r['entry_price'])} / {_g(r['add_price'])}",
            "目標價": f"{_g(r['target1'])} / {_g(r['target2'])}",
            "強制停損價": f"{_g(r['stop_loss'])} (-{r['stop_pct']:.1f}%)",
            "預期盈虧比": f"{r['rr']:.2f}:1" if r["rr"] else "—",
            "籌碼訊號": r["chips"].get("籌碼訊號", "—") if r.get("chips") else "—",
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
.dash-head{padding:14px 16px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.dash-head .t{font-size:1.05rem;font-weight:700;line-height:1.25}
.dash-head .s{font-size:.85rem;opacity:.9}
.dash-head .rating{font-size:1.7rem;font-weight:800;white-space:nowrap}
.dash-body{padding:12px 14px 4px 14px;display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}
.dash-body .l{font-size:.75rem;opacity:.7}
.dash-body .v{font-size:1.15rem;font-weight:700}
.dash-body .v small{font-size:.75rem;font-weight:500;opacity:.75}
.dash-foot{padding:6px 14px 12px 14px;font-size:.85rem;opacity:.9;line-height:1.45}
.dash-warn{margin:0 14px 10px 14px;padding:8px 10px;border-radius:10px;font-size:.8rem;
  background:rgba(234,179,8,.18);border:1px solid rgba(234,179,8,.5)}
</style>
"""
CARD_COLORS = {"買進": ("#16a34a", "#ffffff"), "觀望": ("#eab308", "#1f2937"), "賣出": ("#dc2626", "#ffffff")}


def card_html(r: dict) -> str:
    """綠/黃/紅評級大卡片（手機兩欄格狀，字體放大）。"""
    e = html.escape
    bg, fg = CARD_COLORS[r["rating_final"]]
    chips = (r.get("chips") or {}).get("籌碼訊號", "")
    warn = "".join(f"<div>⚠️ {e(f)}</div>" for f in r["flags"][:2])
    rr = f"{r['rr']:.2f}:1" if r.get("rr") else "—"
    pos = (f"{r['pos_entry']:g}% + {r['pos_add']:g}%" if r["rating_final"] == "買進" else "0%")
    cell = lambda l, v, sub="": f'<div><div class="l">{e(l)}</div><div class="v">{v}{f" <small>{e(sub)}</small>" if sub else ""}</div></div>'
    return f"""
<div class="dash-card">
 <div class="dash-head" style="background:{bg};color:{fg}">
  <div><div class="t">{e(r['code'])} {e(r['name'])}</div>
   <div class="s">{r['price']:.2f}（{r['change']:+.2f}%）｜{e(str(r.get('trend_bias', '')))}</div></div>
  <div class="rating">{RATING_ICON[r['rating_final']]} {e(r['rating_final'])}</div>
 </div>
 <div class="dash-body">
  {cell("建倉點", _g(r['entry_price']), r['entry_basis'])}
  {cell("加碼點", _g(r['add_price']), r['add_type'])}
  {cell("強制停損價", _g(r['stop_loss']), f"-{r['stop_pct']:.1f}%")}
  {cell("目標價", f"{_g(r['target1'])} / {_g(r['target2'])}")}
  {cell("預期盈虧比", rr)}
  {cell("建議部位", pos, "建倉+加碼")}
 </div>
 {f'<div class="dash-warn">{warn}</div>' if warn else ''}
 <div class="dash-foot">{e(str(r.get('reason', '')))}{('<br>' + e(chips)) if chips else ''}</div>
</div>"""


def copy_button(text: str, label: str = "📋 一鍵複製交易指令"):
    """
    點一下複製到剪貼簿（clipboard API 被擋時退回 execCommand）。
    新版 Streamlit 用 st.iframe（components.v1.html 已公告移除），舊版自動退回 components.html。
    文字以 JSON 字串嵌入並跳脫所有「<」，避免內容（含 AI 產生的欄位）被當成 HTML/script 解析。
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


def _secret(name: str, default: str = "") -> str:
    """讀 Streamlit Secrets；沒有 secrets 檔、沒這個鍵都安靜回傳預設值。"""
    try:
        import streamlit as st
        v = st.secrets.get(name, default)
        return str(v).strip() if v else default
    except Exception:
        return default


def _show_detail(st, r):
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
        + f"｜**過熱**：{'是 🔥' if r.get('overheated') else '否'}\n"
        f"- **基本/估值**：{r.get('fundamentals_valuation', '')}\n"
        f"- **產業**：{r.get('industry', '')}\n"
        f"- **籌碼/技術**：{r.get('chips_technical', '')}\n"
        f"- **消息/總經**：{r.get('news_macro', '')}\n"
        f"- **分批執行**：{r.get('batches', '')}\n"
        f"- **⚠️ 風險**：" + "；".join(r.get("risks", [])))
    st.line_chart(r["chart"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**機率模擬**（建倉點，未來 30 交易日）")
        if r.get("sim"):
            st.table(pd.Series({k: v for k, v in r["sim"].items()
                                if k.endswith("%") or "R" in k or "天數" in k}, name="值").astype(str))
    with c2:
        st.markdown("**策略回測**（全期間 / 近1年）")
        st.table(pd.DataFrame({"全期間": r["bt"]["全期間"], "近1年": r["bt"]["近1年"]}).astype(str))
        st.caption("回測有優勢：" + ("✅ 是" if r["bt"]["edge_ok"] else "❌ 否"))
    with c3:
        st.markdown("**估值**")
        st.table(pd.Series(r["valuation"] or {"資料": "無"}, name="值").astype(str))
    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**籌碼（外資/投信/融資）**")
        chips = {k: v for k, v in (r.get("chips") or {}).items() if k != "籌碼分數"}
        st.table(pd.Series(chips or {"資料": "無"}, name="值").astype(str))
    with c5:
        st.markdown("**月營收**")
        rev = {k: v for k, v in (r.get("revenue") or {}).items() if k not in ("月份",)}
        st.table(pd.Series(rev or {"資料": "無"}, name="值").astype(str))
    st.markdown("**最新 3 則新聞**")
    for n in r["news"]:
        tag = f"〔{n['tag']}〕" if n.get("tag") else ""
        st.markdown(f"**[{n['title']}]({n['link']})**  \n{tag}{n['source']} · {n['date']}")
        st.caption((n["summary"] or "（無法取得內文摘要，僅標題）") + f"　〔摘要來源：{n['summary_src']}〕")


def run_streamlit():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    global GEMINI_MODEL

    # use_container_width 已被 Streamlit 移除/棄用，新版改用 width="stretch"；舊版仍用舊參數（相容兩者）
    _ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
    STRETCH = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}

    # 保險：同一次腳本執行裡若 UI 被呼叫第二次（例如被別的檔案 import / exec 又跑一遍），直接略過，
    # 否則同樣的元件會產生兩次 → StreamlitDuplicateElementId。
    # （Streamlit 每次重新執行都會換一個新的 ctx.cursors 字典，拿它的身分當「這一輪」的記號。）
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

    # 選用：公開部署時用密碼擋住陌生人（否則他們會用掉你的 API 額度）
    app_pw = _secret("APP_PASSWORD")
    if app_pw and not st.session_state.get("authed"):
        pw = st.text_input("存取密碼", type="password", key="ui_app_password")
        if pw and pw == app_pw:
            st.session_state["authed"] = True
            st.rerun()
        st.stop()

    backend_key = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    # 設定放在主畫面的收合區（手機不用去點側邊欄）
    with st.expander("⚙️ 股票與分析設定", expanded=not st.session_state.get("results")):
        codes_txt = st.text_input("股票代碼（逗號分隔）", ",".join(DEFAULT_STOCKS), key="ui_codes_input",
                                  help="可輸入 2330、0050、00878、2330.TW 或中文名稱（常見股）；上市/上櫃後綴會自動判斷")
        if backend_key:
            key = backend_key
            st.success("🔑 已使用後台預設 API Key")
        else:
            key = st.text_input("Gemini API Key", type="password", key="ui_gemini_key",
                                help="想免輸入：在 Streamlit Secrets 設定 GEMINI_API_KEY，這個欄位就會自動隱藏")
        go = st.button("🔄 開始 / 重新分析", type="primary", key="ui_run_button", **STRETCH)
    st.caption("買進需同時通過：盈虧比≥2:1、停損不過遠、不追高、回測有優勢、模擬期望值>0。僅供研究，非投資建議。")

    if go:
        if not key:
            st.error("請先輸入 Gemini API Key（或在 Secrets 設定 GEMINI_API_KEY）")
            st.stop()
        codes = [c.strip() for c in re.split(r"[,，、\s]+", codes_txt) if c.strip()]
        bar = st.progress(0.0, text="分析中…")
        st.session_state["results"] = run_all(
            codes, key, lambda p, c: bar.progress(p, text=f"已完成 {c}"))
        st.session_state["time"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        bar.empty()

    tab1, tab2, tab3 = st.tabs(["🃏 卡片", "📋 總表", "🎯 戰績"])
    results = st.session_state.get("results")
    ok = [r for r in (results or []) if "error" not in r]
    with tab1:
        if not results:
            st.info("上方設定好股票代碼，按「開始分析」。")
        else:
            reg = ok[0]["regime"] if ok else {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 買進", sum(r["rating_final"] == "買進" for r in ok))
            c2.metric("🟡 觀望", sum(r["rating_final"] == "觀望" for r in ok))
            c3.metric("🔴 賣出", sum(r["rating_final"] == "賣出" for r in ok))
            c4.metric("大盤", reg.get("趨勢", "—"), f"VIX {reg.get('VIX', '—')}", delta_color="off")
            st.caption(f"更新 {st.session_state['time']}｜資料為最近交易日收盤")
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
                    st.code(cmd, language=None)     # 備援：複製按鈕被瀏覽器擋住時，可用右上角複製圖示
            st.subheader("個股細節")
            for r in ok:
                with st.expander(f"{RATING_ICON[r['rating_final']]} {r['code']} {r['name']}（{r['symbol']}）｜"
                                 f"{r['rating_final']}｜{r.get('reason', '')}"):
                    _show_detail(st, r)
    with tab2:
        if results:
            st.dataframe(summary_table(results), hide_index=True, **STRETCH)
        else:
            st.info("尚無分析結果")
    with tab3:
        df, summ = quant.track_record()
        st.write("每次「買進」建議都會存檔，之後用真實股價結算。**模型預測勝率 vs 實際勝率**越接近，代表機率越可信。")
        st.table(pd.Series(summ, name="值").astype(str))
        if len(df):
            st.dataframe(df, hide_index=True, **STRETCH)


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
    _, summ = quant.track_record()
    print("\n戰績：", summ)


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    run_cli(sys.argv[1:] or DEFAULT_STOCKS)
