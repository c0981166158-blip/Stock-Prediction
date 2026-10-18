"""
台股自動化決策儀表板 v3
-----------------------
資料：yfinance（價量、加權指數、VIX）、FinMind（股票資訊、三大法人、融資券、本益比、月營收、毛利率）、
      Google News RSS（最新 3 則：標題＋內文摘要）
判斷：Gemini（五大面向）＋ quant.py（回測、機率模擬、凱利部位、戰績追蹤）
關卡：「買進」需同時通過 ① 預期盈虧比≥2:1 ② 回測有優勢 ③ 模擬期望值>0，否則強制「觀望」

執行：
    streamlit run app.py                 # 網頁
    python app.py 2330 0050 00878        # 終端機
設定（擇一）：
    Streamlit Secrets（.streamlit/secrets.toml 或雲端後台）：GEMINI_API_KEY（有設定就自動隱藏輸入框）
    環境變數：GEMINI_API_KEY
選填：FINMIND_TOKEN、GEMINI_MODEL、APP_PASSWORD（公開部署時擋陌生人，見 secrets.toml.example）
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
MIN_RR = 2.0
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2412": "中華電",
    "2308": "台達電", "2303": "聯電", "3711": "日月光投控", "2881": "富邦金",
    "2882": "國泰金", "2891": "中信金", "2382": "廣達", "2603": "長榮",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息",
}
NAMES_REV = {v: k for k, v in NAMES.items()}

SYSTEM_PROMPT = """你是精通台股交易的資深量化分析師與風控專家，目標是「賺多賠少、盈虧比高」。
從五大面向分析：基本面（營收/毛利率/現金流/負債；ETF 則「不適用」）、估值面（本益比/股價淨值比的歷史百分位與殖利率）、
產業競爭力、技術與籌碼面（均線、三大法人、融資、支撐壓力）、總經與消息面（大盤趨勢、VIX、最新個股新聞）。
個股新聞附有標題與內文摘要；摘要為空代表只有標題，不得臆測內文。

【嚴格風控規則】（違反即視為錯誤輸出）
1. 平均買進價 = (第一批買進價 + 第二批買進價) / 2（兩批各半，第一批較接近現價、第二批為更低的回檔價）。
2. 預期盈虧比 = (目標價1 − 平均買進價) / (平均買進價 − 強制停損價)。
   輸出前必須把實際數字代入檢算，寫入 expected_rr（數字，小數 2 位）與 rr_calc（算式字串，如 "(112-101)/(101-96)=2.20"）。
3. 若 expected_rr < 2，rating 必須是「觀望」，沒有例外；此時仍要給出明確的第一批/第二批買進價與強制停損價（供等待參考）。
4. 資料中「回測統計」若 edge_ok=false，或籌碼與技術面嚴重矛盾，rating 也必須是「觀望」。
5. 價位須以現價、均線、支撐壓力、ATR 為依據，可參考「基準價位」微調，不得憑空捏造；強制停損必須是明確技術位（跌破某支撐/均線）。
6. 資料沒有的數字不要編造，可說「資料不足」。
rating 只能是「買進」「觀望」「賣出」。trend_bias 表示純方向看法（偏多/中性/偏空），不受盈虧比規則限制，
用來在評級被強制為觀望時，仍讓使用者知道方向。

只輸出 JSON（不要 markdown）：
{"rating":"買進"|"觀望"|"賣出","trend_bias":"偏多"|"中性"|"偏空","risk_level":"高"|"中"|"低",
 "batch1_price":數字,"batch2_price":數字,"target1":數字,"target2":數字,"stop_loss":數字,
 "expected_rr":數字,"rr_calc":"算式字串",
 "position_pct":建議佔總資金百分比(數字，觀望/賣出填0),"batches":"分批說明一句話",
 "fundamentals_valuation":"一句話","industry":"一句話","chips_technical":"一句話","news_macro":"一句話",
 "reason":"評級理由一句話","risks":["風險1","風險2"]}"""


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
    return {
        "最近交易日": d.index[-1].strftime("%Y-%m-%d"),
        "收盤價": r(x["Close"]), "漲跌幅%": r((x["Close"] / prev["Close"] - 1) * 100),
        "成交量(張)": int(x["Volume"] / 1000), "5日均量(張)": int(d["Volume"].tail(5).mean() / 1000),
        "MA5": r(x["ma5"]), "MA20": r(x["ma20"]), "MA60": r(x["ma60"]), "MA120": r(x["ma120"]),
        "RSI14": r(x["rsi"]), "ATR14": r(x["atr"]),
        "20日支撐": r(d["Low"].tail(20).min()), "20日壓力": r(d["High"].tail(20).max()),
        "60日支撐": r(d["Low"].tail(60).min()), "60日壓力": r(d["High"].tail(60).max()),
        "近3年價位百分位%": r((x["Close"] - c.min()) / (c.max() - c.min()) * 100),
    }


def fetch_institutional(code: str) -> dict:
    try:
        df = _finmind("TaiwanStockInstitutionalInvestorsBuySell", code)
        if df.empty:
            return {}
        df["net"] = (df["buy"] - df["sell"]) / 1000

        def grp(n):
            if n.startswith("Foreign"):
                return "外資"
            if n == "Investment_Trust":
                return "投信"
            return "自營商" if n.startswith("Dealer") else "其他"

        df["group"] = df["name"].map(grp)
        p = df.groupby(["date", "group"])["net"].sum().unstack(fill_value=0).sort_index()
        p["合計"] = p.sum(axis=1)
        return {"資料日期": p.index[-1],
                "最新一日(張)": p.iloc[-1].round(0).astype(int).to_dict(),
                "近5日合計(張)": p.tail(5).sum().round(0).astype(int).to_dict()}
    except Exception as e:
        return {"error": f"三大法人抓取失敗：{e}"}


def fetch_margin(code: str) -> dict:
    try:
        d = _finmind("TaiwanStockMarginPurchaseShortSale", code).sort_values("date")
        mb, sb = d["MarginPurchaseTodayBalance"], d["ShortSaleTodayBalance"]
        return {"融資餘額(張)": int(mb.iloc[-1]),
                "融資近5日增減(張)": int(mb.iloc[-1] - mb.iloc[-6]) if len(mb) > 5 else None,
                "融券餘額(張)": int(sb.iloc[-1])}
    except Exception:
        return {}


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
    try:
        df = _finmind("TaiwanStockMonthRevenue", code, days=640).sort_values("date")
        rev = df["revenue"].astype(float).reset_index(drop=True)
        yoy, mom = rev.pct_change(12) * 100, rev.pct_change() * 100
        months = [f"{y}-{int(m):02d}" for y, m in zip(df["revenue_year"], df["revenue_month"])]
        return {"月份": months[-6:], "近6月YoY%": [round(v, 1) for v in yoy.tail(6)],
                "最新MoM%": round(float(mom.iloc[-1]), 1),
                "近3月平均YoY%": round(float(yoy.tail(3).mean()), 1),
                "最新月營收(億)": round(float(rev.iloc[-1]) / 1e8, 1)}
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


def fetch_news(query: str, n: int = 3, with_summary: bool = False) -> list:
    """Google News RSS：取最新 n 則（依發布時間排序）。with_summary=True 會並行抓每則的內文摘要。"""
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
    except Exception:
        return []
    seen, uniq = set(), []
    for it in sorted(items, key=lambda x: x["when"], reverse=True):
        if it["title"] not in seen:
            seen.add(it["title"])
            uniq.append(it)
    items = uniq[:n]
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
        out.append({"title": it["title"], "link": it["link"], "source": it["source"],
                    "date": it["when"].strftime("%Y-%m-%d") if it["when"].year > 1 else "",
                    "summary": s, "summary_src": src or "無"})
    return out


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
LEVEL_KEYS = ("entry_low", "entry_high", "target1", "target2", "stop_loss")


def enforce_rules(r: dict) -> dict:
    """
    關卡 1：程式自己重算預期盈虧比（不信任 AI 自報），價位邏輯或盈虧比 < 2:1 → 一律強制「觀望」。
    內部 entry_high = 第一批買進價，entry_low = 第二批買進價。
    """
    flags, rr, valid = [], None, False
    try:
        if r.get("batch1_price") is not None and r.get("batch2_price") is not None:
            b1, b2 = float(r["batch1_price"]), float(r["batch2_price"])
            r["entry_high"], r["entry_low"] = max(b1, b2), min(b1, b2)
        if r.get("target2") is None:
            r["target2"] = r.get("target1")
        for k in LEVEL_KEYS:
            r[k] = quant.round_tick(float(r[k]))
        mid = (r["entry_low"] + r["entry_high"]) / 2
        if mid - r["stop_loss"] > 0 and r["target1"] - mid > 0:
            rr = (r["target1"] - mid) / (mid - r["stop_loss"])
        valid = (r["entry_low"] <= r["entry_high"] and r["stop_loss"] < r["entry_low"]
                 and r["target1"] > r["entry_high"])
    except (KeyError, TypeError, ValueError):
        pass

    rating = r.get("rating", "觀望")
    if rating not in ("買進", "觀望", "賣出"):
        rating = "觀望"
    if rating != "觀望":
        if not valid:
            rating = "觀望"
            flags.append("AI 價位邏輯不一致，已強制標註為觀望")
        elif rr is None or rr < MIN_RR:
            rating = "觀望"
            flags.append(f"預期盈虧比 {rr:.2f}:1 < 2:1，依規則強制標註為觀望" if rr
                         else "預期盈虧比不合格，依規則強制標註為觀望")
    try:   # AI 自報值與程式重算對不上 → 提醒；AI 自己都說 <2 卻給買進 → 也降級
        ai_rr = float(r.get("expected_rr"))
        if rr is not None and abs(ai_rr - rr) > 0.3:
            flags.append(f"AI 自報盈虧比 {ai_rr:.2f} 與程式重算 {rr:.2f} 不一致，以程式為準")
        if rating == "買進" and ai_rr < MIN_RR:
            rating = "觀望"
            flags.append(f"AI 自評盈虧比 {ai_rr:.2f} < 2:1，已強制標註為觀望")
    except (TypeError, ValueError):
        pass
    r.update({"rating_final": rating, "rr": rr, "flags": flags, "levels_ok": valid})
    return r


def quant_gate(r: dict, sim, bt: dict, regime: dict) -> dict:
    """關卡 2、3：回測優勢＋機率模擬期望值；並以凱利決定部位。"""
    rating, flags, pos = r["rating_final"], r["flags"], 0.0
    if rating == "買進":
        if not bt["edge_ok"]:
            rating = "觀望"
            flags.append("回測無統計優勢（次數/期望值/獲利因子不足），已降為觀望")
        elif sim is None:
            rating = "觀望"
            flags.append("無法模擬此價位計畫，已降為觀望")
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
    r.update({"rating_final": rating, "position_final": round(pos, 1), "sim": sim})
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
    news = fetch_news(f"{code} {name} when:7d", n=3, with_summary=True)
    payload = {
        "股票": f"{code} {name}", "產業別": info.get("industry", "未知"),
        "今日日期": dt.date.today().isoformat(),
        "技術面": tech, "估值(FinMind)": fetch_valuation(code),
        "月營收": fetch_revenue(code), "毛利率趨勢": fetch_margins_trend(code),
        "三大法人": fetch_institutional(code), "融資券": fetch_margin(code),
        "大盤環境": regime, "回測統計": {**bt, "說明": "固定規則多頭拉回策略，R=風險單位，已含交易成本"},
        "基準價位": {"第一批買進價": base["entry_high"], "第二批買進價": base["entry_low"],
                  "強制停損價": base["stop_loss"], "目標價1": base["target1"], "目標價2": base["target2"]},
        "個股最新3則新聞": [{"標題": n["title"], "來源": n["source"], "日期": n["date"],
                         "內文摘要": n["summary"] or "（無摘要，僅標題）"} for n in news],
        "總經新聞標題": [n["title"] for n in macro_news],
    }
    prompt = "請分析以下台股資料並輸出 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    r = enforce_rules(call_gemini(api_key, prompt))
    sim = (quant.simulate_plan(hist["Close"], price, r["entry_low"], r["entry_high"],
                               r["target1"], r["stop_loss"]) if r["levels_ok"] else None)
    r = quant_gate(r, sim, bt, regime)

    chart = d[["Close", "ma20", "ma60"]].tail(120).rename(columns={"Close": "收盤", "ma20": "MA20", "ma60": "MA60"})
    if r["levels_ok"]:
        chart["停損"], chart["第一批"], chart["目標1"] = r["stop_loss"], r["entry_high"], r["target1"]
    chart.index = chart.index.strftime("%m-%d")
    r.update({"code": code, "name": name, "symbol": ticker.ticker, "price": price,
              "change": tech["漲跌幅%"], "news": news, "tech": tech, "bt": bt, "chart": chart,
              "date": tech["最近交易日"], "valuation": payload["估值(FinMind)"],
              "revenue": payload["月營收"]})
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


def summary_table(results: list) -> pd.DataFrame:
    icon = {"買進": "🟢 買進", "觀望": "🟡 觀望", "賣出": "🔴 賣出"}
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"股票代碼": r["code"], "當前評級": "⚠️ 錯誤", "現價": r["error"][:50]})
            continue
        f = lambda k: f"{float(r[k]):.2f}" if r.get(k) is not None else "—"
        s, bt = r.get("sim"), r["bt"]["全期間"]
        rows.append({
            "股票代碼": f"{r['code']} {r['name']}",
            "現價": f"{r['price']:.2f} ({r['change']:+.2f}%)",
            "當前評級": icon.get(r["rating_final"], r["rating_final"]),
            "多空傾向": r.get("trend_bias", "—"),
            "第一批 / 第二批買進價": f"{f('entry_high')} / {f('entry_low')}",
            "目標價": f"{f('target1')} / {f('target2')}",
            "強制停損價": f("stop_loss"),
            "預期盈虧比": f"{r['rr']:.2f}:1" if r["rr"] else "—",
            "達標機率(模擬)": f"{s['先達標機率%']}%" if s else "—",
            "期望值(R)": f"{s['期望值R']:+.2f}" if s else "—",
            "回測勝率/期望值": f"{bt.get('勝率%', '—')}% / {bt.get('期望值R', '—')}R",
            "建議部位%": r["position_final"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 介面
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
    st.markdown(
        f"- **多空傾向**：{r.get('trend_bias', '—')}｜**風險等級**：{r.get('risk_level', '—')}\n"
        f"- **盈虧比**：AI 算式 `{r.get('rr_calc', '—')}`｜程式重算 "
        f"`{rr_txt}`\n"
        f"- **基本/估值**：{r.get('fundamentals_valuation', '')}\n"
        f"- **產業**：{r.get('industry', '')}\n"
        f"- **籌碼/技術**：{r.get('chips_technical', '')}\n"
        f"- **消息/總經**：{r.get('news_macro', '')}\n"
        f"- **分批**：{r.get('batches', '')}\n"
        f"- **⚠️ 風險**：" + "；".join(r.get("risks", [])))
    st.line_chart(r["chart"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**機率模擬**（未來 30 交易日）")
        if r.get("sim"):
            st.table(pd.Series({k: v for k, v in r["sim"].items()
                                if k.endswith("%") or "R" in k or "天數" in k}, name="值").astype(str))
        else:
            st.caption("價位不合邏輯，未模擬")
    with c2:
        st.markdown("**策略回測**（全期間 / 近1年）")
        st.table(pd.DataFrame({"全期間": r["bt"]["全期間"], "近1年": r["bt"]["近1年"]}).astype(str))
        st.caption("回測有優勢：" + ("✅ 是" if r["bt"]["edge_ok"] else "❌ 否"))
    with c3:
        st.markdown("**估值 / 營收**")
        st.table(pd.Series({**r["valuation"], **{k: v for k, v in r["revenue"].items()
                                                  if k in ("近3月平均YoY%", "最新MoM%")}}, name="值").astype(str))
    st.markdown("**最新 3 則新聞**")
    for n in r["news"]:
        st.markdown(f"**[{n['title']}]({n['link']})**  \n{n['source']} · {n['date']}")
        st.caption((n["summary"] or "（無法取得內文摘要，僅標題）") + f"　〔摘要來源：{n['summary_src']}〕")


def run_streamlit():
    import streamlit as st
    global GEMINI_MODEL

    st.set_page_config(page_title="台股決策儀表板", layout="wide")
    st.title("📈 台股自動化決策儀表板")

    # Secrets → 環境變數 / 設定（有設定才套用）
    for name in ("FINMIND_TOKEN",):
        if _secret(name) and not os.getenv(name):
            os.environ[name] = _secret(name)
    if _secret("GEMINI_MODEL"):
        GEMINI_MODEL = _secret("GEMINI_MODEL")

    # 選用：公開部署時用密碼擋住陌生人（否則他們會用掉你的 API 額度）
    app_pw = _secret("APP_PASSWORD")
    if app_pw and not st.session_state.get("authed"):
        pw = st.text_input("存取密碼", type="password")
        if pw and pw == app_pw:
            st.session_state["authed"] = True
            st.rerun()
        st.stop()

    backend_key = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    with st.sidebar:
        codes_txt = st.text_input("股票代碼（逗號分隔）", ",".join(DEFAULT_STOCKS),
                                  help="可輸入 2330、0050、00878、2330.TW 或中文名稱（常見股）；上市/上櫃後綴會自動判斷")
        if backend_key:
            key = backend_key
            st.success("🔑 已使用後台預設 API Key")
        else:
            key = st.text_input("Gemini API Key", type="password",
                                help="想免輸入：在 Streamlit Secrets 設定 GEMINI_API_KEY，這個欄位就會自動隱藏")
        go = st.button("🔄 開始 / 重新分析", type="primary", use_container_width=True)
        st.caption("買進需同時通過：預期盈虧比≥2:1、回測有優勢、模擬期望值>0。僅供研究，非投資建議。")

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

    tab1, tab2 = st.tabs(["📊 總覽", "🎯 戰績（預測準不準）"])
    with tab1:
        results = st.session_state.get("results")
        if not results:
            st.info("左側設定好股票代碼，按「開始分析」。")
        else:
            ok = [r for r in results if "error" not in r]
            reg = ok[0]["regime"] if ok else {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 買進", sum(r["rating_final"] == "買進" for r in ok))
            c2.metric("🟡 觀望", sum(r["rating_final"] == "觀望" for r in ok))
            c3.metric("🔴 賣出", sum(r["rating_final"] == "賣出" for r in ok))
            c4.metric("大盤", reg.get("趨勢", "—"), f"VIX {reg.get('VIX', '—')}", delta_color="off")
            st.caption(f"更新 {st.session_state['time']}｜資料為最近交易日收盤")
            st.dataframe(summary_table(results), use_container_width=True, hide_index=True)
            for r in results:
                if "error" in r:
                    st.error(f"{r['code']}：{r['error']}")
                    continue
                with st.expander(f"{r['code']} {r['name']}（{r['symbol']}）｜{r['rating_final']}｜{r.get('reason', '')}"):
                    _show_detail(st, r)
    with tab2:
        df, summ = quant.track_record()
        st.write("每次「買進」建議都會存檔，之後用真實股價結算。**模型預測勝率 vs 實際勝率**越接近，代表機率越可信。")
        st.table(pd.Series(summ, name="值").astype(str))
        if len(df):
            st.dataframe(df, use_container_width=True, hide_index=True)


def run_cli(codes):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("請先設定環境變數 GEMINI_API_KEY")
    results = run_all(codes, key, lambda p, c: print(f"完成 {c}", file=sys.stderr))
    pd.set_option("display.width", 240)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n台股決策儀表板  {dt.date.today()}\n")
    print(summary_table(results).to_string(index=False))
    for r in results:
        if "error" in r:
            continue
        print(f"\n[{r['code']} {r['name']} / {r['symbol']}] {r['rating_final']}：{r.get('reason', '')}")
        print(f"  第一批 {r['entry_high']} / 第二批 {r['entry_low']}｜停損 {r['stop_loss']}｜目標 {r['target1']} / {r['target2']}")
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

if _in_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    run_cli(sys.argv[1:] or DEFAULT_STOCKS)
