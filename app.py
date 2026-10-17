"""
台股自動化決策儀表板 v2
-----------------------
資料：yfinance（價量、加權指數、VIX）、FinMind（三大法人、融資券、本益比河流、月營收、毛利率）、Google News
判斷：Gemini（五大面向）＋ quant.py（回測、機率模擬、凱利部位、戰績追蹤）
關卡：LLM 說「買進」還要同時通過 ① 盈虧比≥2 ② 回測有優勢 ③ 模擬期望值>0，否則自動降為「觀望」

執行：
    streamlit run app.py                 # 網頁
    python app.py 2330 2317 2454         # 終端機
環境變數：GEMINI_API_KEY（必填）、FINMIND_TOKEN（選填）、GEMINI_MODEL（選填）
"""
import os
import sys
import json
import datetime as dt
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests
import pandas as pd
import yfinance as yf

import quant

# ------------------------------------------------------------------ 設定
DEFAULT_STOCKS = ["2330", "2317", "2454"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
FALLBACK_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]   # gemini-1.5 已下架時自動改用
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
MIN_RR = 2.0

NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2412": "中華電",
    "2308": "台達電", "2303": "聯電", "3711": "日月光投控", "2881": "富邦金",
    "2882": "國泰金", "2891": "中信金", "2382": "廣達", "2603": "長榮",
}

SYSTEM_PROMPT = """你是精通台股交易的資深量化分析師與風險控管專家，目標是「賺多賠少、盈虧比高」。
從五大面向分析：基本面（營收/毛利率/現金流/負債）、估值面（本益比、股價淨值比的歷史百分位與殖利率）、
產業競爭力、技術與籌碼面（均線、三大法人、融資、支撐壓力）、總經與消息面（大盤趨勢、VIX、新聞）。

硬性規則：
- 盈虧比 = (目標價1 − 買進中位價) / (買進中位價 − 停損價) 必須 ≥ 2，否則 rating 必須是「觀望」。
- 資料中「回測統計」若顯示無統計優勢（edge_ok=false），或訊息/籌碼/技術嚴重矛盾，rating 必須是「觀望」。
- 價位必須以提供的現價、均線、支撐壓力、ATR 為依據；可參考「基準價位」並微調，不得憑空捏造。
- 分批進場；停損必須是明確技術位。資料中沒有的數字不要編造，可說「資料不足」。

只輸出 JSON（不要 markdown）：
{"rating":"買進"|"觀望"|"賣出","risk_level":"高"|"中"|"低",
 "entry_low":數字,"entry_high":數字,"target1":數字,"target2":數字,"stop_loss":數字,
 "position_pct":建議佔總資金百分比(數字，觀望/賣出填0),"batches":"分批說明一句話",
 "fundamentals_valuation":"一句話","industry":"一句話","chips_technical":"一句話","news_macro":"一句話",
 "reason":"評級理由一句話","risks":["風險1","風險2"]}
rating 為「觀望」「賣出」時，價位欄位仍請填「若要進場/等待」的參考價位。"""


# ------------------------------------------------------------------ 資料抓取
def fetch_price(code: str):
    for suffix in (".TW", ".TWO"):
        t = yf.Ticker(code + suffix)
        hist = t.history(period="3y", auto_adjust=False)
        if not hist.empty:
            return t, hist.dropna(subset=["Close"])
    raise ValueError(f"找不到 {code} 的股價資料")


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


def _finmind(dataset: str, code: str, days: int = 30) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": code,
              "start_date": (dt.date.today() - dt.timedelta(days=days)).isoformat()}
    if os.getenv("FINMIND_TOKEN"):
        params["token"] = os.getenv("FINMIND_TOKEN")
    r = requests.get(FINMIND_URL, params=params, timeout=25)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("data", []))


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
    """FinMind 月營收：近 6 月 YoY、最新 MoM。"""
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
    """FinMind 財報：近 6 季毛利率、營益率。"""
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
    """大盤環境：加權指數 vs MA60/MA120 判多空，加上 VIX。"""
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


def fetch_news(query: str, n: int = 6) -> list:
    url = (f"https://news.google.com/rss/search?q={quote(query)}"
           "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        return [{"title": it.findtext("title", ""), "date": it.findtext("pubDate", ""),
                 "link": it.findtext("link", "")} for it in list(root.iter("item"))[:n]]
    except Exception:
        return []


# ------------------------------------------------------------------ Gemini
def call_gemini(api_key: str, prompt: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    last = None
    for m in [GEMINI_MODEL] + [x for x in FALLBACK_MODELS if x != GEMINI_MODEL]:
        try:
            resp = client.models.generate_content(
                model=m, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json", temperature=0.2))
            data = json.loads(resp.text)
            data["_model"] = m
            return data
        except Exception as e:
            last = e
    raise RuntimeError(f"Gemini 呼叫失敗：{last}")


# ------------------------------------------------------------------ 風控關卡
LEVEL_KEYS = ("entry_low", "entry_high", "target1", "target2", "stop_loss")


def enforce_rules(r: dict) -> dict:
    """關卡 1：價位邏輯與盈虧比 ≥ 2:1。"""
    flags, rr, valid = [], None, False
    try:
        for k in LEVEL_KEYS:
            r[k] = quant.round_tick(float(r[k]))
        mid = (r["entry_low"] + r["entry_high"]) / 2
        if mid - r["stop_loss"] > 0 and r["target1"] - mid > 0:
            rr = (r["target1"] - mid) / (mid - r["stop_loss"])
        valid = r["entry_low"] <= r["entry_high"] and r["stop_loss"] < r["entry_low"] \
            and r["target1"] > r["entry_high"]
    except (KeyError, TypeError, ValueError):
        pass
    rating = r.get("rating", "觀望")
    if rating == "買進":
        if not valid:
            rating = "觀望"
            flags.append("AI 價位邏輯不一致，已降為觀望")
        elif rr is None or rr < MIN_RR:
            rating = "觀望"
            flags.append(f"盈虧比 {rr:.2f}:1 < 2:1，已降為觀望" if rr else "盈虧比不合格，已降為觀望")
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
def analyze_stock(code: str, api_key: str, macro_news: list, regime: dict) -> dict:
    ticker, hist = fetch_price(code)
    d = quant.add_indicators(hist)
    tech = compute_technicals(d)
    price = tech["收盤價"]
    try:
        info = ticker.info
        en_name = info.get("shortName")
    except Exception:
        en_name = None
    name = NAMES.get(code) or en_name or code

    bt = quant.backtest(d)
    base = quant.propose_levels(d)
    news = fetch_news(f"{code} {name} when:7d")
    payload = {
        "股票": f"{code} {name}", "今日日期": dt.date.today().isoformat(),
        "技術面": tech, "估值(FinMind)": fetch_valuation(code),
        "月營收": fetch_revenue(code), "毛利率趨勢": fetch_margins_trend(code),
        "三大法人": fetch_institutional(code), "融資券": fetch_margin(code),
        "大盤環境": regime, "回測統計": {**bt, "說明": "固定規則多頭拉回策略，R=風險單位，已含交易成本"},
        "基準價位": base,
        "個股新聞標題": [n["title"] for n in news],
        "總經新聞標題": [n["title"] for n in macro_news],
    }
    prompt = "請分析以下台股資料並輸出 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    r = enforce_rules(call_gemini(api_key, prompt))
    sim = (quant.simulate_plan(hist["Close"], price, r["entry_low"], r["entry_high"],
                               r["target1"], r["stop_loss"]) if r["levels_ok"] else None)
    r = quant_gate(r, sim, bt, regime)

    chart = d[["Close", "ma20", "ma60"]].tail(120).rename(columns={"Close": "收盤", "ma20": "MA20", "ma60": "MA60"})
    if r["levels_ok"]:
        chart["停損"], chart["買進上限"], chart["目標1"] = r["stop_loss"], r["entry_high"], r["target1"]
    chart.index = chart.index.strftime("%m-%d")
    r.update({"code": code, "name": name, "price": price, "change": tech["漲跌幅%"],
              "news": news, "tech": tech, "bt": bt, "chart": chart, "date": tech["最近交易日"],
              "valuation": payload["估值(FinMind)"], "revenue": payload["月營收"]})
    return r


def run_all(codes, api_key, progress=None) -> list:
    try:   # 先結算過去的預測，戰績才會是最新的
        quant.evaluate_open(lambda c: quant.add_indicators(fetch_price(c)[1]))
    except Exception:
        pass
    regime = fetch_regime()
    macro = fetch_news("台股 外資 聯準會 總體經濟 when:3d", 6)
    out = []
    for i, code in enumerate(codes):
        try:
            r = analyze_stock(code, api_key, macro, regime)
            quant.log_prediction(r["date"], r)
            r["regime"] = regime
            out.append(r)
        except Exception as e:
            out.append({"code": code, "error": str(e)})
        if progress:
            progress((i + 1) / len(codes), code)
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
            "建議買進價位": f"{f('entry_low')} – {f('entry_high')}",
            "目標價": f"{f('target1')} / {f('target2')}",
            "強制停損價": f("stop_loss"),
            "盈虧比": f"{r['rr']:.1f}:1" if r["rr"] else "—",
            "達標機率(模擬)": f"{s['先達標機率%']}%" if s else "—",
            "期望值(R)": f"{s['期望值R']:+.2f}" if s else "—",
            "回測勝率/期望值": f"{bt.get('勝率%', '—')}% / {bt.get('期望值R', '—')}R",
            "建議部位%": r["position_final"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 介面
def _show_detail(st, r):
    for fl in r["flags"]:
        st.warning(fl)
    st.markdown(
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
    for n in r["news"]:
        st.markdown(f"- [{n['title']}]({n['link']})")


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="台股決策儀表板", layout="wide")
    st.title("📈 台股自動化決策儀表板")
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_key = ""
    with st.sidebar:
        codes_txt = st.text_input("股票代碼（逗號分隔）", ",".join(DEFAULT_STOCKS))
        key = st.text_input("Gemini API Key", os.getenv("GEMINI_API_KEY", default_key), type="password")
        go = st.button("🔄 開始 / 重新分析", type="primary", use_container_width=True)
        st.caption("買進需同時通過：盈虧比≥2:1、回測有優勢、模擬期望值>0。僅供研究，非投資建議。")

    if go:
        if not key:
            st.error("請先輸入 Gemini API Key")
            st.stop()
        codes = [c.strip() for c in codes_txt.replace("，", ",").split(",") if c.strip()]
        bar = st.progress(0.0, text="分析中…")
        st.session_state["results"] = run_all(
            codes, key, lambda p, c: bar.progress(p, text=f"已完成 {c}"))
        st.session_state["time"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        bar.empty()

    tab1, tab2 = st.tabs(["📊 總覽", "🎯 戰績（預測準不準）"])
    with tab1:
        results = st.session_state.get("results")
        if not results:
            st.info("左側設定好股票與 API Key，按「開始分析」。")
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
                with st.expander(f"{r['code']} {r['name']}｜{r['rating_final']}｜{r.get('reason', '')}"):
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
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n台股決策儀表板  {dt.date.today()}\n")
    print(summary_table(results).to_string(index=False))
    for r in results:
        if "error" in r:
            continue
        print(f"\n[{r['code']} {r['name']}] {r['rating_final']}：{r.get('reason', '')}")
        for fl in r["flags"]:
            print("  ⚠", fl)
        print("  風險：" + "；".join(r.get("risks", [])))
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
