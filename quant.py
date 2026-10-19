MAX_STOP_PCT = 0.08  # 設為 0.08 代表最大停損 8%（可依需求調整）
"""
量化引擎（不依賴網路與 LLM，可單獨測試）
1. add_indicators  技術指標
2. backtest        固定規則的歷史回測（含手續費+證交稅、次日開盤進場、同日停損優先）
3. simulate_plan   依「這一檔現在的波動」做蒙地卡羅，算出 先達標 vs 先停損 的機率與期望值
4. kelly_fraction  凱利部位
5. 戰績追蹤        每次「買進」建議寫入 SQLite，之後用真實股價結算，驗證預測準不準
"""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

COST_RT = 0.00585        # 來回成本：手續費 0.1425%×2 + 證交稅 0.3%（未計券商折讓，偏保守）
HOLD_DAYS = 20           # 進場後最長持有交易日
ENTRY_WAIT = 10          # 等待進場區間的最長交易日
DB_PATH = Path(__file__).with_name("predictions.db")


# ------------------------------------------------------------------ 基礎
def round_tick(p: float) -> float:
    """依台股升降單位取整。"""
    tick = (0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100
            else 0.5 if p < 500 else 1 if p < 1000 else 5)
    return round(round(p / tick) * tick, 2)


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


def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    d = hist[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    c = d["Close"]
    for n in (5, 20, 60, 120):
        d[f"ma{n}"] = c.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(),
                    (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    return d


MAX_STOP_PCT = 8.0       # 停損離建倉點最大允許 %（超過 → 觀望）
MAX_STOP_ATR = 4.0       # 停損離建倉點最大允許 ATR 倍數


def propose_levels(d: pd.DataFrame) -> dict:
    """
    機械式基準價位（AI 價位不合邏輯時的備用，也是給 AI 的參考）：
      建倉點：回測 5 日線（現價已在 5 日線下就用現價）
      加碼點：月線(MA20)離建倉點夠遠 → 「回測月線」；否則 → 「突破壓力」（20 日高點上方）
      停損：近 10 日低點 / 建倉−1ATR 取低再減 0.5ATR，並限制在 8% 與 4ATR 之內
      目標：2.5R / 4R
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
    out = {k: round_tick(v) for k, v in {
        "entry_price": entry, "add_price": add, "stop_loss": stop,
        "target1": entry + 2.5 * risk, "target2": entry + 4 * risk}.items()}
    out["add_type"] = add_type
    out["entry_basis"] = "回測5日線" if entry < price else "現價附近"
    return out


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


def chips_features(inst, margin, vol5_lots=None, ret5_pct=None) -> dict:
    """
    把三大法人與融資資料整理成可直接判讀的變數，並依固定規則給出「籌碼訊號」。
    inst:   index=日期、欄位 外資/投信/自營商（單位：張，買超為正）
    margin: index=日期、欄位 融資餘額/融券餘額（單位：張）
    """
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


def backtest(d: pd.DataFrame) -> dict:
    """
    策略（固定規則，無任何擬合參數）：多頭拉回
      訊號：MA20>MA60、收盤>MA60、MA60 向上、收盤在 MA20 −3%~+2%、RSI 35~60
      進場：隔日開盤；停損：近10日低點/進場−1ATR 取低，再減 0.5ATR；目標：2.5R；最長持有 20 日
    """
    o, h, l, c = (d[k].values for k in ("Open", "High", "Low", "Close"))
    ma20, ma60, atr, rsi = (d[k].values for k in ("ma20", "ma60", "atr", "rsi"))
    n = len(d)
    R, idx = [], []
    i = 125
    while i < n - 2:
        ok = (ma20[i] > ma60[i] and c[i] > ma60[i] and ma60[i] > ma60[i - 10]
              and 0.97 * ma20[i] <= c[i] <= 1.02 * ma20[i] and 35 <= rsi[i] <= 60)
        if not ok:
            i += 1
            continue
        entry = o[i + 1]
        stop = min(l[i - 9:i + 1].min(), entry - atr[i]) - 0.5 * atr[i]
        risk = entry - stop
        if risk <= 0 or risk / entry > 0.10:
            i += 1
            continue
        target = entry + 2.5 * risk
        last = min(i + HOLD_DAYS, n - 1)
        exit_px, j = c[last], last
        for j in range(i + 1, last + 1):
            if j > i + 1 and o[j] <= stop:
                exit_px = o[j]
                break
            if l[j] <= stop:
                exit_px = stop
                break
            if h[j] >= target:
                exit_px = target
                break
        R.append((exit_px - entry) / risk - COST_RT * entry / risk)
        idx.append(i)
        i = j + 1
    R, idx = np.array(R), np.array(idx)
    recent = R[idx >= n - 252] if len(R) else R
    allst, recst = _stats(R), _stats(recent)
    edge = (allst["交易次數"] >= 12 and allst["期望值R"] > 0.1 and allst["獲利因子"] > 1.2
            and (recst["交易次數"] < 3 or recst["期望值R"] > 0))
    return {"全期間": allst, "近1年": recst, "edge_ok": bool(edge)}


# ------------------------------------------------------------------ 機率模擬
def simulate_plan(close: pd.Series, price, entry_low, entry_high, target, stop,
                  n=4000, seed=7, drift_shrink=0.5):
    """
    濾波歷史模擬（FHS）：用近 3 年標準化殘差重抽樣，乘上「目前」的 EWMA 波動。
    漂移只採用歷史平均的 drift_shrink 倍（預設 50%），避免把過去的大漲外推到未來。
    價格僅用收盤價路徑判斷，會略低估觸及機率。回傳 None 代表價位不合邏輯。
    """
    mid = (entry_low + entry_high) / 2
    risk = mid - stop
    if not (stop < entry_low <= entry_high and target > entry_high and risk > 0):
        return None
    r = np.log(close.astype(float)).diff().dropna().values[-750:]
    lam = 0.94
    var = np.empty_like(r)
    var[0] = r[:20].var()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    z = r / np.sqrt(var)
    z = (z - z.mean()) / z.std()
    sig_now = np.sqrt(lam * var[-1] + (1 - lam) * r[-1] ** 2)
    mu = r.mean() * drift_shrink

    T = ENTRY_WAIT + HOLD_DAYS + 1
    rng = np.random.default_rng(seed)
    steps = mu + sig_now * rng.choice(z, size=(n, T))
    P = np.hstack([np.full((n, 1), float(price)), price * np.exp(np.cumsum(steps, axis=1))])

    cost_r = COST_RT * mid / risk
    r_win = (target - mid) / risk - cost_r
    r_loss = -1 - cost_r
    BIG = 10 ** 9
    filled = wins = losses = times = 0
    rs, days = [], []
    for p in P:
        hit = np.flatnonzero(p[:ENTRY_WAIT + 1] <= entry_high)
        if hit.size == 0:
            continue
        f = int(hit[0])
        filled += 1
        if p[f] <= stop:
            losses += 1
            rs.append(r_loss)
            continue
        seg = p[f + 1:f + 1 + HOLD_DAYS]
        s_hit, t_hit = np.flatnonzero(seg <= stop), np.flatnonzero(seg >= target)
        s = int(s_hit[0]) if s_hit.size else BIG
        t = int(t_hit[0]) if t_hit.size else BIG
        if s == BIG and t == BIG:
            rs.append((seg[-1] - mid) / risk - cost_r)
            times += 1
        elif s <= t:
            rs.append(r_loss)
            losses += 1
            days.append(s + 1)
        else:
            rs.append(r_win)
            wins += 1
            days.append(t + 1)
    if filled == 0:
        return {"成交機率%": 0.0, "先達標機率%": 0.0, "先停損機率%": 0.0, "期滿未決機率%": 0.0,
                "期望值R": 0.0, "損益兩平勝率%": 0.0, "p_win": 0.0, "p_loss": 0.0,
                "breakeven": 1.0, "ev_r": 0.0, "r_win": r_win, "r_loss": r_loss}
    breakeven = -r_loss / (r_win - r_loss)
    return {
        "成交機率%": round(filled / n * 100, 1),
        "先達標機率%": round(wins / filled * 100, 1),
        "先停損機率%": round(losses / filled * 100, 1),
        "期滿未決機率%": round(times / filled * 100, 1),
        "期望值R": round(float(np.mean(rs)), 2),
        "損益兩平勝率%": round(breakeven * 100, 1),
        "中位數達標天數": int(np.median([x for x in days])) if days else None,
        # 以下給程式判斷用
        "p_win": wins / filled, "p_loss": losses / filled, "breakeven": breakeven,
        "ev_r": float(np.mean(rs)), "r_win": r_win, "r_loss": r_loss,
    }


def kelly_fraction(p_win, p_loss, r_win, r_loss) -> float:
    decided = p_win + p_loss
    if decided <= 0:
        return 0.0
    p, b = p_win / decided, r_win / abs(r_loss)
    return max(p - (1 - p) / b, 0.0)


# ------------------------------------------------------------------ 戰績追蹤
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS preds(
        date TEXT, code TEXT, name TEXT, price REAL, entry_low REAL, entry_high REAL,
        target1 REAL, stop_loss REAL, rr REAL, p_win REAL, ev_r REAL,
        status TEXT DEFAULT 'open', resolved_date TEXT, outcome_r REAL,
        PRIMARY KEY(date, code))""")
    return con


def log_prediction(date: str, r: dict):
    """只記錄「買進」建議（同日同檔只記一次）。"""
    if r.get("rating_final") != "買進" or not r.get("sim"):
        return
    with _conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO preds(date,code,name,price,entry_low,entry_high,target1,"
            "stop_loss,rr,p_win,ev_r) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (date, r["code"], r["name"], r["price"], r["entry_low"], r["entry_high"],
             r["target1"], r["stop_loss"], r["rr"], r["sim"]["p_win"], r["sim"]["ev_r"]))


def resolve_plan(h: pd.DataFrame, lo, hi, t1, sl):
    """用建議日之後的真實日線結算。回傳 (status, R, 結算日) 或 None（仍在進行）。"""
    mid = (lo + hi) / 2
    fill, f = None, None
    for k, (ts, row) in enumerate(h.iterrows()):
        day = ts.strftime("%Y-%m-%d")
        if fill is None:
            if k >= ENTRY_WAIT:
                return "unfilled", 0.0, day
            if row["Low"] <= hi:
                fill = mid if row["Low"] <= mid else hi   # 保守：以較差價成交
                f = k
                if row["Low"] <= sl:
                    return "loss", -1 - COST_RT * fill / max(fill - sl, 1e-9), day
            continue
        risk = fill - sl
        cost_r = COST_RT * fill / risk
        if row["Low"] <= sl:                                # 同日停損優先
            return "loss", -1 - cost_r, day
        if row["High"] >= t1:
            return "win", (t1 - fill) / risk - cost_r, day
        if k - f >= HOLD_DAYS:
            return "timeout", (row["Close"] - fill) / risk - cost_r, day
    return None


def evaluate_open(fetch_hist) -> int:
    """結算所有進行中的預測。fetch_hist(code) -> 含 Open/High/Low/Close 的日線。回傳結算筆數。"""
    done = 0
    with _conn() as con:
        rows = con.execute("SELECT date,code,entry_low,entry_high,target1,stop_loss "
                           "FROM preds WHERE status='open'").fetchall()
        for date, code, lo, hi, t1, sl in rows:
            try:
                h = fetch_hist(code)
                h = h[h.index.strftime("%Y-%m-%d") > date]
            except Exception:
                continue
            res = resolve_plan(h, lo, hi, t1, sl)
            if res:
                con.execute("UPDATE preds SET status=?, outcome_r=?, resolved_date=? "
                            "WHERE date=? AND code=?", (res[0], res[1], res[2], date, code))
                done += 1
    return done


def track_record():
    """回傳 (全部紀錄 DataFrame, 摘要 dict)。"""
    with _conn() as con:
        df = pd.read_sql_query("SELECT * FROM preds ORDER BY date DESC", con)
    closed = df[df["status"].isin(["win", "loss", "timeout"])]
    summ = {"累計建議": len(df), "進行中": int((df["status"] == "open").sum()),
            "未成交": int((df["status"] == "unfilled").sum()), "已結算": len(closed)}
    if len(closed):
        summ.update({
            "實際勝率%": round(float((closed["status"] == "win").mean() * 100), 1),
            "模型預測勝率%": round(float(closed["p_win"].mean() * 100), 1),
            "平均R": round(float(closed["outcome_r"].mean()), 2),
            "模型預測期望值R": round(float(closed["ev_r"].mean()), 2),
        })
    return df, summ
