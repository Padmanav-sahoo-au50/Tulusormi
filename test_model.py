#!/usr/bin/env python3
"""
trade_ms_full.py
Full Live Sniper client + optional local FastAPI model server.
- Uses your Upstox ACCESS_TOKEN + INSTRUMENT_KEY for live LTP
- Loads local lgb_sniper_model.pkl if present (client-side prediction)
- Optionally starts a local FastAPI server with the same model (--start-local-server)
- Or can call a hosted model endpoint (--hosted-url)
"""

import argparse
import threading
import json
import requests
import time
from datetime import datetime, date
import pandas as pd
from collections import deque
import os
import csv
import numpy as np
from colorama import init, Fore, Style

# Optional server imports (fastapi, uvicorn, joblib, lightgbm) — used only if starting local server
try:
    from fastapi import FastAPI, HTTPException, Header
    from pydantic import BaseModel
    import uvicorn
    import joblib
    import lightgbm as lgb
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False

# ---------------- INIT ----------------
init(autoreset=True)

# ---------------- CONFIG (EDIT THESE) ----------------
# Put your Upstox access token and instrument key here (as you already had)
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiJESDIzNzEiLCJqdGkiOiI2OGJhNWNlNDZjZDI1MDM2ZWE2NDc0MzAiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzU3MDQzOTQwLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NTcxMDk2MDB9.xOWbr0HW6h25sDkkQ8PKVZjg_o0-HypHwr6akWA2_CY"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# Hosted model endpoint (if you deploy model to cloud)
# <-- Filled by integration as requested by you -->
HOSTED_MODEL_URL = "http://127.0.0.1:8000/predict"  # e.g. "https://my-hosted-model.example.com/predict"

# Local model file name (put lgb_sniper_model.pkl in same folder)
LOCAL_MODEL_PATH = "lgb_sniper_model.pkl"
START_LOCAL_SERVER_PORT = 8001

# Runtime params
INTERVAL_SEC = 1           # seconds between LTP fetches
AGG_SEC = 30               # aggregate ticks into 1-minute-like candles every AGG_SEC ticks
BUFFER_MIN = 400
MAX_SIGNALS_PER_DAY = 4
SIGNAL_LOG = "signals_log.csv"
MODEL_TIMEOUT = 3.0       # seconds to wait for hosted model response

# TP/SL multipliers (ATR-based)
TP1_ATR_MULT = 0.5
TP2_ATR_MULT = 1.0
SL_ATR_MULT  = 1.0

# Colors / emojis
GREEN = Fore.GREEN
R = Fore.RED
Y = Fore.YELLOW
C = Fore.CYAN
RESET = Style.RESET_ALL

# ---------------- NEW: PREDICTION SWITCH CONFIG ----------------
# Use remote hosted model by default (set False to prefer local model first)
USE_REMOTE = True
# If remote fails, fallback to local model if present
FALLBACK_TO_LOCAL = True

# ---------------- GLOBALS ----------------
prev_ltp = None
candles_buffer = deque(maxlen=BUFFER_MIN)
sec_ticks = []
min_candles = deque(maxlen=BUFFER_MIN)
signals_today = []
live_connected = False
market_closed_warning_printed = False
local_model = None   # if we load model locally for direct predictions

# Ensure signal log header
if not os.path.exists(SIGNAL_LOG):
    with open(SIGNAL_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date","time","side","entry","t1","t2","sl","reason","confidence","ai_confidence"])

# ---------------- HELPERS ----------------
def log_signal(side, entry, prob, t1, t2, sl, reason):
    dt = datetime.now().date().isoformat()
    tm = datetime.now().strftime("%H:%M:%S")
    row=[dt,tm,side,entry,t1,t2,sl,reason,prob]
    with open(SIGNAL_LOG,"a",newline="") as f:
        writer=csv.writer(f); writer.writerow(row)
    signals_today.append({"date":dt,"time":tm,"side":side,"entry":entry,"prob":prob})

def color_bar(score, bar_length=24):
    filled_length = int(bar_length * max(0,min(100,score)) / 100)
    empty_length = bar_length - filled_length
    color = GREEN if score >= 60 else Y if score >= 30 else R
    return color + "█" * filled_length + RESET + "-" * empty_length

# ---------------- UPSTOX LTP ----------------
def fetch_ltp():
    """
    Returns: (ltp, timestamp_str, arrow, score, meter)
    If API errors / market closed -> returns prev_ltp (may be None) and prints clear status message.
    """
    global prev_ltp, live_connected, market_closed_warning_printed
    now = datetime.now().time()
    market_start = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_end = datetime.strptime("15:30:00", "%H:%M:%S").time()
    market_open = market_start <= now <= market_end

    url = f"https://api.upstox.com/v3/market-quote/ltp?instrument_key={INSTRUMENT_KEY}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        j = r.json()
        # The Upstox response nests data by instrument key string (as in their API)
        d = j.get("data", {}).get(INSTRUMENT_KEY) or j.get("data", {}).get(INSTRUMENT_KEY.replace("|",":"))
        if d and d.get("last_price") is not None:
            current_ltp = float(d.get("last_price"))
            timestamp = datetime.now().strftime("%H:%M:%S")
            # connection status print once
            if not live_connected:
                if market_open:
                    print(f"{GREEN}🟢 Now connected to Live LTP (Market Open){RESET}")
                else:
                    print(f"{Y}⚠️ Connected to Upstox — Market Closed (data may be stale){RESET}")
                live_connected = True
            # compute arrow/delta
            if prev_ltp is None:
                arrow = "→"
                delta_pct = 0.0
            else:
                delta_pct = ((current_ltp - prev_ltp) / prev_ltp) * 100 if prev_ltp != 0 else 0.0
                arrow = "🟢↑" if delta_pct > 0 else ("🔴↓" if delta_pct < 0 else "→")
            prev_ltp = current_ltp
            score = min(max((delta_pct + 2) / (2*2) * 100, 0), 100)  # simple mapping
            meter = color_bar(score)
            market_closed_warning_printed = False
            return current_ltp, timestamp, arrow, score, meter
        else:
            # API worked but instrument data not present (market closed or mapping mismatch)
            if not market_closed_warning_printed:
                print(f"{Y}⚠️ Upstox returned no instrument data — Market Closed or wrong instrument key.{RESET}")
                market_closed_warning_printed = True
            return prev_ltp, datetime.now().strftime("%H:%M:%S"), "→", 0, color_bar(0)
    except Exception as e:
        # network error / auth / rate limit
        if not market_closed_warning_printed:
            print(f"{R}🔴 Upstox API error or Market Closed: {e}{RESET}")
            market_closed_warning_printed = True
        return prev_ltp, datetime.now().strftime("%H:%M:%S"), "→", 0, color_bar(0)

# ---------------- CANDLE BUILD ----------------
def append_tick(ltp, ts_str):
    sec_ticks.append({"ts": ts_str, "ltp": ltp})

def build_minute_candle_from_ticks(ticks):
    if not ticks: return None
    opens = [t["ltp"] for t in ticks]
    highs = max(opens)
    lows = min(opens)
    close = opens[-1]
    openp = opens[0]
    tick_volume = len(ticks)
    ts_min = datetime.now().strftime("%Y-%m-%d %H:%M")
    return {"timestamp": ts_min, "open": openp, "high": highs, "low": lows, "close": close, "volume": tick_volume}

# ---------------- INDICATORS ----------------
def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def indicator_vote(df_min):
    last = df_min.iloc[-1]
    votes_buy = 0
    votes_sell = 0
    reasons = []
    # EMA / SMA simple votes
    ema9 = df_min['close'].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = df_min['close'].ewm(span=21, adjust=False).mean().iloc[-1]
    sma5 = df_min['close'].rolling(5).mean().iloc[-1]
    sma20 = df_min['close'].rolling(20).mean().iloc[-1]
    if ema9 > ema21:
        votes_buy += 1; reasons.append('EMA9>EMA21')
    elif ema9 < ema21:
        votes_sell += 1; reasons.append('EMA9<EMA21')
    if sma5 > sma20:
        votes_buy += 1; reasons.append('SMA5>SMA20')
    elif sma5 < sma20:
        votes_sell += 1; reasons.append('SMA5<SMA20')
    # momentum candle
    if last['close'] > last['open']:
        votes_buy += 1; reasons.append('Bull candle')
    elif last['close'] < last['open']:
        votes_sell += 1; reasons.append('Bear candle')
    atr = compute_atr(df_min).iloc[-1]
    return votes_buy, votes_sell, ','.join(reasons), atr

# ---------------- AI/ML CLIENT ----------------
def call_hosted_model(hosted_url, features_dict, auth_token=None):
    try:
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        r = requests.post(hosted_url, json={"features": features_dict}, timeout=MODEL_TIMEOUT, headers=headers)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # minimal debug message, not too spammy
        print(f"{Y}⚠️ Model call failed: {e}{RESET}")
        return None

# ---------------- NEW: Unified prediction wrapper ----------------
def get_prediction_unified(features_dict, hosted_url_param=None, auth_token=None):
    """
    Unified prediction wrapper that respects USE_REMOTE and FALLBACK_TO_LOCAL flags.
    Returns: dict or None. Expected dict structure: {"signal": "...", "confidence": float}
    """
    global local_model
    # decide which URL to use
    chosen_url = hosted_url_param or HOSTED_MODEL_URL
    # Try remote first if USE_REMOTE is True and a URL exists
    if USE_REMOTE and chosen_url:
        try:
            res = call_hosted_model(chosen_url, features_dict, auth_token=auth_token)
            if isinstance(res, dict):
                return res
        except Exception:
            # call_hosted_model already prints minimal info
            res = None
        # If remote failed and fallback is allowed, try local
        if FALLBACK_TO_LOCAL:
            if local_model is None and os.path.exists(LOCAL_MODEL_PATH):
                try:
                    import joblib
                    local_model = joblib.load(LOCAL_MODEL_PATH)
                    print(f"{GREEN}✅ Local AI model loaded into client from {LOCAL_MODEL_PATH}{RESET}")
                except Exception as e:
                    print(f"{Y}⚠️ Failed to load local model during fallback: {e}{RESET}")
            if local_model is not None:
                try:
                    feat_order = ['ema9','ema21','sma5','sma20','rsi14','momentum','body','wick','tick_vol']
                    X = pd.DataFrame([[features_dict[k] for k in feat_order]], columns=feat_order)
                    pred = local_model.predict(X)[0]
                    proba = local_model.predict_proba(X)[0] if hasattr(local_model, "predict_proba") else None
                    map_lbl = {0:'SELL',1:'HOLD',2:'BUY'}
                    ai_signal = map_lbl.get(int(pred), str(pred))
                    ai_conf = float(max(proba)) if proba is not None else 0.0
                    return {"signal": ai_signal, "confidence": ai_conf}
                except Exception as e:
                    print(f"{Y}⚠️ Fallback local prediction failed: {e}{RESET}")
                    return None
        return None
    else:
        # USE_REMOTE is False -> try local model
        if local_model is None and os.path.exists(LOCAL_MODEL_PATH):
            try:
                import joblib
                local_model = joblib.load(LOCAL_MODEL_PATH)
                print(f"{GREEN}✅ Local AI model loaded into client from {LOCAL_MODEL_PATH}{RESET}")
            except Exception as e:
                print(f"{Y}⚠️ Failed to load local model: {e}{RESET}")
        if local_model is not None:
            try:
                feat_order = ['ema9','ema21','sma5','sma20','rsi14','momentum','body','wick','tick_vol']
                X = pd.DataFrame([[features_dict[k] for k in feat_order]], columns=feat_order)
                pred = local_model.predict(X)[0]
                proba = local_model.predict_proba(X)[0] if hasattr(local_model, "predict_proba") else None
                map_lbl = {0:'SELL',1:'HOLD',2:'BUY'}
                ai_signal = map_lbl.get(int(pred), str(pred))
                ai_conf = float(max(proba)) if proba is not None else 0.0
                return {"signal": ai_signal, "confidence": ai_conf}
            except Exception as e:
                print(f"{Y}⚠️ Local model prediction failed: {e}{RESET}")
                return None
        # no local model
        return None

# ---------------- LOCAL FASTAPI SERVER (optional) ----------------
if HAVE_FASTAPI:
    app = FastAPI()
    class PredictRequest(BaseModel):
        features: dict

    @app.post('/predict')
    def predict(req: PredictRequest, authorization: str | None = Header(default=None)):
        # optional token check if you want to secure local endpoint via MODEL_TOKEN env var
        expected_token = os.environ.get("MODEL_TOKEN", "")
        token = (authorization or "").replace("Bearer ", "")
        if expected_token and token != expected_token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise HTTPException(status_code=500, detail=f"Model not found: {LOCAL_MODEL_PATH}")
        mdl = joblib.load(LOCAL_MODEL_PATH)
        # model expects either 9-feature vector or dict -> handle
        feats = req.features
        # If dict mapping provided, convert to list in stable order if applicable
        if isinstance(feats, dict):
            # try common order - if model expects a fixed order on training, ensure you send same order.
            # We'll attempt to convert in a typical order if keys present.
            keys_order = ["ema9","ema21","sma5","sma20","rsi14","momentum","body","wick","tick_vol"]
            if all(k in feats for k in keys_order):
                X = pd.DataFrame([[feats[k] for k in keys_order]])
            else:
                # fallback: convert values order
                X = pd.DataFrame([list(feats.values())])
        else:
            X = pd.DataFrame([feats])
        try:
            preds = mdl.predict(X)
            proba = mdl.predict_proba(X)[0] if hasattr(mdl, "predict_proba") else None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")
        # mapping assumption - adjust to your training label encoding
        signal_map = {0:'SELL',1:'HOLD',2:'BUY'}
        pred_label = signal_map.get(int(preds[0]), str(preds[0]))
        confidence = float(max(proba)) if proba is not None else 0.0
        return {"signal": pred_label, "confidence": confidence}

def start_local_model_server(port=START_LOCAL_SERVER_PORT):
    if not HAVE_FASTAPI:
        print(f"{R}FastAPI/uvicorn not installed. Cannot start local server.{RESET}")
        return None
    def run():
        uvicorn.run(app, host='0.0.0.0', port=port)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"{C}➡️ Local FastAPI model server started on port {port}{RESET}")
    return t

# ---------------- SIGNAL / TARGET calc ----------------
def calc_targets_sl_from_atr(entry, side, atr_val):
    if atr_val is None or np.isnan(atr_val):
        t1 = entry + (10 if side=="BUY" else -10)
        t2 = entry + (20 if side=="BUY" else -20)
        sl = entry - (15 if side=="BUY" else -15)
    else:
        if side=="BUY":
            t1 = entry + TP1_ATR_MULT * atr_val
            t2 = entry + TP2_ATR_MULT * atr_val
            sl = entry - SL_ATR_MULT * atr_val
        else:
            t1 = entry - TP1_ATR_MULT * atr_val
            t2 = entry - TP2_ATR_MULT * atr_val
            sl = entry + SL_ATR_MULT * atr_val
    return round(t1,2), round(t2,2), round(sl,2)

# ---------------- SIGNAL / TARGET calc ----------------
def calc_targets_sl_from_atr(entry, side, atr_val):
    if atr_val is None or np.isnan(atr_val):
        t1 = entry + (10 if side=="BUY" else -10)
        t2 = entry + (20 if side=="BUY" else -20)
        sl = entry - (15 if side=="BUY" else -15)
    else:
        if side=="BUY":
            t1 = entry + TP1_ATR_MULT * atr_val
            t2 = entry + TP2_ATR_MULT * atr_val
            sl = entry - SL_ATR_MULT * atr_val
        else:
            t1 = entry - TP1_ATR_MULT * atr_val
            t2 = entry - TP2_ATR_MULT * atr_val
            sl = entry + SL_ATR_MULT * atr_val
    return round(t1,2), round(t2,2), round(sl,2)

# ---------------- DAILY SIGNAL TRACKER ----------------
class SignalTracker:
    def __init__(self, log_file="daily_signals.csv"):
        self.signals = []  # in-memory
        self.log_file = log_file
        # create file if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date','time','signal','entry','t1','t2','sl','result'])

    def log_signal(self, signal, entry, t1, t2, sl):
        """Log signal with empty result initially"""
        ts = datetime.now()
        record = {
            "date": ts.date(),
            "time": ts.time(),
            "signal": signal,
            "entry": entry,
            "t1": t1,
            "t2": t2,
            "sl": sl,
            "result": ""  # will fill later
        }
        self.signals.append(record)
        # append to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([record['date'], record['time'], signal, entry, t1, t2, sl, ""])

    def update_result(self, index, result):
        """Update hit/miss for a signal"""
        self.signals[index]['result'] = result
        # rewrite CSV fully for simplicity
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date','time','signal','entry','t1','t2','sl','result'])
            for r in self.signals:
                writer.writerow([r['date'], r['time'], r['signal'], r['entry'], r['t1'], r['t2'], r['sl'], r['result']])

def daily_report(self, current_ltp=None):
    """Compute daily summary and update any pending signal results based on current LTP"""
    today = date.today()

    # Update pending signals if current LTP given
    if current_ltp is not None:
        for i, s in enumerate(self.signals):
            result = s.get("result", "")  # safe check
            if result == "":
                if s["signal"] in ("BUY", "BREAKOUT"):   # handle both naming styles
                    if current_ltp >= s["t1"]:
                        self.update_result(i, "T1")
                    elif current_ltp >= s["t2"]:
                        self.update_result(i, "T2")
                    elif current_ltp <= s["sl"]:
                        self.update_result(i, "SL")
                elif s["signal"] in ("SELL", "BREAKDOWN"):  # handle both naming styles
                    if current_ltp <= s["t1"]:
                        self.update_result(i, "T1")
                    elif current_ltp <= s["t2"]:
                        self.update_result(i, "T2")
                    elif current_ltp >= s["sl"]:
                        self.update_result(i, "SL")

    # Filter today's signals
    today_signals = [s for s in self.signals if s.get("date") == today]
    total = len(today_signals)
    t1_hit = sum(1 for s in today_signals if s.get("result") == "T1")
    t2_hit = sum(1 for s in today_signals if s.get("result") == "T2")
    sl_hit = sum(1 for s in today_signals if s.get("result") == "SL")

    # Print report
    print(f"\n📊 Daily Signal Report for {today}")
    print(f"Total Signals: {total}")
    print(f"T1 Hit: {t1_hit}")
    print(f"T2 Hit: {t2_hit}")
    print(f"SL Hit: {sl_hit}")
    if total > 0:
        win_pct = ((t1_hit + t2_hit) / total) * 100
        print(f"Winning %: {win_pct:.2f}%")
    print("--------------------------------------------------")

# ---------------- MAIN LOOP ----------------
def main_loop(hosted_url=None, start_local_server=False, model_auth_token=None):
    tracker = SignalTracker()  # tracker instance
    print(f"{C}=== ✅ Sniper system started ==={RESET}")
    if start_local_server:
        start_local_model_server()
        hosted_url = f"http://127.0.0.1:{START_LOCAL_SERVER_PORT}/predict"
        print(f"{C}➡️ Using local model server at {hosted_url}{RESET}")
    elif hosted_url:
        print(f"{C}➡️ Using hosted model endpoint: {hosted_url}{RESET}")
    else:
        global local_model
        if os.path.exists(LOCAL_MODEL_PATH):
            try:
                import joblib
                local_model = joblib.load(LOCAL_MODEL_PATH)
                print(f"{GREEN}✅ Local AI model loaded into client from {LOCAL_MODEL_PATH}{RESET}")
            except Exception as e:
                print(f"{Y}⚠️ Failed to load local model: {e}{RESET}")
        else:
            print(f"{Y}⚠️ No hosted URL and no local model file found. AI predictions will be skipped.{RESET}")

    secs = 0
    last_print = 0

    while True:
        ltp, ts, arrow, score, meter = fetch_ltp()
        if ltp is None:
            time.sleep(INTERVAL_SEC)
            continue

        append_tick(ltp, ts)
        print(f"{C}{ts}{RESET} | LTP: {ltp} {arrow} | Score: [{meter}] {score:.1f}%")
        secs += 1

        if len(sec_ticks) >= AGG_SEC:
            candle = build_minute_candle_from_ticks(sec_ticks)
            if candle:
                min_candles.append(candle)
            sec_ticks.clear()

            if len(min_candles) >= 6:
                dfmin = pd.DataFrame(min_candles)
                votes_buy, votes_sell, reasons, atr_val = indicator_vote(dfmin)
                features = {
                    'ema9': dfmin['close'].ewm(span=9, adjust=False).mean().iloc[-1],
                    'ema21': dfmin['close'].ewm(span=21, adjust=False).mean().iloc[-1],
                    'sma5': dfmin['close'].rolling(5).mean().iloc[-1],
                    'sma20': dfmin['close'].rolling(20).mean().iloc[-1],
                    'rsi14': float(
                        100 - 100 / (
                            1 + (
                                (dfmin['close'].diff().clip(lower=0).ewm(span=14).mean().iloc[-1]) /
                                (-(dfmin['close'].diff().clip(upper=0).ewm(span=14).mean().iloc[-1]) + 1e-12)
                            )
                        )
                    ),
                    'momentum': dfmin['close'].diff().iloc[-1],
                    'body': dfmin['close'].iloc[-1] - dfmin['open'].iloc[-1],
                    'wick': dfmin['high'].iloc[-1] - dfmin['low'].iloc[-1],
                    'tick_vol': dfmin['volume'].iloc[-1]
                }

                ai_signal = None
                ai_conf = 0.0
                res = None
                try:
                    res = get_prediction_unified(features, hosted_url_param=hosted_url, auth_token=model_auth_token)
                except Exception as e:
                    print(f"{Y}⚠️ Unified get_prediction failed: {e}{RESET}")
                    res = None

                if res is None:
                    if hosted_url:
                        _res = call_hosted_model(hosted_url, features, auth_token=model_auth_token)
                        if isinstance(_res, dict):
                            res = _res
                    elif local_model is not None:
                        try:
                            feat_order = ['ema9','ema21','sma5','sma20','rsi14','momentum','body','wick','tick_vol']
                            X = pd.DataFrame([[features[k] for k in feat_order]], columns=feat_order)
                            pred = local_model.predict(X)[0]
                            proba = local_model.predict_proba(X)[0] if hasattr(local_model, "predict_proba") else None
                            map_lbl = {0:'SELL',1:'HOLD',2:'BUY'}
                            ai_signal = map_lbl.get(int(pred), str(pred))
                            ai_conf = float(max(proba)) if proba is not None else 0.0
                            res = {"signal": ai_signal, "confidence": ai_conf}
                        except Exception as e:
                            print(f"{Y}⚠️ Local model prediction failed: {e}{RESET}")
                            res = None

                if isinstance(res, dict):
                    ai_signal = res.get("signal")
                    ai_conf = float(res.get("confidence", 0.0))

                final = None
                reason = ''
                if votes_buy >= 2 and ai_signal == 'BUY' and ai_conf >= 0.80:
                    final = 'BUY'
                    reason = f"Indicators({votes_buy})+AI({ai_conf:.2f})|{reasons}"
                elif votes_sell >= 2 and ai_signal == 'SELL' and ai_conf >= 0.80:
                    final = 'SELL'
                    reason = f"Indicators({votes_sell})+AI({ai_conf:.2f})|{reasons}"
                else:
                    if votes_buy >= 3:
                        final = 'BUY'; reason = f"Indicators only({votes_buy})|{reasons}"
                    elif votes_sell >= 3:
                        final = 'SELL'; reason = f"Indicators only({votes_sell})|{reasons}"

                if final:
                    entry = dfmin['close'].iloc[-1]
                    t1, t2, sl = calc_targets_sl_from_atr(entry, final, atr_val)
                    emoji = '🟢' if final == 'BUY' else '🔴'
                    conf_str = f"{ai_conf:.2f}"
                    print(f"{GREEN if final=='BUY' else R}{emoji} {final} SIGNAL @ {entry} | T1: {t1} | T2: {t2} | SL: {sl} | AI_conf: {conf_str} | Reasons: {reason}{RESET}")
                    
                    # log signal in tracker
                    tracker.log_signal(final, entry, t1, t2, sl)

        time.sleep(INTERVAL_SEC)

        # Print daily report every 1000 seconds
        if secs % 1000 == 0 and secs != 0:
            tracker.daily_report()

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trade MS Full Live Hosted AI/ML Sniper')
    parser.add_argument('--hosted-url', type=str, default='', help='Hosted model endpoint URL (POST /predict)')
    parser.add_argument('--start-local-server', action='store_true', help='Start local FastAPI model server using local model file')
    parser.add_argument('--model-token', type=str, default='', help='Optional Authorization token for hosted/local model')
    args = parser.parse_args()

    # Use CLI hosted-url if provided, else fallback to top-level HOSTED_MODEL_URL
    url = args.hosted_url or HOSTED_MODEL_URL
    token = args.model_token or os.environ.get("MODEL_TOKEN", "")

    # Respect USE_REMOTE flag: if USE_REMOTE is False, ensure url is empty so main_loop will prefer local model
    if not USE_REMOTE:
        # disable remote usage even if HOSTED_MODEL_URL present
        url = ''

    # Print small startup summary so you see connection status at start
    print(f"{C}=== Starting Trade MS Sniper ==={RESET}")
    print(f"Upstox token present: {'YES' if ACCESS_TOKEN and ACCESS_TOKEN.strip() else 'NO'}")
    print(f"Instrument key: {INSTRUMENT_KEY}")
    print(f"Local model present: {'YES' if os.path.exists(LOCAL_MODEL_PATH) else 'NO'}")
    print(f"Hosted model URL configured: {'YES' if url else 'NO'}")
    print(f"USE_REMOTE flag: {'ENABLED' if USE_REMOTE else 'DISABLED'}")
    print(f"FALLBACK_TO_LOCAL flag: {'ENABLED' if FALLBACK_TO_LOCAL else 'DISABLED'}")
    print("Run mode:", "Start local server" if args.start_local_server else ("Use hosted URL" if url else "Client-local model (if present)"))
    print("---------------------------------------------------------------")

    # Start main loop with chosen URL and options
    try:
        main_loop(hosted_url=url, start_local_server=args.start_local_server, model_auth_token=token)
    except KeyboardInterrupt:
        print(f"{C}\n=== Exiting (KeyboardInterrupt) ==={RESET}")