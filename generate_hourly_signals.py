"""
Generate hourly trading signal report for ETH-USD,
14 May 2026 00:00 UTC s/d 17 May 2026 23:00 UTC.

Output: ETH_signals_14-17May2026.xlsx
Kolom : Datetime (UTC), Raw Posisi, Posisi Final, Score, Last TR
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf


# ============================
# PARAMETER (samakan dgn Analisa.ipynb)
# ============================
MIN_SCORE_FOR_TRADE = 1
TP_ATR_MULT = 1.4
SL_ATR_MULT = 3.5
FALLBACK_TP_PCT = 0.005
FALLBACK_SL_PCT = 0.009
ADX_TREND_THRESHOLD = 20.0
ADX_NO_TRADE_THRESHOLD = 15.0
EXTREME_TR_MULT = 1.6


# ============================
# Indikator
# ============================
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.rolling(window=signal_window).mean()
    return macd, signal


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_adx(high, low, close, period=14):
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=period, min_periods=period).mean()
    plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window=period, min_periods=period).mean()


def compute_signal_from_indicators(df):
    open_ = df["Open"].astype(float)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    ema_fast = compute_ema(close, span=21)
    ema_slow = compute_ema(close, span=50)
    macd, signal = compute_macd(close)
    rsi = compute_rsi(close, period=14)
    atr = compute_atr(high, low, close, period=14)
    adx = compute_adx(high, low, close, period=14)

    last_close = float(close.iloc[-1])
    ema_fast_last = float(ema_fast.iloc[-1])
    ema_slow_last = float(ema_slow.iloc[-1])
    macd_last = float(macd.dropna().iloc[-1]) if macd.dropna().size > 0 else np.nan
    signal_last = float(signal.dropna().iloc[-1]) if signal.dropna().size > 0 else np.nan
    rsi_last = float(rsi.dropna().iloc[-1]) if rsi.dropna().size > 0 else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if atr.dropna().size > 0 else np.nan
    adx_last = float(adx.dropna().iloc[-1]) if adx.dropna().size > 0 else np.nan

    score = 0
    if last_close > ema_fast_last: score += 1
    else: score -= 1
    if ema_fast_last > ema_slow_last: score += 1
    else: score -= 1
    if not np.isnan(macd_last) and not np.isnan(signal_last):
        if macd_last > signal_last: score += 1
        else: score -= 1
    if not np.isnan(macd_last):
        if macd_last > 0: score += 1
        else: score -= 1
    if not np.isnan(rsi_last):
        if rsi_last > 55: score += 1
        elif rsi_last < 45: score -= 1
    if not np.isnan(adx_last) and adx_last >= ADX_TREND_THRESHOLD:
        if ema_fast_last > ema_slow_last: score += 1
        elif ema_fast_last < ema_slow_last: score -= 1

    if score >= MIN_SCORE_FOR_TRADE:
        raw_position = "LONG"
    elif score <= -MIN_SCORE_FOR_TRADE:
        raw_position = "SHORT"
    else:
        raw_position = "NO TRADE"

    position_after_filters = raw_position
    if not np.isnan(adx_last) and adx_last < ADX_NO_TRADE_THRESHOLD:
        position_after_filters = "NO TRADE"

    prev_close = float(close.iloc[-2]) if len(close) > 1 else float(last_close)
    tr1 = float(abs(high.iloc[-1] - low.iloc[-1]))
    tr2 = float(abs(high.iloc[-1] - prev_close))
    tr3 = float(abs(low.iloc[-1] - prev_close))
    last_tr = max(tr1, tr2, tr3)

    if not np.isnan(atr_last) and atr_last > 0 and last_tr > EXTREME_TR_MULT * atr_last:
        position_after_filters = "NO TRADE"

    return {
        "score": score,
        "raw_position": raw_position,
        "position": position_after_filters,
        "last_tr": float(last_tr),
    }


# ============================
# MAIN
# ============================
def main():
    ticker = "ETH-USD"
    interval = "1h"

    # Range target
    range_start = datetime(2026, 5, 14, 0, 0, tzinfo=timezone.utc)
    range_end   = datetime(2026, 5, 17, 23, 0, tzinfo=timezone.utc)

    # Ambil data: 90 hari sebelum range_start sampai range_end + 1h supaya candle 17 Mei 23:00 ikut
    fetch_start = range_start - timedelta(days=90)
    fetch_end   = range_end + timedelta(hours=2)

    print(f"Download {ticker} {interval} dari {fetch_start} sampai {fetch_end} ...")
    data = yf.download(
        ticker,
        start=fetch_start,
        end=fetch_end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty:
        raise RuntimeError("Data dari yfinance kosong. Coba cek koneksi / ticker / tanggal.")

    # Pastikan tz UTC
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data = data.tz_convert("UTC")

    data = data.dropna()
    print(f"Total bar terdownload: {len(data)}  (range: {data.index[0]}  -> {data.index[-1]})")

    # Generate timestamp target per jam
    target_hours = pd.date_range(start=range_start, end=range_end, freq="1H", tz="UTC")

    rows = []
    for ts in target_hours:
        # Cari bar dengan timestamp == ts (yfinance index = waktu open bar)
        if ts not in data.index:
            # bar tsb tidak tersedia (data hilang) -> skip dgn placeholder
            rows.append({
                "Datetime (UTC)": ts.strftime("%Y-%m-%d %H:%M"),
                "Raw Posisi": "N/A (no data)",
                "Posisi Final": "N/A (no data)",
                "Score": None,
                "Last TR": None,
            })
            continue

        # Slice data sampai bar ts inklusif -> sama dgn behavior interactive_plot
        window_df = data.loc[:ts]
        if len(window_df) < 60:
            rows.append({
                "Datetime (UTC)": ts.strftime("%Y-%m-%d %H:%M"),
                "Raw Posisi": "N/A (data <60 bar)",
                "Posisi Final": "N/A (data <60 bar)",
                "Score": None,
                "Last TR": None,
            })
            continue

        res = compute_signal_from_indicators(window_df)
        rows.append({
            "Datetime (UTC)": ts.strftime("%Y-%m-%d %H:%M"),
            "Raw Posisi": res["raw_position"],
            "Posisi Final": res["position"],
            "Score": res["score"],
            "Last TR": round(res["last_tr"], 1),
        })

    out_df = pd.DataFrame(rows)

    out_path = "ETH_signals_14-17May2026.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Hourly Signals")
        ws = writer.sheets["Hourly Signals"]
        # auto-width sederhana
        for col_cells in ws.columns:
            max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = max_len + 2

    print(f"\n[OK] Selesai. File: {out_path}  ({len(out_df)} baris)")
    print(out_df.head(10).to_string(index=False))
    print("...")
    print(out_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
