"""
日次データ取得スクリプト（GitHub Actions想定）

対象：日経225現物（^N225）、1579.T、1357.T
足種：日足・5分足・1分足

CSV仕様：
- 1行目：銘柄コード（例：^N225）
- 2行目：ヘッダー（日足は Date,Open,High,Low,Close／分足は Date,Time,Open,High,Low,Close）
- 3行目以降：データ

設計方針：
- 毎回「取得可能な最大窓」を取りに行き、既存CSVとマージして重複排除する。
- 1分足は1リクエスト7日制限があるため、7日ずつチャンクに分けて
  過去29日分を取得・結合する（境界は必ず30日以内に収める）。
- yfinanceはGitHub Actionsの共有IPからレートリミットを受けることがあるため、
  各取得にリトライ（指数バックオフ）を入れる。

欠損補完（8/23追加、8/23改訂）：
- Yahoo Financeは特定の足（寄り09:00、引け際15:25/15:30等）を配信しないことがある。
- 基本ルール：セッション内（前場09:00-11:30、後場12:30-last_slot）で欠けている
  時刻は、直前の実データの終値でフラットOHLC（O=H=L=C）を作って埋める。
  日の先頭が欠損の場合は、その日最初の実データの始値で埋める。
- 【現物（^N225）のみ】末尾の欠損（15:30）は、直前5分足の終値ではなく
  日足の確定終値で埋める（より確実な値のため）。5分足→1分足の順で
  同じ確定終値を使って埋める（1分足にも15:30枠を新設）。
  ETF（1579/1357）はこの上書きを行わず、従来どおり直前値の横引き。
"""

import os
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TICKERS = {
    "n225": "^N225",
    "etf1579": "1579.T",
    "etf1357": "1357.T",
}

MAX_RETRIES = 4

DAILY_COLS = ["Date", "Open", "High", "Low", "Close"]
INTRADAY_COLS = ["Date", "Time", "Open", "High", "Low", "Close"]


def _download_with_retry(**kwargs) -> pd.DataFrame:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(**kwargs, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                return df
            last_err = "empty result"
        except Exception as e:
            last_err = e
        wait = (2 ** attempt) + random.uniform(0, 1)
        print(f"    リトライ待機 {wait:.1f}s（{attempt + 1}/{MAX_RETRIES}回目、理由: {last_err}）")
        time.sleep(wait)
    print(f"    取得失敗（最終）: {last_err}")
    return pd.DataFrame()


def _to_jst(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df.index = df.index.tz_convert(JST)
    return df


def _session_slots(freq_minutes: int, last_slot: str) -> list[str]:
    slots = []
    t = pd.Timestamp("2000-01-01 09:00")
    end_morning = pd.Timestamp("2000-01-01 11:30")
    while t <= end_morning:
        slots.append(t.strftime("%H:%M"))
        t += pd.Timedelta(minutes=freq_minutes)
    t = pd.Timestamp("2000-01-01 12:30")
    end_afternoon = pd.Timestamp(f"2000-01-01 {last_slot}")
    while t <= end_afternoon:
        slots.append(t.strftime("%H:%M"))
        t += pd.Timedelta(minutes=freq_minutes)
    return slots


def _fill_intraday_gaps(
    df: pd.DataFrame,
    freq_minutes: int,
    last_slot: str,
    close_override: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    slots = _session_slots(freq_minutes, last_slot)
    df = df.sort_values(["Date", "Time"])
    out_rows = []

    for date, group in df.groupby("Date", sort=True):
        existing = {r["Time"]: r for _, r in group.iterrows()}
        day_rows = []
        prev_close = None
        for slot in slots:
            if slot in existing:
                row = dict(existing[slot])
                day_rows.append(row)
                prev_close = row["Close"]
            elif slot == last_slot and close_override and date in close_override:
                v = close_override[date]
                day_rows.append({
                    "Date": date, "Time": slot,
                    "Open": v, "High": v, "Low": v, "Close": v,
                })
                prev_close = v
            elif prev_close is not None:
                day_rows.append({
                    "Date": date, "Time": slot,
                    "Open": prev_close, "High": prev_close,
                    "Low": prev_close, "Close": prev_close,
                })
            else:
                day_rows.append(None)

        first_real_idx = next((i for i, r in enumerate(day_rows) if r is not None), None)
        if first_real_idx is not None:
            fill_val = day_rows[first_real_idx]["Open"]
            for i in range(first_real_idx):
                day_rows[i] = {
                    "Date": date, "Time": slots[i],
                    "Open": fill_val, "High": fill_val,
                    "Low": fill_val, "Close": fill_val,
                }

        out_rows.extend([r for r in day_rows if r is not None])

    return pd.DataFrame(out_rows, columns=INTRADAY_COLS)


def fetch_daily(ticker: str) -> pd.DataFrame:
    df = _download_with_retry(tickers=ticker, period="2y", interval="1d")
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = _to_jst(df)
    out = pd.DataFrame({
        "Date": df.index.strftime("%Y-%m-%d"),
        "Open": df["Open"].values,
        "High": df["High"].values,
        "Low": df["Low"].values,
        "Close": df["Close"].values,
    })
    return out


def fetch_5min(ticker: str) -> pd.DataFrame:
    df = _download_with_retry(tickers=ticker, period="60d", interval="5m")
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = _to_jst(df)
    out = pd.DataFrame({
        "Date": df.index.strftime("%Y-%m-%d"),
        "Time": df.index.strftime("%H:%M"),
        "Open": df["Open"].values,
        "High": df["High"].values,
        "Low": df["Low"].values,
        "Close": df["Close"].values,
    })
    return out


def fetch_1min(ticker: str, days_back: int = 29) -> pd.DataFrame:
    frames = []
    now = datetime.now(JST)
    window_start = now - timedelta(days=days_back)

    chunk_start = window_start
    while chunk_start < now:
        chunk_end = min(chunk_start + timedelta(days=7), now)
        df = _download_with_retry(
            tickers=ticker, start=chunk_start, end=chunk_end, interval="1m",
        )
        if not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            frames.append(_to_jst(df))
        chunk_start = chunk_end
        time.sleep(1)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame({
        "Date": df.index.strftime("%Y-%m-%d"),
        "Time": df.index.strftime("%H:%M"),
        "Open": df["Open"].values,
        "High": df["High"].values,
        "Low": df["Low"].values,
        "Close": df["Close"].values,
    })
    return out


def merge_and_save(
    new_df: pd.DataFrame,
    path: str,
    key_cols: list[str],
    ticker_symbol: str,
    header_cols: list[str],
    fill_freq_minutes: int | None = None,
    fill_last_slot: str | None = None,
    close_override: dict[str, str] | None = None,
) -> pd.DataFrame:
    if new_df.empty:
        print(f"  取得結果が空のためスキップ: {path}")
        if os.path.exists(path):
            return pd.read_csv(path, skiprows=1, dtype=str)
        return pd.DataFrame(columns=header_cols)

    if os.path.exists(path):
        old_df = pd.read_csv(path, skiprows=1, dtype=str)
    else:
        old_df = pd.DataFrame(columns=header_cols)

    combined = pd.concat([old_df, new_df.astype(str)], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols)

    if fill_freq_minutes is not None:
        before = len(combined)
        combined = _fill_intraday_gaps(
            combined, fill_freq_minutes, fill_last_slot, close_override=close_override,
        )
        filled = len(combined) - before
        if filled > 0:
            print(f"    欠損補完: {filled}本追加")

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{ticker_symbol}\n")
    combined.to_csv(path, mode="a", index=False, encoding="utf-8")
    print(f"  保存: {path}（{len(combined)}行）")
    return combined


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, ticker in TICKERS.items():
        print(f"[{name} / {ticker}] 日足取得中...")
        daily_combined = merge_and_save(
            fetch_daily(ticker),
            os.path.join(DATA_DIR, f"{name}_daily.csv"),
            key_cols=["Date"],
            ticker_symbol=ticker,
            header_cols=DAILY_COLS,
        )
        time.sleep(1)

        close_override = None
        if name == "n225" and daily_combined is not None and not daily_combined.empty:
            close_override = dict(zip(daily_combined["Date"], daily_combined["Close"]))

        print(f"[{name} / {ticker}] 5分足取得中...")
        merge_and_save(
            fetch_5min(ticker),
            os.path.join(DATA_DIR, f"{name}_5min.csv"),
            key_cols=["Date", "Time"],
            ticker_symbol=ticker,
            header_cols=INTRADAY_COLS,
            fill_freq_minutes=5,
            fill_last_slot="15:30",
            close_override=close_override,
        )
        time.sleep(1)

        print(f"[{name} / {ticker}] 1分足取得中...")
        merge_and_save(
            fetch_1min(ticker),
            os.path.join(DATA_DIR, f"{name}_1min.csv"),
            key_cols=["Date", "Time"],
            ticker_symbol=ticker,
            header_cols=INTRADAY_COLS,
            fill_freq_minutes=1,
            fill_last_slot=("15:30" if name == "n225" else "15:29"),
            close_override=close_override,
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
