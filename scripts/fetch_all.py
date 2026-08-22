"""
日次データ取得スクリプト（GitHub Actions想定）

対象：日経225現物（^N225）、1579.T、1357.T
足種：日足・5分足・1分足

設計方針：
- 毎回「取得可能な最大窓」を取りに行き、既存CSVとマージして重複排除する。
  （実行を1〜2日飛ばしても、窓の範囲内なら欠損しない）
- 1分足は1リクエスト7日制限があるため、7日ずつチャンクに分けて
  過去29日分を取得・結合する（境界は必ず30日以内に収める）。
- 出力はプロジェクトの既存CSVと同じ日本語カラム名に揃える。
- yfinanceはGitHub Actionsの共有IPからレートリミットを受けることがあるため、
  各取得にリトライ（指数バックオフ）を入れる。
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


def _download_with_retry(**kwargs) -> pd.DataFrame:
    """yf.downloadをリトライ付きで実行する。レートリミット等の一時エラーに対応。"""
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
    """indexをAsia/Tokyoに変換する（tzなしの場合はUTC由来として付与してから変換）"""
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df.index = df.index.tz_convert(JST)
    return df


def fetch_daily(ticker: str) -> pd.DataFrame:
    df = _download_with_retry(tickers=ticker, period="2y", interval="1d")
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = _to_jst(df)
    out = pd.DataFrame({
        "日付": df.index.strftime("%Y-%m-%d"),
        "始値": df["Open"].values,
        "高値": df["High"].values,
        "安値": df["Low"].values,
        "終値": df["Close"].values,
    })
    return out


def fetch_5min(ticker: str) -> pd.DataFrame:
    df = _download_with_retry(tickers=ticker, period="60d", interval="5m")
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = _to_jst(df)
    out = pd.DataFrame({
        "日付": df.index.strftime("%Y-%m-%d"),
        "時刻": df.index.strftime("%H:%M"),
        "始値": df["Open"].values,
        "高値": df["High"].values,
        "安値": df["Low"].values,
        "終値": df["Close"].values,
    })
    return out


def fetch_1min(ticker: str, days_back: int = 29) -> pd.DataFrame:
    """
    過去days_back日分を7日ずつのチャンクで取得する。
    境界は必ず [now - days_back, now] の範囲内に収め、30日制限を超えない。
    """
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
        time.sleep(1)  # 連続リクエストの間隔を空ける

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame({
        "日付": df.index.strftime("%Y-%m-%d"),
        "時刻": df.index.strftime("%H:%M"),
        "始値": df["Open"].values,
        "高値": df["High"].values,
        "安値": df["Low"].values,
        "終値": df["Close"].values,
    })
    return out


def merge_and_save(new_df: pd.DataFrame, path: str, key_cols: list[str]):
    if new_df.empty:
        print(f"  取得結果が空のためスキップ: {path}")
        return
    if os.path.exists(path):
        old_df = pd.read_csv(path, dtype=str)
        combined = pd.concat([old_df, new_df.astype(str)], ignore_index=True)
    else:
        combined = new_df.astype(str)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols)
    combined.to_csv(path, index=False, encoding="utf-8")
    print(f"  保存: {path}（{len(combined)}行）")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, ticker in TICKERS.items():
        print(f"[{name} / {ticker}] 日足取得中...")
        merge_and_save(
            fetch_daily(ticker),
            os.path.join(DATA_DIR, f"{name}_daily.csv"),
            key_cols=["日付"],
        )
        time.sleep(1)

        print(f"[{name} / {ticker}] 5分足取得中...")
        merge_and_save(
            fetch_5min(ticker),
            os.path.join(DATA_DIR, f"{name}_5min.csv"),
            key_cols=["日付", "時刻"],
        )
        time.sleep(1)

        print(f"[{name} / {ticker}] 1分足取得中...")
        merge_and_save(
            fetch_1min(ticker),
            os.path.join(DATA_DIR, f"{name}_1min.csv"),
            key_cols=["日付", "時刻"],
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
