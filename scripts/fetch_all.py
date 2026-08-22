"""
日次データ取得スクリプト（GitHub Actions想定）

対象：日経225現物（^N225）、1579.T、1357.T
足種：日足・5分足・1分足

設計方針：
- 毎回「取得可能な最大窓」を取りに行き、既存CSVとマージして重複排除する。
  （実行を1〜2日飛ばしても、窓の範囲内なら欠損しない）
- 1分足は1リクエスト7日制限があるため、7日ずつチャンクに分けて
  過去29日分を取得・結合する。
- 出力はプロジェクトの既存CSVと同じ日本語カラム名に揃える。
"""

import os
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


def _to_jst(df: pd.DataFrame) -> pd.DataFrame:
    """indexをAsia/Tokyoに変換する（tzなしの場合はUTC由来として付与してから変換）"""
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df.index = df.index.tz_convert(JST)
    return df


def fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
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
    df = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=False)
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
    frames = []
    now = datetime.now(JST)
    i = 0
    while i < days_back:
        chunk_end = now - timedelta(days=i)
        chunk_start = chunk_end - timedelta(days=7)
        df = yf.download(
            ticker, start=chunk_start, end=chunk_end,
            interval="1m", progress=False, auto_adjust=False,
        )
        if not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            frames.append(_to_jst(df))
        i += 7
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

        print(f"[{name} / {ticker}] 5分足取得中...")
        merge_and_save(
            fetch_5min(ticker),
            os.path.join(DATA_DIR, f"{name}_5min.csv"),
            key_cols=["日付", "時刻"],
        )

        print(f"[{name} / {ticker}] 1分足取得中...")
        merge_and_save(
            fetch_1min(ticker),
            os.path.join(DATA_DIR, f"{name}_1min.csv"),
            key_cols=["日付", "時刻"],
        )


if __name__ == "__main__":
    main()
