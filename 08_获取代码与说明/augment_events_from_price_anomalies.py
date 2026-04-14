from __future__ import annotations

from pathlib import Path

import baostock as bs
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FINAL = ROOT / "最终数据"


def load_daily_panel() -> pd.DataFrame:
    files = sorted((OUTPUT / "expanded" / "daily_kline_batches").glob("*.parquet"))
    frame = pd.concat(
        [
            pd.read_parquet(
                file,
                columns=[
                    "date",
                    "code",
                    "code_name",
                    "industry",
                    "open",
                    "high",
                    "low",
                    "close",
                    "preclose",
                    "turn",
                    "pctChg",
                ],
            )
            for file in files
        ],
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["code", "date"]).reset_index(drop=True)


def load_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    login = bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,code,close,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        frame = pd.DataFrame(rows, columns=rs.fields)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["pctChg"] = pd.to_numeric(frame["pctChg"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        return frame
    finally:
        bs.logout()


def classify_event(row: pd.Series) -> tuple[str, str, int]:
    pct = float(row["pctChg"])
    turn = float(row["turn"])
    if pct >= 9.5:
        return "涨停异动", "公司/市场异动事件", 5 if turn >= 10 else 4
    if pct <= -9.5:
        return "跌停异动", "公司/市场异动事件", 5 if turn >= 8 else 4
    return "放量上涨异动", "公司/市场异动事件", 4 if turn >= 15 else 3


def extract_events(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = panel[
        (panel["pctChg"] >= 9.5)
        | (panel["pctChg"] <= -9.5)
        | ((panel["pctChg"] >= 7.0) & (panel["turn"] >= 8.0))
    ].copy()
    candidates = candidates.drop_duplicates(["code", "date"]).reset_index(drop=True)

    events = []
    mappings = []
    for row in candidates.itertuples(index=False):
        event_tag, driver_type, intensity = classify_event(pd.Series(row._asdict()))
        date_str = pd.Timestamp(row.date).strftime("%Y%m%d")
        short_code = row.code.split(".")[1]
        event_id = f"A_{date_str}_{short_code}"
        event_name = f"{event_tag}_{row.code_name}_{pd.Timestamp(row.date).date()}"
        events.append(
            {
                "event_id": event_id,
                "event_name": event_name,
                "event_date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "driver_type": driver_type,
                "impact_cycle": "脉冲型事件",
                "predictability": "突发型事件",
                "industry": row.industry if isinstance(row.industry, str) and row.industry else "未知行业",
                "intensity_score": intensity,
                "source_note": "近一年交易数据规则抽取",
            }
        )
        mappings.append(
            {
                "event_id": event_id,
                "stock_code": short_code,
                "stock_name": row.code_name,
                "market_code": row.code,
                "relation_type": event_tag,
                "relation_strength": 1.0,
                "relation_layer": "直接",
                "note": f"规则抽取：涨跌幅={float(row.pctChg):.2f}%, 换手率={float(row.turn):.2f}%",
            }
        )
    return pd.DataFrame(events), pd.DataFrame(mappings)


def build_summary(mappings: pd.DataFrame, panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    bench = benchmark[["date", "pctChg"]].rename(columns={"pctChg": "bench_pct"})
    grouped = {code: frame.reset_index(drop=True) for code, frame in panel.groupby("code")}
    rows = []
    for mapping in mappings.itertuples(index=False):
        stock_df = grouped.get(mapping.market_code)
        if stock_df is None or stock_df.empty:
            continue
        merged = stock_df.merge(bench, on="date", how="left")
        merged["ar"] = merged["pctChg"] - merged["bench_pct"]
        event_date = pd.to_datetime(mapping.event_id.split("_")[1])
        loc = merged.index[merged["date"] == event_date]
        if len(loc) == 0:
            continue
        idx = int(loc[0])
        if idx == 0:
            continue
        prev_close = float(merged.iloc[idx - 1]["close"])

        def close_ret(offset: int) -> float | None:
            target = idx + offset
            if target >= len(merged):
                return None
            return (float(merged.iloc[target]["close"]) / prev_close - 1) * 100

        def car(offset: int) -> float | None:
            target = idx + offset
            if target >= len(merged):
                return None
            return float(merged.iloc[idx : target + 1]["ar"].sum())

        rows.append(
            {
                **mapping._asdict(),
                "event_day_pct": float(merged.iloc[idx]["pctChg"]),
                "event_day_ar": float(merged.iloc[idx]["ar"]),
                "ret_close_t2": close_ret(2),
                "ret_close_t5": close_ret(5),
                "ret_close_t7": close_ret(7),
                "car_t2": car(2),
                "car_t5": car(5),
                "car_t7": car(7),
            }
        )
    return pd.DataFrame(rows)


def sync_final_folder() -> None:
    targets = {
        FINAL / "01_事件识别与分类" / "events.csv": OUTPUT / "events.csv",
        FINAL / "02_事件关联公司" / "event_company_map.csv": OUTPUT / "event_company_map.csv",
        FINAL / "02_事件关联公司" / "relation_stats.csv": OUTPUT / "relation_stats.csv",
        FINAL / "03_事件影响预测" / "event_window_summary.csv": OUTPUT / "event_window_summary.csv",
        FINAL / "03_事件影响预测" / "price_panel.csv": OUTPUT / "price_panel.csv",
        FINAL / "08_获取代码与说明" / "augment_events_from_price_anomalies.py": ROOT / "scripts" / "augment_events_from_price_anomalies.py",
    }
    for dst, src in targets.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())


def main() -> None:
    panel = load_daily_panel()
    benchmark = load_benchmark(
        start_date=panel["date"].min().strftime("%Y-%m-%d"),
        end_date=panel["date"].max().strftime("%Y-%m-%d"),
    )

    existing_events = pd.read_csv(OUTPUT / "events.csv")
    existing_map = pd.read_csv(OUTPUT / "event_company_map.csv")
    existing_summary = pd.read_csv(OUTPUT / "event_window_summary.csv")

    new_events, new_map = extract_events(panel)
    new_summary = build_summary(new_map, panel, benchmark)

    events = (
        pd.concat([existing_events, new_events], ignore_index=True)
        .drop_duplicates(subset=["event_id"])
        .sort_values(["event_date", "event_id"])
        .reset_index(drop=True)
    )
    event_map = (
        pd.concat([existing_map, new_map], ignore_index=True)
        .drop_duplicates(subset=["event_id", "stock_code"])
        .reset_index(drop=True)
    )
    summary = (
        pd.concat([existing_summary, new_summary], ignore_index=True)
        .drop_duplicates(subset=["event_id", "stock_code"])
        .reset_index(drop=True)
    )
    relation_stats = (
        summary.groupby("relation_layer")[["car_t2", "car_t5", "car_t7", "ret_close_t7"]]
        .mean()
        .reset_index()
    )

    events.to_csv(OUTPUT / "events.csv", index=False, encoding="utf-8-sig")
    event_map.to_csv(OUTPUT / "event_company_map.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "event_window_summary.csv", index=False, encoding="utf-8-sig")
    relation_stats.to_csv(OUTPUT / "relation_stats.csv", index=False, encoding="utf-8-sig")

    sync_final_folder()
    print(
        {
            "events_total": len(events),
            "new_events": len(new_events),
            "event_map_total": len(event_map),
            "summary_total": len(summary),
        }
    )


if __name__ == "__main__":
    main()
