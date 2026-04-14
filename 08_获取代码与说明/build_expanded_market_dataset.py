from __future__ import annotations

from datetime import date, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import baostock as bs
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "expanded"
DAILY_DIR = OUT / "daily_kline_batches"
EVENT_DIR = OUT / "corporate_events"
MACRO_DIR = OUT / "macro_events"

FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
    "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)
WORKERS = max(2, min(8, cpu_count()))


def result_to_frame(rs) -> pd.DataFrame:
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    frame = pd.DataFrame(rows, columns=rs.fields)
    for col in [column for column in frame.columns if column not in {"date", "code", "code_name", "industry", "industryClassification"}]:
        frame[col] = pd.to_numeric(frame[col], errors="ignore")
    return frame


def ensure_dirs() -> None:
    for folder in [OUT, DAILY_DIR, EVENT_DIR, MACRO_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def get_trade_universe(snapshot_date: str) -> pd.DataFrame:
    rs = bs.query_all_stock(snapshot_date)
    universe = result_to_frame(rs)
    mask = universe["code"].str.match(r"^(sh\.60|sh\.68|sz\.00|sz\.30|bj\.)")
    return universe.loc[mask].reset_index(drop=True)


def save_basic_tables(snapshot_date: str) -> pd.DataFrame:
    universe = get_trade_universe(snapshot_date)

    basic = result_to_frame(bs.query_stock_basic())
    industry = result_to_frame(bs.query_stock_industry())

    universe = universe.merge(basic, on="code", how="left", suffixes=("", "_basic"))
    universe = universe.merge(industry[["code", "industry", "industryClassification"]], on="code", how="left")

    universe.to_csv(OUT / "a_share_universe.csv", index=False, encoding="utf-8-sig")
    universe.to_parquet(OUT / "a_share_universe.parquet", index=False)

    basic.to_csv(OUT / "stock_basic_all.csv", index=False, encoding="utf-8-sig")
    industry.to_csv(OUT / "stock_industry_all.csv", index=False, encoding="utf-8-sig")

    for name, func in [
        ("hs300", bs.query_hs300_stocks),
        ("sz50", bs.query_sz50_stocks),
        ("zz500", bs.query_zz500_stocks),
    ]:
        frame = result_to_frame(func())
        frame.to_csv(OUT / f"{name}_constituents.csv", index=False, encoding="utf-8-sig")

    return universe


def fetch_daily_kline_batches(universe: pd.DataFrame, start_date: str, end_date: str, batch_size: int = 100) -> pd.DataFrame:
    stats: list[dict] = []
    total = len(universe)
    for start in range(0, total, batch_size):
        batch = universe.iloc[start : start + batch_size]
        batch_id = start // batch_size
        batch_path = DAILY_DIR / f"daily_kline_batch_{batch_id:03d}.parquet"
        if batch_path.exists():
            existing = pd.read_parquet(batch_path)
            stats.append(
                {
                    "batch_id": batch_id,
                    "stock_count": len(batch),
                    "row_count": len(existing),
                    "start_code": batch.iloc[0]["code"],
                    "end_code": batch.iloc[-1]["code"],
                    "path": str(batch_path),
                }
            )
            print(f"skip batch {batch_id:03d} rows={len(existing)}")
            continue
        payload = [
            {
                "code": row["code"],
                "code_name": row["code_name"],
                "industry": row.get("industry", ""),
                "start_date": start_date,
                "end_date": end_date,
            }
            for _, row in batch.iterrows()
        ]
        with Pool(processes=WORKERS) as pool:
            frames = [frame for frame in pool.map(fetch_single_stock_daily, payload) if not frame.empty]
        if not frames:
            continue
        batch_frame = pd.concat(frames, ignore_index=True)
        batch_frame.to_parquet(batch_path, index=False)
        stats.append(
            {
                "batch_id": batch_id,
                "stock_count": len(batch),
                "row_count": len(batch_frame),
                "start_code": batch.iloc[0]["code"],
                "end_code": batch.iloc[-1]["code"],
                "path": str(batch_path),
            }
        )
        print(f"saved batch {batch_id:03d} rows={len(batch_frame)} stocks={len(batch)}")
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT / "daily_kline_batches_index.csv", index=False, encoding="utf-8-sig")
    return stats_df


def fetch_single_stock_daily(payload: dict) -> pd.DataFrame:
    login = bs.login()
    if login.error_code != "0":
        return pd.DataFrame()
    try:
        rs = bs.query_history_k_data_plus(
            payload["code"],
            FIELDS,
            start_date=payload["start_date"],
            end_date=payload["end_date"],
            frequency="d",
            adjustflag="2",
        )
        frame = result_to_frame(rs)
        if frame.empty:
            return frame
        frame.insert(1, "code_name", payload["code_name"])
        frame.insert(2, "industry", payload["industry"])
        return frame
    finally:
        bs.logout()


def fetch_macro_tables() -> None:
    result_to_frame(bs.query_deposit_rate_data()).to_csv(MACRO_DIR / "deposit_rate.csv", index=False, encoding="utf-8-sig")
    result_to_frame(bs.query_loan_rate_data()).to_csv(MACRO_DIR / "loan_rate.csv", index=False, encoding="utf-8-sig")
    result_to_frame(bs.query_required_reserve_ratio_data()).to_csv(
        MACRO_DIR / "required_reserve_ratio.csv", index=False, encoding="utf-8-sig"
    )
    result_to_frame(bs.query_money_supply_data_month()).to_csv(
        MACRO_DIR / "money_supply_month.csv", index=False, encoding="utf-8-sig"
    )
    result_to_frame(bs.query_money_supply_data_year()).to_csv(
        MACRO_DIR / "money_supply_year.csv", index=False, encoding="utf-8-sig"
    )


def collect_performance_express(universe: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for code in universe["code"]:
        rs = bs.query_performance_express_report(code, start_date=start_date, end_date=end_date)
        frame = result_to_frame(rs)
        if not frame.empty:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(EVENT_DIR / "performance_express.csv", index=False, encoding="utf-8-sig")
    if not result.empty:
        result.to_parquet(EVENT_DIR / "performance_express.parquet", index=False)
    return result


def collect_forecast_report(universe: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for code in universe["code"]:
        rs = bs.query_forecast_report(code, start_date=start_date, end_date=end_date)
        frame = result_to_frame(rs)
        if not frame.empty:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(EVENT_DIR / "forecast_report.csv", index=False, encoding="utf-8-sig")
    if not result.empty:
        result.to_parquet(EVENT_DIR / "forecast_report.parquet", index=False)
    return result


def collect_dividend_data(universe: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        year_frames: list[pd.DataFrame] = []
        for code in universe["code"]:
            rs = bs.query_dividend_data(code=code, year=year, yearType="report")
            frame = result_to_frame(rs)
            if not frame.empty:
                frame.insert(1, "report_year", year)
                year_frames.append(frame)
        year_result = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()
        year_result.to_csv(EVENT_DIR / f"dividend_{year}.csv", index=False, encoding="utf-8-sig")
        if not year_result.empty:
            year_result.to_parquet(EVENT_DIR / f"dividend_{year}.parquet", index=False)
        frames.append(year_result)
    result = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(EVENT_DIR / "dividend_all.csv", index=False, encoding="utf-8-sig")
    if not result.empty:
        result.to_parquet(EVENT_DIR / "dividend_all.parquet", index=False)
    return result


def build_summary(
    universe: pd.DataFrame,
    batch_stats: pd.DataFrame,
    performance_express: pd.DataFrame,
    forecast_report: pd.DataFrame,
    dividend_all: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    daily_rows = int(batch_stats["row_count"].sum()) if not batch_stats.empty else 0
    daily_files = int(len(batch_stats))
    trade_days = pd.bdate_range(start_date, end_date).size
    lines: list[str] = []
    lines.append("# 扩展版A股数据项目树说明\n")
    lines.append(f"- 数据时间范围：{start_date} 至 {end_date}\n")
    lines.append(f"- A股股票覆盖数：{len(universe)}\n")
    lines.append(f"- 日频交易数据总行数：{daily_rows}\n")
    lines.append(f"- 日频批次文件数：{daily_files}\n")
    lines.append(f"- 近似交易日数量（工作日口径）：{trade_days}\n")
    lines.append("\n## 项目树\n")
    lines.append("- `output/expanded/a_share_universe.csv`：近一年A股股票池快照，含代码、名称、状态、行业\n")
    lines.append("- `output/expanded/stock_basic_all.csv`：股票基础信息全表\n")
    lines.append("- `output/expanded/stock_industry_all.csv`：行业分类全表\n")
    lines.append("- `output/expanded/hs300_constituents.csv`：沪深300成分股\n")
    lines.append("- `output/expanded/sz50_constituents.csv`：上证50成分股\n")
    lines.append("- `output/expanded/zz500_constituents.csv`：中证500成分股\n")
    lines.append("- `output/expanded/daily_kline_batches/`：近一年A股日频交易面板分批Parquet文件\n")
    lines.append("- `output/expanded/daily_kline_batches_index.csv`：日频批次索引，记录每批股票范围和行数\n")
    lines.append("- `output/expanded/corporate_events/performance_express.csv`：业绩快报事件表\n")
    lines.append("- `output/expanded/corporate_events/forecast_report.csv`：业绩预告事件表\n")
    lines.append("- `output/expanded/corporate_events/dividend_2024.csv`：2024年分红送转事件表\n")
    lines.append("- `output/expanded/corporate_events/dividend_2025.csv`：2025年分红送转事件表\n")
    lines.append("- `output/expanded/macro_events/deposit_rate.csv`：存款利率事件表\n")
    lines.append("- `output/expanded/macro_events/loan_rate.csv`：贷款利率事件表\n")
    lines.append("- `output/expanded/macro_events/required_reserve_ratio.csv`：存款准备金率事件表\n")
    lines.append("- `output/expanded/macro_events/money_supply_month.csv`：月度货币供应量事件表\n")
    lines.append("- `output/expanded/macro_events/money_supply_year.csv`：年度货币供应量事件表\n")
    lines.append("\n## 交易字段说明\n")
    lines.append("- `open`：开盘价\n")
    lines.append("- `high`：最高价\n")
    lines.append("- `low`：最低价\n")
    lines.append("- `close`：收盘价\n")
    lines.append("- `preclose`：前收盘价\n")
    lines.append("- `volume`：成交量\n")
    lines.append("- `amount`：成交额\n")
    lines.append("- `turn`：换手率\n")
    lines.append("- `pctChg`：涨跌幅\n")
    lines.append("- `peTTM`：滚动市盈率\n")
    lines.append("- `pbMRQ`：市净率\n")
    lines.append("- `psTTM`：滚动市销率\n")
    lines.append("- `pcfNcfTTM`：滚动市现率\n")
    lines.append("- `tradestatus`：交易状态\n")
    lines.append("- `isST`：是否ST\n")
    lines.append("\n## 事件覆盖说明\n")
    lines.append(f"- 业绩快报记录数：{len(performance_express)}\n")
    lines.append(f"- 业绩预告记录数：{len(forecast_report)}\n")
    lines.append(f"- 分红送转记录数：{len(dividend_all)}\n")
    lines.append("- 宏观事件表可与个股日频面板按日期联结\n")
    lines.append("- 公司事件表可与股票池表按 `code` 联结，再与交易面板按 `code + date` 或窗口日期联结\n")
    lines.append("\n## 推荐主键\n")
    lines.append("- 股票级联结：`code`\n")
    lines.append("- 股票-日期联结：`code + date`\n")
    lines.append("- 公司事件研究：`code + 公告/预告日期`\n")

    (OUT / "DATA_TREE.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    end_date = "2026-03-05"
    start_date = (date(2026, 3, 5) - timedelta(days=365)).isoformat()

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)

    try:
        universe = save_basic_tables(end_date)
        batch_stats = fetch_daily_kline_batches(universe, start_date, end_date, batch_size=100)
        fetch_macro_tables()
        performance_express = collect_performance_express(universe, start_date, end_date)
        forecast_report = collect_forecast_report(universe, start_date, end_date)
        dividend_all = collect_dividend_data(universe, years=["2024", "2025"])
        build_summary(universe, batch_stats, performance_express, forecast_report, dividend_all, start_date, end_date)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
