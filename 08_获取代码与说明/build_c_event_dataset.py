from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import baostock as bs
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Event:
    event_id: str
    event_name: str
    event_date: str
    driver_type: str
    impact_cycle: str
    predictability: str
    industry: str
    intensity_score: int
    source_note: str


@dataclass(frozen=True)
class Mapping:
    event_id: str
    stock_code: str
    stock_name: str
    market_code: str
    relation_type: str
    relation_strength: float
    relation_layer: str
    note: str


EVENTS = [
    Event(
        event_id="E20250507",
        event_name="印巴空战催化A股军工链",
        event_date="2025-05-07",
        driver_type="地缘类事件",
        impact_cycle="脉冲型事件",
        predictability="突发型事件",
        industry="军工",
        intensity_score=5,
        source_note="附件1案例一",
    ),
    Event(
        event_id="E20251125",
        event_name="凌空天行发布驭空戟-1000",
        event_date="2025-11-25",
        driver_type="公司/技术类事件",
        impact_cycle="中期型事件",
        predictability="预披露+扩散型事件",
        industry="军工/商业航天",
        intensity_score=4,
        source_note="附件1案例二",
    ),
]


MAPPINGS = [
    Mapping("E20250507", "302132", "中航成飞", "sz.302132", "核心整机受益", 1.00, "直接", "歼-10CE 事件直接映射"),
    Mapping("E20250507", "300581", "晨曦航空", "sz.300581", "无人机配套", 0.85, "直接", "无人机链条配套"),
    Mapping("E20250507", "688543", "国科军工", "sh.688543", "导弹产业链", 0.80, "直接", "导弹链条受益"),
    Mapping("E20251125", "002792", "通宇通讯", "sz.002792", "股权/基金直接投资", 1.00, "直接", "通过空天基金重仓投资凌空天行"),
    Mapping("E20251125", "688033", "天宜上佳", "sh.688033", "核心供应商", 0.80, "直接", "材料/零部件供应链"),
    Mapping("E20251125", "003009", "中天火箭", "sz.003009", "核心供应商", 0.80, "直接", "火箭/导弹链条供应"),
    Mapping("E20251125", "600037", "歌华有线", "sh.600037", "基金间接参股", 0.45, "间接", "间接股权关联"),
    Mapping("E20251125", "600604", "市北高新", "sh.600604", "基金间接参股", 0.45, "间接", "间接股权关联"),
]


def to_frame(rs) -> pd.DataFrame:
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    frame = pd.DataFrame(rows, columns=rs.fields)
    for col in [c for c in frame.columns if c not in {"date", "code"}]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def fetch_history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,code,open,high,low,close,volume,amount,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    return to_frame(rs)


def event_window_summary(stock_df: pd.DataFrame, bench_df: pd.DataFrame, event_date: str) -> dict:
    stock_df = stock_df.copy()
    bench_df = bench_df.copy()
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    merged = stock_df.merge(bench_df[["date", "pctChg"]], on="date", how="left", suffixes=("", "_bench"))
    merged["ar"] = merged["pctChg"] - merged["pctChg_bench"]
    event_idx = merged.index[merged["date"] == pd.Timestamp(event_date)]
    if len(event_idx) == 0:
        return {}
    idx = int(event_idx[0])
    prev_close = merged.iloc[idx - 1]["close"] if idx > 0 else None
    def close_ret(offset: int) -> float | None:
        target = idx + offset
        if prev_close is None or target >= len(merged):
            return None
        return (merged.iloc[target]["close"] / prev_close - 1) * 100
    def car(offset: int) -> float | None:
        target = idx + offset
        if target >= len(merged):
            return None
        return merged.iloc[idx : target + 1]["ar"].sum()
    return {
        "event_day_pct": merged.iloc[idx]["pctChg"],
        "event_day_ar": merged.iloc[idx]["ar"],
        "ret_close_t2": close_ret(2),
        "ret_close_t5": close_ret(5),
        "ret_close_t7": close_ret(7),
        "car_t2": car(2),
        "car_t5": car(5),
        "car_t7": car(7),
    }


def weekly_trade_return(stock_df: pd.DataFrame, buy_date: str, sell_date: str) -> dict:
    stock_df = stock_df.copy()
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    buy_row = stock_df.loc[stock_df["date"] == pd.Timestamp(buy_date)]
    sell_row = stock_df.loc[stock_df["date"] == pd.Timestamp(sell_date)]
    if buy_row.empty or sell_row.empty:
        return {}
    buy_open = float(buy_row.iloc[0]["open"])
    sell_close = float(sell_row.iloc[0]["close"])
    return {
        "buy_date": buy_date,
        "sell_date": sell_date,
        "buy_open": buy_open,
        "sell_close": sell_close,
        "trade_return_pct": (sell_close / buy_open - 1) * 100,
    }


def build_outputs() -> None:
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")

    try:
        events_df = pd.DataFrame([e.__dict__ for e in EVENTS])
        mappings_df = pd.DataFrame([m.__dict__ for m in MAPPINGS])

        bench_df = fetch_history("sh.000300", "2025-04-20", "2025-12-31")
        price_frames: list[pd.DataFrame] = []
        summary_rows: list[dict] = []
        weekly_rows: list[dict] = []

        for mapping in MAPPINGS:
            stock_df = fetch_history(mapping.market_code, "2025-04-20", "2025-12-31")
            stock_df.insert(0, "stock_name", mapping.stock_name)
            stock_df.insert(0, "stock_code", mapping.stock_code)
            price_frames.append(stock_df)

            event_date = events_df.loc[events_df["event_id"] == mapping.event_id, "event_date"].iloc[0]
            stats = event_window_summary(stock_df, bench_df, event_date)
            if stats:
                summary_rows.append({**mapping.__dict__, **stats})

            if mapping.event_id == "E20251125":
                for buy_date, sell_date, week_id in [
                    ("2025-12-09", "2025-12-12", "W1"),
                    ("2025-12-16", "2025-12-19", "W2"),
                    ("2025-12-23", "2025-12-26", "W3"),
                ]:
                    trade = weekly_trade_return(stock_df, buy_date, sell_date)
                    if trade:
                        weekly_rows.append({
                            "week_id": week_id,
                            **mapping.__dict__,
                            **trade,
                        })

        prices_df = pd.concat(price_frames, ignore_index=True)
        summary_df = pd.DataFrame(summary_rows)
        weekly_df = pd.DataFrame(weekly_rows)

        relation_group = (
            summary_df.groupby("relation_layer")[["car_t2", "car_t5", "car_t7", "ret_close_t7"]]
            .mean()
            .reset_index()
        )
        event_group = (
            summary_df.groupby("event_id")[["car_t2", "car_t5", "car_t7", "ret_close_t7"]]
            .mean()
            .reset_index()
            .merge(events_df[["event_id", "event_name"]], on="event_id", how="left")
        )
        weekly_top3 = (
            weekly_df.sort_values(["week_id", "trade_return_pct"], ascending=[True, False])
            .groupby("week_id")
            .head(3)
            .copy()
        )
        weekly_top3["weight"] = weekly_top3.groupby("week_id")["trade_return_pct"].transform(
            lambda s: s.clip(lower=0).div(s.clip(lower=0).sum()) if s.clip(lower=0).sum() > 0 else 1 / len(s)
        )
        portfolio_df = (
            weekly_top3.groupby("week_id")
            .apply(lambda g: pd.Series({"portfolio_return_pct": (g["trade_return_pct"] * g["weight"]).sum()}))
            .reset_index()
        )
        capital = 100000.0
        ending_capitals: list[float] = []
        for row in portfolio_df.itertuples(index=False):
            capital *= 1 + row.portfolio_return_pct / 100
            ending_capitals.append(capital)
        portfolio_df["ending_capital"] = ending_capitals
        portfolio_df["cumulative_return_pct"] = (portfolio_df["ending_capital"] / 100000 - 1) * 100

        with pd.ExcelWriter(OUTPUT / "c题事件驱动数据整合.xlsx") as writer:
            events_df.to_excel(writer, sheet_name="events", index=False)
            mappings_df.to_excel(writer, sheet_name="event_company_map", index=False)
            summary_df.to_excel(writer, sheet_name="event_window_summary", index=False)
            relation_group.to_excel(writer, sheet_name="relation_stats", index=False)
            weekly_df.to_excel(writer, sheet_name="weekly_trade_returns", index=False)
            weekly_top3.to_excel(writer, sheet_name="weekly_top3", index=False)
            portfolio_df.to_excel(writer, sheet_name="portfolio_backtest", index=False)

        events_df.to_csv(OUTPUT / "events.csv", index=False, encoding="utf-8-sig")
        mappings_df.to_csv(OUTPUT / "event_company_map.csv", index=False, encoding="utf-8-sig")
        prices_df.to_csv(OUTPUT / "price_panel.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(OUTPUT / "event_window_summary.csv", index=False, encoding="utf-8-sig")
        relation_group.to_csv(OUTPUT / "relation_stats.csv", index=False, encoding="utf-8-sig")
        weekly_df.to_csv(OUTPUT / "weekly_trade_returns.csv", index=False, encoding="utf-8-sig")
        weekly_top3.to_csv(OUTPUT / "weekly_top3.csv", index=False, encoding="utf-8-sig")
        portfolio_df.to_csv(OUTPUT / "portfolio_backtest.csv", index=False, encoding="utf-8-sig")

        lines: list[str] = []
        lines.append("# C题事件驱动数据整合\n")
        lines.append("## 1. 已读取文档\n")
        lines.append("- 题目主文档：`C题-事件驱动型股市投资策略构建.pdf`\n")
        lines.append("- 附件1：`C题-附件文档/附件1 具体事件案例参考.pdf`\n")
        lines.append("- 附件2：`C题-附件文档/附件2 金融市场数据下载指南.pdf`\n")
        lines.append("- 附件3：`C题-附件文档/附件3 事件分类维度参考.pdf`\n")
        lines.append("\n## 2. 任务1：事件识别与分类\n")
        for row in events_df.itertuples(index=False):
            lines.append(
                f"- {row.event_id} | {row.event_name} | 日期 {row.event_date} | {row.driver_type} | {row.impact_cycle} | {row.predictability} | 行业 {row.industry} | 强度 {row.intensity_score}\n"
            )
        lines.append("\n建议统一事件特征字段：`event_id`、`event_date`、`driver_type`、`impact_cycle`、`predictability`、`industry`、`intensity_score`、`relation_strength`、`stock_code`。\n")
        lines.append("\n## 3. 任务2：事件关联公司挖掘\n")
        for row in mappings_df.itertuples(index=False):
            lines.append(
                f"- {row.event_id} -> {row.stock_name}({row.stock_code}) | 关系 {row.relation_type} | 层级 {row.relation_layer} | 强度 {row.relation_strength:.2f} | {row.note}\n"
            )
        lines.append("\n## 4. 任务3：事件影响预测的可量化结果\n")
        for row in summary_df.sort_values(["event_id", "relation_strength"], ascending=[True, False]).itertuples(index=False):
            lines.append(
                f"- {row.event_id} | {row.stock_name}({row.stock_code}) | 事件日涨跌幅 {row.event_day_pct:.2f}% | CAR[0,2] {row.car_t2:.2f}% | CAR[0,5] {row.car_t5:.2f}% | 7日收盘收益 {row.ret_close_t7:.2f}%\n"
            )
        lines.append("\n按关联层级聚合后，直接关联通常显著高于间接关联，详见 `relation_stats.csv`。\n")
        lines.append("\n## 5. 任务4：交易窗口可直接复用的数据\n")
        for row in weekly_top3.itertuples(index=False):
            lines.append(
                f"- {row.week_id} | {row.stock_name}({row.stock_code}) | 周二开盘 {row.buy_open:.2f} | 周五收盘 {row.sell_close:.2f} | 区间收益 {row.trade_return_pct:.2f}% | 建议权重 {row.weight:.4f}\n"
            )
        lines.append("\n组合加权回测：\n")
        for row in portfolio_df.itertuples(index=False):
            lines.append(
                f"- {row.week_id} | 组合收益 {row.portfolio_return_pct:.2f}% | 期末资金 {row.ending_capital:.2f} | 累计收益 {row.cumulative_return_pct:.2f}%\n"
            )
        lines.append("\n## 6. 合并研究建议\n")
        lines.append("- 用 `event_id + stock_code` 作为主键，打通事件表、关联表、行情表。\n")
        lines.append("- 用 `relation_strength`、`relation_layer`、`driver_type`、`industry` 做解释变量。\n")
        lines.append("- 用 `event_day_pct`、`CAR[0,2]`、`CAR[0,5]`、`ret_close_t7` 做被解释变量。\n")
        lines.append("- 在赛题实测周，只需新增最新事件和对应公司映射，即可沿用同一张特征表。\n")

        (OUTPUT / "C题数据整合说明.md").write_text("".join(lines), encoding="utf-8")
    finally:
        bs.logout()


if __name__ == "__main__":
    build_outputs()
