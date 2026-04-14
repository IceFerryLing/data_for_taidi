# C题项目说明

本目录用于 **泰迪杯 C 题：事件驱动型股市投资策略构建**。

当前文件已按“最终可交付数据 + 支撑数据 + 获取代码”重新整理，统一放入 `最终数据` 文件夹中。

## 题目最终数据分为几部分

按 `C题-事件驱动型股市投资策略构建.pdf` 的任务要求，核心最终数据分为 **4 部分**：

- 事件识别与分类数据
- 事件关联公司数据
- 事件影响预测数据
- 投资策略与回测数据

为便于直接使用和后续扩展，当前又整理成 **8 个子目录**：

- `最终数据/01_事件识别与分类`
- `最终数据/02_事件关联公司`
- `最终数据/03_事件影响预测`
- `最终数据/04_投资策略与回测`
- `最终数据/05_A股近一年交易数据`
- `最终数据/06_股票基础信息`
- `最终数据/07_宏观事件数据`
- `最终数据/08_获取代码与说明`


### `最终数据/01_事件识别与分类`

- `events.csv`
  - 事件主表
  - 包含事件编号、事件名称、事件日期、事件分类、影响周期、强度等字段
  - 当前事件量已扩充到约 `6615` 条

### `最终数据/02_事件关联公司`

- `event_company_map.csv`
  - 事件与上市公司映射表
- `relation_stats.csv`
  - 按直接/间接关联聚合后的统计结果
  - 当前映射记录约 `6621` 条

### `最终数据/03_事件影响预测`

- `event_window_summary.csv`
  - 事件窗收益、异常收益、CAR 等统计
- `price_panel.csv`
  - 题目案例相关股票行情面板
  - 当前事件窗统计约 `6598` 条

### `最终数据/04_投资策略与回测`

- `weekly_trade_returns.csv`
  - 周度个股交易收益
- `weekly_top3.csv`
  - 每周前 3 只候选股票及建议权重
- `portfolio_backtest.csv`
  - 组合回测结果
- `c题事件驱动数据整合.xlsx`
  - 综合整合版 Excel

### `最终数据/05_A股近一年交易数据`

- `daily_kline_batches_index.csv`
  - 近一年日频批次索引
- `daily_kline_batches/`
  - 近一年 A 股交易数据批次文件
  - 当前已落地约 `1200` 只股票、`282586` 行日频记录
  - 时间范围：`2025-03-05` 至 `2026-03-05`

日频字段包括：

- `date`
- `code`
- `open`
- `high`
- `low`
- `close`
- `preclose`
- `volume`
- `amount`
- `turn`
- `pctChg`
- `peTTM`
- `pbMRQ`
- `psTTM`
- `pcfNcfTTM`
- `tradestatus`
- `isST`

### `最终数据/06_股票基础信息`

- `a_share_universe.csv`
  - A 股股票池快照
- `a_share_universe.parquet`
  - 股票池 Parquet 版本
- `stock_basic_all.csv`
  - 股票基础信息
- `stock_industry_all.csv`
  - 股票行业分类
- `hs300_constituents.csv`
  - 沪深 300 成分股
- `sz50_constituents.csv`
  - 上证 50 成分股
- `zz500_constituents.csv`
  - 中证 500 成分股

### `最终数据/07_宏观事件数据`

- `deposit_rate.csv`
- `loan_rate.csv`
- `money_supply_month.csv`
- `money_supply_year.csv`
- `required_reserve_ratio.csv`

这些表可作为宏观事件、政策环境和市场背景变量使用。

### `最终数据/08_获取代码与说明`

- `build_c_event_dataset.py`
  - 题目案例整合代码
- `build_expanded_market_dataset.py`
  - 近一年 A 股扩展数据获取代码
- `augment_events_from_price_anomalies.py`
  - 按当前最终数据格式，从近一年交易面板批量抽取规则化事件并补充到事件主表
- `README.md`
  - 当前说明文档

## 推荐主键

- 股票静态表联结：`code`
- 交易面板联结：`code + date`
- 题目事件联结：`event_id + stock_code`
- 宏观事件联结：`date`

## 如何重新生成

- 重新生成题目案例整合结果：
  - `python scripts/build_c_event_dataset.py`
- 继续扩充近一年 A 股交易数据：
  - `python scripts/build_expanded_market_dataset.py`

## 说明
若后续需要继续补齐更多股票批次或公司事件表，可直接续跑 `build_expanded_market_dataset.py`。
