# 前情提要，如要运行，务必安装numpy,pandas,xgboost,sklearn等依赖库


import argparse
import logging
import os
import re
import warnings
import zipfile
from typing import Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConstantProbClassifier:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


class EventGraphPriceCompactStrategy:
    """
    减少特征，以免过拟合
    """

    def __init__(self, paths: dict):
        self.paths = paths
        self.events = pd.DataFrame()
        self.ecm = pd.DataFrame()
        self.market = pd.DataFrame()
        self.universe = pd.DataFrame()
        self.stock_industry = pd.DataFrame()
        self.stock_basic = pd.DataFrame()
        self.hs300 = pd.DataFrame()
        self.sz50 = pd.DataFrame()
        self.zz500 = pd.DataFrame()

        self.weekly_returns = pd.DataFrame()
        self.decision_samples = pd.DataFrame()
        self.price_snapshot = pd.DataFrame()

        self.feature_columns: List[str] = []
        self.model_reg = None
        self.model_cls = None

        self.latest_event_date = pd.NaT
        self.latest_market_date = pd.NaT
        self.effective_prediction_date = pd.NaT

        self.min_train_weeks = 12
        self.use_index_features = False
        self.use_industry_features = True
        self.use_stock_basic_features = True

    @staticmethod
    def _read_table(path: Optional[str]) -> pd.DataFrame:
        if path is None or not os.path.exists(path):
            return pd.DataFrame()
        if zipfile.is_zipfile(path):
            try:
                return pd.read_excel(path)
            except Exception:
                pass
        for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                try:
                    return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
                except Exception:
                    continue
        raise ValueError(f"无法读取文件: {path}")

    @staticmethod
    def _safe_numeric(s, default=0.0):
        return pd.to_numeric(s, errors="coerce").fillna(default)

    @staticmethod
    def _normalize_market_code(value) -> str:
        if pd.isna(value):
            return np.nan
        x = str(value).strip().lower()
        x = re.sub(r"\s+", "", x)
        if x in {"", "nan", "none", "null", "na"}:
            return np.nan
        if re.fullmatch(r"(sh|sz|bj)\.\d{6}", x):
            return x
        if re.fullmatch(r"(sh|sz|bj)\d{6}", x):
            ex = "bj" if x.startswith("bj") else x[:2]
            return f"{ex}.{x[len(ex):]}"
        if re.fullmatch(r"\d{6}\.(sh|sz|bj)", x):
            code, ex = x.split(".")
            return f"{ex}.{code}"
        if re.fullmatch(r"\d+", x):
            code = x.zfill(6)
            if code.startswith(("5", "6", "9")):
                return f"sh.{code}"
            if code.startswith(("0", "2", "3")):
                return f"sz.{code}"
            if code.startswith(("4", "8")):
                return f"bj.{code}"
            return code
        return x

    @staticmethod
    def _align_to_tuesday(dt: pd.Timestamp) -> pd.Timestamp:
        dt = pd.to_datetime(dt).normalize()
        monday = dt - pd.to_timedelta(dt.weekday(), unit="D")
        return monday + pd.Timedelta(days=1)

    @staticmethod
    def _next_decision_tuesday(event_date: pd.Series) -> pd.Series:
        event_date = pd.to_datetime(event_date).dt.normalize()
        monday = event_date - pd.to_timedelta(event_date.dt.weekday, unit="D")
        this_tuesday = monday + pd.Timedelta(days=1)
        use_this_week = event_date.dt.weekday == 0
        return pd.to_datetime(np.where(use_this_week, this_tuesday, this_tuesday + pd.Timedelta(days=7)))

    @staticmethod
    def _bucket_driver_type(s: pd.Series) -> pd.Series:
        t = s.fillna("未知").astype(str)
        out = pd.Series("other", index=t.index)
        out[t.str.contains("公司|企业|市场", na=False)] = "company"
        out[t.str.contains("行业|产业", na=False)] = "industry"
        out[t.str.contains("宏观", na=False)] = "macro"
        out[t.str.contains("地缘|国际|战争|冲突", na=False)] = "geo"
        out[t.str.contains("政策|监管", na=False)] = "policy"
        return out

    @staticmethod
    def _bucket_impact_cycle(s: pd.Series) -> pd.Series:
        t = s.fillna("未知").astype(str)
        out = pd.Series("other", index=t.index)
        out[t.str.contains("脉冲", na=False)] = "pulse"
        out[t.str.contains("中期", na=False)] = "mid"
        out[t.str.contains("长尾", na=False)] = "long"
        return out

    @staticmethod
    def _bucket_predictability(s: pd.Series) -> pd.Series:
        t = s.fillna("未知").astype(str)
        out = pd.Series("other", index=t.index)
        out[t.str.contains("突发", na=False)] = "sudden"
        out[t.str.contains("预披露|可预期|预告", na=False)] = "pre"
        return out

    def load_data(self):
        logger.info("加载数据源")
        self.events = self._read_table(self.paths.get("events"))
        self.ecm = self._read_table(self.paths.get("event_company_map"))
        self.universe = self._read_table(self.paths.get("a_share_universe"))
        self.stock_industry = self._read_table(self.paths.get("stock_industry_all"))
        self.stock_basic = self._read_table(self.paths.get("stock_basic_all"))
        self.hs300 = self._read_table(self.paths.get("hs300_constituents"))
        self.sz50 = self._read_table(self.paths.get("sz50_constituents"))
        self.zz500 = self._read_table(self.paths.get("zz500_constituents"))

        for df in [self.events, self.ecm, self.universe, self.stock_industry, self.stock_basic, self.hs300, self.sz50, self.zz500]:
            if not df.empty:
                df.columns = [str(c).replace("ï»¿", "").strip() for c in df.columns]

        required_events = {"event_id", "event_name", "event_date"}
        required_ecm = {"event_id", "stock_code", "market_code", "relation_strength", "relation_layer"}
        if not required_events.issubset(self.events.columns):
            raise ValueError(f"events.csv 缺少必要列: {sorted(required_events - set(self.events.columns))}")
        if not required_ecm.issubset(self.ecm.columns):
            raise ValueError(f"event_company_map.csv 缺少必要列: {sorted(required_ecm - set(self.ecm.columns))}")

        self.events["event_date"] = pd.to_datetime(self.events["event_date"], errors="coerce").dt.normalize()
        self.events = self.events.dropna(subset=["event_id", "event_date"]).copy()
        self.latest_event_date = self.events["event_date"].max()

        if "market_code" in self.ecm.columns:
            self.ecm["market_code"] = self.ecm["market_code"].apply(self._normalize_market_code)
        if "stock_code" in self.ecm.columns:
            self.ecm["stock_code"] = self.ecm["stock_code"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(6)
            self.ecm.loc[self.ecm["stock_code"] == "000nan", "stock_code"] = np.nan
        if "relation_strength" in self.ecm.columns:
            self.ecm["relation_strength"] = self._safe_numeric(self.ecm["relation_strength"], 0.0)

        for df in [self.universe, self.stock_industry, self.stock_basic, self.hs300, self.sz50, self.zz500]:
            if not df.empty and "code" in df.columns:
                df["code"] = df["code"].apply(self._normalize_market_code)

        if not self.stock_basic.empty and "ipoDate" in self.stock_basic.columns:
            self.stock_basic["ipoDate"] = pd.to_datetime(self.stock_basic["ipoDate"], errors="coerce").dt.normalize()

        self._load_market()
        logger.info(
            "数据加载完成 | events=%d | event_company_map=%d | market_rows=%d | latest_event_date=%s | latest_market_date=%s",
            len(self.events), len(self.ecm), len(self.market),
            self.latest_event_date.date() if pd.notna(self.latest_event_date) else "NA",
            self.latest_market_date.date() if pd.notna(self.latest_market_date) else "NA",
        )

    def _load_market(self):
        frames = []
        mdir = self.paths.get("daily_kline_batches_dir")
        mfile = self.paths.get("price_panel")

        if mdir and os.path.isdir(mdir):
            logger.info("读取公司股价")
            for name in sorted(os.listdir(mdir)):
                if name.lower().endswith(".csv"):
                    fp = os.path.join(mdir, name)
                    try:
                        frames.append(self._read_table(fp))
                    except Exception as e:
                        logger.warning("跳过无法读取的行情文件 %s: %s", fp, e)
        elif mfile and os.path.exists(mfile):
            logger.warning("未找到 daily_kline_batches 目录，回退使用 price_panel.csv；覆盖股票可能较少")
            frames = [self._read_table(mfile)]
        else:
            raise FileNotFoundError("未找到 daily_kline_batches 目录或 price_panel.csv")

        if not frames:
            raise ValueError("未读取到任何行情文件")

        mkt = pd.concat(frames, ignore_index=True, sort=False)
        mkt.columns = [str(c).strip() for c in mkt.columns]
        rename = {}
        if "date" in mkt.columns:
            rename["date"] = "trade_date"
        if "code" in mkt.columns:
            rename["code"] = "market_code"
        mkt = mkt.rename(columns=rename)

        required = {"trade_date", "market_code", "open", "close"}
        if not required.issubset(mkt.columns):
            raise ValueError(f"行情文件缺少必要列: {sorted(required - set(mkt.columns))}")

        mkt["trade_date"] = pd.to_datetime(mkt["trade_date"], errors="coerce").dt.normalize()
        mkt["market_code"] = mkt["market_code"].apply(self._normalize_market_code)
        for c in ["open", "high", "low", "close", "volume", "amount", "pctChg", "tradestatus"]:
            if c in mkt.columns:
                mkt[c] = self._safe_numeric(mkt[c], np.nan if c in {"open", "close"} else 0.0)

        if "tradestatus" in mkt.columns:
            mkt = mkt[mkt["tradestatus"].fillna(1) == 1].copy()
        mkt = mkt.dropna(subset=["trade_date", "market_code", "open", "close"]).copy()
        mkt = mkt[(mkt["open"] > 0) & (mkt["close"] > 0)].copy()
        mkt = mkt.sort_values(["market_code", "trade_date"]).reset_index(drop=True)
        self.market = mkt
        self.latest_market_date = mkt["trade_date"].max()

    def build_weekly_labels(self):
        df = self.market.copy()
        df["week_start"] = df["trade_date"] - pd.to_timedelta(df["trade_date"].dt.weekday, unit="D")
        df["weekday"] = df["trade_date"].dt.weekday

        buy = (
            df[df["weekday"] >= 1]
            .groupby(["market_code", "week_start"], as_index=False)
            .first()[["market_code", "week_start", "trade_date", "open"]]
            .rename(columns={"trade_date": "buy_date", "open": "buy_open"})
        )
        sell = (
            df[df["weekday"] <= 4]
            .groupby(["market_code", "week_start"], as_index=False)
            .last()[["market_code", "week_start", "trade_date", "close"]]
            .rename(columns={"trade_date": "sell_date", "close": "sell_close"})
        )
        weekly = buy.merge(sell, on=["market_code", "week_start"], how="inner")
        weekly["decision_tuesday"] = weekly["week_start"] + pd.Timedelta(days=1)
        weekly["target_return"] = (weekly["sell_close"] - weekly["buy_open"]) / weekly["buy_open"]
        weekly["target_return"] = weekly["target_return"].clip(-0.5, 0.5)
        self.weekly_returns = weekly[["decision_tuesday", "market_code", "target_return"]].copy()
        logger.info(
            "标签 | rows=%d | weeks=%d | codes=%d",
            len(self.weekly_returns),
            self.weekly_returns["decision_tuesday"].nunique(),
            self.weekly_returns["market_code"].nunique(),
        )

    def _attach_static(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self.use_industry_features:
            if not self.universe.empty and {"code", "industry"}.issubset(self.universe.columns):
                uni = self.universe[["code", "industry"]].drop_duplicates("code").rename(columns={"code": "market_code", "industry": "stock_industry_u"})
                out = out.merge(uni, on="market_code", how="left")
            if not self.stock_industry.empty and {"code", "industry"}.issubset(self.stock_industry.columns):
                ind = self.stock_industry[["code", "industry"]].drop_duplicates("code").rename(columns={"code": "market_code", "industry": "stock_industry_i"})
                out = out.merge(ind, on="market_code", how="left")
            out["stock_industry"] = out.get("stock_industry", pd.Series(index=out.index, dtype=object))
            out["stock_industry"] = out["stock_industry"].fillna(out.get("stock_industry_u")).fillna(out.get("stock_industry_i"))
            out["industry_match"] = (
                out.get("industry", pd.Series("", index=out.index)).fillna("").astype(str)
                == out.get("stock_industry", pd.Series("", index=out.index)).fillna("").astype(str)
            ).astype(int)
        else:
            out["stock_industry"] = "未知"
            out["industry_match"] = 0

        if self.use_stock_basic_features and not self.stock_basic.empty and {"code", "ipoDate"}.issubset(self.stock_basic.columns):
            basic = self.stock_basic[["code", "ipoDate"]].drop_duplicates("code").rename(columns={"code": "market_code", "ipoDate": "ipo_date"})
            out = out.merge(basic, on="market_code", how="left")
        else:
            out["ipo_date"] = pd.NaT

        for feat_name, idx_df in [("is_hs300", self.hs300), ("is_sz50", self.sz50), ("is_zz500", self.zz500)]:
            if self.use_index_features and (not idx_df.empty) and "code" in idx_df.columns:
                codes = set(idx_df["code"].dropna().astype(str))
                out[feat_name] = out["market_code"].astype(str).isin(codes).astype(int)
            else:
                out[feat_name] = 0
        return out

    def _build_price_snapshot(self):
        df = self.market.copy().sort_values(["market_code", "trade_date"]).reset_index(drop=True)
        df["daily_ret"] = df.groupby("market_code")["close"].pct_change()
        df["close_mom_5"] = df.groupby("market_code")["close"].pct_change(5)
        df["close_mom_20"] = df.groupby("market_code")["close"].pct_change(20)
        df["volatility_5"] = df.groupby("market_code")["daily_ret"].transform(lambda s: s.rolling(5, min_periods=3).std())
        df["volatility_20"] = df.groupby("market_code")["daily_ret"].transform(lambda s: s.rolling(20, min_periods=5).std())
        if "amount" in df.columns:
            df["amount_ma_5"] = df.groupby("market_code")["amount"].transform(lambda s: s.rolling(5, min_periods=3).mean())
            df["amount_ma_20"] = df.groupby("market_code")["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
            df["amount_ratio_5_20"] = df["amount_ma_5"] / (df["amount_ma_20"].replace(0, np.nan))
        else:
            df["amount_ratio_5_20"] = 1.0
        if "volume" in df.columns:
            df["volume_ma_5"] = df.groupby("market_code")["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
            df["volume_ma_20"] = df.groupby("market_code")["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
            df["volume_ratio_5_20"] = df["volume_ma_5"] / (df["volume_ma_20"].replace(0, np.nan))
        else:
            df["volume_ratio_5_20"] = 1.0
        df["last_pctchg"] = (df["pctChg"] / 100.0) if "pctChg" in df.columns else df["daily_ret"]
        self.price_snapshot = df[[
            "market_code", "trade_date", "close_mom_5", "close_mom_20",
            "volatility_5", "volatility_20", "amount_ratio_5_20",
            "volume_ratio_5_20", "last_pctchg"
        ]].copy()

    def _build_prev_week_context(self):
        weekly = self.weekly_returns.copy().sort_values(["market_code", "decision_tuesday"])
        stock_prev = weekly.rename(columns={"target_return": "prev_week_return"}).copy()
        stock_prev["decision_tuesday"] = stock_prev["decision_tuesday"] + pd.Timedelta(days=7)
        stock_prev["prev_week_up"] = (stock_prev["prev_week_return"] > 0).astype(int)
        stock_prev = stock_prev[["decision_tuesday", "market_code", "prev_week_return", "prev_week_up"]]

        market_prev = weekly.groupby("decision_tuesday")["target_return"].agg(
            market_prev_week_mean="mean",
            market_prev_week_std="std",
        ).reset_index()
        market_prev["market_prev_week_up_ratio"] = weekly.groupby("decision_tuesday")["target_return"].apply(lambda s: float((s > 0).mean())).values
        market_prev["decision_tuesday"] = market_prev["decision_tuesday"] + pd.Timedelta(days=7)
        return stock_prev, market_prev

    def _weighted_category_sums(self, df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
        tmp = df[group_keys + ["row_weight", "driver_bucket", "impact_bucket", "predict_bucket"]].copy()
        tmp["driver_bucket"] = tmp["driver_bucket"].fillna("other")
        tmp["impact_bucket"] = tmp["impact_bucket"].fillna("other")
        tmp["predict_bucket"] = tmp["predict_bucket"].fillna("other")

        driver = (
            tmp.pivot_table(index=group_keys, columns="driver_bucket", values="row_weight", aggfunc="sum", fill_value=0.0)
            .reset_index()
        )
        driver.columns = [f"driver_w_{c}" if c not in group_keys else c for c in driver.columns]

        impact = (
            tmp.pivot_table(index=group_keys, columns="impact_bucket", values="row_weight", aggfunc="sum", fill_value=0.0)
            .reset_index()
        )
        impact.columns = [f"impact_w_{c}" if c not in group_keys else c for c in impact.columns]

        pred = (
            tmp.pivot_table(index=group_keys, columns="predict_bucket", values="row_weight", aggfunc="sum", fill_value=0.0)
            .reset_index()
        )
        pred.columns = [f"predict_w_{c}" if c not in group_keys else c for c in pred.columns]

        out = driver.merge(impact, on=group_keys, how="outer").merge(pred, on=group_keys, how="outer")
        return out.fillna(0.0)

    def build_decision_samples(self):
        logger.info("构建决策周样本")
        df = self.ecm.copy().merge(self.events, on="event_id", how="left")
        df = df[df["event_date"].notna()].copy()
        df = self._attach_static(df)

        df["decision_tuesday"] = self._next_decision_tuesday(df["event_date"])
        df["decision_gap_days"] = (df["decision_tuesday"] - df["event_date"]).dt.days.clip(lower=1, upper=30)
        df["decay_to_decision"] = np.exp(-np.log(2) * df["decision_gap_days"] / 7.0)

        df["is_direct"] = df.get("relation_layer", pd.Series("", index=df.index)).astype(str).str.contains("直接", na=False).astype(int)
        df["is_indirect"] = df.get("relation_layer", pd.Series("", index=df.index)).astype(str).str.contains("间接", na=False).astype(int)

        event_name = df.get("event_name", pd.Series("", index=df.index)).astype(str)
        relation_type = df.get("relation_type", pd.Series("", index=df.index)).astype(str)
        note = df.get("note", pd.Series("", index=df.index)).astype(str)
        pos_mask = (
            event_name.str.contains("涨停|放量上涨|中标|订单|突破|增长|景气", na=False)
            | relation_type.str.contains("核心整机受益|核心供应商|直接投资|配套|受益", na=False)
            | note.str.contains("直接映射|受益|配套|供应", na=False)
        )
        neg_mask = (
            event_name.str.contains("跌停|减持|处罚|亏损|违约|下调", na=False)
            | relation_type.str.contains("风险|利空", na=False)
            | note.str.contains("风险|利空|减值", na=False)
        )
        df["event_sign"] = 0
        df.loc[pos_mask, "event_sign"] = 1
        df.loc[neg_mask, "event_sign"] = -1
        df["positive_event"] = (df["event_sign"] > 0).astype(int)
        df["negative_event"] = (df["event_sign"] < 0).astype(int)

        df["intensity_score"] = self._safe_numeric(df.get("intensity_score", 0.0), 0.0)
        df["relation_strength"] = self._safe_numeric(df.get("relation_strength", 0.0), 0.0)
        df["row_weight"] = np.maximum(df["relation_strength"], 0.05) * (1.0 + df["intensity_score"] / 5.0) * df["decay_to_decision"] * (1.0 + 0.5 * df["is_direct"])

        df["driver_bucket"] = self._bucket_driver_type(df.get("driver_type", pd.Series("未知", index=df.index)))
        df["impact_bucket"] = self._bucket_impact_cycle(df.get("impact_cycle", pd.Series("未知", index=df.index)))
        df["predict_bucket"] = self._bucket_predictability(df.get("predictability", pd.Series("未知", index=df.index)))

        group_keys = ["decision_tuesday", "market_code"]

        head = (
            df.sort_values(["decision_tuesday", "market_code", "row_weight", "event_date"], ascending=[True, True, False, False])
              .drop_duplicates(group_keys, keep="first")
              [["decision_tuesday", "market_code", "stock_code", "stock_name", "event_name", "event_date", "ipo_date"]]
              .rename(columns={"event_date": "last_event_date"})
        )

        counts = (
            df.groupby(group_keys, as_index=False)
              .agg(
                  event_count=("event_id", "size"),
                  unique_event_count=("event_id", "nunique"),
                  positive_event_count=("positive_event", "sum"),
                  negative_event_count=("negative_event", "sum"),
              )
        )

        num_aggs = (
            df.groupby(group_keys, as_index=False)
              .agg(
                  row_weight_sum=("row_weight", "sum"),
                  row_weight_max=("row_weight", "max"),
                  row_weight_mean=("row_weight", "mean"),
                  relation_strength_max=("relation_strength", "max"),
                  relation_strength_mean=("relation_strength", "mean"),
                  intensity_score_max=("intensity_score", "max"),
                  intensity_score_mean=("intensity_score", "mean"),
                  industry_match_mean=("industry_match", "mean"),
                  industry_match_max=("industry_match", "max"),
                  event_sign_sum=("event_sign", "sum"),
                  event_sign_mean=("event_sign", "mean"),
                  decision_gap_days_min=("decision_gap_days", "min"),
                  decision_gap_days_mean=("decision_gap_days", "mean"),
                  decay_to_decision_max=("decay_to_decision", "max"),
                  decay_to_decision_mean=("decay_to_decision", "mean"),
                  direct_event_share=("is_direct", "mean"),
                  indirect_event_share=("is_indirect", "mean"),
                  direct_event_count=("is_direct", "sum"),
                  indirect_event_count=("is_indirect", "sum"),
                  is_hs300=("is_hs300", "max"),
                  is_sz50=("is_sz50", "max"),
                  is_zz500=("is_zz500", "max"),
              )
        )

        cat_w = self._weighted_category_sums(df, group_keys)

        feat = head.merge(counts, on=group_keys, how="left")
        feat = feat.merge(num_aggs, on=group_keys, how="left")
        feat = feat.merge(cat_w, on=group_keys, how="left")

        self._build_price_snapshot()
        snap = self.price_snapshot.sort_values(["market_code", "trade_date"]).copy()
        req = feat[["decision_tuesday", "market_code"]].drop_duplicates().sort_values(["market_code", "decision_tuesday"]).copy()
        snap_parts = []
        for code, left_grp in req.groupby("market_code", sort=False):
            right_grp = snap[snap["market_code"] == code].copy()
            if right_grp.empty:
                temp = left_grp.copy()
                for c in ["close_mom_5", "close_mom_20", "volatility_5", "volatility_20", "amount_ratio_5_20", "volume_ratio_5_20", "last_pctchg"]:
                    temp[c] = np.nan
                snap_parts.append(temp)
                continue
            merged = pd.merge_asof(
                left_grp.sort_values("decision_tuesday"),
                right_grp.drop(columns=["market_code"], errors="ignore").sort_values("trade_date"),
                left_on="decision_tuesday",
                right_on="trade_date",
                direction="backward",
                allow_exact_matches=False,
            )
            merged["market_code"] = code
            merged = merged.drop(columns=["trade_date"], errors="ignore")
            snap_parts.append(merged)
        snap_feat = pd.concat(snap_parts, ignore_index=True, sort=False)
        feat = feat.merge(snap_feat, on=["decision_tuesday", "market_code"], how="left")

        if self.use_stock_basic_features and "ipo_date" in feat.columns:
            feat["listed_days"] = (pd.to_datetime(feat["decision_tuesday"]) - pd.to_datetime(feat["ipo_date"])).dt.days.clip(lower=0)
            feat["is_recent_ipo_180d"] = (feat["listed_days"] <= 180).astype(float)
            feat["is_recent_ipo_365d"] = (feat["listed_days"] <= 365).astype(float)
        else:
            feat["listed_days"] = np.nan
            feat["is_recent_ipo_180d"] = 0.0
            feat["is_recent_ipo_365d"] = 0.0

        stock_prev, market_prev = self._build_prev_week_context()
        feat = feat.merge(stock_prev, on=["decision_tuesday", "market_code"], how="left")
        feat = feat.merge(market_prev, on="decision_tuesday", how="left")
        feat = feat.merge(self.weekly_returns, on=["decision_tuesday", "market_code"], how="left")

        defaults = [
            ("prev_week_return", 0.0),
            ("prev_week_up", 0.0),
            ("market_prev_week_mean", 0.0),
            ("market_prev_week_std", 0.0),
            ("market_prev_week_up_ratio", 0.5),
            ("close_mom_5", 0.0),
            ("close_mom_20", 0.0),
            ("volatility_5", 0.0),
            ("volatility_20", 0.0),
            ("amount_ratio_5_20", 1.0),
            ("volume_ratio_5_20", 1.0),
            ("last_pctchg", 0.0),
            ("listed_days", 3650.0),
        ]
        for col, default in defaults:
            if col in feat.columns:
                feat[col] = feat[col].fillna(default)

        feat = feat.sort_values(["decision_tuesday", "market_code"]).reset_index(drop=True)
        self.decision_samples = feat

        compact_features = [
            "event_count", "unique_event_count", "positive_event_count", "negative_event_count",
            "row_weight_sum", "row_weight_max", "row_weight_mean",
            "relation_strength_max", "relation_strength_mean",
            "intensity_score_max", "intensity_score_mean",
            "industry_match_mean", "industry_match_max",
            "event_sign_sum", "event_sign_mean",
            "decision_gap_days_min", "decision_gap_days_mean",
            "decay_to_decision_max", "decay_to_decision_mean",
            "direct_event_share", "indirect_event_share",
            "direct_event_count", "indirect_event_count",
            "driver_w_company", "driver_w_industry", "driver_w_macro", "driver_w_geo", "driver_w_policy",
            "impact_w_pulse", "impact_w_mid", "impact_w_long",
            "predict_w_sudden", "predict_w_pre",
            "close_mom_5", "close_mom_20", "volatility_5", "volatility_20",
            "amount_ratio_5_20", "volume_ratio_5_20", "last_pctchg",
            "prev_week_return", "prev_week_up",
            "market_prev_week_mean", "market_prev_week_std", "market_prev_week_up_ratio",
            "listed_days", "is_recent_ipo_180d", "is_recent_ipo_365d",
            "is_hs300", "is_sz50", "is_zz500",
        ]
        self.feature_columns = [c for c in compact_features if c in feat.columns]
        if not self.use_index_features:
            self.feature_columns = [c for c in self.feature_columns if c not in {"is_hs300", "is_sz50", "is_zz500"}]
        if not self.use_stock_basic_features:
            self.feature_columns = [c for c in self.feature_columns if c not in {"listed_days", "is_recent_ipo_180d", "is_recent_ipo_365d"}]
        if not self.use_industry_features:
            self.feature_columns = [c for c in self.feature_columns if c not in {"industry_match_mean", "industry_match_max"}]

        logger.info(
            "决策周样本 | rows=%d | labeled_rows=%d | weeks=%d | features=%d",
            len(feat), feat["target_return"].notna().sum(), feat["decision_tuesday"].nunique(), len(self.feature_columns)
        )

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.feature_columns].copy()
        for col in x.columns:
            if not pd.api.types.is_numeric_dtype(x[col]):
                x[col] = pd.to_numeric(x[col], errors="coerce")
        return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _new_models(self):
        reg = xgb.XGBRegressor(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=8,
            subsample=0.72,
            colsample_bytree=0.72,
            reg_alpha=0.8,
            reg_lambda=3.0,
            gamma=0.2,
            random_state=42,
            n_jobs=2,
            tree_method="hist",
        )
        cls = xgb.XGBClassifier(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=8,
            subsample=0.72,
            colsample_bytree=0.72,
            reg_alpha=0.8,
            reg_lambda=3.0,
            gamma=0.2,
            random_state=42,
            n_jobs=2,
            tree_method="hist",
            eval_metric="logloss",
        )
        return reg, cls

    def _prepare_prediction_date(self, requested_date: str, allow_roll_back: bool = True) -> pd.Timestamp:
        requested = self._align_to_tuesday(pd.to_datetime(requested_date))
        available = sorted(pd.to_datetime(self.decision_samples["decision_tuesday"].dropna().unique()))
        if not available:
            raise ValueError("无可用决策周样本")
        if requested in available:
            return requested
        if not allow_roll_back:
            return requested
        prior = [d for d in available if d <= requested]
        return prior[-1] if prior else available[-1]

    def _walk_forward_validate(self, decision_date: pd.Timestamp, min_train_weeks: Optional[int] = None, max_val_weeks: int = 10):
        if min_train_weeks is None:
            min_train_weeks = self.min_train_weeks
        base = self.decision_samples.copy()
        base = base[(base["decision_tuesday"] < decision_date) & (base["target_return"].notna())].copy()
        weeks = sorted(pd.to_datetime(base["decision_tuesday"].dropna().unique()))
        if len(weeks) <= min_train_weeks:
            return None
        eval_weeks = weeks[-max_val_weeks:]
        preds_all = []
        for wk in eval_weeks:
            tr = base[base["decision_tuesday"] < wk].copy()
            va = base[base["decision_tuesday"] == wk].copy()
            if tr["decision_tuesday"].nunique() < min_train_weeks or va.empty:
                continue
            X_tr = self._make_features(tr)
            X_va = self._make_features(va)
            y_tr = self._safe_numeric(tr["target_return"], 0.0)
            y_va = self._safe_numeric(va["target_return"], 0.0)
            y_up_tr = (y_tr > 0).astype(int)
            reg, cls = self._new_models()
            reg.fit(X_tr, y_tr)
            if y_up_tr.nunique() < 2:
                cls = ConstantProbClassifier(float(y_up_tr.mean()))
            else:
                cls.fit(X_tr, y_up_tr)
            temp = va[["decision_tuesday", "market_code", "target_return"]].copy()
            temp["pred_ret"] = reg.predict(X_va)
            temp["prob_up"] = cls.predict_proba(X_va)[:, 1]
            preds_all.append(temp)

        if not preds_all:
            return None
        eval_df = pd.concat(preds_all, ignore_index=True)
        y = self._safe_numeric(eval_df["target_return"], 0.0)
        p = self._safe_numeric(eval_df["pred_ret"], 0.0)
        prob = self._safe_numeric(eval_df["prob_up"], 0.5)
        y_up = (y > 0).astype(int)
        auc = roc_auc_score(y_up, prob) if y_up.nunique() > 1 else 0.5
        r2 = r2_score(y, p) if y.std() > 0 else 0.0
        rank_ic = np.corrcoef(p, y)[0, 1] if np.std(p) > 0 and np.std(y) > 0 else 0.0
        hit = float(((prob >= 0.5).astype(int) == y_up).mean())
        scale = np.nanmedian(np.abs(p)) + 1e-6
        eval_df["selection_score"] = 0.60 * prob + 0.40 * (p / scale)
        top3 = []
        for _, grp in eval_df.groupby("decision_tuesday"):
            top3.append(grp.sort_values("selection_score", ascending=False).head(3)["target_return"].mean())
        top3_mean = float(np.mean(top3)) if top3 else 0.0
        return {
            "auc": auc, "r2": r2, "rank_ic": rank_ic, "hit": hit,
            "top3_mean": top3_mean, "weeks": eval_df["decision_tuesday"].nunique(),
        }

    def fit_as_of(self, decision_date: pd.Timestamp):
        decision_date = pd.to_datetime(decision_date).normalize()
        trainable = self.decision_samples[
            (self.decision_samples["decision_tuesday"] < decision_date) & self.decision_samples["target_return"].notna()
        ].copy()
        train_weeks = trainable["decision_tuesday"].nunique()
        effective_min_train_weeks = min(self.min_train_weeks, max(4, train_weeks - 1))
        if train_weeks < 4:
            raise ValueError(f"截至 {decision_date.date()} 可用训练周数过少：{train_weeks}")
        if train_weeks < self.min_train_weeks:
            logger.warning("截至 %s 可用训练周数仅 %d，低于默认 %d，自动降到 %d 周继续训练", decision_date.date(), train_weeks, self.min_train_weeks, effective_min_train_weeks)

        wf = self._walk_forward_validate(decision_date, min_train_weeks=effective_min_train_weeks)
        if wf is not None:
            logger.info(
                " 周数=%d | AUC=%.4f | R2=%.4f | RankIC=%.4f | 上涨命中率=%.4f | Top3周均收益=%.4f",
                wf["weeks"], wf["auc"], wf["r2"], wf["rank_ic"], wf["hit"], wf["top3_mean"]
            )

        X_train = self._make_features(trainable)
        y_train = self._safe_numeric(trainable["target_return"], 0.0)
        y_up_train = (y_train > 0).astype(int)
        self.model_reg, self.model_cls = self._new_models()
        logger.info(
            " decision_date=%s | rows=%d | train_weeks=%d | min_train_weeks=%d | features=%d",
            decision_date.date(), len(trainable), train_weeks, effective_min_train_weeks, len(self.feature_columns)
        )
        self.model_reg.fit(X_train, y_train)
        if y_up_train.nunique() < 2:
            self.model_cls = ConstantProbClassifier(float(y_up_train.mean()))
        else:
            self.model_cls.fit(X_train, y_up_train)

    def generate_decision(
        self,
        target_date_str: str,
        output_file: str = "result.xlsx",
        min_prob_up: float = 0.50,
        max_stocks: int = 3,
        save_debug_csv: Optional[str] = None,
        allow_roll_back: bool = True,
    ):
        effective_date = self._prepare_prediction_date(target_date_str, allow_roll_back=allow_roll_back)
        requested_aligned = self._align_to_tuesday(pd.to_datetime(target_date_str))
        self.effective_prediction_date = effective_date
        if effective_date != requested_aligned:
            logger.warning("请求日期 %s 已回退到最近有样本的决策周二 %s", requested_aligned.date(), effective_date.date())

        self.fit_as_of(effective_date)

        pred_df = self.decision_samples[self.decision_samples["decision_tuesday"] == effective_date].copy()
        if pred_df.empty:
            logger.warning("目标决策周没有可用事件候选，未生成结果文件")
            return None

        X_pred = self._make_features(pred_df)
        pred_df["pred_ret"] = self.model_reg.predict(X_pred)
        pred_df["prob_up"] = self.model_cls.predict_proba(X_pred)[:, 1]
        score_scale = np.nanmedian(np.abs(pred_df["pred_ret"])) + 1e-6
        weight_norm = pred_df.get("row_weight_sum", pd.Series(0.0, index=pred_df.index))
        weight_norm = weight_norm / (weight_norm.max() + 1e-6)
        pred_df["selection_score"] = (
            0.58 * pred_df["prob_up"] +
            0.22 * (pred_df["pred_ret"] / score_scale) +
            0.12 * weight_norm +
            0.08 * pred_df.get("prev_week_return", 0.0)
        )

        selected = pred_df[pred_df["prob_up"] >= min_prob_up].copy()
        if selected.empty:
            logger.warning("按 min_prob_up=%.2f 过滤后为空，自动回退为综合分数 Top%d", min_prob_up, max_stocks)
            selected = pred_df.copy()

        selected = selected.sort_values("selection_score", ascending=False).head(max_stocks).copy()
        if selected.empty:
            logger.warning("最终候选为空，未生成结果文件")
            return None

        raw = selected["selection_score"].astype(float).values
        raw = raw - np.nanmax(raw)
        w = np.exp(raw)
        if not np.isfinite(w).all() or w.sum() <= 0:
            base = np.array([0.5, 0.3, 0.2][: len(selected)], dtype=float)
            w = base / base.sum()
        else:
            w = w / w.sum()
        selected["资金比例"] = w

        out = pd.DataFrame({
            "事件名称": selected["event_name"].astype(str).values,
            "标的（股票）代码": selected["market_code"].astype(str).values,
            "资金比例": np.round(selected["资金比例"].values, 4),
        })
        if len(out) > 0:
            out.loc[out.index[-1], "资金比例"] = round(1.0 - out["资金比例"].iloc[:-1].sum(), 4)

        out.to_excel(output_file, index=False)
        logger.info("结果文件: %s | 使用决策周二=%s", os.path.abspath(output_file), effective_date.date())

        if save_debug_csv:
            cols = [
                "decision_tuesday", "event_name", "market_code", "stock_name", "event_count", "unique_event_count",
                "row_weight_sum", "relation_strength_mean", "intensity_score_mean", "industry_match_mean",
                "close_mom_5", "close_mom_20", "volatility_5", "volatility_20",
                "prev_week_return", "market_prev_week_mean", "pred_ret", "prob_up", "selection_score"
            ]
            cols = [c for c in cols if c in pred_df.columns]
            pred_df.sort_values("selection_score", ascending=False)[cols].to_csv(save_debug_csv, index=False, encoding="utf-8-sig")
            logger.info("调试文件导出: %s", os.path.abspath(save_debug_csv))
        return out


def auto_detect_paths(root_dir: str) -> dict:
    files = {
        "events": "events.csv",
        "event_company_map": "event_company_map.csv",
        "price_panel": "price_panel.csv",
        "a_share_universe": "a_share_universe.csv",
        "stock_industry_all": "stock_industry_all.csv",
        "stock_basic_all": "stock_basic_all.csv",
        "hs300_constituents": "hs300_constituents.csv",
        "sz50_constituents": "sz50_constituents.csv",
        "zz500_constituents": "zz500_constituents.csv",
    }
    out = {}
    for key, fname in files.items():
        found = None
        for base, _, names in os.walk(root_dir):
            if fname in names:
                found = os.path.join(base, fname)
                break
        out[key] = found

    daily_dir = None
    for base, dirs, _ in os.walk(root_dir):
        if "daily_kline_batches" in dirs:
            daily_dir = os.path.join(base, "daily_kline_batches")
            break
    out["daily_kline_batches_dir"] = daily_dir
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="事件+关联图谱+股价 压缩特征严格版")
    parser.add_argument("--root-dir", type=str, default=".", help="数据根目录，会递归查找所需文件")
    parser.add_argument("--target-date", type=str, required=True, help="目标日期，会自动对齐到周二")
    parser.add_argument("--output", type=str, default="result.xlsx", help="输出 Excel 文件")
    parser.add_argument("--min-prob-up", type=float, default=0.50, help="最低上涨概率阈值")
    parser.add_argument("--max-stocks", type=int, default=3, help="最多输出股票数")
    parser.add_argument("--save-debug-csv", type=str, default="", help="可选，导出调试 CSV")
    parser.add_argument("--allow-roll-back", action="store_true", help="当目标周无样本时，允许回退到最近可用周")
    parser.add_argument("--min-train-weeks", type=int, default=12, help="至少使用多少个历史决策周训练")
    parser.add_argument("--use-index-features", action="store_true", help="显式启用指数成分特征；默认关闭")
    parser.add_argument("--disable-industry-features", action="store_true", help="关闭行业辅助特征")
    parser.add_argument("--disable-stock-basic-features", action="store_true", help="关闭股票基础信息辅助特征")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("目录: %s", os.path.abspath(args.root_dir))
    paths = auto_detect_paths(args.root_dir)
    for k, v in paths.items():
        if v:
            logger.info("%s: %s", k, v)

    must_have = ["events", "event_company_map"]
    missing = [k for k in must_have if not paths.get(k)]
    if missing:
        raise FileNotFoundError(f"缺少必要文件: {missing}")
    if not paths.get("daily_kline_batches_dir") and not paths.get("price_panel"):
        raise FileNotFoundError("缺少 daily_kline_batches 目录和 price_panel.csv，至少要有一类股价数据")

    strategy = EventGraphPriceCompactStrategy(paths)
    strategy.min_train_weeks = max(4, int(args.min_train_weeks))
    strategy.use_index_features = bool(args.use_index_features)
    strategy.use_industry_features = not bool(args.disable_industry_features)
    strategy.use_stock_basic_features = not bool(args.disable_stock_basic_features)

    strategy.load_data()
    strategy.build_weekly_labels()
    strategy.build_decision_samples()
    strategy.generate_decision(
        target_date_str=args.target_date,
        output_file=args.output,
        min_prob_up=args.min_prob_up,
        max_stocks=args.max_stocks,
        save_debug_csv=args.save_debug_csv or None,
        allow_roll_back=args.allow_roll_back,
    )


if __name__ == "__main__":
    main()
