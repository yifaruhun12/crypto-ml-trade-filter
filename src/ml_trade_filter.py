#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用多模型对均值回归候选交易进行滚动样本外过滤。"""

import gc
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


BACKTEST_PATH = ROOT / "backtest_engine.py"
MODEL_THRESHOLD = 0.55
LABEL_HOURS = 84


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("mean_reversion_backtest", BACKTEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    for _, group in panel.groupby("symbol", sort=False):
        group = group.sort_values("dt").copy()
        close_return = group["close"].pct_change()
        previous_close = group["close"].shift(1)
        true_range = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - previous_close).abs(),
                (group["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        short_volume = group["volume"].rolling(6, min_periods=6).mean()
        long_volume = group["volume"].rolling(42, min_periods=42).mean()
        group["f_ret_1"] = group["close"].pct_change(1).shift(1)
        group["f_ret_6"] = group["close"].pct_change(6).shift(1)
        group["f_ret_18"] = group["close"].pct_change(18).shift(1)
        group["f_ret_42"] = group["close"].pct_change(42).shift(1)
        group["f_vol_6"] = close_return.rolling(6, min_periods=6).std().shift(1)
        group["f_vol_18"] = close_return.rolling(18, min_periods=18).std().shift(1)
        group["f_vol_42"] = close_return.rolling(42, min_periods=42).std().shift(1)
        group["f_atr_14"] = (true_range.rolling(14, min_periods=14).mean() / group["close"]).shift(1)
        group["f_volume_ratio"] = (short_volume / long_volume).shift(1)
        group["f_d_slope"] = group["D"].diff().shift(1)
        group["f_w_slope"] = group["w"].diff().shift(1)
        parts.append(group)

    featured = pd.concat(parts, ignore_index=True).sort_values(["dt", "symbol"]).reset_index(drop=True)
    featured["row_id"] = np.arange(len(featured))
    cross_section = featured.groupby("dt").agg(
        f_market_breadth=("f_ret_6", lambda values: float((values > 0).mean())),
        f_market_dispersion=("f_ret_6", "std"),
        f_market_median_return=("f_ret_6", "median"),
    )
    featured = featured.join(cross_section, on="dt")

    btc = featured[featured["symbol"].eq("BTC/USDT")][
        ["dt", "f_ret_6", "f_ret_18", "f_vol_18"]
    ].rename(
        columns={
            "f_ret_6": "f_btc_ret_6",
            "f_ret_18": "f_btc_ret_18",
            "f_vol_18": "f_btc_vol_18",
        }
    )
    featured = featured.merge(btc, on="dt", how="left")
    featured["f_side"] = np.where(featured["signal_buy"], 1.0, -1.0)
    featured["f_signal_d"] = featured["signal_D"]
    featured["f_signal_w"] = featured["signal_w"]
    featured["f_signal_lev"] = featured["signal_lev"]

    feature_columns = [
        "f_side",
        "f_signal_d",
        "f_signal_w",
        "f_signal_lev",
        "f_ret_1",
        "f_ret_6",
        "f_ret_18",
        "f_ret_42",
        "f_vol_6",
        "f_vol_18",
        "f_vol_42",
        "f_atr_14",
        "f_volume_ratio",
        "f_d_slope",
        "f_w_slope",
        "f_market_breadth",
        "f_market_dispersion",
        "f_market_median_return",
        "f_btc_ret_6",
        "f_btc_ret_18",
        "f_btc_vol_18",
    ]
    return featured, feature_columns


def label_candidates(featured: pd.DataFrame, module) -> pd.DataFrame:
    labeled_parts = []
    round_trip_cost = 2.0 * (module.FEE_RATE + module.SLIPPAGE_RATE)
    for symbol, group in featured.groupby("symbol", sort=False):
        group = group.sort_values("dt").reset_index(drop=True)
        candidate_positions = np.flatnonzero((group["signal_buy"] | group["signal_sell"]).to_numpy())
        records = []
        for position in candidate_positions:
            if position + LABEL_HOURS // module.BAR_HOURS >= len(group):
                continue
            entry = group.iloc[position]
            side = "LONG" if bool(entry["signal_buy"]) else "SHORT"
            sign = 1.0 if side == "LONG" else -1.0
            entry_price = float(entry["open"])
            exit_row = None
            exit_reason = None
            max_steps = LABEL_HOURS // module.BAR_HOURS
            for step in range(1, max_steps + 1):
                current = group.iloc[position + step]
                signed_return = sign * (float(current["open"]) / entry_price - 1.0)
                if signed_return <= -module.STOP_LOSS_RETURN:
                    exit_row = current
                    exit_reason = "stop_loss"
                    break
                if step == max_steps:
                    exit_row = current
                    exit_reason = "timeout"
                    break
                signal_d = current["signal_D"]
                signal_w = current["signal_w"]
                if side == "LONG" and (
                    signal_d >= module.MEAN_EXIT_D or signal_w >= module.MEAN_EXIT_W
                ):
                    exit_row = current
                    exit_reason = "mean_reversion"
                    break
                if side == "SHORT" and (
                    signal_d <= module.MEAN_EXIT_D or signal_w <= module.MEAN_EXIT_W
                ):
                    exit_row = current
                    exit_reason = "mean_reversion"
                    break
            holding_hours = (exit_row["dt"] - entry["dt"]).total_seconds() / 3600.0
            carry = (
                module.SHORT_CARRY_RATE_ANNUAL * holding_hours / (365 * 24)
                if side == "SHORT"
                else 0.0
            )
            price_return = sign * (float(exit_row["open"]) / entry_price - 1.0)
            net_return = price_return - round_trip_cost - carry
            records.append(
                {
                    "row_id": int(entry["row_id"]),
                    "symbol": symbol,
                    "dt": entry["dt"],
                    "side": side,
                    "target_return": net_return,
                    "target_win": int(net_return > 0),
                    "label_exit_reason": exit_reason,
                    "label_holding_hours": holding_hours,
                }
            )
        labeled_parts.append(pd.DataFrame(records))
    labels = pd.concat(labeled_parts, ignore_index=True)
    return labels.merge(featured, on=["row_id", "symbol", "dt"], how="left")


def model_factories():
    return {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=15,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=3,
                        learning_rate=0.03,
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        n_jobs=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=350,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=4,
                        min_child_samples=20,
                        reg_lambda=2.0,
                        class_weight="balanced",
                        verbosity=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def walk_forward_predictions(samples: pd.DataFrame, feature_columns: list[str]):
    start = pd.Timestamp("2025-02-20 04:00:00", tz="UTC")
    end = samples["dt"].max()
    boundaries = pd.to_datetime(
        np.linspace(start.value, end.value + 1, 5).astype("int64"), utc=True
    ).floor("4h")
    predictions = []
    fold_rows = []
    for model_name, estimator in model_factories().items():
        for fold in range(4):
            test_start = boundaries[fold]
            test_end = boundaries[fold + 1]
            train_end = test_start - pd.Timedelta(hours=LABEL_HOURS)
            train = samples[samples["dt"].lt(train_end)]
            test = samples[samples["dt"].ge(test_start) & samples["dt"].lt(test_end)]
            fitted = clone(estimator)
            fitted.fit(train[feature_columns], train["target_win"])
            score = fitted.predict_proba(test[feature_columns])[:, 1]
            prediction = test[["row_id", "dt", "symbol", "side", "target_return", "target_win"]].copy()
            prediction["model"] = model_name
            prediction["fold"] = fold + 1
            prediction["ml_score"] = score
            predictions.append(prediction)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold + 1,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "positive_rate": test["target_win"].mean(),
                    "roc_auc": roc_auc_score(test["target_win"], score),
                    "average_precision": average_precision_score(test["target_win"], score),
                    "brier_score": brier_score_loss(test["target_win"], score),
                }
            )
            del fitted
            gc.collect()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows), boundaries


def portfolio_comparison(featured, raw, predictions, fold_metrics, boundaries, module):
    evaluation_start = boundaries[0]
    evaluation_end = boundaries[-1]
    evaluation_panel = featured[
        featured["dt"].ge(evaluation_start) & featured["dt"].lt(evaluation_end)
    ].copy()
    evaluation_raw = raw[raw["dt"].ge(evaluation_start) & raw["dt"].lt(evaluation_end)].copy()
    result_rows = []

    module.ML_SCORE_THRESHOLD = None
    baseline_curve, baseline_trades, baseline_summaries, _ = module.run_backtest(evaluation_panel)
    baseline = module.calculate_metrics(
        baseline_curve, baseline_trades, baseline_summaries, evaluation_raw
    ).iloc[0]
    result_rows.append(
        {
            "model": "baseline_no_filter",
            "threshold": np.nan,
            "total_return": baseline.total_return,
            "sharpe": baseline.sharpe,
            "max_drawdown": baseline.max_drawdown,
            "trades": int(baseline.completed_trades),
            "win_rate": baseline.win_rate,
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "brier_score": np.nan,
        }
    )

    for model_name in predictions["model"].unique():
        model_predictions = predictions[predictions["model"].eq(model_name)][
            ["row_id", "ml_score"]
        ]
        scored_panel = evaluation_panel.merge(model_predictions, on="row_id", how="left")
        module.ML_SCORE_THRESHOLD = MODEL_THRESHOLD
        curve, trades, summaries, _ = module.run_backtest(scored_panel)
        metric = module.calculate_metrics(curve, trades, summaries, evaluation_raw).iloc[0]
        model_fold_metrics = fold_metrics[fold_metrics["model"].eq(model_name)]
        result_rows.append(
            {
                "model": model_name,
                "threshold": MODEL_THRESHOLD,
                "total_return": metric.total_return,
                "sharpe": metric.sharpe,
                "max_drawdown": metric.max_drawdown,
                "trades": int(metric.completed_trades),
                "win_rate": metric.win_rate,
                "roc_auc": model_fold_metrics["roc_auc"].mean(),
                "average_precision": model_fold_metrics["average_precision"].mean(),
                "brier_score": model_fold_metrics["brier_score"].mean(),
            }
        )
        del scored_panel, curve, trades, summaries
        gc.collect()
    module.ML_SCORE_THRESHOLD = None
    return pd.DataFrame(result_rows).sort_values("total_return", ascending=False)


def main():
    module = load_backtest_module()
    module.POSITION_SIZING_MODE = "equity_fraction"
    module.POSITION_EQUITY_FRACTION = 0.075
    raw = module.load_market_data(module.DATA_PATH)
    panel = module.prepare_panel(raw)
    featured, feature_columns = add_features(panel)
    samples = label_candidates(featured, module)
    predictions, fold_metrics, boundaries = walk_forward_predictions(samples, feature_columns)
    comparison = portfolio_comparison(
        featured, raw, predictions, fold_metrics, boundaries, module
    )

    samples.to_csv(ROOT / "ml_trade_samples.csv", index=False)
    predictions.to_csv(ROOT / "ml_oos_predictions.csv", index=False)
    fold_metrics.to_csv(ROOT / "ml_fold_metrics.csv", index=False)
    comparison.to_csv(ROOT / "ml_model_comparison.csv", index=False)

    print(f"Samples: {len(samples):,}, positive rate: {samples['target_win'].mean():.2%}")
    print(f"OOS range: {boundaries[0]} -> {boundaries[-1]}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
