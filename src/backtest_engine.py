#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4H 加密货币均值回归策略回测。"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().with_name("top100_4h_730d.csv")
OUTPUT_DIR = DATA_PATH.parent

WINDOW_TECH = 35
D_WINDOW = 21
SHORT_VOLUME_WINDOW = 6
LONG_VOLUME_WINDOW = 180

INITIAL_EQUITY = 300_000.0
SLOT_SIZE = 15_000.0
POSITION_SIZING_MODE = "equity_fraction"
POSITION_EQUITY_FRACTION = 0.075
ML_SCORE_THRESHOLD = None
FEE_RATE = 0.0008
SLIPPAGE_RATE = 0.0005
SHORT_CARRY_RATE_ANNUAL = 0.10
MAX_HOLD_HOURS = 84
MAX_NEW_PER_BAR_PER_SIDE = 3
STOP_LOSS_RETURN = 0.40

LONG_ENTRY_D = 10.0
LONG_ENTRY_W = -95.0
SHORT_ENTRY_D = 90.0
SHORT_ENTRY_W = -5.0
MEAN_EXIT_D = 50.0
MEAN_EXIT_W = -40.0

BAR_HOURS = 4
BARS_PER_YEAR = 6 * 365
EPSILON = 1e-12


def load_market_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"datetime_utc", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"输入数据缺列：{sorted(missing)}")

    frame["dt"] = pd.to_datetime(frame["datetime_utc"], utc=True, errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame[["dt", "symbol", *numeric]].isna().any().any():
        raise ValueError("输入数据存在无法解析的时间、标的或数值")
    if frame.duplicated(["symbol", "dt"]).any():
        raise ValueError("输入数据存在重复的 symbol/datetime")

    invalid_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame[["open", "high", "low", "close", "volume"]] < 0).any(axis=1)
    )
    if invalid_ohlc.any():
        raise ValueError(f"输入数据存在 {int(invalid_ohlc.sum())} 行异常 OHLCV")

    frame = frame.sort_values(["symbol", "dt"]).reset_index(drop=True)
    expected_delta = pd.Timedelta(hours=BAR_HOURS)
    deltas = frame.groupby("symbol")["dt"].diff().dropna()
    if not deltas.eq(expected_delta).all():
        raise ValueError(f"输入数据存在 {int((deltas != expected_delta).sum())} 个非 4H 断点")

    end_by_symbol = frame.groupby("symbol")["dt"].max()
    if end_by_symbol.nunique() != 1:
        raise ValueError("各标的结束时间不一致，禁止在缺失尾部行情上回测")
    return frame


def compute_indicators(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("dt").copy()
    rolling_high = group["high"].rolling(WINDOW_TECH, min_periods=WINDOW_TECH).max()
    rolling_low = group["low"].rolling(WINDOW_TECH, min_periods=WINDOW_TECH).min()
    price_range = (rolling_high - rolling_low).replace(0, np.nan)

    group["K"] = 100.0 * (group["close"] - rolling_low) / price_range
    group["D"] = group["K"].rolling(D_WINDOW, min_periods=D_WINDOW).mean()
    group["w"] = 100.0 * (group["close"] - rolling_high) / price_range

    short_volume = group["volume"].rolling(
        SHORT_VOLUME_WINDOW, min_periods=SHORT_VOLUME_WINDOW
    ).mean()
    long_volume = group["volume"].rolling(
        LONG_VOLUME_WINDOW, min_periods=LONG_VOLUME_WINDOW
    ).mean()
    leverage = pd.Series(np.nan, index=group.index, dtype=float)
    ready = long_volume.notna()
    leverage.loc[ready] = 1.0
    leverage.loc[ready & short_volume.ge(long_volume) & short_volume.lt(long_volume * 1.2)] = 2.0
    leverage.loc[ready & short_volume.ge(long_volume * 1.2)] = 3.0
    group["lev"] = leverage

    group["buy"] = group["D"].le(LONG_ENTRY_D) & group["w"].le(LONG_ENTRY_W) & ready
    group["sell"] = group["D"].ge(SHORT_ENTRY_D) & group["w"].ge(SHORT_ENTRY_W) & ready

    lag_columns = ["D", "w", "lev", "buy", "sell"]
    group[[f"signal_{column}" for column in lag_columns]] = group[lag_columns].shift(1)
    group["signal_buy"] = group["signal_buy"].eq(True)
    group["signal_sell"] = group["signal_sell"].eq(True)
    return group


def prepare_panel(raw: pd.DataFrame) -> pd.DataFrame:
    parts = [compute_indicators(group) for _, group in raw.groupby("symbol", sort=False)]
    return pd.concat(parts, ignore_index=True).sort_values(["dt", "symbol"])


def select_candidates(frame: pd.DataFrame, side: str, limit: int, excluded: set[str]) -> list[dict]:
    signal_column = "signal_buy" if side == "LONG" else "signal_sell"
    candidates = frame[frame[signal_column] & ~frame["symbol"].isin(excluded)].copy()
    if ML_SCORE_THRESHOLD is not None:
        if "ml_score" not in candidates.columns:
            return []
        candidates = candidates[candidates["ml_score"].ge(ML_SCORE_THRESHOLD)]
    if candidates.empty or limit <= 0:
        return []
    median_d = candidates["signal_D"].median()
    candidates["d_distance"] = (candidates["signal_D"] - median_d).abs()
    sort_columns = ["d_distance", "volume", "symbol"]
    ascending = [True, False, True]
    if ML_SCORE_THRESHOLD is not None:
        sort_columns.insert(0, "ml_score")
        ascending.insert(0, False)
    candidates = candidates.sort_values(sort_columns, ascending=ascending)
    return candidates.head(limit).to_dict("records")


def interleave(long_items: list[dict], short_items: list[dict]) -> list[tuple[str, dict]]:
    combined = []
    for index in range(max(len(long_items), len(short_items))):
        if index < len(long_items):
            combined.append(("LONG", long_items[index]))
        if index < len(short_items):
            combined.append(("SHORT", short_items[index]))
    return combined


def side_sign(side: str) -> int:
    return 1 if side == "LONG" else -1


def exposure(positions: dict[str, dict[str, dict]]) -> tuple[float, float, float]:
    gross_long = sum(position["notional"] for position in positions["LONG"].values())
    gross_short = sum(position["notional"] for position in positions["SHORT"].values())
    return gross_long, gross_short, gross_long - gross_short


def margin_used(positions: dict[str, dict[str, dict]]) -> float:
    return sum(
        position["notional"] / position["lev"]
        for side_positions in positions.values()
        for position in side_positions.values()
    )


def position_notional(equity: float) -> float:
    if POSITION_SIZING_MODE == "fixed":
        return SLOT_SIZE
    if POSITION_SIZING_MODE == "equity_fraction":
        return max(0.0, equity * POSITION_EQUITY_FRACTION)
    raise ValueError(f"不支持的仓位模式：{POSITION_SIZING_MODE}")


def trading_cost(notional_change: float) -> tuple[float, float, float]:
    fee = abs(notional_change) * FEE_RATE
    slippage = abs(notional_change) * SLIPPAGE_RATE
    return fee, slippage, fee + slippage


def append_trade(
    rows: list[dict],
    timestamp: pd.Timestamp,
    side: str,
    action: str,
    reason: str | None,
    symbol: str,
    from_notional: float,
    to_notional: float,
    execution_price: float,
    leverage: float,
) -> float:
    fee, slippage, cost = trading_cost(to_notional - from_notional)
    rows.append(
        {
            "dt": timestamp,
            "side": side,
            "action": action,
            "reason": reason,
            "symbol": symbol,
            "execution_price": execution_price,
            "from_notional": from_notional,
            "to_notional": to_notional,
            "fee_notional": abs(to_notional - from_notional),
            "fee_paid": fee,
            "slippage_paid": slippage,
            "trading_cost": cost,
            "to_lev": leverage,
        }
    )
    return cost


def close_position(
    positions: dict[str, dict[str, dict]],
    side: str,
    symbol: str,
    timestamp: pd.Timestamp,
    price: float,
    reason: str,
    trade_rows: list[dict],
    summary_rows: list[dict],
) -> float:
    position = positions[side].pop(symbol)
    cost = append_trade(
        trade_rows,
        timestamp,
        side,
        "CLOSE",
        reason,
        symbol,
        position["notional"],
        0.0,
        price,
        0.0,
    )
    fee, slippage, _ = trading_cost(position["notional"])
    position["fees"] += fee
    position["slippage"] += slippage
    position["costs"] += cost
    summary_rows.append(
        {
            "side": side,
            "symbol": symbol,
            "open_dt": position["entry_dt"],
            "close_dt": timestamp,
            "holding_hours": (timestamp - position["entry_dt"]).total_seconds() / 3600.0,
            "entry_px": position["entry_px"],
            "exit_px": price,
            "entry_notional": position["entry_notional"],
            "gross_pnl": position["gross_pnl"],
            "fees_paid": position["fees"],
            "slippage_paid": position["slippage"],
            "carry_paid": position["carry"],
            "net_pnl": position["gross_pnl"] - position["costs"] - position["carry"],
            "price_return": side_sign(side) * (price / position["entry_px"] - 1.0),
            "num_adjusts": position["adjustments"],
            "close_reason": reason,
        }
    )
    return cost


def settle_interval(
    positions: dict[str, dict[str, dict]],
    previous_time: pd.Timestamp,
    current_time: pd.Timestamp,
    previous_prices: dict[str, float],
    current_prices: dict[str, float],
) -> tuple[float, float, float]:
    pnl_long = 0.0
    pnl_short = 0.0
    carry_cost = 0.0
    elapsed_hours = (current_time - previous_time).total_seconds() / 3600.0
    for side in ("LONG", "SHORT"):
        sign = side_sign(side)
        for symbol, position in positions[side].items():
            if symbol not in previous_prices or symbol not in current_prices:
                raise ValueError(f"持仓期间缺少价格：{symbol} {previous_time} -> {current_time}")
            bar_return = current_prices[symbol] / previous_prices[symbol] - 1.0
            pnl = sign * position["notional"] * bar_return
            carry = 0.0
            if side == "SHORT" and SHORT_CARRY_RATE_ANNUAL > 0:
                carry = position["notional"] * SHORT_CARRY_RATE_ANNUAL * elapsed_hours / (365 * 24)
            position["gross_pnl"] += pnl
            position["carry"] += carry
            carry_cost += carry
            if side == "LONG":
                pnl_long += pnl
            else:
                pnl_short += pnl
    return pnl_long, pnl_short, carry_cost


def should_close(position: dict, side: str, row: pd.Series, timestamp: pd.Timestamp) -> str | None:
    holding_hours = (timestamp - position["entry_dt"]).total_seconds() / 3600.0
    current_price = float(row["open"])
    signed_return = side_sign(side) * (current_price / position["entry_px"] - 1.0)
    if signed_return <= -STOP_LOSS_RETURN:
        return "stop_loss"
    if holding_hours >= MAX_HOLD_HOURS - EPSILON:
        return "timeout"
    signal_d = row["signal_D"]
    signal_w = row["signal_w"]
    if pd.isna(signal_d) or pd.isna(signal_w):
        return None
    if side == "LONG" and (signal_d >= MEAN_EXIT_D or signal_w >= MEAN_EXIT_W):
        return "mean_reversion"
    if side == "SHORT" and (signal_d <= MEAN_EXIT_D or signal_w <= MEAN_EXIT_W):
        return "mean_reversion"
    return None


def run_backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshots = {timestamp: group.set_index("symbol") for timestamp, group in panel.groupby("dt", sort=True)}
    times = list(snapshots)
    positions = {"LONG": {}, "SHORT": {}}
    equity = INITIAL_EQUITY
    curve_rows = []
    trade_rows = []
    summary_rows = []
    pick_rows = []
    previous_time = None
    previous_open_prices = None
    bankrupt = False

    for index, timestamp in enumerate(times):
        snapshot = snapshots[timestamp]
        open_prices = snapshot["open"].astype(float).to_dict()
        pnl_long = 0.0
        pnl_short = 0.0
        carry_cost = 0.0
        turnover_cost = 0.0

        if previous_time is not None:
            pnl_long, pnl_short, carry_cost = settle_interval(
                positions, previous_time, timestamp, previous_open_prices, open_prices
            )
            equity += pnl_long + pnl_short - carry_cost

        if equity <= 0:
            bankrupt = True
            curve_rows.append(
                {
                    "dt": timestamp,
                    "equity": equity,
                    "pnl_long": pnl_long,
                    "pnl_short": pnl_short,
                    "turnover_cost": 0.0,
                    "carry_cost": carry_cost,
                    "num_long": len(positions["LONG"]),
                    "num_short": len(positions["SHORT"]),
                    "gross_long_notional": exposure(positions)[0],
                    "gross_short_notional": exposure(positions)[1],
                    "net_notional": exposure(positions)[2],
                    "margin_used": margin_used(positions),
                    "bankrupt": True,
                }
            )
            break

        is_final_bar = index == len(times) - 1
        if not is_final_bar:
            for side in ("LONG", "SHORT"):
                for symbol in list(positions[side]):
                    if symbol not in snapshot.index:
                        raise ValueError(f"调仓时缺少持仓标的行情：{symbol} {timestamp}")
                    row = snapshot.loc[symbol]
                    reason = should_close(positions[side][symbol], side, row, timestamp)
                    if reason:
                        turnover_cost += close_position(
                            positions,
                            side,
                            symbol,
                            timestamp,
                            float(row["open"]),
                            reason,
                            trade_rows,
                            summary_rows,
                        )

            for side in ("LONG", "SHORT"):
                for symbol, position in list(positions[side].items()):
                    row = snapshot.loc[symbol]
                    if pd.isna(row["signal_lev"]):
                        continue
                    new_leverage = float(row["signal_lev"])
                    old_leverage = position["lev"]
                    if new_leverage == old_leverage:
                        continue
                    current_margin = margin_used(positions)
                    next_margin = (
                        current_margin
                        - position["notional"] / old_leverage
                        + position["notional"] / new_leverage
                    )
                    if next_margin > equity - turnover_cost:
                        continue
                    append_trade(
                        trade_rows,
                        timestamp,
                        side,
                        "SET_LEVERAGE",
                        None,
                        symbol,
                        position["notional"],
                        position["notional"],
                        float(row["open"]),
                        new_leverage,
                    )
                    position["lev"] = new_leverage
                    position["adjustments"] += 1

            held_symbols = set(positions["LONG"]) | set(positions["SHORT"])
            long_picks = select_candidates(
                snapshot.reset_index(),
                "LONG",
                MAX_NEW_PER_BAR_PER_SIDE,
                held_symbols,
            )
            short_picks = select_candidates(
                snapshot.reset_index(),
                "SHORT",
                MAX_NEW_PER_BAR_PER_SIDE,
                held_symbols,
            )

            for side, row in interleave(long_picks, short_picks):
                symbol = str(row["symbol"])
                if symbol in positions["LONG"] or symbol in positions["SHORT"]:
                    continue
                leverage = float(row["signal_lev"])
                notional = position_notional(equity - turnover_cost)
                order_cost = trading_cost(notional)[2]
                required_margin = notional / leverage
                available_margin = equity - turnover_cost - margin_used(positions)
                if required_margin + order_cost > available_margin:
                    continue
                price = float(row["open"])
                cost = append_trade(
                    trade_rows,
                    timestamp,
                    side,
                    "OPEN",
                    None,
                    symbol,
                    0.0,
                    notional,
                    price,
                    leverage,
                )
                fee, slippage, _ = trading_cost(notional)
                positions[side][symbol] = {
                    "notional": notional,
                    "lev": leverage,
                    "entry_notional": notional,
                    "entry_dt": timestamp,
                    "entry_px": price,
                    "gross_pnl": 0.0,
                    "fees": fee,
                    "slippage": slippage,
                    "costs": cost,
                    "carry": 0.0,
                    "adjustments": 0,
                }
                turnover_cost += cost
                pick_rows.append(
                    {
                        "dt": timestamp,
                        "signal_dt": timestamp - pd.Timedelta(hours=BAR_HOURS),
                        "side": side,
                        "symbol": symbol,
                        "execution_price": price,
                        "D": float(row["signal_D"]),
                        "w": float(row["signal_w"]),
                        "lev": leverage,
                        "volume": float(row["volume"]),
                        "notional": notional,
                    }
                )

            equity -= turnover_cost
        else:
            close_prices = snapshot["close"].astype(float).to_dict()
            intrabar_long, intrabar_short, intrabar_carry = settle_interval(
                positions, timestamp, timestamp + pd.Timedelta(hours=BAR_HOURS), open_prices, close_prices
            )
            pnl_long += intrabar_long
            pnl_short += intrabar_short
            carry_cost += intrabar_carry
            equity += intrabar_long + intrabar_short - intrabar_carry
            for side in ("LONG", "SHORT"):
                for symbol in list(positions[side]):
                    turnover_cost += close_position(
                        positions,
                        side,
                        symbol,
                        timestamp,
                        close_prices[symbol],
                        "end_of_test",
                        trade_rows,
                        summary_rows,
                    )
            equity -= turnover_cost

        gross_long, gross_short, net = exposure(positions)
        curve_rows.append(
            {
                "dt": timestamp,
                "equity": equity,
                "pnl_long": pnl_long,
                "pnl_short": pnl_short,
                "turnover_cost": turnover_cost,
                "carry_cost": carry_cost,
                "num_long": len(positions["LONG"]),
                "num_short": len(positions["SHORT"]),
                "gross_long_notional": gross_long,
                "gross_short_notional": gross_short,
                "net_notional": net,
                "margin_used": margin_used(positions),
                "bankrupt": bankrupt,
            }
        )
        previous_time = timestamp
        previous_open_prices = open_prices

    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(pick_rows),
    )


def calculate_metrics(
    curve: pd.DataFrame, trades: pd.DataFrame, summaries: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    returns = curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    elapsed_years = max(
        (curve["dt"].iloc[-1] - curve["dt"].iloc[0]).total_seconds() / (365.25 * 24 * 3600),
        EPSILON,
    )
    total_return = curve["equity"].iloc[-1] / INITIAL_EQUITY - 1.0
    annual_return = (
        (curve["equity"].iloc[-1] / INITIAL_EQUITY) ** (1.0 / elapsed_years) - 1.0
        if curve["equity"].iloc[-1] > 0
        else -1.0
    )
    annual_volatility = returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR) if len(returns) > 1 else 0.0
    sharpe = (
        returns.mean() / returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR)
        if len(returns) > 1 and returns.std(ddof=0) > 0
        else 0.0
    )
    drawdown = curve["equity"] / curve["equity"].cummax() - 1.0
    max_drawdown = -drawdown.min()
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    btc = raw[raw["symbol"].eq("BTC/USDT")].sort_values("dt")
    btc_return = btc["close"].iloc[-1] / btc["open"].iloc[0] - 1.0 if not btc.empty else np.nan
    completed = len(summaries)
    win_rate = float((summaries["net_pnl"] > 0).mean()) if completed else 0.0

    return pd.DataFrame(
        [
            {
                "start": curve["dt"].iloc[0],
                "end": curve["dt"].iloc[-1],
                "bars": len(curve),
                "initial_equity": INITIAL_EQUITY,
                "position_sizing_mode": POSITION_SIZING_MODE,
                "fixed_slot_size": SLOT_SIZE,
                "equity_fraction": POSITION_EQUITY_FRACTION,
                "final_equity": curve["equity"].iloc[-1],
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "calmar": calmar,
                "btc_buy_hold_return": btc_return,
                "completed_trades": completed,
                "win_rate": win_rate,
                "fees_paid": trades["fee_paid"].sum() if not trades.empty else 0.0,
                "slippage_paid": trades["slippage_paid"].sum() if not trades.empty else 0.0,
                "carry_paid": curve["carry_cost"].sum(),
                "max_gross_leverage": (
                    (curve["gross_long_notional"] + curve["gross_short_notional"])
                    / curve["equity"].clip(lower=EPSILON)
                ).max(),
                "max_margin_utilization": (
                    curve["margin_used"] / curve["equity"].clip(lower=EPSILON)
                ).max(),
                "max_positions": (curve["num_long"] + curve["num_short"]).max(),
                "bankrupt": bool(curve["bankrupt"].any()),
            }
        ]
    )


def main():
    raw = load_market_data(DATA_PATH)
    panel = prepare_panel(raw)
    curve, trades, summaries, picks = run_backtest(panel)
    metrics = calculate_metrics(curve, trades, summaries, raw)

    curve.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    summaries.to_csv(OUTPUT_DIR / "trades_summary.csv", index=False)
    picks.to_csv(OUTPUT_DIR / "picks_per_reb.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    result = metrics.iloc[0]
    print("=== 两年均值回归回测 ===")
    print(f"区间         : {result['start']} -> {result['end']}")
    print(f"期末资金     : {result['final_equity']:,.2f}")
    print(f"总收益       : {result['total_return']:.2%}")
    print(f"年化收益     : {result['annual_return']:.2%}")
    print(f"年化波动率   : {result['annual_volatility']:.2%}")
    print(f"夏普率       : {result['sharpe']:.2f}")
    print(f"最大回撤     : {result['max_drawdown']:.2%}")
    print(f"BTC同期收益  : {result['btc_buy_hold_return']:.2%}")
    print(f"完成交易     : {int(result['completed_trades'])}")
    print(f"胜率         : {result['win_rate']:.2%}")
    print(f"最高名义/权益: {result['max_gross_leverage']:.2f}x")
    print(f"最高保证金占比: {result['max_margin_utilization']:.2%}")
    print(f"是否爆仓     : {'是' if result['bankrupt'] else '否'}")


if __name__ == "__main__":
    main()
