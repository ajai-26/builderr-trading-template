from __future__ import annotations
import math
from statistics import pstdev
from typing import Any, Optional

MOM_LONG: int = 42
MOM_SHORT: int = 21
MOM_W_LONG: float = 0.50
MOM_W_SHORT: float = 0.30
MOM_W_GAP: float = 0.20
NAME_SMA: int = 50
IDX_SMA_FAST: int = 20
IDX_SMA_SLOW: int = 50
ENTER_BAND: float = 0.002
EXIT_BAND: float = 0.008
VOL_SIZE: int = 20
VOL_BRAKE: int = 10
VOL_FULL_MAX: float = 0.30
BRAKE_VOL10: float = 0.40
BRAKE_R3: float = -0.040
BREADTH_MIN: float = 0.50
TOP_N_MAX: int = 5
NAME_CAP: float = 0.26
CORE_FULL: float = 0.67
CORE_NEUTRAL: float = 0.85
SLEEVE_DOLLAR_FULL: float = 0.33
SLEEVE_DOLLAR_ULTRA: float = 0.44   # boosted sleeve when trend is exceptionally clean
ULTRA_VOL_MAX: float = 0.18         # QQQ 20-day vol must be below this for ultra mode
ULTRA_BREADTH_MIN: float = 0.70     # 70%+ of leaders above 50d SMA = very healthy market
MAX_BETA_GROSS: float = 1.45
DD_HALF: float = -0.04
DD_LOCK: float = -0.07
TAPER_HALF: float = 0.50
TAPER_LOCK: float = 0.25
TRAIL_STOP: float = 0.08
STOP_COOLDOWN_DAYS: int = 2
REBALANCE_DAYS: int = 3
COOLDOWN_DAYS: int = 2
DRIFT_LIMIT: float = 0.28
MIN_TRADE_PCT: float = 0.03
CASH_BUFFER: float = 0.98
MAX_ORDERS: int = 45
MIN_BARS: int = 51
THRUST_LOOKBACK: int = 10
THRUST_MIN_RET: float = 0.10

INDEX_REF: tuple[str, ...] = ("SPY", "QQQ")
LEADER_STOCKS: tuple[str, ...] = (
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL", "AVGO", "AMD", "MU", "MRVL",
    "NFLX", "TSLA", "PLTR", "ORCL", "CRM", "JPM", "V", "MA", "COST", "LLY",
    "PANW", "ANET", "NOW", "SNOW", "CRWD",
)
LEADER_ETFS: tuple[str, ...] = (
    "QQQ", "SPY", "SMH", "XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLV",
    "XLP", "XLU", "XLRE", "DIA", "IWM", "SOXX",
)
LEADER_POOL: tuple[str, ...] = tuple(dict.fromkeys(LEADER_STOCKS + LEADER_ETFS))
SLEEVE: tuple[str, ...] = ("QLD", "SSO")
BETA: dict[str, float] = {
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0, "FAS": 3.0,
    "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0, "UDOW": 3.0, "NAIL": 3.0,
}
_STATE_RANK: dict[str, int] = {"CASH": 0, "NEUTRAL": 1, "FULL": 2}

_state: str = "NEUTRAL"
_cooldown: int = 0
_peak_equity: float = 0.0
_pos_high: dict[str, float] = {}
_stop_block: dict[str, int] = {}
_last_rebalance_date: Optional[str] = None
_last_seen_date: Optional[str] = None
_prev_state: str = "NEUTRAL"
_prev_taper_mult: float = 1.0


def _beta(ticker: str) -> float:
    return BETA.get(ticker, 1.0)


def _date_of(ts: Any) -> str:
    return str(ts)[:10]


def _closes_of(
    market_state: dict[str, Any],
    ticker: str,
    cache: dict[str, Optional[list[float]]],
) -> Optional[list[float]]:
    if ticker in cache:
        return cache[ticker]
    closes: Optional[list[float]] = None
    bars = market_state.get(ticker)
    if bars:
        try:
            closes = [float(bar["close"]) for bar in bars]
        except (KeyError, TypeError, ValueError):
            closes = None
    cache[ticker] = closes
    return closes


def _computable(closes: Optional[list[float]]) -> bool:
    return closes is not None and len(closes) >= MIN_BARS and closes[-1] > 0.0


def _sma(closes: list[float], n: int) -> Optional[float]:
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _ret(closes: list[float], k: int) -> Optional[float]:
    if len(closes) < k + 1:
        return None
    start = closes[-(k + 1)]
    if start <= 0.0:
        return None
    return closes[-1] / start - 1.0


def _vol(closes: list[float], n: int) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    window = closes[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0.0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 2:
        return None
    return pstdev(rets) * math.sqrt(252.0)


def _trend_gap(closes: list[float]) -> Optional[float]:
    sma50 = _sma(closes, NAME_SMA)
    if sma50 is None or sma50 <= 0.0:
        return None
    return closes[-1] / sma50 - 1.0


def _momentum_score(closes: list[float]) -> Optional[float]:
    r_long = _ret(closes, MOM_LONG)
    r_short = _ret(closes, MOM_SHORT)
    gap = _trend_gap(closes)
    if r_long is None or r_short is None or gap is None:
        return None
    r_fast = _ret(closes, 5)
    accel_bonus = (r_fast * 0.10) if r_fast is not None else 0.0
    base = MOM_W_LONG * r_long + MOM_W_SHORT * r_short + MOM_W_GAP * gap
    if base <= 0.0:
        return None
    return base + accel_bonus


def _resolve_cash(portfolio_state: dict[str, Any], cash: float) -> float:
    try:
        return float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        try:
            return float(cash)
        except (TypeError, ValueError):
            return 0.0


def _aggregate_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        try:
            ticker = str(raw["ticker"]).upper()
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if qty <= 0.0:
            continue
        if ticker in out:
            existing = out[ticker]
            total = existing["quantity"] + qty
            existing["avg_cost"] = (
                (existing["avg_cost"] * existing["quantity"] + avg_cost * qty) / total
                if total > 0.0 else avg_cost
            )
            existing["quantity"] = total
        else:
            out[ticker] = {"quantity": qty, "avg_cost": avg_cost}
    return out


def _mark_price(
    ticker: str,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
) -> Optional[float]:
    lp = last_prices.get(ticker)
    try:
        if lp is not None and float(lp) > 0.0:
            return float(lp)
    except (TypeError, ValueError):
        pass
    closes = _closes_of(market_state, ticker, cache)
    if closes and closes[-1] > 0.0:
        return closes[-1]
    return None


def _exec_price(
    ticker: str,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
) -> Optional[float]:
    closes = _closes_of(market_state, ticker, cache)
    if closes and closes[-1] > 0.0:
        return closes[-1]
    lp = last_prices.get(ticker)
    try:
        if lp is not None and float(lp) > 0.0:
            return float(lp)
    except (TypeError, ValueError):
        pass
    return None


def _compute_equity(
    positions: dict[str, dict[str, float]],
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
    cash_value: float,
) -> float:
    total = cash_value
    for ticker in sorted(positions):
        pos = positions[ticker]
        price = _mark_price(ticker, market_state, cache, last_prices)
        if price is None:
            price = pos["avg_cost"] if pos["avg_cost"] > 0.0 else 0.0
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _build_targets(
    state: str,
    taper_mult: float,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    stop_block: dict[str, int],
    ultra_mode: bool = False,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    if state == "CASH":
        return weights

    sleeve_present: list[str] = []
    if state == "FULL":
        sleeve_present = [
            s for s in SLEEVE
            if s not in stop_block and _computable(_closes_of(market_state, s, cache))
        ]

    if state == "FULL":
        sleeve_budget = SLEEVE_DOLLAR_ULTRA if (ultra_mode and sleeve_present) else SLEEVE_DOLLAR_FULL
        core_base = CORE_FULL if sleeve_present else (CORE_FULL + SLEEVE_DOLLAR_FULL)
    else:
        sleeve_budget = 0.0
        core_base = CORE_NEUTRAL
    core_budget = core_base * taper_mult

    qualifiers: list[tuple[float, str]] = []
    for ticker in LEADER_POOL:
        if ticker in stop_block:
            continue
        closes = _closes_of(market_state, ticker, cache)
        if not _computable(closes):
            continue
        score = _momentum_score(closes)
        if score is None:
            continue
        sma50 = _sma(closes, NAME_SMA)
        if sma50 is None:
            continue
        if score > 0.0 and closes[-1] > sma50:
            qualifiers.append((score, ticker))

    qualifiers.sort(key=lambda pair: (-pair[0], pair[1]))
    selected = qualifiers[:TOP_N_MAX]
    n = len(selected)

    if n > 0:
        sum_raw = n * (n + 1) / 2.0
        for rank, (_, ticker) in enumerate(selected, start=1):
            raw = float(n - rank + 1)
            weight = min(core_budget * raw / sum_raw, NAME_CAP)
            if weight > 0.0:
                weights[ticker] = weight

    if sleeve_present:
        per = (sleeve_budget * taper_mult) / len(sleeve_present)
        for s in sleeve_present:
            weight = min(per, NAME_CAP)
            if weight > 0.0:
                weights[s] = weight

    beta_gross = sum(w * _beta(t) for t, w in weights.items())
    if beta_gross > MAX_BETA_GROSS and beta_gross > 0.0:
        scale = MAX_BETA_GROSS / beta_gross
        weights = {t: w * scale for t, w in weights.items()}

    return weights


def _generate_orders(
    do_rebalance: bool,
    weights: dict[str, float],
    positions: dict[str, dict[str, float]],
    forced_stops: list[tuple[str, float]],
    equity: float,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
    cash_value: float,
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    sold: set[str] = set()
    proceeds = 0.0
    min_trade = MIN_TRADE_PCT * equity

    for ticker, qty in forced_stops:
        if qty > 0.0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
            sold.add(ticker)
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is not None:
                proceeds += qty * price

    if do_rebalance:
        for ticker in sorted(positions):
            if ticker in sold:
                continue
            held = positions[ticker]["quantity"]
            if held <= 0.0:
                continue
            target_w = weights.get(ticker, 0.0)
            price = _exec_price(ticker, market_state, cache, last_prices)
            if target_w == 0.0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": held})
                sold.add(ticker)
                if price is not None and price > 0.0:
                    proceeds += held * price
                continue
            if price is None or price <= 0.0:
                continue
            target_shares = math.floor(target_w * equity / price)
            delta = target_shares - held
            if delta < 0 and (-delta) * price >= min_trade:
                sell_qty = float(int(min(-delta, held)))
                if sell_qty > 0.0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                    sold.add(ticker)
                    proceeds += sell_qty * price

        spendable = cash_value + CASH_BUFFER * proceeds
        for ticker in sorted(weights, key=lambda t: (-weights[t], t)):
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is None or price <= 0.0:
                continue
            held = positions[ticker]["quantity"] if ticker in positions else 0.0
            target_shares = math.floor(weights[ticker] * equity / price)
            deficit = target_shares - held
            if deficit > 0 and deficit * price >= min_trade:
                affordable = math.floor(min(deficit * price, spendable) / price)
                if affordable > 0:
                    orders.append({"ticker": ticker, "side": "buy", "quantity": float(affordable)})
                    spendable -= affordable * price

    if len(orders) > MAX_ORDERS:
        sells = [o for o in orders if o["side"] == "sell"]
        buys = [o for o in orders if o["side"] == "buy"]
        orders = (sells + buys)[:MAX_ORDERS]

    return [o for o in orders if o["quantity"] > 0.0]


def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    global _state, _cooldown, _peak_equity, _pos_high, _stop_block
    global _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult

    snapshot = (
        _state, _cooldown, _peak_equity, dict(_pos_high), dict(_stop_block),
        _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult,
    )
    try:
        return _run(market_state or {}, portfolio_state or {}, cash)
    except Exception:
        (
            _state, _cooldown, _peak_equity, _pos_high, _stop_block,
            _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult,
        ) = snapshot
        return []


def _run(
    market_state: dict[str, Any],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict[str, Any]]:
    global _state, _cooldown, _peak_equity, _pos_high, _stop_block
    global _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult

    if not market_state:
        return []

    cache: dict[str, Optional[list[float]]] = {}
    last_prices: dict[str, Any] = {}
    for key, value in (portfolio_state.get("last_prices", {}) or {}).items():
        last_prices[str(key).upper()] = value
    cash_value = _resolve_cash(portfolio_state, cash)

    spy_bars = market_state.get("SPY")
    spy = _closes_of(market_state, "SPY", cache)
    qqq = _closes_of(market_state, "QQQ", cache)
    current_date: Optional[str] = None
    if spy_bars:
        ts = spy_bars[-1].get("ts")
        current_date = _date_of(ts) if ts is not None else str(len(spy_bars))

    if not _computable(spy) or not _computable(qqq):
        positions = _aggregate_positions(portfolio_state)
        orders: list[dict[str, Any]] = []
        for ticker in sorted(positions):
            if market_state.get(ticker):
                qty = positions[ticker]["quantity"]
                if qty > 0.0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
        _prev_state = _state
        _prev_taper_mult = 1.0
        if current_date is not None:
            _last_seen_date = current_date
        return orders

    spy_closes: list[float] = spy
    qqq_closes: list[float] = qqq

    positions = _aggregate_positions(portfolio_state)

    is_new_day = current_date != _last_seen_date
    if is_new_day:
        if _cooldown > 0:
            _cooldown -= 1
        if _stop_block:
            decayed: dict[str, int] = {}
            for ticker, days in _stop_block.items():
                remaining = days - 1
                if remaining > 0:
                    decayed[ticker] = remaining
            _stop_block = decayed

    equity = _compute_equity(positions, market_state, cache, last_prices, cash_value)
    if equity <= 0.0:
        _prev_state = _state
        _prev_taper_mult = 1.0
        _last_seen_date = current_date
        return []
    _peak_equity = max(_peak_equity, equity)
    dd = (equity / _peak_equity - 1.0) if _peak_equity > 0.0 else 0.0
    if dd <= DD_LOCK:
        taper_mult = TAPER_LOCK
    elif dd <= DD_HALF:
        taper_mult = TAPER_HALF
    else:
        taper_mult = 1.0

    spy_close = spy_closes[-1]
    qqq_close = qqq_closes[-1]
    spy_sma_fast = _sma(spy_closes, IDX_SMA_FAST)
    spy_sma_slow = _sma(spy_closes, IDX_SMA_SLOW)
    qqq_sma_fast = _sma(qqq_closes, IDX_SMA_FAST)
    qqq_sma_slow = _sma(qqq_closes, IDX_SMA_SLOW)
    qqq_vol20 = _vol(qqq_closes, VOL_SIZE)
    qqq_r3 = _ret(qqq_closes, 3)
    qqq_vol10 = _vol(qqq_closes, VOL_BRAKE)

    n_comp = 0
    n_up = 0
    for ticker in LEADER_POOL:
        closes = _closes_of(market_state, ticker, cache)
        if not _computable(closes):
            continue
        sma50 = _sma(closes, NAME_SMA)
        if sma50 is None:
            continue
        n_comp += 1
        if closes[-1] > sma50:
            n_up += 1
    breadth = (n_up / n_comp) if n_comp > 0 else 0.0

    prev_cycle_state = _state
    brake_fired = (
        (qqq_r3 is not None and qqq_r3 < BRAKE_R3)
        or (qqq_vol10 is not None and qqq_vol10 > BRAKE_VOL10)
        or (_ret(qqq_closes, 1) is not None and _ret(qqq_closes, 1) < -0.025)
        or (_ret(spy_closes, 1) is not None and _ret(spy_closes, 1) < -0.025)
    )
    hard_cash = (
        brake_fired
        or (spy_sma_slow is not None and spy_close < spy_sma_slow * (1.0 - EXIT_BAND))
        or (qqq_sma_slow is not None and qqq_close < qqq_sma_slow * (1.0 - EXIT_BAND))
    )
    reclaim = (
        spy_sma_slow is not None and spy_close > spy_sma_slow * (1.0 + ENTER_BAND)
        and qqq_sma_slow is not None and qqq_close > qqq_sma_slow * (1.0 + ENTER_BAND)
    )
    qqq_ret10 = _ret(qqq_closes, THRUST_LOOKBACK)
    thrust_signal = (
        qqq_ret10 is not None and qqq_ret10 > THRUST_MIN_RET
        and qqq_sma_fast is not None and qqq_close > qqq_sma_fast
        and len(qqq_closes) >= 2 and qqq_close > qqq_closes[-2]
        and qqq_vol20 is not None and qqq_vol20 < 0.32
    )
    if (
        prev_cycle_state == "CASH"
        and not brake_fired
        and not reclaim
        and thrust_signal
    ):
        _state = "NEUTRAL"
    elif hard_cash:
        _state = "CASH"
        _cooldown = COOLDOWN_DAYS
    elif prev_cycle_state == "CASH" and not reclaim:
        _state = "CASH"
    elif _cooldown > 0:
        _state = "NEUTRAL"
    else:
        full_conditions = (
            spy_sma_fast is not None and spy_close > spy_sma_fast
            and qqq_sma_fast is not None and qqq_close > qqq_sma_fast
            and spy_sma_slow is not None and spy_close > spy_sma_slow * (1.0 + ENTER_BAND)
            and qqq_sma_slow is not None and qqq_close > qqq_sma_slow * (1.0 + ENTER_BAND)
            and breadth >= BREADTH_MIN
            and qqq_vol20 is not None and qqq_vol20 < VOL_FULL_MAX
        )
        _state = "FULL" if full_conditions else "NEUTRAL"

    for ticker in list(_pos_high):
        if ticker not in positions:
            del _pos_high[ticker]
    forced_stops: list[tuple[str, float]] = []
    for ticker in sorted(positions):
        price = _exec_price(ticker, market_state, cache, last_prices)
        if price is None:
            continue
        high = _pos_high.get(ticker, price)
        if price > high:
            high = price
        _pos_high[ticker] = high
        if high > 0.0 and price < high * (1.0 - TRAIL_STOP):
            forced_stops.append((ticker, positions[ticker]["quantity"]))
            _stop_block[ticker] = STOP_COOLDOWN_DAYS
            if ticker in _pos_high:
                del _pos_high[ticker]

    if _last_rebalance_date is None:
        do_rebalance = True
    else:
        elapsed_dates: set[str] = set()
        for bar in spy_bars:
            ts = bar.get("ts")
            bar_date = _date_of(ts) if ts is not None else ""
            if bar_date > _last_rebalance_date:
                elapsed_dates.add(bar_date)
        days_since = len(elapsed_dates)
        derisk_state = _STATE_RANK[_state] < _STATE_RANK[_prev_state]
        derisk_taper = taper_mult < _prev_taper_mult
        drift = False
        for ticker, pos in positions.items():
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is not None and equity > 0.0 and (pos["quantity"] * price / equity) > DRIFT_LIMIT:
                drift = True
                break
        do_rebalance = (
            days_since >= REBALANCE_DAYS or derisk_state or derisk_taper or drift
        )
    if _last_rebalance_date == current_date:
        do_rebalance = False

    ultra_mode = (
        _state == "FULL"
        and qqq_vol20 is not None and qqq_vol20 < ULTRA_VOL_MAX
        and breadth >= ULTRA_BREADTH_MIN
        and taper_mult == 1.0
    )

    weights = (
        _build_targets(_state, taper_mult, market_state, cache, _stop_block, ultra_mode)
        if do_rebalance
        else {}
    )

    orders = _generate_orders(
        do_rebalance, weights, positions, forced_stops,
        equity, market_state, cache, last_prices, cash_value,
    )
    if do_rebalance and len(orders) >= 1:
        _last_rebalance_date = current_date

    _prev_state = _state
    _prev_taper_mult = taper_mult
    _last_seen_date = current_date
    return orders
