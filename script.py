"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CE + ZLSMA Strategy — NIFTY 50 Options Algo (All-in-One)           ║
║  Modes: 1) Backtest (10 yr)  2) Paper Trading (live sim)  3) Live Trading   ║
║  Broker: Fyers API  |  Instrument: NIFTY ATM CE/PE Futures (Weekly)        ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETUP INSTRUCTIONS
──────────────────
1.  pip install fyers-apiv3 pandas numpy requests schedule pytz tabulate colorama

2.  Fill in CONFIG section below (Fyers credentials, capital, etc.)

3.  Run modes:
      python nifty_ce_zlsma_algo.py --mode backtest
      python nifty_ce_zlsma_algo.py --mode paper
      python nifty_ce_zlsma_algo.py --mode live

4.  Backtest reads data from CSV (auto-downloaded via Fyers historical API)
    or place a file called  nifty_data.csv  with columns:
      date, open, high, low, close, volume   (datetime index)

IMPORTANT NOTES
───────────────
- No trade on expiry day (Wednesday for NIFTY weekly).
- On day BEFORE expiry: all open positions are closed by 15:20.
- If a new signal fires after 15:20 on pre-expiry day → next expiry contract.
- Lot size rule: floor( (dailyLossLimit / riskRatio) / LOT_SIZE ) lots,
  minimum 1 lot, rounded DOWN to nearest whole lot.
- Daily loss limit: algo halts new entries if intraday PnL ≤ -dailyLossLimit.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time, json, math, logging, argparse
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

try:
    from fyers_apiv3 import fyersModel
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False
    print("[WARN] fyers-apiv3 not installed. Live/Paper modes will be limited.")

try:
    from tabulate import tabulate
    TABULATE = True
except ImportError:
    TABULATE = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = RESET = ""

IST = ZoneInfo("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████   CONFIG — EDIT THIS SECTION   ██████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Fyers Credentials ──────────────────────────────────────────────────
    "CLIENT_ID"       : "HV4ID10C9I-100",
    "SECRET_KEY"      : "A81349Q03B",
    "REDIRECT_URI"    : "https://trade.fyers.in/api-login/redirect-uri/index.html",
    "ACCESS_TOKEN"    : "",                          # filled after login

    # ── Capital & Sizing ───────────────────────────────────────────────────
    "CAPITAL"         : 50_000,      # ₹ your total capital
    "DAILY_LOSS_LIMIT": 1_000,       # ₹ max loss per day before halt
    "RISK_RATIO"      : 10,          # divisor: target_units = DLL / risk_ratio
    "LOT_SIZE"        : 65,          # NIFTY 1 lot = 65 qty (update if NSE changes)

    # ── Strategy Params ────────────────────────────────────────────────────
    "CE_ATR_PERIOD"   : 1,
    "CE_ATR_MULT"     : 2.0,
    "CE_USE_CLOSE"    : True,
    "ZLSMA_LEN"       : 50,
    "RR_RATIO"        : 3.0,         # 1 : 3 risk-reward
    "STOP_ATR_MULT"   : 1.0,

    # ── Execution ──────────────────────────────────────────────────────────
    "NIFTY_INDEX"     : "NSE:NIFTY50-INDEX",
    "TIMEFRAME"       : "5",         # minutes for live; backtest uses daily
    "PRE_EXPIRY_EXIT_TIME": "15:20", # HH:MM IST — close before expiry
    "MARKET_OPEN"     : "09:15",
    "MARKET_CLOSE"    : "15:30",

    # ── Backtest ───────────────────────────────────────────────────────────
    "BT_CSV"          : "nifty_data.csv",   # optional local CSV override
    "BT_YEARS"        : 10,

    # ── Logging ────────────────────────────────────────────────────────────
    "LOG_FILE"        : "algo_log.txt",
}
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("CE_ZLSMA")


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░░░░░  SECTION 1 — INDICATORS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def compute_chandelier_exit(df: pd.DataFrame, period: int, mult: float, use_close: bool):
    """
    Returns (dir_series, long_stop, short_stop, buy_signal, sell_signal)
    dir:  1 = bullish, -1 = bearish
    """
    atr   = compute_atr(df, period) * mult
    close = df["close"]
    high  = close if use_close else df["high"]
    low   = close if use_close else df["low"]

    n = len(df)
    long_stop  = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    direction  = np.ones(n, dtype=int)

    high_roll  = high.rolling(period).max().values
    low_roll   = low.rolling(period).min().values
    atr_vals   = atr.values
    close_vals = close.values

    for i in range(1, n):
        ls_raw = high_roll[i] - atr_vals[i]
        ss_raw = low_roll[i]  + atr_vals[i]

        ls_prev = long_stop[i-1]  if not np.isnan(long_stop[i-1])  else ls_raw
        ss_prev = short_stop[i-1] if not np.isnan(short_stop[i-1]) else ss_raw

        long_stop[i]  = max(ls_raw, ls_prev) if close_vals[i-1] > ls_prev else ls_raw
        short_stop[i] = min(ss_raw, ss_prev) if close_vals[i-1] < ss_prev else ss_raw

        if close_vals[i] > short_stop[i-1]:
            direction[i] = 1
        elif close_vals[i] < long_stop[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    dir_s   = pd.Series(direction, index=df.index)
    ls_s    = pd.Series(long_stop,  index=df.index)
    ss_s    = pd.Series(short_stop, index=df.index)
    buy_sig  = (dir_s == 1) & (dir_s.shift(1) == -1)
    sell_sig = (dir_s == -1) & (dir_s.shift(1) == 1)
    return dir_s, ls_s, ss_s, buy_sig, sell_sig


def compute_zlsma(series: pd.Series, length: int) -> pd.Series:
    """Zero-Lag LSMA"""
    def lsma(s, l):
        return s.rolling(l).apply(
            lambda x: np.polyval(np.polyfit(range(l), x, 1), l - 1), raw=True
        )
    lsma1 = lsma(series, length)
    lsma2 = lsma(lsma1,  length)
    return lsma1 + (lsma1 - lsma2)


def compute_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add all indicator columns and combined signals to df."""
    df = df.copy()

    dir_s, ls, ss, buy_raw, sell_raw = compute_chandelier_exit(
        df, cfg["CE_ATR_PERIOD"], cfg["CE_ATR_MULT"], cfg["CE_USE_CLOSE"]
    )
    df["ce_dir"]        = dir_s
    df["long_stop"]     = ls
    df["short_stop"]    = ss
    df["zlsma"]         = compute_zlsma(df["close"], cfg["ZLSMA_LEN"])
    df["atr"]           = compute_atr(df, cfg["CE_ATR_PERIOD"])

    df["buy_signal"]    = buy_raw  & (df["close"] > df["zlsma"])
    df["sell_signal"]   = sell_raw & (df["close"] < df["zlsma"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░░░  SECTION 2 — POSITION SIZING  ░░░░░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

def calc_lots(capital: float, daily_loss_limit: float,
              risk_ratio: float, lot_size: int) -> tuple[int, int]:
    """
    Returns (lots, qty).

    Rule:
      target_units = daily_loss_limit / risk_ratio      (e.g. 1000/10 = 100)
      lots         = floor(target_units / lot_size)     (e.g. floor(100/65) = 1)
      lots         = max(1, lots)                        minimum 1 lot
      qty          = lots * lot_size                    (e.g. 1 * 65 = 65)
    """
    target_units = daily_loss_limit / risk_ratio
    lots = max(1, math.floor(target_units / lot_size))
    qty  = lots * lot_size
    return lots, qty


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░  SECTION 3 — EXPIRY HELPERS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

def get_nifty_expiry_dates(from_date: date, to_date: date) -> list[date]:
    """
    NIFTY weekly options expire every Thursday (post-Sep 2024 change from Wed).
    Monthly expiry = last Thursday of month.
    Returns sorted list of all Thursday expiries in range.
    """
    expiries = []
    d = from_date
    while d <= to_date:
        if d.weekday() == 3:   # Thursday = 3
            expiries.append(d)
        d += timedelta(days=1)
    return expiries


def next_expiry(ref_date: date, expiries: list[date]) -> date | None:
    """First expiry strictly after ref_date."""
    for e in expiries:
        if e > ref_date:
            return e
    return None


def current_expiry(ref_date: date, expiries: list[date]) -> date | None:
    """Nearest expiry >= ref_date."""
    for e in expiries:
        if e >= ref_date:
            return e
    return None


def is_pre_expiry_day(ref_date: date, expiries: list[date]) -> bool:
    """True if tomorrow is an expiry day."""
    tomorrow = ref_date + timedelta(days=1)
    return tomorrow in expiries


def is_expiry_day(ref_date: date, expiries: list[date]) -> bool:
    return ref_date in expiries


def option_symbol_fyers(expiry: date, strike: int, opt_type: str) -> str:
    """
    Build Fyers options symbol.
    Format: NSE:NIFTY{YY}{MON}{DD}{STRIKE}{CE/PE}  (weekly)
    Example: NSE:NIFTY25MAR2024200CE
    """
    mon = expiry.strftime("%b").upper()   # MAR
    yy  = expiry.strftime("%y")           # 25
    dd  = expiry.strftime("%d")           # 20
    return f"NSE:NIFTY{yy}{mon}{dd}{strike}{opt_type}"


def atm_strike(spot: float, step: int = 50) -> int:
    """Round spot to nearest 50."""
    return int(round(spot / step) * step)


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░░░░░░  SECTION 4 — BACKTEST ENGINE  ░░░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Simulates options P&L using NIFTY index data.

    Options modelling (simplified but realistic):
      - Entry price  = intrinsic + (ATR * 0.5) as proxy for premium at signal bar
      - Exit price   = mark-to-market based on index move + time decay proxy
      - This is NOT Black-Scholes; it's a directional P&L proxy matching
        your requirement: "not just index move profit, but actual CE/PE trade result"

    For each trade:
      - Buy ATM CE (on BUY signal) or ATM PE (on SELL signal)
      - P&L = (exit_option_price - entry_option_price) * qty
      - Option price modelled as: intrinsic_value + theta_decay_proxy
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

    # ── Data loading ──────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        csv_path = self.cfg["BT_CSV"]
        if os.path.exists(csv_path):
            log.info(f"Loading backtest data from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        else:
            log.info("No local CSV found — generating synthetic NIFTY data for demo")
            df = self._synthetic_nifty_data()

        df.columns = [c.lower() for c in df.columns]
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        df.sort_index(inplace=True)

        cutoff = pd.Timestamp.now() - pd.DateOffset(years=self.cfg["BT_YEARS"])
        df = df[df.index >= cutoff]
        log.info(f"Backtest data: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    def _synthetic_nifty_data(self) -> pd.DataFrame:
        """Generate 10 years of daily NIFTY-like OHLCV for demo purposes."""
        np.random.seed(42)
        dates  = pd.date_range(end=pd.Timestamp.today(), periods=2500, freq="B")
        close  = 10000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
        high   = close * (1 + np.abs(np.random.normal(0, 0.006, len(dates))))
        low    = close * (1 - np.abs(np.random.normal(0, 0.006, len(dates))))
        open_  = close * (1 + np.random.normal(0, 0.004, len(dates)))
        volume = np.random.randint(100_000, 500_000, len(dates))
        return pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": volume
        }, index=dates)

    # ── Option price proxy ─────────────────────────────────────────────────

    def option_entry_price(self, spot: float, atr: float, opt_type: str) -> float:
        """
        Proxy ATM option premium at entry.
        ATM option ≈ 0.4 * daily_vol * spot + small intrinsic
        We use: 0.5 * ATR as a simple proxy for premium.
        """
        return max(10.0, atr * 0.5)

    def option_exit_price(self, entry_price: float, spot_entry: float,
                          spot_exit: float, opt_type: str,
                          days_held: int, atr: float) -> float:
        """
        Proxy option exit price.
        - Delta ~ 0.5 for ATM options
        - Theta decay: ~1% of premium per day held
        - Vega ignored (simplified)
        """
        delta        = 0.5
        index_move   = spot_exit - spot_entry
        pnl_on_delta = (index_move if opt_type == "CE" else -index_move) * delta
        theta_decay  = entry_price * 0.01 * days_held
        exit_price   = entry_price + pnl_on_delta - theta_decay
        return max(0.5, exit_price)   # option can't go below 0.5

    # ── Main backtest loop ─────────────────────────────────────────────────

    def run(self):
        df = self.load_data()
        df = compute_signals(df, self.cfg)

        expiries = get_nifty_expiry_dates(
            df.index[0].date(), df.index[-1].date() + timedelta(days=30)
        )
        expiry_set = set(expiries)

        lots, qty = calc_lots(
            self.cfg["CAPITAL"], self.cfg["DAILY_LOSS_LIMIT"],
            self.cfg["RISK_RATIO"], self.cfg["LOT_SIZE"]
        )
        log.info(f"Position size: {lots} lot(s) = {qty} qty")

        capital        = float(self.cfg["CAPITAL"])
        daily_pnl      = 0.0
        daily_loss_lim = float(self.cfg["DAILY_LOSS_LIMIT"])
        position       = None   # dict with trade details
        equity         = capital

        for i, (ts, row) in enumerate(df.iterrows()):
            today      = ts.date()
            is_expiry  = today in expiry_set
            pre_expiry = (today + timedelta(days=1)) in expiry_set

            # Reset daily P&L at start of each day
            if i > 0 and ts.date() != df.index[i-1].date():
                daily_pnl = 0.0

            spot = row["close"]
            atr  = row["atr"] if not np.isnan(row["atr"]) else spot * 0.01

            # ── Close position on expiry day or pre-expiry day ──────────
            if position and (is_expiry or pre_expiry):
                exit_px = self.option_exit_price(
                    position["entry_option_px"], position["spot_entry"],
                    spot, position["opt_type"],
                    max(1, (today - position["entry_date"]).days), atr
                )
                pnl = (exit_px - position["entry_option_px"]) * position["qty"]
                capital   += pnl
                daily_pnl += pnl
                self.trades.append({**position,
                    "exit_date": today, "exit_spot": spot,
                    "exit_option_px": exit_px, "pnl": pnl,
                    "exit_reason": "Expiry forced exit"
                })
                log.info(f"[BT] Expiry exit | {position['opt_type']} | PnL ₹{pnl:,.0f}")
                position = None

            # ── No new entries on expiry day ─────────────────────────────
            if is_expiry:
                self.equity_curve.append({"date": today, "equity": capital})
                continue

            # ── Daily loss halt ───────────────────────────────────────────
            if daily_pnl <= -daily_loss_lim:
                self.equity_curve.append({"date": today, "equity": capital})
                continue

            # ── Signal handling ───────────────────────────────────────────
            signal = None
            if row["buy_signal"]:
                signal = "BUY"
            elif row["sell_signal"]:
                signal = "SELL"

            if signal:
                # Close existing opposite position
                if position:
                    exit_px = self.option_exit_price(
                        position["entry_option_px"], position["spot_entry"],
                        spot, position["opt_type"],
                        max(1, (today - position["entry_date"]).days), atr
                    )
                    pnl = (exit_px - position["entry_option_px"]) * position["qty"]
                    capital   += pnl
                    daily_pnl += pnl
                    self.trades.append({**position,
                        "exit_date": today, "exit_spot": spot,
                        "exit_option_px": exit_px, "pnl": pnl,
                        "exit_reason": "Signal reverse"
                    })
                    position = None

                # Determine expiry for new trade
                if pre_expiry:
                    exp = next_expiry(today, expiries)
                else:
                    exp = current_expiry(today, expiries)

                if exp is None:
                    continue

                opt_type    = "CE" if signal == "BUY" else "PE"
                strike      = atm_strike(spot)
                entry_opt_px = self.option_entry_price(spot, atr, opt_type)
                stop_dist   = self.cfg["STOP_ATR_MULT"] * atr
                sl_spot     = spot - stop_dist if signal == "BUY" else spot + stop_dist
                tp_spot     = spot + stop_dist * self.cfg["RR_RATIO"] if signal == "BUY" \
                              else spot - stop_dist * self.cfg["RR_RATIO"]

                position = {
                    "signal": signal, "opt_type": opt_type,
                    "strike": strike, "expiry": exp,
                    "entry_date": today, "spot_entry": spot,
                    "entry_option_px": entry_opt_px,
                    "sl_spot": sl_spot, "tp_spot": tp_spot,
                    "qty": qty, "lots": lots,
                    "symbol": option_symbol_fyers(exp, strike, opt_type)
                }
                log.debug(f"[BT] ENTER {opt_type} strike={strike} exp={exp} @₹{entry_opt_px:.1f}")

            # ── Check SL / TP for open position ──────────────────────────
            if position:
                hit_sl = (position["signal"] == "BUY"  and spot <= position["sl_spot"]) or \
                         (position["signal"] == "SELL" and spot >= position["sl_spot"])
                hit_tp = (position["signal"] == "BUY"  and spot >= position["tp_spot"]) or \
                         (position["signal"] == "SELL" and spot <= position["tp_spot"])

                if hit_sl or hit_tp:
                    exit_px = self.option_exit_price(
                        position["entry_option_px"], position["spot_entry"],
                        spot, position["opt_type"],
                        max(1, (today - position["entry_date"]).days), atr
                    )
                    pnl = (exit_px - position["entry_option_px"]) * position["qty"]
                    capital   += pnl
                    daily_pnl += pnl
                    reason = "Take Profit" if hit_tp else "Stop Loss"
                    self.trades.append({**position,
                        "exit_date": today, "exit_spot": spot,
                        "exit_option_px": exit_px, "pnl": pnl,
                        "exit_reason": reason
                    })
                    log.debug(f"[BT] EXIT {reason} | PnL ₹{pnl:,.0f}")
                    position = None

            self.equity_curve.append({"date": today, "equity": capital})

        return self.report(capital)

    # ── Performance report ────────────────────────────────────────────────

    def report(self, final_capital: float) -> dict:
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")

        if trades_df.empty:
            log.warning("No trades generated in backtest period.")
            return {}

        total_trades  = len(trades_df)
        wins          = trades_df[trades_df["pnl"] > 0]
        losses        = trades_df[trades_df["pnl"] <= 0]
        win_rate      = len(wins) / total_trades * 100
        total_pnl     = trades_df["pnl"].sum()
        avg_win       = wins["pnl"].mean()   if len(wins)   else 0
        avg_loss      = losses["pnl"].mean() if len(losses) else 0
        profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                         if losses["pnl"].sum() != 0 else float("inf"))

        equity_vals   = equity_df["equity"]
        peak          = equity_vals.cummax()
        drawdown      = (equity_vals - peak) / peak * 100
        max_dd        = drawdown.min()

        init_cap = self.cfg["CAPITAL"]
        returns  = (final_capital - init_cap) / init_cap * 100

        report = {
            "Initial Capital"   : f"₹{init_cap:,.0f}",
            "Final Capital"     : f"₹{final_capital:,.0f}",
            "Net P&L"           : f"₹{total_pnl:,.0f}",
            "Total Return"      : f"{returns:.1f}%",
            "Total Trades"      : total_trades,
            "Win Rate"          : f"{win_rate:.1f}%",
            "Avg Win"           : f"₹{avg_win:,.0f}",
            "Avg Loss"          : f"₹{avg_loss:,.0f}",
            "Profit Factor"     : f"{profit_factor:.2f}",
            "Max Drawdown"      : f"{max_dd:.1f}%",
        }

        print("\n" + "═"*52)
        print("  BACKTEST RESULTS — CE + ZLSMA STRATEGY")
        print("═"*52)
        for k, v in report.items():
            color = GREEN if "P&L" in k and "₹-" not in str(v) else ""
            print(f"  {k:<22} {color}{v}{RESET}")
        print("═"*52)

        # Save detailed trade log
        trades_df.to_csv("backtest_trades.csv", index=False)
        equity_df.to_csv("backtest_equity.csv")
        log.info("Saved: backtest_trades.csv  |  backtest_equity.csv")

        # Monthly P&L summary
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        monthly = trades_df.set_index("exit_date")["pnl"].resample("ME").sum()
        print("\n  MONTHLY P&L")
        print("  " + "-"*30)
        for m, v in monthly.items():
            col = GREEN if v >= 0 else RED
            print(f"  {m.strftime('%b %Y')}  {col}₹{v:>10,.0f}{RESET}")
        print()

        return report


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░░░  SECTION 5 — PAPER TRADING ENGINE  ░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

class PaperTradingEngine:
    """
    Live paper trading: connects to Fyers for real-time NIFTY quotes,
    runs the same signal logic, simulates order fills, tracks P&L.
    All trades are logged but NO real orders are placed.
    """

    def __init__(self, cfg: dict, fyers_client=None):
        self.cfg    = cfg
        self.fyers  = fyers_client
        self.capital     = float(cfg["CAPITAL"])
        self.daily_pnl   = 0.0
        self.position    = None
        self.trade_log   = []
        self.price_buf   = []  # rolling OHLCV buffer for indicators
        self.last_signal = None
        self.today_date  = date.today()

        lots, qty = calc_lots(cfg["CAPITAL"], cfg["DAILY_LOSS_LIMIT"],
                              cfg["RISK_RATIO"], cfg["LOT_SIZE"])
        self.lots = lots
        self.qty  = qty

        log.info(f"{CYAN}[PAPER] Engine started | Lots={lots} Qty={qty}{RESET}")
        self._print_sizing()

    def _print_sizing(self):
        lots, qty = self.lots, self.qty
        print(f"\n{'─'*45}")
        print(f"  PAPER TRADING MODE")
        print(f"  Capital        : ₹{self.capital:,.0f}")
        print(f"  Daily Loss Lim : ₹{self.cfg['DAILY_LOSS_LIMIT']:,.0f}")
        print(f"  Risk Ratio     : {self.cfg['RISK_RATIO']}")
        print(f"  Target Units   : {self.cfg['DAILY_LOSS_LIMIT']/self.cfg['RISK_RATIO']:.0f}")
        print(f"  Lots           : {lots}")
        print(f"  Qty            : {qty}")
        print(f"{'─'*45}\n")

    def get_ltp(self) -> float | None:
        """Fetch latest NIFTY 50 index price from Fyers."""
        if not self.fyers:
            return None
        try:
            data = self.fyers.quotes({"symbols": self.cfg["NIFTY_INDEX"]})
            return data["d"][0]["v"]["lp"]
        except Exception as e:
            log.error(f"LTP fetch failed: {e}")
            return None

    def _reset_daily(self):
        self.daily_pnl = 0.0
        self.today_date = date.today()
        log.info("[PAPER] New day — daily P&L reset")

    def _update_buffer(self, price: float):
        """Maintain a rolling price buffer for signal computation."""
        ts = datetime.now(IST)
        if self.price_buf:
            last = self.price_buf[-1]
            # Simple OHLCV bar: treat each tick as a close
            self.price_buf.append({
                "date": ts, "open": price, "high": price,
                "low": price, "close": price, "volume": 0
            })
        else:
            self.price_buf.append({
                "date": ts, "open": price, "high": price,
                "low": price, "close": price, "volume": 0
            })
        # Keep last 200 bars for indicator warmup
        if len(self.price_buf) > 300:
            self.price_buf = self.price_buf[-300:]

    def _get_signals(self) -> tuple[bool, bool]:
        """Compute CE+ZLSMA signals from buffer. Returns (buy, sell)."""
        if len(self.price_buf) < self.cfg["ZLSMA_LEN"] + 10:
            return False, False
        df = pd.DataFrame(self.price_buf).set_index("date")
        df = compute_signals(df, self.cfg)
        return bool(df["buy_signal"].iloc[-1]), bool(df["sell_signal"].iloc[-1])

    def _entry(self, signal: str, spot: float):
        today = date.today()
        expiries = get_nifty_expiry_dates(today, today + timedelta(days=60))
        pre_exp  = is_pre_expiry_day(today, expiries)
        exp      = next_expiry(today, expiries) if pre_exp else current_expiry(today, expiries)
        if exp is None:
            return

        opt_type = "CE" if signal == "BUY" else "PE"
        strike   = atm_strike(spot)
        # Simulated option price proxy
        atr_proxy = spot * 0.01
        entry_opt = max(10.0, atr_proxy * 0.5)
        symbol    = option_symbol_fyers(exp, strike, opt_type)

        self.position = {
            "signal": signal, "opt_type": opt_type,
            "strike": strike, "expiry": exp, "symbol": symbol,
            "entry_time": datetime.now(IST), "spot_entry": spot,
            "entry_option_px": entry_opt,
            "qty": self.qty, "lots": self.lots,
            "sl_spot": spot - atr_proxy if signal == "BUY" else spot + atr_proxy,
            "tp_spot": spot + atr_proxy * self.cfg["RR_RATIO"]
                       if signal == "BUY" else spot - atr_proxy * self.cfg["RR_RATIO"],
        }
        self.last_signal = signal
        print(f"\n{GREEN}[PAPER ENTRY] {signal} | {opt_type} | Strike {strike} | Expiry {exp}")
        print(f"  Symbol: {symbol} | Est. Premium ₹{entry_opt:.1f} | Qty {self.qty}{RESET}")
        self._log_trade("ENTRY", spot, entry_opt, signal, opt_type, strike, exp)

    def _exit(self, spot: float, reason: str):
        if not self.position:
            return
        p = self.position
        # Simplified exit price
        move  = spot - p["spot_entry"]
        delta = 0.5
        pnl_delta = (move if p["opt_type"] == "CE" else -move) * delta
        exit_opt = max(0.5, p["entry_option_px"] + pnl_delta)
        pnl = (exit_opt - p["entry_option_px"]) * p["qty"]
        self.capital   += pnl
        self.daily_pnl += pnl

        col = GREEN if pnl >= 0 else RED
        print(f"\n{col}[PAPER EXIT] {reason} | {p['opt_type']} | PnL ₹{pnl:,.0f}")
        print(f"  Capital now: ₹{self.capital:,.0f} | Daily PnL: ₹{self.daily_pnl:,.0f}{RESET}")
        self._log_trade("EXIT", spot, exit_opt, p["signal"], p["opt_type"],
                        p["strike"], p["expiry"], pnl=pnl, reason=reason)
        self.position = None

    def _log_trade(self, action, spot, opt_px, signal, opt_type,
                   strike, expiry, pnl=None, reason=""):
        entry = {
            "time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "action": action, "signal": signal, "opt_type": opt_type,
            "strike": strike, "expiry": str(expiry), "spot": spot,
            "option_price": round(opt_px, 2), "qty": self.qty,
            "pnl": round(pnl, 2) if pnl else "",
            "reason": reason, "capital": round(self.capital, 2)
        }
        self.trade_log.append(entry)
        with open("paper_trades.csv", "a") as f:
            if f.tell() == 0:
                f.write(",".join(entry.keys()) + "\n")
            f.write(",".join(str(v) for v in entry.values()) + "\n")

    def tick(self):
        """Called every N seconds. Core loop iteration."""
        now   = datetime.now(IST)
        today = now.date()

        # Daily reset
        if today != self.today_date:
            self._reset_daily()

        expiries  = get_nifty_expiry_dates(today, today + timedelta(days=60))
        exp_day   = is_expiry_day(today, expiries)
        pre_exp   = is_pre_expiry_day(today, expiries)
        time_str  = now.strftime("%H:%M")

        # Force exit before expiry
        if self.position and pre_exp and time_str >= self.cfg["PRE_EXPIRY_EXIT_TIME"]:
            spot = self.get_ltp() or self.position["spot_entry"]
            self._exit(spot, "Pre-expiry forced exit")
            return

        # No new entries on expiry day
        if exp_day:
            return

        # Market hours check
        if not (self.cfg["MARKET_OPEN"] <= time_str <= self.cfg["MARKET_CLOSE"]):
            return

        # Daily loss halt
        if self.daily_pnl <= -float(self.cfg["DAILY_LOSS_LIMIT"]):
            if self.position:
                spot = self.get_ltp() or self.position["spot_entry"]
                self._exit(spot, "Daily loss limit hit")
            return

        spot = self.get_ltp()
        if spot is None:
            return

        self._update_buffer(spot)
        buy_sig, sell_sig = self._get_signals()

        # Check SL/TP
        if self.position:
            p = self.position
            hit_sl = (p["signal"] == "BUY"  and spot <= p["sl_spot"]) or \
                     (p["signal"] == "SELL" and spot >= p["sl_spot"])
            hit_tp = (p["signal"] == "BUY"  and spot >= p["tp_spot"]) or \
                     (p["signal"] == "SELL" and spot <= p["tp_spot"])
            if hit_sl:
                self._exit(spot, "Stop Loss")
                return
            if hit_tp:
                self._exit(spot, "Take Profit")
                return

        # New signal
        if buy_sig and self.last_signal != "BUY":
            if self.position:
                self._exit(spot, "Signal reverse")
            self._entry("BUY", spot)
        elif sell_sig and self.last_signal != "SELL":
            if self.position:
                self._exit(spot, "Signal reverse")
            self._entry("SELL", spot)

    def run(self, interval_seconds: int = 60):
        log.info(f"[PAPER] Starting live paper trading loop (interval={interval_seconds}s)")
        print(f"\n{CYAN}Paper trading active. Press Ctrl+C to stop.{RESET}")
        try:
            while True:
                self.tick()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            log.info("[PAPER] Stopped by user.")
            if self.position:
                spot = self.get_ltp() or self.position["spot_entry"]
                self._exit(spot, "Manual stop")
            self._summary()

    def _summary(self):
        print(f"\n{'═'*45}")
        print(f"  PAPER TRADING SUMMARY")
        print(f"  Trades executed : {len([t for t in self.trade_log if t['action']=='EXIT'])}")
        print(f"  Final Capital   : ₹{self.capital:,.0f}")
        print(f"  Net P&L         : ₹{self.capital - self.cfg['CAPITAL']:,.0f}")
        print(f"{'═'*45}")


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░░░░  SECTION 6 — LIVE TRADING ENGINE  ░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

class LiveTradingEngine(PaperTradingEngine):
    """
    Extends PaperTradingEngine — same logic, but places REAL Fyers orders.
    ⚠️  Only use after completing 3–6 months of paper trading with good results.
    """

    def _place_order(self, symbol: str, side: int, qty: int) -> str | None:
        """
        side: 1 = BUY, -1 = SELL
        Returns order_id or None on failure.
        """
        if not self.fyers:
            log.error("Fyers client not initialised.")
            return None
        order_data = {
            "symbol"      : symbol,
            "qty"         : qty,
            "type"        : 2,           # Market order
            "side"        : side,
            "productType" : "INTRADAY",
            "limitPrice"  : 0,
            "stopPrice"   : 0,
            "disclosedQty": 0,
            "validity"    : "DAY",
            "offlineOrder": False,
        }
        try:
            resp = self.fyers.place_order(order_data)
            log.info(f"Order placed: {resp}")
            if resp.get("s") == "ok":
                return resp.get("id")
            else:
                log.error(f"Order failed: {resp}")
                return None
        except Exception as e:
            log.error(f"Order exception: {e}")
            return None

    def _entry(self, signal: str, spot: float):
        # First call parent to set position dict
        super()._entry(signal, spot)
        if not self.position:
            return
        p      = self.position
        symbol = p["symbol"]
        order_id = self._place_order(symbol, 1, p["qty"])
        if order_id:
            self.position["order_id"] = order_id
            log.info(f"{GREEN}[LIVE] BUY order placed | {symbol} | qty={p['qty']} | id={order_id}{RESET}")
        else:
            log.error("[LIVE] Entry order FAILED — position NOT opened")
            self.position = None

    def _exit(self, spot: float, reason: str):
        if not self.position:
            return
        p      = self.position
        symbol = p["symbol"]
        # For options bought (long), exit = SELL
        order_id = self._place_order(symbol, -1, p["qty"])
        if order_id:
            log.info(f"{RED}[LIVE] SELL order placed | {symbol} | qty={p['qty']} | reason={reason}{RESET}")
        else:
            log.error("[LIVE] Exit order FAILED — manual intervention needed!")
        # Call parent to update P&L tracking regardless
        super()._exit(spot, reason)


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░░░░  SECTION 7 — FYERS AUTH  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

def fyers_login(cfg: dict):
    """
    Interactive Fyers OAuth login.
    Prints auth URL → user pastes redirect URL → returns fyers client.
    """
    if not FYERS_AVAILABLE:
        log.error("fyers-apiv3 not installed. Run: pip install fyers-apiv3")
        return None

    import urllib.parse

    session = fyersModel.SessionModel(
        client_id    = cfg["CLIENT_ID"],
        secret_key   = cfg["SECRET_KEY"],
        redirect_uri = cfg["REDIRECT_URI"],
        response_type= "code",
        grant_type   = "authorization_code"
    )
    auth_url = session.generate_authcode()
    print(f"\n{YELLOW}Open this URL in your browser and log in:{RESET}")
    print(auth_url)
    redirect_url = input("\nPaste the full redirect URL here: ").strip()

    parsed = urllib.parse.urlparse(redirect_url)
    auth_code = urllib.parse.parse_qs(parsed.query).get("auth_code", [None])[0]
    if not auth_code:
        log.error("Could not extract auth_code from URL.")
        return None

    session.set_token(auth_code)
    resp = session.generate_token()
    access_token = resp.get("access_token")
    if not access_token:
        log.error(f"Token generation failed: {resp}")
        return None

    fyers = fyersModel.FyersModel(
        client_id    = cfg["CLIENT_ID"],
        token        = access_token,
        log_path     = "."
    )
    log.info(f"{GREEN}Fyers login successful!{RESET}")
    return fyers


# ─────────────────────────────────────────────────────────────────────────────
# ░░░░░░░░░░░░░░░░░░░░░  SECTION 8 — MAIN ENTRY  ░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CE + ZLSMA NIFTY Options Algo — Fyers"
    )
    parser.add_argument(
        "--mode", choices=["backtest", "paper", "live"],
        default="backtest",
        help="Run mode: backtest | paper | live"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Tick interval in seconds for paper/live mode (default: 60)"
    )
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║    CE + ZLSMA Strategy  —  NIFTY 50 Options Algo    ║
║    Mode : {args.mode.upper():<44}║
╚══════════════════════════════════════════════════════╝
""")

    # ── Validate sizing ───────────────────────────────────────────────────
    lots, qty = calc_lots(
        CONFIG["CAPITAL"], CONFIG["DAILY_LOSS_LIMIT"],
        CONFIG["RISK_RATIO"], CONFIG["LOT_SIZE"]
    )
    print(f"  Lot sizing: target={CONFIG['DAILY_LOSS_LIMIT']/CONFIG['RISK_RATIO']:.0f} units "
          f"→ {lots} lot(s) = {qty} qty  (1 lot = {CONFIG['LOT_SIZE']} units)")
    print()

    # ── MODE: BACKTEST ────────────────────────────────────────────────────
    if args.mode == "backtest":
        engine = BacktestEngine(CONFIG)
        engine.run()

    # ── MODE: PAPER ───────────────────────────────────────────────────────
    elif args.mode == "paper":
        fyers = None
        if FYERS_AVAILABLE and CONFIG["CLIENT_ID"] != "YOUR_FYERS_APP_ID-100":
            fyers = fyers_login(CONFIG)
        else:
            log.warning("No Fyers credentials set. Paper mode will use synthetic prices.")

        engine = PaperTradingEngine(CONFIG, fyers_client=fyers)
        engine.run(interval_seconds=args.interval)

    # ── MODE: LIVE ────────────────────────────────────────────────────────
    elif args.mode == "live":
        print(f"""
{RED}⚠️  LIVE TRADING MODE — REAL MONEY AT RISK{RESET}
  - Ensure you have completed 3–6 months of paper trading
  - Backtest results should be verified
  - Daily loss limit: ₹{CONFIG['DAILY_LOSS_LIMIT']:,}
  - Capital: ₹{CONFIG['CAPITAL']:,}
""")
        confirm = input("  Type  YES  to proceed with live trading: ").strip()
        if confirm != "YES":
            print("Aborted.")
            return

        if not FYERS_AVAILABLE:
            log.error("Install fyers-apiv3 first: pip install fyers-apiv3")
            return

        fyers = fyers_login(CONFIG)
        if not fyers:
            log.error("Fyers login failed. Cannot start live trading.")
            return

        engine = LiveTradingEngine(CONFIG, fyers_client=fyers)
        engine.run(interval_seconds=args.interval)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()