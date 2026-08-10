"""Stock universes for backtesting.

Static lists of liquid, large/mid-cap US stocks (survivorship bias applies, as
with any static list — noted in the README).
"""

# Small universe for quick tests / first backtest pass.
SMALL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "CRM", "NFLX", "COST", "JPM", "UNH", "V", "LLY",
    "XOM", "HD", "MRK", "ADBE",
]

# Extended universe: ~S&P 100 + liquid growth names.
EXTENDED_UNIVERSE = sorted(set(SMALL_UNIVERSE + [
    "ABBV", "ABT", "ACN", "AIG", "AMAT", "AMGN", "AMT", "AXP", "BA", "BAC",
    "BK", "BKNG", "BLK", "BMY", "C", "CAT", "CB", "CI", "CL", "CMCSA",
    "COF", "COP", "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR",
    "ETN", "FDX", "GD", "GE", "GILD", "GM", "GS", "HON", "IBM", "INTC",
    "INTU", "ISRG", "JNJ", "KO", "LIN", "LMT", "LOW", "LRCX", "MA", "MCD",
    "MDLZ", "MDT", "MET", "MMC", "MMM", "MO", "MS", "MU", "NEE", "NKE",
    "NOW", "ORCL", "PEP", "PFE", "PG", "PGR", "PLTR", "PM", "PANW", "QCOM",
    "RTX", "SBUX", "SCHW", "SHOP", "SO", "SPGI", "T", "TGT", "TJX", "TMO",
    "TMUS", "TXN", "UBER", "UNP", "UPS", "USB", "VRTX", "VZ", "WFC", "WMT",
    "ANET", "APP", "CRWD", "DDOG", "DELL", "KLAC", "MRVL", "NET", "SMCI", "SNOW",
]))
