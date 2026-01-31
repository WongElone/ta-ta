import pandas as pd

def k_zigzag_s(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    up_deviation_percent: float,
    down_deviation_percent: float
) -> pd.Series:
    """
    Identifies zigzag peaks and troughs on OHLCV data using High and Low prices.
    
    The algorithm identifies significant swing highs (peaks) and swing lows (troughs)
    based on minimum percentage reversals.
    
    - This is a repainting indicator (the last point updates as new data arrives).
    - To handle the initial direction ambiguity, the function runs the core algorithm twice
      (once assuming initial uptrend, once assuming initial downtrend) and selects the version 
      that detects the first pivot earliest.
    - Returns a pd.Series with True for peaks, False for troughs, and None elsewhere.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain high_col and low_col columns (case-sensitive). Index can be integer or datetime.
    high_col : str
        Column name for high prices.
    low_col : str
        Column name for low prices.
    up_deviation_percent : float
        Minimum percentage for upward reversal (from low to high).
    down_deviation_percent : float
        Minimum percentage for downward reversal (from high to low).
    
    Returns:
    --------
    pd.Series
        Series with 1 for peaks (swing highs), -1 for troughs (swing lows), 
        and 0 for other datapoints.
    """
    ...
