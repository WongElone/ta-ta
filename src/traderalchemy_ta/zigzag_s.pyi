import pandas as pd

def zigzag_s(
    df: pd.DataFrame, 
    col: str, 
    up_deviation_percent,
    down_deviation_percent
) -> pd.Series:
    """
    Computes the ZigZag indicator on any time series data.
    
    The algorithm identifies significant swing highs and swing lows based on minimum percentage reversals.
    
    - This is a repainting indicator (the last leg updates as new data arrives).
    - To handle the initial direction ambiguity, the function runs the core algorithm twice
      (once assuming initial uptrend from a potential high, once assuming initial downtrend
      from a potential low) and selects the version that detects the first swing pivot earliest.
    - Returns a pd.Series with the ZigZag price at pivot points (High at swing highs,
      Low at swing lows) and NaN elsewhere. Plotting this series will connect the pivots
      with straight lines in most charting libraries.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain col (case-sensitive). Index can be integer or datetime.
    col : str
        Column name for analysis.
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