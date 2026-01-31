import pandas as pd

def k_zigzag(
    df: pd.DataFrame, 
    high_col: str, 
    low_col: str, 
    up_deviation_percent: float,
    down_deviation_percent: float
) -> pd.Series:
    """
    Computes the ZigZag indicator on k-lines data using High and Low prices.
    
    The algorithm identifies significant swing highs (using High) and swing lows (using Low)
    based on minimum percentage reversals.
    
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
        ZigZag values aligned with the input DataFrame index.
    """
    ...