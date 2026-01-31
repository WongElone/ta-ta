import pandas as pd

def k_pips(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    pts: int
) -> pd.Series:
    """
    Computes the PIP (Perceptually Important Points) indicator on k-lines data using High and Low prices.
    (Or you can pass any other time-series data which contains two columns, one presents higher values, one presents lower values)
    
    The algorithm identifies significant swing highs (using High) and swing lows (using Low)
    based on perceptual importance via recursive line simplification.
    
    1. Start with the first and last point of the time series — these are always PIPs.
       - Selection logic: if high[0] - low[-1] > high[-1] - low[0], use high for first and low for last, vice versa.
    2. Determine an imaginary straight line between them in point-slope form. 
       (Row number is used as x-axis to keep consistent regardless of DataFrame index type)
    3. Find the point between them that has the largest absolute vertical distance to this line. 
       Take high or low whichever provides larger absolute vertical distance to the line.
    4. Add that farthest point as the next PIP → now you have two line segments.
    5. Repeat the process on each new segment, always processing the segment with the largest maximum distance.
    6. Continue until you reach the desired number of PIPs or all segments cannot be further divided 
       (i.e. there are no more gaps between adjacent PIPs).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain high_col and low_col columns (case-sensitive). Index can be integer or datetime.
    high_col : str
        Column name for high prices.
    low_col : str
        Column name for low prices.
    pts : int
        Desired number of perceptually important points
    return : pd.Series (int)
        a series of int, 1 if high is a PIP, -1 if low is a PIP, 0 if not a PIP
    
    Returns:
    --------
    pd.Series
        PIPs aligned with the input DataFrame index.
    """