import pandas as pd

def pips(
    df: pd.DataFrame,
    col: str,
    pts: int
) -> pd.Series:
    """
    Computes the PIPs (Perceptually Important Points) indicator on time series data.
    
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
        Must contain col column (case-sensitive). Index can be integer or datetime.
    col : str
        Column name for analysis.
    pts : int
        Desired number of perceptually important points
    return : pd.Series (int)
        a series of int, 1 if col value is a PIP above the line, -1 if below the line, 0 if not a PIP
    
    Returns:
    --------
    pd.Series
        PIPs aligned with the input DataFrame index.
        Values: 1 for PIP concave down, -1 for PIP concave up, 0 for non-PIP.
    """