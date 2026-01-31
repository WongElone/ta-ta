# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
import pandas as pd
cimport numpy as cnp
cimport cython
from libc.math cimport NAN

# Type definitions for better performance
ctypedef cnp.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_t, ndim=1] _core_zigzag(
    const DTYPE_t[::1] highs,
    const DTYPE_t[::1] lows,
    double up_percent,
    double down_percent,
    bint initial_up
):
    """
    Core ZigZag computation with Cython optimizations.
    
    Parameters:
    -----------
    highs : memoryview of float64 array
        High prices
    lows : memoryview of float64 array  
        Low prices
    up_percent : double
        Upward deviation percentage as decimal (e.g., 0.05 for 5%)
    down_percent : double
        Downward deviation percentage as decimal (e.g., 0.05 for 5%)
    initial_up : bint
        True if assuming initial uptrend, False for downtrend
    
    Returns:
    --------
    cnp.ndarray[DTYPE_t, ndim=1]
        ZigZag values with NaN for non-pivot points
    """
    cdef Py_ssize_t length = highs.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] zz = np.full(length, NAN, dtype=np.float64)
    
    cdef double extreme_price
    cdef Py_ssize_t extreme_idx = 0
    cdef int trend  # 1 for up, -1 for down
    cdef Py_ssize_t i
    cdef double current_high, current_low
    cdef double reversal_threshold
    
    # Initialize based on direction
    if initial_up:
        extreme_price = highs[0]
        trend = 1  # looking up
    else:
        extreme_price = lows[0]
        trend = -1  # looking down
    
    # Main loop - optimized for speed
    for i in range(1, length):
        current_high = highs[i]
        current_low = lows[i]
        
        if trend == 1:  # Looking for swing high
            # Update potential high
            if current_high > extreme_price:
                extreme_price = current_high
                extreme_idx = i
            
            # Check for downward reversal
            reversal_threshold = extreme_price * (1.0 - down_percent)
            if current_low <= reversal_threshold:
                zz[extreme_idx] = extreme_price  # Confirm swing high
                trend = -1
                extreme_price = current_low
                extreme_idx = i
        else:  # trend == -1, looking for swing low
            # Update potential low
            if current_low < extreme_price:
                extreme_price = current_low
                extreme_idx = i
            
            # Check for upward reversal
            reversal_threshold = extreme_price * (1.0 + up_percent)
            if current_high >= reversal_threshold:
                zz[extreme_idx] = extreme_price  # Confirm swing low
                trend = 1
                extreme_price = current_high
                extreme_idx = i
    
    return zz

cpdef k_zigzag(df, high_col, low_col, up_deviation_percent, down_deviation_percent):
    """
    Computes the ZigZag indicator on OHLCV data using High and Low prices.
    
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
    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError("DataFrame must contain high_col and low_col columns.")
    
    if len(df) < 2:
        return pd.Series(np.full(len(df), np.nan), index=df.index, name='ZigZag')
    
    cdef double up_percent = (up_deviation_percent / 100.0)
    cdef double down_percent = (down_deviation_percent / 100.0)
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] highs_array = df[high_col].to_numpy(dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] lows_array = df[low_col].to_numpy(dtype=np.float64)
    
    cdef const DTYPE_t[::1] highs_view = highs_array
    cdef const DTYPE_t[::1] lows_view = lows_array
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] zz_up = _core_zigzag(highs_view, lows_view, up_percent, down_percent, True)
    cdef cnp.ndarray[DTYPE_t, ndim=1] zz_down = _core_zigzag(highs_view, lows_view, up_percent, down_percent, False)
    
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] pivots_up = np.where(~np.isnan(zz_up))[0]
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] pivots_down = np.where(~np.isnan(zz_down))[0]
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] data
    cdef cnp.npy_intp first_up_idx, first_down_idx
    
    if len(pivots_up) == 0:
        data = zz_down
    elif len(pivots_down) == 0:
        data = zz_up
    else:
        first_up_idx = pivots_up[0]
        first_down_idx = pivots_down[0]
        data = zz_up if first_up_idx <= first_down_idx else zz_down
    
    return pd.Series(data, index=df.index, name='ZigZag')