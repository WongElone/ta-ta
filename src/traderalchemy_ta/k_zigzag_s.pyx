# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
import pandas as pd
cimport numpy as cnp
cimport cython

# Type definitions for better performance
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int8_t INT8_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[INT8_t, ndim=1] _core_zigzag_peaks(
    const DTYPE_t[::1] highs,
    const DTYPE_t[::1] lows,
    double up_percent,
    double down_percent,
    bint initial_up
):
    """
    Core zigzag peak/trough detection with Cython optimizations.
    
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
    cnp.ndarray[INT8_t, ndim=1]
        Array with 1 for peaks, -1 for troughs, 0 for other points
    """
    cdef Py_ssize_t length = highs.shape[0]
    cdef cnp.ndarray[INT8_t, ndim=1] result = np.zeros(length, dtype=np.int8)
    
    cdef double extreme_price
    cdef Py_ssize_t extreme_idx = 0
    cdef int trend
    cdef Py_ssize_t i
    cdef double current_high, current_low
    cdef double reversal_threshold
    
    if initial_up:
        extreme_price = highs[0]
        trend = 1
    else:
        extreme_price = lows[0]
        trend = -1
    
    for i in range(1, length):
        current_high = highs[i]
        current_low = lows[i]
        
        if trend == 1:
            if current_high > extreme_price:
                extreme_price = current_high
                extreme_idx = i
            
            reversal_threshold = extreme_price * (1.0 - down_percent)
            if current_low <= reversal_threshold:
                result[extreme_idx] = 1
                trend = -1
                extreme_price = current_low
                extreme_idx = i
        else:
            if current_low < extreme_price:
                extreme_price = current_low
                extreme_idx = i
            
            reversal_threshold = extreme_price * (1.0 + up_percent)
            if current_high >= reversal_threshold:
                result[extreme_idx] = -1
                trend = 1
                extreme_price = current_high
                extreme_idx = i
    
    return result

cpdef k_zigzag_s(df, high_col, low_col, up_deviation_percent, down_deviation_percent):
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
        DataFrame containing OHLCV data.
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
    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError("DataFrame must contain high_col and low_col columns.")
    
    if len(df) < 2:
        return pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index, name='ZigZagPeaks')
    
    cdef double up_percent = up_deviation_percent / 100.0
    cdef double down_percent = down_deviation_percent / 100.0
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] highs_array = df[high_col].to_numpy(dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] lows_array = df[low_col].to_numpy(dtype=np.float64)
    
    cdef const DTYPE_t[::1] highs_view = highs_array
    cdef const DTYPE_t[::1] lows_view = lows_array
    
    cdef cnp.ndarray[INT8_t, ndim=1] zz_up = _core_zigzag_peaks(highs_view, lows_view, up_percent, down_percent, True)
    cdef cnp.ndarray[INT8_t, ndim=1] zz_down = _core_zigzag_peaks(highs_view, lows_view, up_percent, down_percent, False)
    
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] pivots_up = np.where(zz_up != 0)[0]
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] pivots_down = np.where(zz_down != 0)[0]
    
    cdef cnp.ndarray[INT8_t, ndim=1] data
    cdef cnp.npy_intp first_up_idx, first_down_idx
    
    if len(pivots_up) == 0:
        data = zz_down
    elif len(pivots_down) == 0:
        data = zz_up
    else:
        first_up_idx = pivots_up[0]
        first_down_idx = pivots_down[0]
        data = zz_up if first_up_idx <= first_down_idx else zz_down
    
    return pd.Series(data, index=df.index, name='ZigZagPeaks')
