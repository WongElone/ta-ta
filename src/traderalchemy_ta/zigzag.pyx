# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
import pandas as pd
cimport numpy as cnp
cimport cython
from libc.math cimport NAN

ctypedef cnp.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_t, ndim=1] _core_zigzag(
    const DTYPE_t[::1] values,
    double up_percent,
    double down_percent,
    bint initial_up
):
    """
    Core ZigZag computation with Cython optimizations.
    
    Parameters:
    -----------
    values : memoryview of float64 array
        Price values
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
    cdef Py_ssize_t length = values.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] zz = np.full(length, NAN, dtype=np.float64)
    
    cdef double extreme_price
    cdef Py_ssize_t extreme_idx = 0
    cdef int trend
    cdef Py_ssize_t i
    cdef double current_value
    cdef double reversal_threshold
    
    if initial_up:
        extreme_price = values[0]
        trend = 1
    else:
        extreme_price = values[0]
        trend = -1
    
    for i in range(1, length):
        current_value = values[i]
        
        if trend == 1:
            if current_value > extreme_price:
                extreme_price = current_value
                extreme_idx = i
            
            reversal_threshold = extreme_price * (1.0 - down_percent)
            if current_value <= reversal_threshold:
                zz[extreme_idx] = extreme_price
                trend = -1
                extreme_price = current_value
                extreme_idx = i
        else:
            if current_value < extreme_price:
                extreme_price = current_value
                extreme_idx = i
            
            reversal_threshold = extreme_price * (1.0 + up_percent)
            if current_value >= reversal_threshold:
                zz[extreme_idx] = extreme_price
                trend = 1
                extreme_price = current_value
                extreme_idx = i
    
    return zz

cpdef zigzag(df, col, up_deviation_percent, down_deviation_percent):
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
        ZigZag values aligned with the input DataFrame index.
    """
    if col not in df.columns:
        raise ValueError(f"DataFrame must contain column '{col}'.")
    
    if len(df) < 2:
        return pd.Series(np.full(len(df), np.nan), index=df.index, name='ZigZag')
    
    cdef double up_percent = (up_deviation_percent / 100.0)
    cdef double down_percent = (down_deviation_percent / 100.0)
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] values_array = df[col].to_numpy(dtype=np.float64)
    cdef const DTYPE_t[::1] values_view = values_array
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] zz_up = _core_zigzag(values_view, up_percent, down_percent, True)
    cdef cnp.ndarray[DTYPE_t, ndim=1] zz_down = _core_zigzag(values_view, up_percent, down_percent, False)
    
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
