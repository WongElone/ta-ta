# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
import pandas as pd
cimport numpy as cnp
cimport cython
from libc.math cimport fabs
from cpython.mem cimport PyMem_Malloc, PyMem_Free

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int8_t INT8_t


cdef struct Segment:
    Py_ssize_t start_idx
    Py_ssize_t end_idx
    double max_distance
    Py_ssize_t max_idx
    int direction


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_max_distance_point(
    const DTYPE_t[::1] values,
    Py_ssize_t start_idx,
    Py_ssize_t end_idx,
    double y_start,
    double y_end,
    double* out_max_distance,
    Py_ssize_t* out_max_idx,
    int* out_direction
) noexcept nogil:
    """
    Find the point with maximum vertical distance to the line segment.
    
    Parameters:
    -----------
    values : memoryview
        Value array
    start_idx, end_idx : Py_ssize_t
        Segment boundaries (row indices)
    y_start, y_end : double
        Y-values at start and end of segment
    out_max_distance : double*
        Output: maximum distance found
    out_max_idx : Py_ssize_t*
        Output: index of point with max distance
    out_direction : int*
        Output: 1 if point is above line, -1 if below
    """
    cdef Py_ssize_t i
    cdef double slope, y_line, dist, diff
    cdef Py_ssize_t span = end_idx - start_idx
    
    out_max_distance[0] = 0.0
    out_max_idx[0] = -1
    out_direction[0] = 0
    
    if span <= 1:
        return
    
    slope = (y_end - y_start) / span
    
    for i in range(start_idx + 1, end_idx):
        y_line = y_start + slope * (i - start_idx)
        diff = values[i] - y_line
        dist = fabs(diff)
        
        if dist > out_max_distance[0]:
            out_max_distance[0] = dist
            out_max_idx[0] = i
            out_direction[0] = 1 if diff > 0 else -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void heap_push(Segment* heap, Py_ssize_t* heap_size, Segment seg) noexcept nogil:
    """Push segment onto max-heap (max distance at root)."""
    cdef Py_ssize_t i = heap_size[0]
    cdef Py_ssize_t parent
    
    heap[i] = seg
    heap_size[0] += 1
    
    while i > 0:
        parent = (i - 1) // 2
        if heap[parent].max_distance >= heap[i].max_distance:
            break
        heap[i], heap[parent] = heap[parent], heap[i]
        i = parent


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Segment heap_pop(Segment* heap, Py_ssize_t* heap_size) noexcept nogil:
    """Pop segment with max distance from heap."""
    cdef Segment result = heap[0]
    cdef Py_ssize_t i, left, right, largest
    cdef Segment temp
    
    heap_size[0] -= 1
    if heap_size[0] > 0:
        heap[0] = heap[heap_size[0]]
        
        i = 0
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i
            
            if left < heap_size[0] and heap[left].max_distance > heap[largest].max_distance:
                largest = left
            if right < heap_size[0] and heap[right].max_distance > heap[largest].max_distance:
                largest = right
            
            if largest == i:
                break
            
            temp = heap[i]
            heap[i] = heap[largest]
            heap[largest] = temp
            i = largest
    
    return result


cpdef pips(df, col, pts):
    """
    Computes the PIPs (Perceptually Important Points) indicator on time series data.
    
    The algorithm identifies significant swing points based on perceptual importance 
    via recursive line simplification.
    
    1. Start with the first and last point of the time series — these are always PIPs.
    2. Determine an imaginary straight line between them in point-slope form. 
       (Row number is used as x-axis to keep consistent regardless of DataFrame index type)
    3. Find the point between them that has the largest absolute vertical distance to this line.
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
    
    Returns:
    --------
    pd.Series
        Series with int values: 1 if PIP is above the line, -1 if below, 0 if not a PIP.
    """
    if col not in df.columns:
        raise ValueError("DataFrame must contain col column.")
    
    cdef Py_ssize_t length = len(df)
    
    if length < 2:
        return pd.Series(np.zeros(length, dtype=np.int8), index=df.index, name='PIP')
    
    if pts < 2:
        pts = 2
    
    if pts > length:
        pts = length
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] values_array = df[col].to_numpy(dtype=np.float64)
    cdef cnp.ndarray[INT8_t, ndim=1] result = np.zeros(length, dtype=np.int8)
    
    cdef const DTYPE_t[::1] values = values_array
    cdef INT8_t[::1] result_view = result
    
    cdef Py_ssize_t start_idx = 0
    cdef Py_ssize_t end_idx = length - 1
    cdef double y_start, y_end
    
    y_start = values[0]
    y_end = values[end_idx]
    
    # Reason: First and last points are PIPs with arbitrary direction of 1
    result_view[start_idx] = 1
    result_view[end_idx] = 1
    
    cdef Py_ssize_t current_pips = 2
    
    if pts <= 2 or length <= 2:
        return pd.Series(result, index=df.index, name='PIP')
    
    cdef Segment* heap = <Segment*>PyMem_Malloc(length * sizeof(Segment))
    if not heap:
        raise MemoryError("Failed to allocate heap memory")
    
    cdef Py_ssize_t heap_size = 0
    cdef Segment initial_seg, current_seg, new_seg1, new_seg2
    cdef double max_distance
    cdef Py_ssize_t max_idx
    cdef int direction
    
    try:
        find_max_distance_point(
            values, start_idx, end_idx, y_start, y_end,
            &max_distance, &max_idx, &direction
        )
        
        if max_idx > 0:
            initial_seg.start_idx = start_idx
            initial_seg.end_idx = end_idx
            initial_seg.max_distance = max_distance
            initial_seg.max_idx = max_idx
            initial_seg.direction = direction
            heap_push(heap, &heap_size, initial_seg)
        
        while heap_size > 0 and current_pips < pts:
            current_seg = heap_pop(heap, &heap_size)
            
            result_view[current_seg.max_idx] = current_seg.direction
            current_pips += 1
            
            y_start = values[current_seg.start_idx]
            y_end = values[current_seg.max_idx]
            
            find_max_distance_point(
                values, current_seg.start_idx, current_seg.max_idx, y_start, y_end,
                &max_distance, &max_idx, &direction
            )
            
            if max_idx > 0:
                new_seg1.start_idx = current_seg.start_idx
                new_seg1.end_idx = current_seg.max_idx
                new_seg1.max_distance = max_distance
                new_seg1.max_idx = max_idx
                new_seg1.direction = direction
                heap_push(heap, &heap_size, new_seg1)
            
            y_start = values[current_seg.max_idx]
            y_end = values[current_seg.end_idx]
            
            find_max_distance_point(
                values, current_seg.max_idx, current_seg.end_idx, y_start, y_end,
                &max_distance, &max_idx, &direction
            )
            
            if max_idx > 0:
                new_seg2.start_idx = current_seg.max_idx
                new_seg2.end_idx = current_seg.end_idx
                new_seg2.max_distance = max_distance
                new_seg2.max_idx = max_idx
                new_seg2.direction = direction
                heap_push(heap, &heap_size, new_seg2)
        
        return pd.Series(result, index=df.index, name='PIP')
    
    finally:
        PyMem_Free(heap)
