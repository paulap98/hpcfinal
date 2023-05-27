# src/hpcfinal/regression/metrics.pyx

import numpy as np
cimport numpy as np
from math import sqrt

cpdef double mean_squared_error(np.ndarray[np.double_t, ndim=1] y_true,
                                np.ndarray[np.double_t, ndim=1] y_pred):
    cdef int n = y_true.shape[0]
    cdef double mse = np.sum(np.square(y_true - y_pred)) / n
    return mse

cpdef double root_mean_squared_error(np.ndarray[np.double_t, ndim=1] y_true,
                                     np.ndarray[np.double_t, ndim=1] y_pred):
    cdef double mse = mean_squared_error(y_true, y_pred)
    cdef double rmse = sqrt(mse)
    return rmse

cpdef double mean_absolute_error(np.ndarray[np.double_t, ndim=1] y_true,
                                 np.ndarray[np.double_t, ndim=1] y_pred):
    cdef int n = y_true.shape[0]
    cdef double mae = np.sum(np.abs(y_true - y_pred)) / n
    return mae
