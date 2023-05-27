# src/hpcfinal/classification/metrics.pyx

import numpy as np
cimport numpy as np

cpdef np.ndarray[np.int_t, ndim=2] confusion_matrix(np.ndarray[np.int_t, ndim=1] y_true,
                                                   np.ndarray[np.int_t, ndim=1] y_pred,
                                                   int num_classes):
    cdef np.ndarray[np.int_t, ndim=2] cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    cdef int i, n = y_true.shape[0]
    
    for i in range(n):
        cm[y_true[i], y_pred[i]] += 1
    
    return cm

cpdef double accuracy(np.ndarray[np.int_t, ndim=1] y_true,
                      np.ndarray[np.int_t, ndim=1] y_pred):
    cdef int n_correct = np.sum(y_true == y_pred)
    cdef double acc = <double>n_correct / y_true.shape[0]
    return acc

cpdef double precision(np.ndarray[np.int_t, ndim=1] y_true,
                       np.ndarray[np.int_t, ndim=1] y_pred,
                       int class_label):
    cdef np.ndarray[np.int_t, ndim=2] cm = confusion_matrix(y_true, y_pred, num_classes)
    cdef int tp = cm[class_label, class_label]
    cdef int fp = np.sum(cm[:, class_label]) - tp
    
    if tp + fp == 0:
        return 0.0
    
    cdef double prec = <double>tp / (tp + fp)
    return prec

cpdef double recall(np.ndarray[np.int_t, ndim=1] y_true,
                    np.ndarray[np.int_t, ndim=1] y_pred,
                    int class_label):
    cdef np.ndarray[np.int_t, ndim=2] cm = confusion_matrix(y_true, y_pred, num_classes)
    cdef int tp = cm[class_label, class_label]
    cdef int fn = np.sum(cm[class_label, :]) - tp
    
    if tp + fn == 0:
        return 0.0
    
    cdef double rec = <double>tp / (tp + fn)
    return rec

cpdef double f1_score(np.ndarray[np.int_t, ndim=1] y_true,
                      np.ndarray[np.int_t, ndim=1] y_pred,
                      int class_label):
    cdef double prec = precision(y_true, y_pred, class_label)
    cdef double rec = recall(y_true, y_pred, class_label)
    
    if prec + rec == 0.0:
        return 0.0
    
    cdef double f1 = 2.0 * (prec * rec) / (prec + rec)
    return f1
