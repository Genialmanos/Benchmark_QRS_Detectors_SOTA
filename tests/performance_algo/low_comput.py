from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import scipy
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def low_comput(ecg_signal, fs):
    k = None
    if fs == 250:
        k = fs//50
    else:
        k = fs//60
    filtered_signal = apply_fir_filter(ecg_signal, k)
    squared_signal = filtered_signal ** 2
    QRS_width = int(0.15 * fs) # test 0.15 comme l'article au lieu de 0.12
    smoothed_signal = moving_average(squared_signal, QRS_width)
    normalized_signal = normalize_operation(smoothed_signal)
    thv = threshold(normalized_signal, QRS_width)
    r_peaks_01 = [1 if thv[i] <= normalized_signal[i] else 0 for i in range(len(thv))]
    r_peaks = [i for i in range(len(r_peaks_01)) if r_peaks_01[i] == 1]
    r_peaks = regroup_nul(np.array(r_peaks), 36)
    Rsi = int(0.24 * fs)
    a = - Rsi
    tab = []

    for value in r_peaks:
        if value >= a + Rsi:
            tab.append(value)
            a = value
            
    r_peaks_real = super_filtre(ecg_signal, tab, Rsi)

    meanos = np.mean(normalized_signal)
    listos2 = [a for a in r_peaks_real if normalized_signal[a] >= meanos]

    return listos2

def calcul_AVE_RR(list_frames):
    return sum(np.diff(list_frames[-10:]))/len(list_frames)


def super_filtre(ecg_signal, r_peaks, Rsi):
    WQRS = Rsi-1 if Rsi%2 == 0 else Rsi         #(12)
    H_WQRS = WQRS//2
    list_peak = []
    L = 10
    for idx, peak in enumerate(r_peaks[10:]):
        sous_tab = r_peaks[idx-L:idx+L]
        
        direction = ( (1/(2*L)) * sum([ecg_signal[aaa] for aaa in sous_tab]) ) #13.a
        if list_peak == []:
            for p in r_peaks[:10]:
                list_peak.append(choose_max(ecg_signal, H_WQRS, p, -1))
                continue
        list_peak.append(choose_max(ecg_signal, H_WQRS, peak, direction))
    
    return list_peak

def threshold(signal, QRS_width):
    N = QRS_width
    thv = [1]
    for i in signal:
        thv.append(((N-1) * thv[-1] + i )/ N)
    return thv[1:]

def regroup_nul(peaks, thr):
    diff = peaks[1:]-peaks[:-1]
    gps = np.concatenate([[0], np.cumsum(diff>=thr)])
    temp = [peaks[gps==i] for i in range(gps[-1]+1)]
    return [np.mean(sublist).astype(int) for sublist in temp]

def choose_max(ecg_signal, H_WQRS, peak, direction):
    if direction > 0:
        return peak - H_WQRS + np.argmax(ecg_signal[peak-(H_WQRS):peak+(H_WQRS)])
    else:
        return peak - H_WQRS + np.argmin(ecg_signal[peak-(H_WQRS):peak+(H_WQRS)])
    
def apply_fir_filter(signal, k):
    y0 = np.zeros_like(signal)
    #bk = [1, 0, 0, 0, 0, 0, -1]
    for i in range(k, len(signal)):
        y0[i] = signal[i] + (-1 * signal[i-k])
    #y0[k:] = signal[k:] - signal[:-k]
    return y0

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def normalize_operation(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))