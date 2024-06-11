import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import csv
import os
from scipy.signal import butter, filtfilt
import scipy
#from PyEMD import EMD
import emd
import time
from scipy.interpolate import CubicSpline

def full_emd(signal, fs):
    M = int(fs*0.12)
    no_baseline = remove_baseline_wander(signal, fs)

    IMFs = decompose_signal_into_imfs(no_baseline)
    sntks = [moving_window_integration(nonlinear_transform(imf), M) for imf in IMFs]
    z_n = np.sum(sntks, axis=0)
    z_filtered = low_pass_filter(z_n, 1 , fs)
    final_ecg = normalize_data(z_filtered)
    r_positions = qrs_localization(final_ecg, fs)
    return r_positions

def decompose_signal_into_imfs(signal):
    # Utilisation de 'emd.sift.sift' pour calculer les IMFs
    imf = emd.sift.sift(signal)
    imfs = np.array(imf).T[:2]
    return imfs

def moving_window_integration(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def low_pass_filter(data, cutoff_frequency, fs):
    order = 1
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def nonlinear_transform(x):
    y = np.zeros_like(x)
    for n in range(2, len(x)):
        if np.sign(x[n]) == np.sign(x[n-1]) == np.sign(x[n-2]):
            y[n] = abs(x[n] * x[n-1] * x[n-2])
        else:
            y[n] = 0
    return y

def normalize_data(data):
    #return data
    max_value =  np.max(data) # * 0.1 
    normalized_signal = np.where(data >= max_value, 1, data / max_value)
    return normalized_signal

def remove_baseline_wander(signal, fs):
    order = 5
    nyquist = 0.5 * fs
    normal_cutoff = 1 / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def qrs_localization(h, fs):
    i = np.argmax(h[:int(fs*0.5)])
    list_peaks = [i]
    limite_suite = int(fs*0.12)
    previous_val = h[i+limite_suite]
    last_peak = i
    i = i + limite_suite
    while i < len(h):
        if h[i] > previous_val:
            peakos = i + np.argmax(h[i:i+int(fs*0.4)])
            if last_peak != list_peaks[-1]:
                if peakos - last_peak < int(fs*0.3):
                    list_peaks.append(peakos)
                else:
                    list_peaks.append(last_peak)
                
            last_peak = peakos
            i = peakos + limite_suite

        if i < len(h):
            previous_val = h[i]
        i += 1

    return list_peaks