import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from scipy.signal import resample
from scipy.interpolate import UnivariateSpline
import pickle

def CNN(ecg_signal, fs):
    cleaned_ecg = preprocessing(ecg_signal, fs)
    windows = extract_windows(cleaned_ecg, fs)
    
    # Redimensionner pour le modèle
    R = windows.reshape((windows.shape[0], 145, 1))    #X.reshape((X.shape[0], X.shape[1], 1))
    model =  pickle.load(open("model_CNN", 'rb'))
    predictos = model.predict(R).flatten()
    pred_frame = [a+36 for a in range(len(predictos)) if predos[a] >= 0.5]
    final_pred = regroup(np.array(pred_frame), 20)
    return final_pred

def regroup(peaks, thr):
    diff = peaks[1:]-peaks[:-1]
    gps = np.concatenate([[0], np.cumsum(diff>=thr)])
    temp = [peaks[gps==i] for i in range(gps[-1]+1)]
    max_sublist = []
    return [np.mean(sublist).astype(int) for sublist in temp]

def extract_windows(signal, fs):
    # Points avant et après basés sur la fréquence d'échantillonnage
    points_before = int(100 * fs / 1000)
    points_after = int(300 * fs / 1000)
    total_points = points_before + points_after + 1
    
    if total_points != 145:
        raise ValueError(f"La fenêtre calculée a {total_points} points, mais 145 sont attendus.")
    
    # Extraire les fenêtres pour chaque point dans cleaned_ecg
    windows = []
    for i in range(len(signal)):
        start = i - points_before
        end = i + points_after + 1
        if start >= 0 and end <= len(signal):
            window = signal[start:end]
            if len(window) == 145:
                windows.append(window)
    
    return np.array(windows)

def preprocessing(signal, fs):
    clean_baseline = baseline_wander_removal(signal)
    normalization = normalize_signal(clean_baseline)
    return normalization

def baseline_wander_removal(ecg_signal, window_size=4, sampling_rate=360, subsample_rate=200):
    # Convert window size to number of samples
    window_samples = window_size * sampling_rate
    
    # Initialize an empty array to store the corrected signal
    corrected_signal = np.zeros_like(ecg_signal)
    
    # Process the signal in windows
    for start in range(0, len(ecg_signal), window_samples):
        end = min(start + window_samples, len(ecg_signal))
        segment = ecg_signal[start:end]
        
        # Resample segment to reduce computational load
        resampled_segment = resample(segment, subsample_rate)
        
        # Perform LOESS regression
        x = np.linspace(0, len(resampled_segment) - 1, len(resampled_segment))
        spline = UnivariateSpline(x, resampled_segment, s=len(resampled_segment))
        baseline = spline(x)
        
        # Upsample the baseline back to the original sampling rate
        baseline_full = resample(baseline, len(segment))
        
        # Subtract the baseline from the original segment
        corrected_signal[start:end] = segment - baseline_full
    
    return corrected_signal

def normalize_signal(ecg_signal):
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    
    # Subtract mean and divide by standard deviation
    normalized_signal = (ecg_signal - mean_val) / std_val
    
    return normalized_signal