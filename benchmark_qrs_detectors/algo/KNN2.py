from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import scipy
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import os
import random

def KNN2(sig, fs):
    band_passed = bandpass_filter(sig, fs)
    gradient = np.square(np.gradient(band_passed).tolist())
    x = 35
    gradient = moving_average(gradient, x+1)
    gradient = np.concatenate((gradient, np.zeros(x)))
    qrs = qrs_localization(gradient)

    if np.any(np.diff(qrs) >= fs * 1):
        gradient = np.abs(np.gradient(band_passed).tolist())
        x = 35
        gradient = moving_average(gradient, x+1)
        gradient = np.concatenate((gradient, np.zeros(x)))
        qrs = qrs_localization(gradient)
    return qrs

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size), 'valid') / window_size

def bandpass_filter(x, fs):
    nyq = 0.5 * fs
    f_high = 5 / nyq  # Highpass filter cutoff frequency
    f_low = 12 / nyq  # Lowpass filter cutoff frequency
    order_high = 4  # Highpass filter order
    order_low = 4  # Lowpass filter order

    # Highpass filter
    b_high, a_high = scipy.signal.butter(order_high, f_high, 'high')
    x_highpass = scipy.signal.lfilter(b_high, a_high, x)

    # Lowpass filter
    b_low, a_low = scipy.signal.butter(order_low, f_low, 'low')
    y = scipy.signal.lfilter(b_low, a_low, x_highpass)

    return y

def qrs_localization(h):
    qrs_indices = []
    sous_groupe = []
    w = 5000
    threshold = np.mean(h) #XX * max(h[:w])
    for i in range(len(h)):
        #if i%w == 0 and i != 0:
        #    threshold = XX * max(h[i-(w//2):i+(w//2)])
        #print(h[i])
        if h[i] >= threshold:
            sous_groupe.append(i)
        elif sous_groupe != [] and i - sous_groupe[-1] >= 36: # 36 = valeur donnée dans l'article
            qrs_indices.append(max([(x, h[x]) for x in sous_groupe], key=lambda x: x[1])[0])
            sous_groupe = []
    if sous_groupe != []:
        qrs_indices.append(max([(x, h[x]) for x in sous_groupe], key=lambda x: x[1])[0])
    return qrs_indices