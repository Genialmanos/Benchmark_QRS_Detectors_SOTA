import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import csv
import os
from scipy.signal import butter, filtfilt
#from PyEMD import EMD
import emd
import time
import pywt 
import biosppy.signals.ecg as bsp_ecg
import biosppy.signals.tools as bsp_tools
from scipy.signal import medfilt
from scipy.stats import norm
from scipy.signal import firwin, lfilter, stft, istft
from scipy.io import wavfile
from scipy.signal import stft
from scipy.signal.windows import blackman
from scipy.stats import skew, kurtosis

def clustering(ecg_signal, fs):
    ecg_signal_filtered = reduce_baseline_wandering(ecg_signal, fs)
    COWA = cowa_filter(ecg_signal_filtered, M1=5, M2=5, P=3, sigma=2.2)
    bandpassed = bandpass_filter(COWA, fs)
    dq_signal = diff_and_square(bandpassed)
    final_ecg = smooth_signal(dq_signal, fs)

    full_r_peaks = []
    step = len(ecg_signal)//5 #10000 #1000
    a = []
    for zone_min in range(0, len(final_ecg), step):
        zone_max = min(len(final_ecg), step + zone_min)
        A_th = fuzzy_c_median_clustering(final_ecg[zone_min:zone_max])
        a.append(A_th)
        peak_grouped = [a for a in range(zone_min, zone_max) if final_ecg[a] > A_th]
        r_peaks = regroup(final_ecg, np.array(peak_grouped), 3)#, zone_min)
        full_r_peaks.extend(r_peaks) #])

    A_th = np.mean(a)
    slopes = calculate_slopes(final_ecg)
    slope_th = np.percentile(slopes, 95)

    Q_n = np.zeros(len(final_ecg))
    Q_n[full_r_peaks] = final_ecg[full_r_peaks]
    peaks = find_r_peaks(Q_n, A_th, slope_th, fs, full_r_peaks[:])
    peaks = regroup(final_ecg, np.array(peaks), fs//3)

    return peaks #full_r_peaks


def find_r_peaks(x, Ath, slope_th, fs, TR):
    # Constants
    C = round(0.110 * fs)
    d = round(0.083 * fs)

    # Initialize Peaks array
    Peaks = []
    Peaks.append(TR[0])
    
    # Process each peak in TR
    for i in range(1, len(TR)):
        r0 = TR[i] - C
        r1 = TR[i] + C
        
        # Ensure indices are within bounds
        r0 = max(r0, 0)
        r1 = min(r1, len(x) - 1)
        
        # Determine the range and find the local max
        range_indices = range(r0, r1 + 1)
        maxVal, maxInd = max((x[j], j) for j in range_indices)
        TR[i] = maxInd
        
        # Determine dL and dR
        dL = TR[i] - range_indices[0]
        dR = range_indices[-1] - TR[i]
        
        # Calculate the slopes
        t_left = np.arange(TR[i] - dL, TR[i] + 1)
        t_right = np.arange(TR[i], TR[i] + dR + 1)
        
        #print(f"t_left = {t_left}")
        
        if len(t_left) == 1 or len(t_right) == 1:
            continue
        
        slope_l_max = max(np.gradient(x[t_left], t_left))
        slope_r_max = max(np.abs(np.gradient(x[t_right], t_right)))
        
        slope_min = min(slope_l_max, slope_r_max)
        
        # Calculate amplitude differences
        Delta_l = x[TR[i]] - x[max(TR[i] - d, 0)]
        Delta_r = x[min(TR[i] + d, len(x) - 1)] - x[TR[i]]
        
        # Check conditions
        if Delta_l > 0 and Delta_r < 0:
            #Ath = fuzzy_c_median_clustering(x[min(0,i-5000):max(len(x), i+5000)])
            if x[TR[i]] > Ath:
                if TR[i] - TR[i-1] > 0.25 * fs:
                    if slope_min > slope_th:
                        Peaks.append(TR[i])
    
    return Peaks




def calculate_slopes(ecg_signal, window_size=110, fs=360):
    slopes = []
    for i in range(0, len(ecg_signal) - window_size, window_size):
        segment = ecg_signal[i:i+window_size]
        slope = np.max(np.abs(np.gradient(segment)))
        slopes.append(slope)
    return slopes

def fuzzy_c_median_clustering(x, m=2, epsilon=1e-5, max_iter=100):
    """
    Fuzzy c-median clustering to determine the amplitude threshold.
    
    Parameters:
    x (array): Detection function samples. Expected shape (N,) or (N, 1)
    m (float): Fuzziness parameter.
    epsilon (float): Convergence threshold.
    max_iter (int): Maximum number of iterations.
    
    Returns:
    A_th (float): Amplitude threshold.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]  # Convert x from (N,) to (N, 1) if necessary
    
    N = x.shape[0]
    c = 3  # Number of clusters
    u = np.random.rand(c, N)  # Random initialization of membership values
    u = u / np.sum(u, axis=0)
    
    for _ in range(max_iter):
        u_old = u.copy()
        
        # Compute cluster centers v using L1 norm
        v = np.sum((u**m).reshape(c, N, 1) * x, axis=1) / np.sum(u**m, axis=1).reshape(c, 1)
        
        # Compute absolute difference using L1 norm
        d = np.abs(x - v[:, np.newaxis]).sum(axis=2)
        
        # Avoid division by zero
        d = np.where(d == 0, 1e-10, d)
        
        # Update membership values
        u = 1 / (d**(1/(m-1)) + 1e-10)
        u = u / np.sum(u, axis=0)
        
        # Check for convergence based on the change in J
        J = np.sum((u**m) * d)
        if np.abs(J - np.sum((u_old**m) * d)) < epsilon:
            break
    
    # Identify noise group and determine amplitude threshold
    noise_group_index = np.argmin(np.max(v, axis=1))
    A_th = np.max(x[np.argmax(u[noise_group_index], axis=0)])
    
    return A_th

def regroup(ecg_signal, peaks, thr, maxi = 0):
    diff = peaks[1:]-peaks[:-1]
    gps = np.concatenate([[0], np.cumsum(diff>=thr)])
    temp = [peaks[np.where(gps == i)] for i in range(gps[-1]+1)]
    returnos = [maxi + i[np.argmax([ecg_signal[j] for j in i])] for i in temp]
    return returnos


def reduce_baseline_wandering(ecg_signal, fs):
    d1 = int(0.5 * fs)
    
    if d1 % 2 == 0:
        d1 += 1
        
    median_filtered = medfilt(ecg_signal, kernel_size=d1)
    filtered_ecg_signal = ecg_signal - median_filtered
    
    return filtered_ecg_signal

def gaussian_weights(M, sigma=2.2):
    # Indices pour lesquels on calcule la gaussienne
    r = np.arange(1, M + 1)
    # Calcul des poids gaussiens centrés à (M+1)/2 et normalisés
    weights = norm.pdf(r, loc=(M+1)/2, scale=sigma)
    return weights / np.sum(weights)

def owa_filter(signal, M, weights):
    half_M = M // 2
    padded_signal = np.pad(signal, (half_M, half_M), 'edge')
    filtered_signal = np.zeros_like(signal)
    
    for i in range(len(signal)):
        # Fenêtre de données ordonnée
        window = np.sort(padded_signal[i:i+M])
        # Produit scalaire de la fenêtre ordonnée avec les poids
        filtered_signal[i] = np.dot(weights, window)
    return filtered_signal

def cowa_filter(signal, M1, M2, P, sigma=2.2):
    # Générer les poids pour les deux filtres OWA
    weights1 = gaussian_weights(M1, sigma)
    weights2 = gaussian_weights(M2, sigma)
    
    # Première couche de filtres OWA
    x01 = owa_filter(signal, M1, weights1)
    x02 = owa_filter(signal, M2, weights2)
    
    # Calcul final de x0(n)
    x0 = 0.5 * (x01 + x02)
    return x0

def bandpass_filter(signal, fs):
    f1 = 0.8  # Fréquence de coupure basse en Hz
    f2 = 20   # Fréquence de coupure haute en Hz
    N = int(1.55 * fs)  # Longueur du filtre
    if N % 2 == 0:
        N += 1  # Assure que N est impair
    nyquist = fs / 2
    f1 /= nyquist
    f2 /= nyquist
    beta = 5.655  # Facteur de transition de la fenêtre de Kaiser
    taps = firwin(N, [f1, f2], window=('kaiser', beta), pass_zero=False)
    x1 = filtfilt(taps, 1.0, signal)
    return x1

def diff_and_square(signal):
    derivative = np.diff(signal)
    squared_signal = derivative ** 2
    return squared_signal


def smooth_signal(signal, fs):
    M = int(0.160 * fs)  # Longueur du filtre MA
    d2 = int(0.03 * fs)  # Longueur du filtre médian
    
    if M % 2 == 0:
        M += 1
    if d2 % 2 == 0:
        d2 += 1
    
    x3 = np.convolve(signal, np.ones(M)/M, mode='same')
    x = medfilt(x3, kernel_size=d2)
    return x