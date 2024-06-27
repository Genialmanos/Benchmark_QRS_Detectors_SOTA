import numpy as np
from scipy.signal import butter, filtfilt
import pywt
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler


def full_wavelets(signal, sampling_rate):

    h = wavelet_decomposition(signal, 5)
    qrs = np.array(qrs_localization_wave(h, 0.15, sampling_rate)) #0.15
    qrs = delete_contraction(qrs, sampling_rate, sampling_rate//10)
    qrs = qrs + searchback_missed_qrs(h, qrs, sampling_rate)
    #qrs = delete_contraction(qrs, sampling_rate, sampling_rate//10)
    #qrs = delete_stat(qrs, sampling_rate)
    return qrs

def delete_stat(qrs, fs):
    return qrs

def upsample_signal(original_signal, upsampled_length):
    # Coordonnées originales du signal
    original_indices = np.linspace(0, len(original_signal) - 1, num=len(original_signal))
    
    # Coordonnées désirées du signal upsampled
    upsampled_indices = np.linspace(0, len(original_signal) - 1, num=upsampled_length)
    
    # Interpolation spline cubique
    cs = CubicSpline(original_indices, original_signal)
    
    # Signal upsampled
    upsampled_signal = cs(upsampled_indices)
    
    return upsampled_signal

def wavelet_decomposition(sig, idx):
    # Décomposition en ondelettes
    coeffs = pywt.wavedec(sig, "haar", level=idx)
    
    # Prendre les coefficients aux niveaux idx-4 et idx-5
    w4 = coeffs[-2]
    w5 = coeffs[-1]
    
    # Upsample les coefficients à la longueur du signal original
    w4_upsampled = upsample_signal(w4, len(sig))
    w5_upsampled = upsample_signal(w5, len(sig))
    
    # Multiplication des coefficients upsampled et prise de la valeur absolue
    resultat = np.abs(w4_upsampled * w5_upsampled)
    
    return resultat

def qrs_localization_wave(h, XX, fs):
    qrs_indices = []
    sous_groupe = []
    w = 1000
    threshold = XX * max(h[:w])
    for i in range(len(h)):
        if i%w == 0 and i != 0:
            threshold = XX * max(h[i-(w//2):i+(w//2)])
        #print(h[i])
        if h[i] >= threshold:
            sous_groupe.append(i)
        elif sous_groupe != [] and i - sous_groupe[-1] >= (fs//10): # 36 = valeur donnée dans l'article
            qrs_indices.append(max([(x, h[x]) for x in sous_groupe], key=lambda x: x[1])[0])
            sous_groupe = []
    if sous_groupe != []:
        qrs_indices.append(max([(x, h[x]) for x in sous_groupe], key=lambda x: x[1])[0])
    return qrs_indices

def delete_contraction(r_peaks, fs, T):
    new_r_peaks = []
    i = 0
    while i < len(r_peaks) -1:
        new_r_peaks.append(r_peaks[i])
        if r_peaks[i] + T >= r_peaks[i+1] : # 5 * (fs / 2 )200 correspond à un mouvement du coeur particulier après un battement
            i += 2
        else:
            i += 1
    return new_r_peaks

def find_hidden_r(h, threshold, fs):
    qrs_indices = []
    sous_groupe = []
    for i in range(len(h)):
        if h[i] >= threshold:
            sous_groupe.append(i)
        elif sous_groupe != [] and i - sous_groupe[-1] >= (fs//10): # 36 = valeur donnée dans l'article
            qrs_indices.append(max([(x, h[x]) for x in sous_groupe], key=lambda x: x[1])[0])
            sous_groupe = []
    if sous_groupe != []:
        qrs_indices.append(max([(x, h[x]) for x in sous_groupe], key=lambda x: x[1])[0])
    return qrs_indices

def searchback_missed_qrs(ecg_signal, filtered_indices, fs):
    rr_interval = np.diff(filtered_indices)
    missed_peaks = []
    for i in range(1, len(filtered_indices) - 1):
        if filtered_indices[i+1] - filtered_indices[i] > rr_interval[i-1] * 1.5:
            intervening_segment = ecg_signal[filtered_indices[i]+(fs//10):filtered_indices[i+1]-(fs//10)]
            if len(intervening_segment) == 0:
                continue
            seuil = ((ecg_signal[filtered_indices[i]] + ecg_signal[filtered_indices[i+1]]) /2) # ((ecg_signal[filtered_indices[i]] + ecg_signal[filtered_indices[i+1]]) / 2) * 0.5
            a = find_hidden_r(intervening_segment, seuil, fs)
            for i_a in a:
                missed_peaks.append(i_a+filtered_indices[i])
    return missed_peaks