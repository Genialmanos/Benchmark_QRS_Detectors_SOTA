import numpy as np
from scipy.signal import butter, filtfilt
import pywt
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
import time

def main():
    df = pd.read_csv('../data_csv/mit_bih_Arrhythmia/100.csv') #207
    ecg_signal = np.array(df["MLII"], dtype=np.float32)#[:10000]
    fs = 360
    start = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
    full_wavelets(ecg_signal, fs)
    elapsed = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID) - start
    print(f'{elapsed} s')

def full_wavelets(signal, sampling_rate):
    h = wavelet_decomposition(signal, 5, 4)
    qrs = np.array(qrs_localization_wave(h, 0.15, sampling_rate)) #0.15
    qrs = delete_contraction(qrs, sampling_rate, sampling_rate//10)
    qrs = qrs + searchback_missed_qrs(h, qrs, sampling_rate, 2)
    qrs = delete_contraction(qrs, sampling_rate, sampling_rate//10)
    return qrs

def wavelet_decomposition(sig, idx, nb_wave):
    listos = pywt.wavedec(sig, "haar", level=idx)
    resultat = np.array(insert_zeros(listos[-1], 1))
    resultat = resultat[:len(sig)]
    for i in range(1, idx-nb_wave):
        f = listos[-(i+1)]
        for j in range(i+1):
            f = insert_zeros(f, 1)
        f = f[:len(sig)]
        resultat = np.abs(np.multiply(resultat, np.array(f)))
    return resultat

def insert_zeros(lst, x):
    arr = np.array(lst)
    result = np.zeros(len(arr) + (len(arr) - 1) * x, dtype=arr.dtype)
    result[::x + 1] = arr
    result = np.append(result, 0)
    return result.tolist()

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

def searchback_missed_qrs(ecg_signal, filtered_indices, fs, v):
    rr_interval = np.diff(filtered_indices)
    search_interval = v * rr_interval//1
    missed_peaks = []
    for i in range(1, len(filtered_indices) - 1):
        if filtered_indices[i+1] - filtered_indices[i] > rr_interval[i-1] * 1.5:
            intervening_segment = ecg_signal[filtered_indices[i]+(fs//5):filtered_indices[i+1]-(fs//5)]
            seuil = 0.4 # ((ecg_signal[filtered_indices[i]] + ecg_signal[filtered_indices[i+1]]) / 2) * 0.5
            a = find_hidden_r(intervening_segment, seuil, fs)
            for i_a in a:
                missed_peaks.append(i_a+filtered_indices[i])
    return missed_peaks


main()