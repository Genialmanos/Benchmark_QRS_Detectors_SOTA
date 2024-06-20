import numpy as np
from scipy.signal import butter, filtfilt
import pywt
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
from algo.seuil_dynamique import detect_peaks


def multi_lvl(ecg_signal, fs):
    clean_signal = preprocessing(ecg_signal, fs)
    #L = detect_peaks(clean_signal, distance= int(fs*0.3)).tolist()
    #P = [clean_signal[a] for a in L]
    #L = KNN2(ecg_signal, fs)
    #P = [clean_signal[i] for i in L]
    #qrs_detector_wow(clean_signal, fs)
    L, P = qrs_detector_5(clean_signal, fs)
    T = sum(P) / len(P)
    M = T.copy()
    R = T * 0.6

    Vsl = M
    Vnl = R

    Bw = 0

    i = 6
    H = L[:6]
    G = P[:6]
    B_i = 0
    while i < len(P):
        if P[i] >= M:
            G.append(P[i])
            H.append(L[i])
            Vsl = make_vsl(P[i] , Vsl)

        elif P[i] < M and P[i] >= R:
            B_i = make_B_i(H, fs, len(H)-2)
            Bw = make_Bw(B_i, len(H)-1, H, fs)

            if B_i <= Bw:
                G.append(P[i])
                H.append(L[i])
                Vsl = make_vsl(P[i] , Vsl) 
            else:
                Vnl = make_vnl(P[i], Vnl)
                
        M = make_M(Vnl, Vsl)
        R = M/2
        i += 1

    C = 50000
    G_seg, H_seg = divide_into_segments(G, H, C)
    i = 1
    while i < len(G_seg):
        Y = H_seg[i]
        Z = G_seg[i]
        F = 0.8 * sum(Z) / C
        Dfp = make_Dfp(Y)
        k = 0.95 # à voir
        W = k * sum(Dfp) / len(Dfp)
        for idx, zi in enumerate(Z):
            if zi < F and Dfp[idx-1] < W and Dfp[idx] < W:
                Y.pop(idx)
                Z.pop(idx)
        i += 1
    A = [element for sous_liste in H_seg for element in H_seg][0]
    E = [element for sous_liste in H_seg for element in G_seg][0]
    Dsb =  [A[x] - A[x-1] for x in range(1, len(A))]
    S = 1.75 * sum(Dsb) / len(Dsb)
    n = 1
    Q = len(Dsb)
    final_peaks = []
    while n < Q:
        if Dsb[n] >= S:
            sub_signal = clean_signal[A[n]:A[n+1]+1]
            M_local = max(sub_signal) * 0.25
            f = qrs_detector_5(sub_signal, fs, seuil = M_local, delay = S/2)
            final_peaks.extend(f[0][1:])
        final_peaks.append(A[n])
        n += 1
    return final_peaks

def make_B_i(H, fs, i):
    return (60 * fs) / (H[i+1] - H[i])

def make_Bw(B_i, Q, H, fs):
    return 1.75 * sum([make_B_i(H, fs, i) for i in range(3, Q-1)]) / (Q - 4) 

def make_vsl(a, vsl):
    return a * 0.1 + 0.9 * vsl

def make_vnl(a, vnl):
    return 0.2 * a + 0.8 * vnl

def make_M(Vnl, Vsl):
    return 0.6 * Vnl + (0.6 * (Vsl - Vnl))

def divide_into_segments(G, H, C):
    G_segments = [G[i:i+C] for i in range(0, len(G), C)]
    H_segments = [H[i:i+C] for i in range(0, len(H), C)]
    return G_segments, H_segments

def make_Dfp(Y):
    return [Y[x] - Y[x-1] for x in range(1, len(Y))]

def preprocessing(ecg_signal, fs):
    band_pass_filter = scipy.signal.butter(1, [5, 35], btype='bandpass', fs=fs)
    filtered_signal = scipy.signal.filtfilt(band_pass_filter[0], band_pass_filter[1], ecg_signal)
    
    absolute_signal = np.abs(filtered_signal)
    window_length = int(0.05 * fs)
    smoothed_signal = np.convolve(absolute_signal, np.ones(window_length)/window_length, mode='same')
    
    return smoothed_signal

def qrs_detector_5(signal, fs, seuil = 0, delay = 0):
    if seuil == 0:
        seuil = max(signal) / 6
    if delay == 0: 
        delay = fs//10
    i = 0
    peak_position = []
    peak_amplitude = []
    sous_groupe = []
    while i < len(signal):
        if signal[i]>= seuil:
            sous_groupe.append(i)
        elif sous_groupe != [] and i - sous_groupe[-1] >= delay:
            max_value = max((signal[j], j) for j in sous_groupe)
            peak_amplitude.append(max_value[0])
            peak_position.append(max_value[1])
            sous_groupe = []
            i += int(fs * 0.28) # interval minimum donné dans l'article
        i += 1
    if sous_groupe != []:
        max_value = max((signal[j], j) for j in sous_groupe)
        peak_amplitude.append(max_value[0])
        peak_position.append(max_value[1])
    return peak_position, peak_amplitude
