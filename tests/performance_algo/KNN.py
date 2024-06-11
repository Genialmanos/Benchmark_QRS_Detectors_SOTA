from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import scipy
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore")

def KNN(ecg_signal, fs):
    knn = None
    with open('modele_arrhythmia.pkl', 'rb') as fichier:  # 'rb' signifie lecture en mode binaire
        knn = pickle.load(fichier)
    X = 15
    band_passed = bandpass_filter(ecg_signal, fs)
    gradient = np.square(np.gradient(band_passed).tolist())[X:]
    y_pred = knn.predict(gradient.reshape(-1, 1))
    r_peaks = np.where(np.array(y_pred) == 1)[0]
    clean_r_peaks = regroup(gradient, r_peaks, 36, 0)
    return clean_r_peaks

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


"""def regroup(ecg_signal, peaks, thr, maxi):
    if len(peaks) < 2:
        return [maxi + p for p in peaks]

    # Calcul des groupes basés sur le seuil 'thr'
    groups = np.split(peaks, np.where(np.diff(peaks) >= thr)[0] + 1)

    # Trouver le pic avec le maximum de valeur de signal dans chaque groupe
    return [maxi + group[np.argmax(ecg_signal[group])] for group in groups]

"""
def regroup(ecg_signal, peaks, thr, maxi):
    diff = peaks[1:]-peaks[:-1]
    gps = np.concatenate([[0], np.cumsum(diff>=thr)])
    print(gps)
    #temp = [peaks[np.where(gps == i)] for i in range(gps[-1]+1)]
    return peaks
    #peaks[np.where(gps == i)]
    #returnos = [maxi + i[np.argmax([ecg_signal[j] for j in i])] for i in temp]
    #return returnos


