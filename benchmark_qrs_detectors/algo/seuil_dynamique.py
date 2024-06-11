import numpy as np
from scipy.signal import butter, filtfilt, medfilt, firwin, lfilter

def seuil_dynamique(sig, freq_sampling):
    cleaned_ecg = preprocess_ecg(sig, freq_sampling, 5, 15)
    peaks = detect_peaks(cleaned_ecg, distance = int(freq_sampling * 0.222))
    qrs_indices = threshold_detection(cleaned_ecg, peaks, freq_sampling, initial_search_samples= int(freq_sampling * 0.83), long_peak_distance=int(freq_sampling*1.111))
    return qrs_indices

def detect_peaks(cleaned_ecg, distance=0, no_peak_distance=300):

    last_max = -np.inf  # The most recent encountered maximum value
    last_max_pos = -1  # Position of the last_max in the array
    peaks = []  # Detected peaks positions
    peak_values = []  # Detected peaks values
    
    for i in range(len(cleaned_ecg)):
        current_value = cleaned_ecg[i]
        
        if current_value > last_max:
            last_max = current_value
            last_max_pos = i
        
        if current_value <= last_max / 2 or (i - last_max_pos >= no_peak_distance and last_max_pos != -1):
            if last_max_pos != -1:
                peaks.append(last_max_pos)
                peak_values.append(last_max)
            
            last_max = current_value
            last_max_pos = i
    
    peaks = np.array(peaks)
    peak_values = np.array(peak_values)
    
    refined_peaks = []
    i = 0
    while i < len(peaks):
        peak_group_start = i
        while i < len(peaks) - 1 and peaks[i + 1] - peaks[peak_group_start] < distance:
            i += 1
        
        # End of group
        peak_group_end = i
        
        # Find the largest peak in this group
        if peak_group_start == peak_group_end:
            refined_peaks.append(peaks[i])
        else:
            # Select the peak with the maximum value in this group
            max_peak = peak_values[peak_group_start:peak_group_end+1].argmax()
            refined_peaks.append(peaks[peak_group_start + max_peak])
        
        i += 1
    
    return np.array(refined_peaks)


def threshold_detection(cleaned_ecg, peaks, fs, initial_search_samples=300, long_peak_distance=400):
    M_VAL = np.max(cleaned_ecg[:initial_search_samples])
    
    SPK = 0.13 * M_VAL
    NPK = 0.1 * SPK
    THRESHOLD = 0.25 * SPK + 0.75 * NPK
    
    qrs_peaks = []
    noise_peaks = []
    qrs_buffer = []
    last_qrs_time = 0
    min_distance = int(fs * 0.12)
    
    for peak in peaks:
        peak_value = cleaned_ecg[peak]
        
        if peak_value > THRESHOLD:
            if qrs_peaks and (peak - qrs_peaks[-1] < min_distance):
                if peak_value > cleaned_ecg[qrs_peaks[-1]]:
                    qrs_peaks[-1] = peak
            else:
                qrs_peaks.append(peak)
                last_qrs_time = peak
            
            SPK = 0.25 * peak_value + 0.75 * SPK
            
            qrs_buffer.append(peak)
            if len(qrs_buffer) > 10:
                qrs_buffer.pop(0)
        else:
            noise_peaks.append(peak)
            NPK = 0.25 * peak_value + 0.75 * NPK
        
        THRESHOLD = 0.25 * SPK + 0.75 * NPK
        
        if peak - last_qrs_time > long_peak_distance:
            SPK *= 0.5
            THRESHOLD = 0.25 * SPK + 0.75 * NPK
            for lookback_peak in peaks:
                if last_qrs_time < lookback_peak < peak:
                    if cleaned_ecg[lookback_peak] > THRESHOLD:
                        qrs_peaks.append(lookback_peak)
                        SPK = 0.875 * SPK + 0.125 * cleaned_ecg[lookback_peak]
                        THRESHOLD = 0.25 * SPK + 0.75 * NPK
                        last_qrs_time = lookback_peak
                        break
        
        if len(qrs_buffer) > 1:
            rr_intervals = np.diff(qrs_buffer)
            mean_rr = np.mean(rr_intervals)
            if peak - last_qrs_time > 1.5 * mean_rr:
                SPK *= 0.5
                THRESHOLD = 0.25 * SPK + 0.75 * NPK
    
    return np.array(qrs_peaks)

def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    y = filtfilt(b, a, data)
    return y

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    b, a = butter(order, low, btype='low')
    y = filtfilt(b, a, data)
    return y

def differentiate(data):
    return np.diff(data, prepend=data[0])

def squaring(data):
    return np.square(data)

def moving_window_integration(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def preprocess_ecg(data, fs, high, low):
    high_passed_data = highpass_filter(data, high, fs)
    low_passed_data = lowpass_filter(high_passed_data, low, fs)
    differentiated_data = differentiate(low_passed_data)
    squared_data = squaring(differentiated_data)
    integrated_data = moving_window_integration(squared_data, int(0.0667 * fs))
    return integrated_data

