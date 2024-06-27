import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def print_signal_qrs(signal, qrs, true_qrs, mini = 0, maxi = 1, description = "a"):
    if maxi == 1:
        maxi = len(signal)
    cut_qrs = [a - mini for a in qrs if a < maxi and a > mini]
    true_cut_qrs = [a - mini for a in true_qrs if a < maxi and a > mini]
    print(f"signal de longueur: {len(signal)}")
    plt.figure()
    signal_cut = signal[mini:maxi]
    plt.plot(signal_cut)
    plt.scatter(cut_qrs , [signal_cut[i] * 1.05 for i in cut_qrs ], color='blue', label = 'predicted')
    plt.scatter(true_cut_qrs , [signal_cut[i] for i in true_cut_qrs ], color='green', label = 'true')
    plt.title(label= description)
    plt.legend()
    plt.show()

def print_signal(signal, description= "A", y = False):
    print(f"signal de longueur: {len(signal)}")
    plt.figure(figsize = (10, 3))
    plt.plot(range(len(signal)), signal)
    plt.xlabel("frame du signal")
    if y == True:
        plt.ylabel("mV")

    plt.title(label= description)
    plt.show()
    
def calcul_f1(TP, FP, FN):
    return (2 * TP) / (2 * TP + FN + FP)

def perf(labels, peaks, minmax, printos = False):
    x = np.concatenate([np.array(labels), np.array(peaks)]) #list(set(QRS + r_peaks))
    x.sort()
    diff = x[1:]-x[:-1]
    gps = np.concatenate([[0], np.cumsum(diff>=minmax)])
    temp = [x[gps==i] for i in range(gps[-1]+1)]
    TP = 0
    FP = 0
    FN = 0
    list_F = []
    for sublist in temp:
        if len(sublist) == 2:
            if sublist[0] == sublist[1]:
                TP += 1
            elif sublist[0] in peaks and sublist[1] in peaks:
                FP += 2
            elif sublist[0] in labels and sublist[1] in labels:
                FN += 2
            else:
                TP += 1
        else:
            list_F.append(sublist)
            FN += 1
            FP += 1
    if printos:
        return TP, FP, FN, calcul_f1(TP, FP, FN), list_F
    #print(f"TP = {TP}, FP = {FP}, FN = {FN}, F1_score = {calcul_f1(TP, FP, FN)}")
    return TP, FP, FN, calcul_f1(TP, FP, FN)