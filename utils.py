import os
import numpy as np


def calc_bandwidth(lambd, sigma):
    r = np.pi*sigma/lambd
    c = np.sqrt(np.log(2)/2)
    return np.log2((r + c)/(r - c))

def calc_sigma(lambd, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return lambd * c / np.pi  * (p + 1) / (p - 1)

def calc_lambda(sigma, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return sigma * np.pi / c  * (p - 1) / (p + 1)
