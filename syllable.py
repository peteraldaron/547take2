import numpy as np, scipy
from scipy import signal
import math
import fileio, feature
from scipy.signal import argrelextrema

#moving average:
def moving_average(samples, length):
    #add padding:
    return np.convolve(samples, np.ones(length)/length)[:len(samples)]

#moving average:
def moving_window(samples, length):
    #add padding:
    return np.convolve(samples, signal.windows.hann(length))[:len(samples)]
    
def local_max_locations(sample):
    #note: discrete...can't do zeros
    '''
    first_order = np.diff(sample)
    second_order = np.diff(sample, n=2)
    #where second order is negative
    return np.intersect1d(np.where(first_order==0), np.where(second_order < 0))
    '''
    return argrelextrema(sample, np.greater)

def top_n_local_max_locations(sample, n=5):
    #return locations
    locs = local_max_locations(sample)
    return np.sort(locs[sample[locs].argsort()[::-1][:n]])

def filter_distinct_peaks(samples, max_locations, filtering_percentage=0.03):
    #forward looking:
    smax = max(samples)
    return [max_locations[i] for i in range(len(max_locations)-1)
        if abs((samples[max_locations[i]]-samples[max_locations[i+1]])/smax) >= filtering_percentage]
            
#attempts to segment a given input through frequency/amplitude analysis:
def segmentation(wave, seglen=256):
    sr = wave.sr
    fft_result = [feature.fft_extract(x) for x in fileio.segment(wave.data, seglen)]
    freq_at_each_step = [feature.fft_to_freq(fft_res, sr, topn=1) for fft_res in fft_result]
    amp_at_each_step = [x.real.max() for x in fft_result]
    return amp_at_each_step, np.hstack(freq_at_each_step)
