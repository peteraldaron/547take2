import numpy as np, scipy
from scipy import signal
from numpy import linalg
import math
import fileio, feature
from scipy.signal import argrelextrema
#from pylab import *

#moving average:
def moving_average(samples, length):
    #add padding:
    return np.convolve(samples, np.ones(length)/length)[:len(samples)]

#moving average:
def moving_window(samples, length, windowFunc=signal.windows.hann):
    #add padding:
    return np.convolve(samples, windowFunc(length))[:len(samples)]
    
def local_max_locations(sample):
    #note: discrete...can't do zeros
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
def segmentation(wave, seglen=256, topn=1):
    sr = wave.sr
    fft_result = [feature.fft_extract(x) for x in fileio.segment(wave.data, seglen)]
    freq_at_each_step = [feature.fft_to_freq(fft_res, sr, topn) for fft_res in fft_result]
    amp_at_each_step = [x.real.max() for x in fft_result]
    return amp_at_each_step, np.hstack(freq_at_each_step)

#len: seconds
def kk_detection(wave, window_size=10, amp_threshold=0.03, freq_threshold=0.7, len_threshold = 0.05):
    #normalize:
    seglen=64
    amp, freq = segmentation(wave, seglen)
    windowed_amp = moving_window(amp/linalg.norm(amp), window_size)
    windowed_freq = moving_window(freq/linalg.norm(freq), window_size)
    trim = 0.1 * len(windowed_amp)
    windowed_amp=windowed_amp[trim:-trim]
    windowed_freq=windowed_freq[trim:-trim]
    #find if it exists where both are below a certain threshold.
    #return np.where(windowed_amp<max(windowed_amp)*amp_threshold), windowed_amp, windowed_freq
    intersection = np.intersect1d(np.where(windowed_amp<max(windowed_amp)*amp_threshold),
                          np.where(windowed_freq<max(windowed_freq)*freq_threshold))
    continuous=np.split(intersection, np.where(np.diff(intersection)!=1)[0]+1)
    
    length = (wave.sr / seglen)*len_threshold
    return (any(map(lambda segment:len(segment) > length, continuous)),
            windowed_amp,
            windowed_freq)

def s_detection(wave, window_size=10, amp_threshold=0.7, freq_threshold=3000, len_threshold = 0.02):
    #continuous region that's greater than 3000
    #normalize:
    seglen=64
    amp, freq = segmentation(wave, seglen)
    freq = moving_average(freq, 10)
    #plot(freq)
    #show()

    windowed_amp = moving_window(amp/linalg.norm(amp), window_size)
    length = (wave.sr / seglen)*len_threshold
    candidates = np.intersect1d(np.where(freq>3000), np.where(windowed_amp < max(windowed_amp)*amp_threshold))
    continuous=np.split(candidates, np.where(np.diff(candidates)!=1)[0]+1)
    filtered = [seg for seg in continuous if len(seg)>=length]
    #location: 3 state: 0, 1, 2
    location = -1 
    centralFramePercent = -1
    if len(filtered) > 0:
        longest = filtered[0];
        for seq in filtered:
            if len(seq) > len(longest):
                longest = seq
        #centralFrame = longest[len(longest)/2]
        #use 1st frame?
        centralFrame = longest[0]
        centralFramePercent = centralFrame/len(amp)
        if centralFramePercent < 0.3:
            location = 0
        elif centralFramePercent <=0.75:
            location = 1
        else:
            location = 2
    return filtered, location, centralFramePercent, windowed_amp, freq 


def guess_syllables(wave, window_size=10):
    amp, _ = segmentation(wave)
    windowed_amp = moving_window(amp/linalg.norm(amp), window_size)
    maximi = local_max_locations(windowed_amp)
    return len(maximi[0])