import numpy as np
from scipy import fftpack, signal
#import pandas as pd
import syllable, fileio
from pylab import *

FEATURE_DIM = 64
DCT_LEN = 128

#simple wrapper
def feature_extract(samples):
    ft=fftpack.dct(samples *signal.hann(len(samples)))
    #normalization:
    absmax = np.absolute(ft).max()
    if absmax!=0:
        ft = ft/absmax
    #extract the first couple:
    if (len(ft) > FEATURE_DIM):
        return ft[:FEATURE_DIM]
        #return np.absolute(ft[:FEATURE_DIM:2])
    else:
        return np.hstack((ft, np.zeros(FEATURE_DIM - len(ft))))

def normalize(vector):
    c=vector/np.linalg.norm(vector)
    return c/c.max()

def interpolate_wave(segment, inter_length):
    return signal.resample(segment, inter_length)

def diff(vec1, vec2, d1=False, debugflag=""):
    if d1:
        vec1=np.diff(vec1)
        vec2=np.diff(vec2)
    protection = min(vec1.size, vec2.size)
    if debugflag!="":
        title(debugflag)
        plot(vec1, label="local")
        plot(vec2, label="candidate")
        legend()
        show()
    return np.add.reduce((vec1[:protection]-vec2[:protection])**2)

def align_peaks(sample, alignment_target, num_peaks, peak_r1=1, peak_r2=64):
    #doing a very direct and coarse alignment:
    #peak_s = signal.find_peaks_cwt(sample, np.arange(peak_r1,peak_r2))
    
    peak_s = list(signal.argrelmax(sample, order=32)[0])
    peak_s_vals = np.array([sample[x] for x in peak_s]).argsort()[-num_peaks:]
    peak_s = np.sort([peak_s[x] for x in peak_s_vals])

    #topnpeak_t = signal.find_peaks_cwt(alignment_target, np.arange(peak_r1,peak_r2))
    topnpeak_t = list(signal.argrelmax(sample, order=32)[0])
    peak_t_vals = np.array([alignment_target[x] for x in topnpeak_t]).argsort()[-num_peaks:]
    topnpeak_t = np.sort([topnpeak_t[x] for x in peak_t_vals])

    sample_partitions = np.split(sample, peak_s)
    chunk_len = []

    chunk_len.append(topnpeak_t[0])
    for i in range(1, len(topnpeak_t)):
        chunk_len.append(topnpeak_t[i]-topnpeak_t[i-1])
    chunk_len.append(len(alignment_target)-topnpeak_t[-1])
    new_samples = [interpolate_wave(seg[0], seg[1]) for seg in zip(sample_partitions, chunk_len)]
    #return np.hstack(new_samples)
    return normalize(syllable.moving_window(np.hstack(new_samples), 16))

#decompose fft into frequency:
#returns first half
def fft_extract(samples, wf=signal.hann):
    ft = fftpack.fft(samples * wf(len(samples)))
    realVal = ft[:len(ft)/2].real
    return realVal

def abstract_cartoon(wave,
                     window_size=64,
                     resample_size=32768,
                     window_func=signal.hann,
                     freq=False,
                     norm=True):
    #whole wave analysis using a set transform window
    #TODO: resample the "cartoon" to a certain length and find trend.
    #resample wave to a set length
    resampled_wave = signal.resample(wave.data, resample_size)
    sr = wave.sr
    #segmentation:
    segments = fileio.segment(resampled_wave, window_size)

    #TODO: test this
    #try frequency:
    if freq:
        transformed_segments = [fft_to_freq(fft_extract(seg, window_func), sr, 1)[0] for seg in segments]
    else:
        transformed_segments = [fft_extract(seg, window_func).max() for seg in segments]

    #ft = fftpack.dct(wave.data * window_type(len(wave.data)))
    #window this?
    transformed_segments = syllable.moving_window(transformed_segments, window_size, window_func)
    if norm:
        transformed_segments = normalize(transformed_segments)
    #1st derivative for trend detection:
    #return np.diff(transformed_segments)
    return transformed_segments

def fft_to_freq(fft_res, sr, topn=10):
    temp = np.sort(fft_res)[-topn]
    bins = np.where(fft_res >= temp)[0]
    return bins*(sr/(len(fft_res)*2))

def partial_logic_1(wave):
    syllables = syllable.guess_syllables(wave)
    kk, _, _ = syllable.kk_detection(wave, window_size=40)
    _, sloc, sperc,_,_ = syllable.s_detection(wave, window_size=40)
    print(syllables, kk, sloc, sperc)
    candidates = []
    if (kk and syllables < 9 and syllables >= 3):
        candidates = ["rakkaampaa", "kukkulaa"]
    elif (kk and syllables < 3):
        candidates = ["kukkulaa"]
    elif (sloc != -1):
        if sloc == 0:
            if syllables <=3:
                if sperc > 0.2:
                    candidates = ["isien"]
                else:
                    if(syllables<=2):
                        candidates = ["sana", "suomi"]
                    else:
                        candidates = ["synnyinmaa"]
            elif syllables >=2 and syllables <3:
                if sperc > 0.2:
                    candidates = ["isien"]
                else:
                    candidates = ["suomi"]
            elif syllables >=3:
                candidates = ["synnyinmaa"]
        elif sloc == 1:
            if syllables <4 and sperc < 0.4:
                candidates = ["isien"]
            else:
                candidates = ["laaksoa"]
        elif sloc == 2:
            if syllables <= 10:
                candidates = ["kallis"]
            else:
                print("virhe sloc 2")
    #first stage syllable filtering:
    elif syllables == 1:
        candidates = ["ei", "maa", "kuin"]
    elif syllables == 2 :
        candidates = ["rantaa", "maa"]
    elif syllables >= 4:
        candidates = [
            "kultainen",
            "pohjoinen"]
    else:
        candidates = [
            "kultainen",
            "rantaa",
            "kotimaa",
            "pohjoinen"]
    return candidates

def partial_logic_2(candidates, wave, samples, samples_freq):
    localcartoon = abstract_cartoon(wave)
    localcartoon_freq = abstract_cartoon(wave, freq=True)
    difference = []
    #DEBUG:
    #query both candidates AND all other options:
    #for key in samples.keys():
    min = 99999
    for key in candidates:
        aligned_local_wave = align_peaks(localcartoon, samples[key], syllable.vocab_syllables[key])
        differ = diff(aligned_local_wave, samples[key], d1=False,debugflag=key)

        aligned_local_wave_freq = align_peaks(localcartoon_freq, samples_freq[key], syllable.vocab_syllables[key])
        differ += diff(aligned_local_wave_freq, samples_freq[key], d1=True,debugflag=key)
        #differ = diff(localcartoon, samples[key], key)
        #title(key)
        #plot(aligned_local_wave)
        #plot(samples[key])
        #show()
        difference.append((key, differ))
        if differ<min:
            min=differ
    print(candidates)
    for x in difference:
        if x[1] == min:
            print(x[0])

