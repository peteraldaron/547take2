import numpy as np
from scipy import fftpack, signal
#import pandas as pd
import syllable, fileio

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

def align_peaks(sample, alignment_target, num_peaks):
    #doing a very direct and coarse alignment:
    sample_partitions = np.split(sample, sample.argsort()[-num_peaks:].sort())
    chunk_len = []
    topnpeak_t = alignment_target.argsort()[-num_peaks:].sort()
    for i in range(1, len(topnpeak_t)):
        chunk_len.append(topnpeak_t[i]-topnpeak_t[i-1])
    new_samples = [interpolate_wave(seg[0], seg[1]) for seg in zip(sample_partitions, chunk_len)]
    return np.hstack(new_samples)
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
                     norm=True):
    #whole wave analysis using a set transform window
    #TODO: resample the "cartoon" to a certain length and find trend.
    #resample wave to a set length
    resampled_wave = signal.resample(wave.data, resample_size)
    #segmentation:
    segments = fileio.segment(resampled_wave, window_size)
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
    if (kk and syllables < 9 and syllables >= 4):
        candidates = ["rakkaampaa", "kukkulaa"]
    elif (kk and syllables <= 4):
        candidates = ["kukkulaa"]
    elif (sloc != -1):
        if sloc == 0:
            if syllables <=2:
                candidates = ["sana"]
            elif syllables <=3:
                if sperc > 0.2:
                    candidates = ["isien"]
                else:
                    candidates = ["sana", "suomi"]
            elif syllables >=2 and syllables <=4:
                if sperc > 0.2:
                    candidates = ["isien"]
                else:
                    candidates = ["suomi"]
            elif syllables >4:
                candidates = ["synnyinmaa"]
        elif sloc == 1:
            if syllables <4 or sperc < 0.4:
                candidates = ["isien"]
            elif syllables >=3 and syllables <=6:
                candidates = ["laaksoa"]
            elif syllables >5:
                print("virhe sloc 1")
        elif sloc == 2:
            if syllables <= 10:
                candidates = ["kallis"]
            else:
                print("virhe sloc 2")
    #first stage syllable filtering:
    elif syllables == 1:
        candidates = ["ei", "maa"]
    elif syllables < 3:
        candidates = ["ei", "maa", "kuin"]
    elif syllables >= 5:
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

def partial_logic_2(candidates, sperc, kk, wave):
    pass
