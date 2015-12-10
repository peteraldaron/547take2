import numpy as np, scipy
from scipy import fftpack, signal
#import pandas as pd
import syllable

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

#decompose fft into frequency:
#returns first half
def fft_extract(samples):
    ft = fftpack.fft(samples * signal.hann(len(samples)))
    realVal = ft[:len(ft)/2].real
    return realVal

def dct2(samples, sr, topn=10):
    ft=fftpack.dct(samples *signal.hann(len(samples)))
    temp = np.sort(ft)[-topn]
    bins = np.where(ft >= temp)[0]
    return bins*(sr/(len(ft)))

def fft_to_freq(fft_res, sr, topn=10):
    temp = np.sort(fft_res)[-topn]
    bins = np.where(fft_res >= temp)[0]
    return bins*(sr/(len(fft_res)*2))

#returns a realized list
def separate_frames(token, samples):
    #window sample data:
    samples = samples.data * signal.hann(samples.data.size)
    if (any(map(lambda x: x in token, syllable.mult_dip))):
        #trim the first 3rd (sort of)
        #division padding:
        samples = np.hstack(np.split(np.hstack((np.zeros(3-(len(samples) % 3)),samples)),3)[1:])
    #padding for stdct:
    samples = np.hstack((np.zeros(DCT_LEN-(len(samples) % 3)), samples))
    splitted = np.split(samples, len(samples)/DCT_LEN)
    return [feature_extract(x) for x in splitted]




