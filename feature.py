import numpy as np, scipy
from scipy import fftpack, signal
#import pandas as pd
import syllable

FEATURE_DIM = 32
DCT_LEN = 128

#simple wrapper
def feature_extract(samples):
    ft=scipy.fftpack.dct(samples)
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




