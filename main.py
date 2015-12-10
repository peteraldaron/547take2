import numpy as np
import scipy, fileio, feature, syllable
from numpy import linalg
from scipy import fftpack
from pylab import *

def pipeline():
    syllables = fileio.readAllFilesInDirectory("./audio/syllables/")
    sr = syllables[0][0].sr
    features = [(feature.fft_extract(x[0].data), x[1]) for x in syllables]
    for i,j in features:
        title(j)
        plot(i)
        print(feature.fft_to_freq(i, sr))
        show()


#pipeline()

def single_wav_analysis():
    audiofiles = fileio.readAllFilesInDirectory("./audio/")
    sr = audiofiles[0][0].sr
    for file in audiofiles:
        print(feature.dct2(file[0].data, sr, topn=3))
        title(file[1])
        amp, freq = syllable.segmentation(file[0])
        plot(amp/linalg.norm(amp))
        plot(freq/linalg.norm(freq))
        show()
single_wav_analysis()
