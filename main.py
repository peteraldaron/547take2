import numpy as np
import scipy, fileio, feature, syllable
from numpy import linalg
from scipy import fftpack
from pylab import *

SAMPLES_DIR = "./audio/samples/"
WORD_DIR = "./audio/"
SYLLABLE_DIR = "./audio/syllables/"
BLIND_DIR = "./audio/blinddata/"
'''
def single_wav_analysis():
    audiofiles = fileio.readAllFilesInDirectory("./audio/")
    sr = audiofiles[0][0].sr
    for file in audiofiles:
        title(file[1])
        amp, freq = syllable.segmentation(file[0])
        windowed_amp = syllable.moving_window(amp/linalg.norm(amp), 10)
        plot(windowed_amp, label="amp")
        #plot(syllable.moving_average(freq/linalg.norm(freq), 10),label="freq")
        plot(np.repeat(np.mean(windowed_amp),len(amp)), label='avga')
        maximi = syllable.local_max_locations(windowed_amp)
        print(maximi)
        #print(syllable.filter_distinct_peaks(windowed_amp, maximi))
        #plot(np.repeat(np.mean(freq/linalg.norm(freq)),len(amp)),label='avgf')
        show()
        #title(file[1])
        #plot(freq)
        #avgfreq = np.mean(freq)
        #plot(np.repeat(avgfreq, len(freq)))
#single_wav_analysis()
#syllable_debug_test()
def testing():
    audiofiles = fileio.readAllFilesInDirectory(BLIND_DIR)
    e = None;
    for file in audiofiles:
        #candidates = feature.partial_logic_1(file[0])
        #print(candidates)
        if e is None:
            e = feature.abstract_cartoon(file[0])
            #plot(e)
        else:
            f = feature.abstract_cartoon(file[0])
            #plot(f)
            e+=f
        #print(feature.abstract_cartoon(file[0]))
    test = feature.abstract_cartoon(fileio.Wave("test.wav").data)
    e = feature.normalize(e)
    test = feature.align_peaks(test, e, 2)
    plot(e)
    plot(test)
    show()
'''
#testing()
def pipeline(sampleData=None, sampleFreq=None):
    if sampleData is None:
        sampleData=fileio.populateSampleData(SAMPLES_DIR)
    audiofile = fileio.Wave("test.wav")
    candidates = feature.partial_logic_1(audiofile)
    #print(candidates)
    feature.partial_logic_2(candidates, audiofile, sampleData, sampleFreq)

#plotting aux:
def plotstuff(filename):
    a=fileio.Wave("audio/samples/"+filename+".wav")
    am, fr = syllable.segmentation(a)
    title(filename)
    plot(feature.normalize(syllable.moving_window(am, 10)), label="amp")
    #plot(feature.normalize(syllable.moving_window(fr, 10)), label="freq")
    #plot(feature.normalize(am), label="amp")
    #plot(feature.normalize(fr), label="freq")
    legend(framealpha=0.5);
    show()
