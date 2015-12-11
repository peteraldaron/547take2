import numpy as np
import scipy, fileio, feature, syllable
from numpy import linalg
from scipy import fftpack
from pylab import *
from feature import *
from syllable import *

TEST_DIR = "./audio/testdata/"
WORD_DIR = "./audio/"
SYLLABLE_DIR = "./audio/syllables/"
BLIND_DIR = "./audio/blinddata/"

vocab = set([
"suomi",
"synnyinmaa",
"sana",
"kultainen",
"laaksoa",
"kukkulaa",
"ei",
"rantaa",
"rakkaampaa",
"kuin",
"kotimaa",
"pohjoinen",
"maa",
"kallis",
"isien"
 ])

vocab_syllables = { 
"ei":1,
"isien":4,
"kallis":4,
"kotimaa":4,
"kuin":3,
"kukkulaa":4,
"kultainen":6,
"laaksoa":5,
"maa":2,
"pohjoinen":6,
"rakkaampaa":6,
"rantaa":4,
"sana":2,
"suomi":3,
"synnyinmaa":7,
}

syllables_to_vocab ={}
for i in range(max(map(lambda l :l, vocab_syllables.values()))):
    syllables_to_vocab[i] = []

for i in vocab:
    syllables_to_vocab[vocab_syllables[i]] = i

def syllable_debug_test():
    audiofiles = fileio.readAllFilesInDirectory(WORD_DIR)
    sr = audiofiles[0][0].sr
    for file in audiofiles:
        print(file[1]+" guess: "+str(guess_syllables(file[0])))
        print(file[1]+ " actual: "+str(vocab_syllables[file[1]]))
        kk, am, fr = syllable.kk_detection(file[0], window_size=40)
        s, sloc, _,_ = syllable.s_detection(file[0], window_size=40)
        print("kk: ", kk)
        print("s: ", s, sloc)
        print()
        title(file[1])
        plot(am, label="amp")
        plot(fr, label="fre")
        show()

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
    for file in audiofiles:
        candidates = feature.partial_logic_1(file[0])
        print(candidates)
testing()