import numpy as np, scipy
import pandas as pd
import math
import fileio, feature

'''
implemented based on description in
Xie, Niyogi (2006)
'''
def sample_autocovar(samples, h):
    upper_bound = len(samples) - h
    return (1/n)*sum(map(lambda t: samples[t+h]*samples[t], range(0, upper_bound)))

#samples: 1-d np array
#todo: optimize
def periodicity(samples, p):
    n = len(samples)
    return (sample_autocovar(samples, p)/(n-p))/(sample_autocovar(samples, 0)/n)

def frame_energy(samples):
    return math.log(sample_autocovar(samples, 0), 2)

'''
end X&N
'''

#moving average:
def mean_amplitude(samples, length):
    #add padding:
    return np.hstack((np.zeros(length-1), pd.rolling_mean(np.absolute(samples),
            length)[length-1:]))

def coarse_mean_amplitude(samples, length):
    return np.hstack(map(lambda x: np.ones(length)*x,
        np.mean(np.hstack((samples, samples[-1]*np.ones(length-(len(samples)%length))))
            .reshape(-1, length), axis=1)))


dip = set([
    "k",
    "p",
    "t"
    #'l'
]);

mult_dip = set([
    "kk",
    "pp",
    "tt"
])

consonants = set([
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "w",
    "x",
    "z"
])

vowels = set([
    "a",
    "o",
    "i",
    "e",
    "u",
    "ä",
    "ö",
    "y"
])


#reimplemented from c++
def tokenize_syllable(word):
    wordlen = len(word)

    tokens = []

    def is_dip(w):
        return w in dip

    def is_consonant(w):
        return w in consonants

    def is_vowel(w):
        return w in vowels

    if (len(word) == 0):
        return tokens

    i = 0


    while i < wordlen:
        if (i == 0):
            curr = word[i];
            buff = word[i];
            if (is_consonant(curr)):
                #look ahead:
                if (i+1 < wordlen and is_consonant(word[i+1])):
                    #if (is_vowel(word[i+2])):
                    #    tokens.append(buff);
                    #    i = i+1; continue
                    #else:
                    #    print("virhe!")
                    #    i=i+1; continue;

                    tokens.append(buff);
                    i = i+1; continue
                elif (i+1 <wordlen and is_vowel(word[i+1])):
                    buff += word[i+1];
                    tokens.append(buff);
                    i = i+1;
                    i=i+1; continue;
                else:
                    print("virhe!")
                    i = i+1
            elif (is_vowel(curr)):
                #look ahead:
                if (i+1 < wordlen) :
                    #next is vowel:
                    if (is_vowel(word[i+1])) :
                        tokens.append(buff);
                        i=i+1; continue;
                    elif (is_consonant(word[i+1])) :
                        #look further ahead:
                        if (i+2 < wordlen and word[i+1] == word[i+2]):
                            if (is_dip(word[i+1])) :
                                #ditch, next iteration should pick up.
                                #a-kk-u, a-pp-u, a-tt-u
                                tokens.append(buff);
                                i=i+1; continue;
                            else:
                                #lex next as token:
                                buff += word[i+1];
                                tokens.append(buff);
                                i = i+1;
                                i=i+1; continue;
                        #if is consonant and next is not equal:
                        else:
                            if (i+2 < wordlen and is_consonant(word[i+2])):
                                #lex next as token:
                                buff += word[i+1];
                                tokens.append(buff);
                                i = i+1;
                                i=i+1; continue;
                            else :
                                tokens.append(buff);
                                i=i+1; continue;
                #last vowel, return.
                else :
                    tokens.append(buff);
                    i=i+1; continue;
        #not first symbol
        else:
            curr = word[i];
            buff = word[i];
            if (is_consonant(curr)):
                #look ahead:
                if (i+1<wordlen):
                    if (is_consonant(word[i+1])):
                        if (i+2 < wordlen and is_vowel(word[i+2]) and curr == word[i+1] and
                                is_dip(curr)) :
                            #only case for long consonants:
                            buff += word[i+1];
                            buff += word[i+2];
                            tokens.append(buff);
                            i = i+2;
                            i=i+1; continue;
                        #otherwise just add:
                        else :
                            tokens.append(buff);
                            i=i+1; continue;
                    if (is_vowel(word[i+1])):
                        #lex whole thing
                        buff += word[i+1];
                        tokens.append(buff);
                        i = i+1;
                        i=i+1; continue;
                else :
                    tokens.append(buff);
                    i=i+1; continue;
            elif (is_vowel(curr)) :
                #look ahead:
                if (i+1 < wordlen) :
                    #next is vowel:
                    if (is_vowel(word[i+1])) :
                        tokens.append(buff);
                        i=i+1; continue;
                    elif (is_consonant(word[i+1])) :
                        #look further ahead:
                        if (i+2 < wordlen and word[i+1] == word[i+2] and is_dip(word[i+1])) :
                            #ditch, next iteration should pick up.
                            #a-kk-u, a-pp-u, a-tt-u
                            tokens.append(buff);
                            i=i+1; continue;
                        #if is consonant and next is not equal:
                        else :
                            tokens.append(buff);
                            i=i+1; continue
                #last vowel, return.
                else :
                    tokens.append(buff);
                    i=i+1; continue;
    return tokens

#assuming trimmed audio on both ends
#wave: waveData or wave
def waveDivisionBySyllable(wave, word):
    tokens = tokenize_syllable(word);
    wordlen = len(word)
    per_letter_len = wave.data.size / wordlen
    frame_marks = np.hstack((np.zeros(1), np.array([len(x) for x in tokens]).cumsum() * per_letter_len))
    return list(map(lambda x : {
            'token': x[2],
            'samples':fileio.WaveData(wave.data[int(x[0]):int(x[1])], wave.sr)
        }, zip(frame_marks[:-1], frame_marks[1:], tokens)))


#attempts to segment a given input through frequency/amplitude analysis:
def segmentation(wave):
    sr = wave.sr
    fft_result = [feature.fft_extract(x) for x in fileio.segment(wave.data, 256)]
    freq_at_each_step = [feature.fft_to_freq(fft_res, sr, topn=1) for fft_res in fft_result]
    amp_at_each_step = [x.real.max() for x in fft_result]
    return amp_at_each_step, np.hstack(freq_at_each_step)
