import numpy as np, sys, wave, struct, os
import soundfile as sf
import feature

#doing everything in numpy

class Wave:
    def __init__(self, filename):
        data, self.sr = sf.read(filename)
        self.rawdata = np.array(data)
        if len(self.rawdata.shape) == 1:
            self.frames = self.rawdata.shape[0]
            self.data = np.array(self.rawdata)
        else:
            self.frames, self.channels = self.rawdata.shape
            self.data = self.rawdata[:,0]
        self.length = self.frames/self.sr

    # returns all frames with given length
    def fft(self, fftlength = 1024):
        return list(map(lambda x:
                    np.fft.fft(x, fftlength),
                    np.split(self.data, range(fftlength, self.frames,
                        fftlength))))

    def ifft(fft_data):
        return np.fft.ifft(fft_data).real

#wrapper
class WaveData:
    def __init__(self, data, sr):
        self.data = data;
        self.frames = data.size
        self.sr = sr

    # returns all frames with given length
    def fft(self, fftlength = 1024):
        return list(map(lambda x:
                    np.fft.fft(x, fftlength),
                    np.split(self.data, range(fftlength, self.frames,
                        fftlength))))

def readAudioFileWithName(directory, filename):
    #tokenize file name:
    name = filename.split(".")[0]
    return (Wave(directory+filename), name)

def readAllFilesInDirectory(directory):
    #debug:
    #for file in os.listdir(directory):
    #    if file.endswith(".wav"):
    #        print(file)
    #        readAudioFileWithName(file)
    filelist = []
    with open("./resources/audiolist", "r") as files:
        for line in files:
            filelist.append(line[:-1]);
    return [readAudioFileWithName(directory, file) for file in filelist]

def writeWavesToFiles(waves, sr, prefix="", namelist=[]):
    if len(namelist) == 0:
        namelist = list(map(lambda x : prefix+str(x), range(len(waves))))
    for i in range(len(waves)):
        sf.write(namelist[i]+".wav", waves[i], sr)

def dumpFeaturesToNNFFile(filename, feature_matrix, feature_size, result_size):
    fp = open(filename, "w")
    print(str(feature_size)+" "+str(result_size), file=fp)
    print(len(feature_matrix), file=fp)
    for i in feature_matrix:
        print(" ".join([str(x)[:10] for x in i]), file=fp)
    fp.close()

def token_to_code(tokens, mode="binary"):
    obj = {}
    if mode == "binary":
        digits = int.bit_length(len(tokens))
        i = 0
        for token in tokens:
            le = np.array(list(bin(i)[2:]))
            if len(le) < digits:
                le = np.hstack((np.zeros(digits - len(le)), le))
            #obj[tokens[i]] = list(len(bin(i)[:2])
            obj[token] = le.astype(np.float64)
            i+=1
        #print(obj)
        return obj
    elif mode == "decimal":
        i = 0
        for token in tokens:
            obj[token] = np.array(i);
            i+=1
        return obj

def exportToNNF(tokens, frames):
    mode = "decimal"
    outputlen = 1 if mode == "decimal" else int.bit_length(len(tokens))
    token_codes = token_to_code(tokens, mode)
    #for x in frames:
    #    print(x[1], token_codes[x[0]])
    aggregated_data = [np.hstack((frame, token_codes[x[0]]))
            for x in frames for frame in x[1]]
    dumpFeaturesToNNFFile("feature_data.txt", aggregated_data,
            feature.FEATURE_DIM, outputlen)

