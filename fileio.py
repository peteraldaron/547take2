import numpy as np, sys, wave, struct, os
import soundfile as sf
import feature, math, syllable

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

def segment(data, length):
        splitAt = [x*length for x in range(1,math.ceil(len(data)/length))]
        res = np.split(data, splitAt)
        lastPadding = length - len(res[-1])
        if lastPadding > 0:
            begin = math.floor(lastPadding/2)
            end = lastPadding - begin
            res[-1] = np.pad(res[-1], (begin, end), mode='constant',
                    constant_values=0)
        return res

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
    filelist = []
    with open(directory+"filelist", "r") as files:
        for line in files:
            filelist.append(line[:-1]);
    return [readAudioFileWithName(directory, file) for file in filelist]

def writeWavesToFiles(waves, sr, prefix="", namelist=[]):
    if len(namelist) == 0:
        namelist = list(map(lambda x : prefix+str(x), range(len(waves))))
    for i in range(len(waves)):
        sf.write(namelist[i]+".wav", waves[i], sr)

def populateSampleData(sample_directory, freq=False):
    print("loading data")
    files = readAllFilesInDirectory(sample_directory);
    sampleData = {}
    abstractSampleData={}
    fileNames = set([x[1] for x in files])
    for name in fileNames:
        sampleData[name] = []
        abstractSampleData[name] =[]
    for file in files:
        sampleData[file[1]].append(file[0])
    #compute signatures:
    for key in sampleData.keys():
        processed=[]
        for sample in sampleData[key]:
            processed.append(feature.normalize(feature.abstract_cartoon(sample, freq=freq)))
        print(key, len(processed))
        for i in range(1,len(processed)):
            processed[i] = feature.align_peaks(processed[i], processed[i-1], 1)+processed[i-1]
        
        #abstractSampleData[key] = feature.normalize(feature.align_peaks(processed[0], processed[1], 1)   + processed[1])
        abstractSampleData[key] = feature.normalize(processed[-1])
    return abstractSampleData