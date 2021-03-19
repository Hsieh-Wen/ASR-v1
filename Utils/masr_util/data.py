import torch
import librosa
import wave
import numpy as np
import scipy
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

sample_rate = 16000
window_size = 0.02
window_stride = 0.01
n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)
window = "hamming"


def load_audio(wav_path, normalize=True):  # -> numpy array
    with wave.open(wav_path) as wav:
        wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
        wav = wav.astype("float")
    if normalize:
        return (wav - wav.mean()) / wav.std()
    else:
        return wav


def spectrogram(wav, normalize=True):
    D = librosa.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )

    spec, phase = librosa.magphase(D)
    spec = np.log1p(spec)
    spec = torch.FloatTensor(spec)

    if normalize:
        spec = (spec - spec.mean()) / spec.std()

    return spec


class MASRDataset(Dataset):
    def __init__(self, index_path, labels_path):
        with open(index_path) as f:
            idx = f.readlines()
        idx = [x.strip().split(",", 1) for x in idx]
        self.idx = idx
        with open(labels_path) as f:
            labels = json.load(f)
        self.labels = dict([(labels[i], i) for i in range(len(labels))])
        self.labels_str = labels

    def __getitem__(self, index):
        wav, transcript = self.idx[index]
        wav = load_audio(wav)
        spect = spectrogram(wav)
        transcript = list(filter(None, [self.labels.get(x) for x in transcript]))

        return spect, transcript

    def __len__(self):
        return len(self.idx)

class MASR_train_Dataset(Dataset):
    def __init__(self, index_path, labels_path):
        with open(index_path) as f:
            idx = f.readlines()
        idx = [x.strip().split(",", 2) for x in idx]
        idx_sorted = sorted(idx, key=lambda sample: float(sample[2]), reverse=False) # all training waves sorted by time (smaal-large)
        
        # Get regions from histogram
        wav_time = [float(t) for i,l,t in idx_sorted]
        hist, bin_edges = np.histogram(np.array(wav_time))
        hist = np.cumsum(hist)
        
        regions = []
        for num in range(len(hist)):
            if num == 0:
                regions.append(idx_sorted[0:hist[0]])
            else:
                regions.append(idx_sorted[hist[num-1]:hist[num]])
                
        # idx_random = idx_sorted 
        
        # Every regions to random sorted       
        for reg_num in range(len(regions)):
            random.shuffle(regions[reg_num])
        idx_random = []
        for i in range(len(regions)):
            idx_random += regions[i]            
            
        self.idx = idx_random
        with open(labels_path) as f:
            labels = json.load(f)
        self.labels = dict([(labels[i], i) for i in range(len(labels))])
        self.labels_str = labels

    def __getitem__(self, index):
        wav, transcript,wav_time = self.idx[index]
        wav = load_audio(wav)
        spect = spectrogram(wav)
        transcript = list(filter(None, [self.labels.get(x) for x in transcript]))

        return spect, transcript

    def __len__(self):
        return len(self.idx)

def _collate_fn(batch): # batch = train_dataset
    def func(p):
        return p[0].size(1)

    # sorted by wave length --> sorted in batch[i][0].size(1)    [i = 0 ~ number of waves] 
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True) 
    # print("batch :"+str(len(batch)))
    # print(str(batch[0][0].size())+"\n")   
    longest_sample = max(batch, key=func)[0] # find the longest wave in batch
    freq_size = longest_sample.size(0) # freq_size = dimension of spect --> 161
    minibatch_size = len(batch) # number of waves
    max_seqlength = longest_sample.size(1) # the longest wave's length
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength) # create zeros [number of waves, dimension of spect(161), the longest wave's length]
    input_lens = torch.IntTensor(minibatch_size)
    target_lens = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0] # wave's tensor -> [161,2100]
        target = sample[1] # label's number (word to number) -> list : 40
        seq_length = tensor.size(1) # wave's length-> 2100
        inputs[x].narrow(1, 0, seq_length).copy_(tensor) # input.size() : [torch.zeros(minibatch_size, freq_size, max_seqlength)]-> [2,161,2100]
        input_lens[x] = seq_length
        target_lens[x] = len(target)
        targets.extend(target) # append labels' number
    targets = torch.IntTensor(targets) # labels to tensor
    return inputs, targets, input_lens, target_lens


class MASRDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MASRDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

