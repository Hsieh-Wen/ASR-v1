import librosa
import wave
import numpy as np
import torch
import soundfile

sample_rate = 16000
window_size = 0.02
window_stride = 0.01
n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)
window = "hamming"


def load_audio(wav_path, normalize=True):  # -> numpy array
#    audio, rate = soundfile.read(wav_path)
#    if rate != 16000:
##        print(f"{audio.shape=}")
#        audio = librosa.resample(audio[:,0], rate, 16000)    
#    return audio
    
    
    with wave.open(wav_path) as wav:
#        sr = wav.getframerate()
        wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
        wav = wav.astype("float")
    if normalize:
        wav = (wav - wav.mean()) / wav.std()
#        if sr != sample_rate:
#            wav = librosa.resample(wav, sr, sample_rate)
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
