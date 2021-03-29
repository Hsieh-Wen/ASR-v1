#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:36:26 2021

@author: c95hcw
"""

import glob
import os
import librosa
import soundfile as sf

path = "/home/c95hcw/ASR_Data/Dataset/raw_data/waves/20210322_speech_tw/wav/UwF7h"
new_path = path+"_normalize/"
os.mkdir(new_path)

wav_path = glob.glob(path+"/*.wav")

for p in wav_path:
    wav, sr = librosa.load(p, sr=16000)
    wav_name = p.split("/")[-1]
    sf.write(new_path+wav_name, wav, sr)
   
path = "/home/c95hcw/ASR_Data/Dataset/raw_data/waves/20210322_speech_tw/wav_normal/UwF7h/20210317_UwF7h_00107.wav"
wav, sr = librosa.load(path, sr=16000)
text = {"speaker":wav.tolist()}
a = requests.post("http://172.16.120.123:6000/"+ "speaker" ,json=text)
json.loads(a.text)