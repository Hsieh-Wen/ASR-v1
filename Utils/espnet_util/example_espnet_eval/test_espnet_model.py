#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:51:40 2021

@author: c95hcw
"""

import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
#d = ModelDownloader(cachedir="ASR_model")
#speech2text = Speech2Text(
#**d.download_and_unpack("kamo-naoyuki/aishell_conformer"),
#    device="cuda",
#    # Decoding parameters are not included in the model file
#    maxlenratio=0.0,
#    minlenratio=0.0,
#    beam_size=20,
#    ctc_weight=0.3,
#    lm_weight=0.5,
#    penalty=0.0,
#    nbest=1
#)

speech2text = Speech2Text(
    asr_train_config = "ASR_model/asr_aishell_v0/config.yaml", 
    asr_model_file="ASR_model/asr_aishell_v0/valid.acc.ave_10best.pth", 
    device="cuda",
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1
)

audio, rate = soundfile.read("ytb_000000.wav")
nbests = speech2text(audio)
text, *_ = nbests[0]
print(text)