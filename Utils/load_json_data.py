#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 08:28:33 2021

@author: c95hcw
"""

import json
import numpy as np

json_path = "/home/c95hcw/1100326_demo/json_file/video6.json"

asr_time = []
google_time = []
speaker_time = []

with open(json_path, 'r') as f:
    data = json.load(f) 

for item in data[1:]:
    asr_time.append(item['asr_recognition_time'])
    google_time.append(item['google_recognition_time'])
    speaker_time.append(item['speaker_recognition_time'])
    
    
avg_asr_time = np.sum(asr_time)/len(asr_time)
avg_google_time = np.sum(google_time)/len(google_time)
avg_speaker_time = np.sum(speaker_time)/len(speaker_time)


print(f"{avg_asr_time=}")
print(f"{avg_google_time=}")
print(f"{avg_speaker_time=}")