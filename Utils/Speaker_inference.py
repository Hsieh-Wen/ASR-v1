#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:48:44 2021

@author: c95hcw
"""

import librosa
import requests
import json
import numpy as np
import time

class SpeakerInference():
    def __init__(self,url):
        self.url = url

    def speaker_predict(self, wav_name, sample_rate):
        print(wav_name)
        wav, sr = librosa.load(wav_name, sr=sample_rate)
        text = {"speaker":wav.tolist()}
        a = requests.post(self.url + "speaker" ,json=text)
        
        speaker = json.loads(a.text)["name"]    
        # speaker = "None"
        return speaker
 
    def Speaker_Eval(self, sample_rate, wav_list, speaker_truth_list):
        speaker_result = []
        speaker_wer_list = []
        speaker_list = []
        speaker_recognition_time =[]
        
        for wav, speaker_truth in zip(wav_list, speaker_truth_list):
            s_time = time.time()
            speaker_id = self.speaker_predict(wav, sample_rate)  
            e_time = time.time() 
            speaker_recognition_time.append(e_time-s_time)
            
            
            speaker_result.append(speaker_id)
            speakers = speaker_truth.split("#")[1:]
            if len(speakers) > 1 :
                print(f"more speaker !!")
                speaker = ','.join(speakers)
            else:
                speaker = speakers[-1]
                if speaker_id == speaker:
                    wer = 0.0
                else:
                    wer = 1.0                
            speaker_list.append(speaker)            
            speaker_wer_list.append(wer) 
        speaker_wer_avg = (np.sum(speaker_wer_list))/len(speaker_list)  
        
        return speaker_wer_avg, speaker_wer_list, speaker_result, speaker_list, speaker_recognition_time
    
    
# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    url = "http://192.168.0.110:8002/ailab"
    
    wav_list = ["/home/ASR_Data/Dataset/raw_data/waves/Training_dataset/Wav/JcpyA/20201029_JcpyA_00000.wav","/home/ASR_Data/Dataset/raw_data/waves/Training_dataset/Wav/JcpyA/20201029_JcpyA_00005.wav"]
    speaker_truth_list = ["#江啟臣","#江啟臣#呂文忠"]
    sample_rate = 16000
    
    speaker_infer = SpeakerInference(url)
    speaker_wer_avg, speaker_wer_list, speaker_result, speaker_list, speaker_recognition_time = speaker_infer.Speaker_Eval(sample_rate, wav_list, speaker_truth_list)