#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:16 2021
@author: c95hcw
"""

from MASR_inference import MasrInference
from Quartznet_inference import QuartznetInference
from Espnet_inference import EspnetInference
from Google_inference import GoogleInference
from Speaker_inference import SpeakerInference
from wer_eval import eval_asr

import os
import re
import configparser
import pandas as pd
import numpy as np
import json
import time

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf



class SpeakerSpeechPredict():
    def __init__(self, config):
        # Load config
        # parameters of initial function
        input_csv = config['asr_inference'].get('INFERENCE_FILE')
        _ = self.load_csv_data(input_csv)

        # save parameter 
        self.save_folder = config['asr_inference'].get('SAVE_PATH')
        self.save_json_name = config['asr_inference'].get('SAVE_NAME')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)  
        self.save_result_data = []
        # parameters of ASR function
        self.wav_folder = config['asr_inference'].get('WAV_FOLDER') 
        self.asr_mode = config['asr_inference'].get('ASR_MODE')
        self.device = config['asr_inference'].get('DEVICE')

        model_path = config['asr_inference'].get('MODEL_PATH') 
        self.model_path = re.sub(" ","",model_path) 

        self.convert_word = config['asr_inference'].get('DEVICE') 

        self.speaker_url = config['asr_inference'].get('SPEAKER_URL') 
        self.sample_rate = config['asr_inference'].getint('SAMPLE_RATE') 

    def data_predict(self):

        asr_wer_avg, asr_wer_list, self.asr_result, self.asr_recognition_time = self.ASR_Predict(self.asr_mode, 
                                                                self.model_path, self.device, 
                                                                self.wav_folder, self.convert_word)
        
        _, google_wer_list, self.google_result, self.google_recognition_time = self.ASR_Predict("Google", 
                                                        self.model_path, self.device, 
                                                        self.wav_folder, self.convert_word)

        speaker_wer_avg, speaker_wer_list, self.speaker_result, self.speaker_truth, self.speaker_recognition_time = self.Speaker_Predict(self.wav_folder)
        
        if self.eng_idx != []:
            for e_idx in self.eng_idx:
                asr_wer_list.pop(e_idx)
            asr_wer_avg = np.sum(asr_wer_list)/len(asr_wer_list)



        return asr_wer_avg, speaker_wer_avg
        

    def output_json(self):
        asr_wer_avg, speaker_wer_avg = self.data_predict()

        # data_list = list(zip(self.wav_start, self.wav_end,  
        #                     self.asr_truth, self.asr_result, self.asr_wer_list, 
        #                     self.speaker_truth, self.speaker_result, self.speaker_wer_list))
        
        
        self.save_result_data.append({'speech_accuracy': 1-asr_wer_avg , 'speaker_accuracy': 1-speaker_wer_avg})
        for i in range(len(self.asr_truth)):
            kwargs = {'start_time':  self.wav_start[i], 'end_time':  self.wav_end[i], 
                    'MASR_results': self.speaker_result[i] + ":" + self.asr_result[i], 
                    'google_asr_results': self.google_result[i],
                    'labels': self.speaker_truth[i] + ":" + self.asr_truth[i],
                    'asr_recognition_time': self.asr_recognition_time[i],
                    'google_recognition_time': self.google_recognition_time[i],
                    'speaker_recognition_time': self.speaker_recognition_time[i]}

            self.save_result_data.append(kwargs)
        with open(self.save_folder + self.save_json_name, 'w') as fp:
            json.dump(self.save_result_data, fp, indent=len(kwargs),ensure_ascii=False)        
        print("Already save data !!")

    def load_csv_data(self, input_csv):
        # Open the CSV file for reading
        csv_data = pd.read_csv(open(input_csv)) # , sep=r",|\t"

        # csv_data = csv_data.iloc[0:2]

        self.wave_names = csv_data.Wave_path       
        self.wav_start = csv_data.Time_start
        self.wav_end = csv_data.Time_end
        self.speaker_truth = csv_data.Speaker_name

        language_data = csv_data.Other_Lang
        asr_truth = csv_data.Labels   

       # Remove punctuation
        self.asr_truth = self.remove_punctuation(asr_truth)
       # Get english index
        self.eng_idx = []
        self.taiwane_idx = []
        for i,lang in  enumerate(language_data):
            if lang == "English":
                self.eng_idx.append(i)
            elif lang == "Taiwanese":
                self.taiwane_idx.append(i)
            else:
                pass

        return csv_data


    
    def ASR_Predict(self, asr_mode, model_path, device, wav_folder, convert_word):
        """
        目前有四種 ASR Model(MASR、Quartznet、Espnet、Google_ASR).
        
        若一次 inference 同時須輸入2個模型，以 && 分割
            EX: Expnet 同時需要 ASR與LM
            `MODEL_PATH = ./Models/ASR_models/Espnet/espnet_v1/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.ave_10best.pth &&
                        ./Models/ASR_models/Espnet/espnet_v1/lm_train_lm_transformer_zh_char/valid.loss.ave_10best.pth`     
        """
        if "&&" in model_path:
            model_path = model_path.split("&&")
        else:
            model_path = model_path

        if asr_mode == "MASR":
            masr_infer = MasrInference(model_path, device)            
            asr_result = []
            asr_recognition_time =[]
            for i in range(len(self.wave_names)):             
                wav = wav_folder + self.wave_names[i]
                s_time = time.time() 
                text = masr_infer.masr_predict(wav, convert_word)  
                e_time = time.time() 
                asr_recognition_time.append(e_time-s_time)
                asr_result.append(text)    
            # eval
            asr_wer_avg, asr_wer_list = eval_asr(self.asr_truth, asr_result)             
            
        elif asr_mode == "Espnet":
            espnet_infer = EspnetInference(model_path, device) 
            asr_result = []  
            asr_recognition_time = []
            for i in range(len(self.wave_names)):
                wav = wav_folder + self.wave_names[i]
                s_time = time.time() 
                text = espnet_infer.espnet_predict(wav, convert_word)        
                e_time = time.time() 
                asr_recognition_time.append(e_time-s_time)
                print(f"{self.wave_names[i]}: {text}.")
                asr_result.append(text)
            # evaluation
            asr_wer_avg, asr_wer_list = eval_asr(self.asr_truth, asr_result)
        
        elif asr_mode == "Google":
            google_infer = GoogleInference() 
            asr_result = []        
            asr_recognition_time = []
            for i in range(len(self.wave_names)):
                wav = wav_folder + self.wave_names[i]            

                s_time = time.time() 
                text = google_infer.google_predict(wav)        
                print(f"{self.wave_names[i]}: {text}.")                
                e_time = time.time() 
                asr_recognition_time.append(e_time-s_time)
                
                asr_result.append(text)
            # evaluation
            asr_wer_avg, asr_wer_list = eval_asr(self.asr_truth, asr_result)     

        else:            
            print(f"No {asr_mode} Exists !!")

        return asr_wer_avg, asr_wer_list, asr_result, asr_recognition_time


        
    def Speaker_Predict(self,wav_folder):
        wav_list = []        
        for i in range(len(self.wave_names)):
            wav = wav_folder + self.wave_names[i]
            wav_list.append(wav)
            
        speaker_infer = SpeakerInference(self.speaker_url)
        speaker_wer_avg, speaker_wer_list, speaker_result, speaker_list, speaker_recognition_time = speaker_infer.Speaker_Eval(self.sample_rate, wav_list, self.speaker_truth)
 
        return speaker_wer_avg, speaker_wer_list, speaker_result, speaker_list, speaker_recognition_time
    
    @staticmethod        
    def remove_punctuation(asr_truth):
        """
        去除 csv 資料(asr_truth)中的標點符號.
        """
        puncs = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fa5]') 
        new_labels = []
        for text in asr_truth:
            text = puncs.sub("",text)
            new_labels.append(text)
        return new_labels  
# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    path = "inference_json.ini"
    config = read_config(path)
    infer_json = SpeakerSpeechPredict(config)
    infer_json.output_json()