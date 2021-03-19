#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:29:00 2021

@author: c95hcw
"""

import speech_recognition as sr

from CSV_utils import read_csv_file, delete_blank
from wer_eval import eval_asr

class GoogleInference():
    def __init__(self):
        self.google_model = sr.Recognizer()
    
    @staticmethod
    def google_load_model():
        google_model = sr.Recognizer()
        return google_model
    
    def google_predict(self, wav):
        with sr.WavFile(wav) as source:
                audio_data =sr.Recognizer().record(source)         
        try:
            asr_predict = self.google_model.recognize_google(audio_data, language="zh-TW")
            # print(result)                    
        except sr.UnknownValueError:
            asr_predict ="無法辨識"
            # print('this utterance have some error w/ ' + str(sr.UnknownValueError))
        return asr_predict 
    
    
    def google_recognition(self, wav_folder, inference_file):
        # load csv data 
        wave_names, asr_truth = read_csv_file(inference_file)     
        asr_truth = delete_blank(asr_truth)        
        # google predict        
        asr_predict_result = []        
        for i in range(len(wave_names)):
            wav = wav_folder + wave_names[i]            
            asr_predict = self.google_predict(wav)        
            print(f"{wave_names[i]}: {asr_predict}.")
            asr_predict_result.append(asr_predict)
        # evaluation
        ASR_wer_avg, wer_list = eval_asr(asr_truth, asr_predict_result)              
        return ASR_wer_avg, asr_truth, asr_predict_result, wer_list
    
if __name__ == "__main__":    
    wav_folder = "../Dataset/raw_data/waves/"
    inference_file = "../Dataset/raw_data/csvs/TW_test.csv"
    
    google_infer = GoogleInference() 
    ASR_wer_avg, asr_truth, asr_predict_result, wer_list = google_infer.google_recognition(wav_folder, inference_file)       
    