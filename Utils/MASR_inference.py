#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:23:32 2021

@author: c95hcw
"""
import sys
sys.path.append("./masr_util")
from models.conv import GatedConv
from CSV_utils import read_mycsv_file, delete_blank, remove_punctuation
from opencc import OpenCC
from wer_eval import eval_asr

class MasrInference():
    def __init__(self, model_path, device):
        self.device = device        
        self.masr_model = self.masr_load_model(model_path, device)
    
    @staticmethod
    def masr_load_model(model_path, device):
        masr_model = GatedConv.load(model_path)         
        if device =="gpu":
            masr_model.to("cuda")
        print("Load MASR model !")        
        return masr_model  
    
    def masr_predict(self, wav, convert_word):
        asr_predict = self.masr_model.predict(wav, self.device)
        # print(asr_predict)
        if convert_word:
            cc = OpenCC('s2tw')
            asr_predict = cc.convert(asr_predict)            
        return asr_predict  

    
    def masr_recognition(self, wav_folder, inference_file, convert_word):
        # load csv data 
        wave_names, asr_truth, _ = read_mycsv_file(inference_file)  
        asr_truth = delete_blank(asr_truth)
        asr_truth = remove_punctuation(asr_truth)
        
        # MASR predict
        asr_predict_result = []
        for i in range(len(wave_names)):             
            wav = wav_folder + wave_names[i]
            text = self.masr_predict(wav, convert_word)  
            asr_predict_result.append(text)    
        # eval
        ASR_wer_avg, wer_list = eval_asr(asr_truth, asr_predict_result)              
        return ASR_wer_avg, asr_truth, asr_predict_result, wer_list





if __name__ == "__main__":
    model_path = "../Models/ASR_models/MASR_v1.2.2/model_ma.pth"
    device = "gpu"

    wav_folder = "../Dataset/raw_data/waves/"
    inference_file = "../Dataset/raw_data/csvs/TW_test.csv"
    convert_word = False
    
    masr_infer = MasrInference(model_path, device)
    ASR_wer_avg, asr_truth, asr_predict_result, wer_list = masr_infer.masr_recognition(wav_folder, inference_file, convert_word)
    