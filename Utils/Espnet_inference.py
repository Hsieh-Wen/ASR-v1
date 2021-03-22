#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:26:10 2021

@author: c95hcw
"""

import soundfile
from opencc import OpenCC
from espnet2.bin.asr_inference import Speech2Text

from CSV_utils import read_csv_file, delete_blank
from wer_eval import eval_asr

class EspnetInference():
    def __init__(self, model_path, device):
        assert len(model_path) == 2 , "Espnet need two model path !!"
        asr_model_path = model_path[0]
        lm_model_path = model_path[1]
        self.espnet_model = self.espnet_load_model(asr_model_path, lm_model_path, device)            

    @staticmethod
    def espnet_load_model(asr_model_path, lm_model_path, device):     
        
        asr_model_folders = asr_model_path.split("/")
        asr_config_folder = "/".join(asr_model_folders[0:-1])
        asr_train_config_path = asr_config_folder + "/config.yaml" 
        
        if lm_model_path != "None":
            lm_folders = lm_model_path.split("/")
            lm_folder = "/".join(lm_folders[0:-1])
            lm_train_config = lm_folder + "/config.yaml"
        else:
            lm_train_config = None
        
    
        if device =="gpu":
            
            device_mode = "cuda"
        else:
            device_mode = "cpu"

        espnet_model = Speech2Text(
            asr_train_config = asr_train_config_path, 
            asr_model_file = asr_model_path, 
            device = device_mode,
            lm_train_config = lm_train_config,
            lm_file = lm_model_path,
            # Decoding parameters are not included in the model file
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=20,
            ctc_weight=0.3,
            lm_weight=0.5,
            penalty=0.0,
            nbest=1
        )
        return espnet_model


    def espnet_predict(self, wav, convert_word):
        audio, rate = soundfile.read(wav)
        nbests = self.espnet_model(audio)
        asr_predict, *_ = nbests[0]
        # print(asr_predict)
        if convert_word:
            cc = OpenCC('s2tw')
            asr_predict = cc.convert(asr_predict)    
            
        if asr_predict == "":
            asr_predict ="無法辨識"
        return asr_predict    
        
        
    
    def espnet_recognition(self, wav_folder, inference_file, convert_word):
        # load csv data 
        wave_names, asr_truth = read_csv_file(inference_file)     
        asr_truth = delete_blank(asr_truth)                 
        # espnet predict        
        asr_predict_result = []        
        for i in range(len(wave_names)):
            wav = wav_folder + wave_names[i]            
            asr_predict = self.espnet_predict(wav, convert_word)        
            print(f"{wave_names[i]}: {asr_predict}.")
            asr_predict_result.append(asr_predict)
        # evaluation
        ASR_wer_avg, wer_list = eval_asr(asr_truth, asr_predict_result)              
        return ASR_wer_avg, asr_truth, asr_predict_result, wer_list        
    

    
if __name__ == "__main__":
    model_path = ["/home/c95hcw/ASR_Data/Models/ASR_models/Espnet/espnet_v1.0/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.best.pth", "/home/c95hcw/ASR_Data/Models/ASR_models/Espnet/espnet_v1.0/lm_train_lm_transformer_zh_char/valid.loss.ave_10best.pth"]
    device = "gpu"
    
    wav_folder = "/home/c95hcw/ASR_Data/Dataset/raw_data/waves/"
    inference_file = "/home/c95hcw/ASR_Data/Dataset/raw_data/csvs/TW_test.csv"
    convert_word = False
    
    espnet_infer = EspnetInference(model_path, device) 
    ASR_wer_avg, asr_truth, asr_predict_result, wer_list = espnet_infer.espnet_recognition(wav_folder, inference_file, convert_word)   


     
