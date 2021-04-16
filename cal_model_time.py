#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to calculate recognition time of ASR Model.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:53:48 2021

@author: c95hcw

整合4個模型(MASR、Quartznet、Espnet、Google) 的inference code:
"""

import re
import os
import time
import configparser
import numpy as np
import librosa

import sys
sys.path.append("./Utils")
sys.path.append("./Utils/masr_util")
sys.path.append("./Utils/quartznet_util")

sys.path.append("../")
from Save_Inference_Result import SaveResult
from CSV_utils import read_mycsv_file, delete_blank, remove_punctuation
from wer_eval import eval_asr

from MASR_inference import MasrInference
from Quartznet_inference import QuartznetInference
from Espnet_inference import EspnetInference
from Google_inference import GoogleInference


def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf


class InferenceASRmodelsTime():
    def __init__(self, config):
        # Load config
        # parameters of initial function
        asr_modes = config['Stage0_asr_inference'].get('ASR_MODE')
        self.device = config['Stage0_asr_inference'].get('DEVICE')    
        model_paths = config['Stage0_asr_inference'].get('MODEL_PATH') 

        
        # parameters of inference function
        self.wav_folder = config['Stage0_asr_inference'].get('WAV_FOLDER')  
        if self.wav_folder == "None":
            self.wav_folder = ""  
        inference_files = config['Stage0_asr_inference'].get('INFERENCE_FILE') 
        self.convert_word = config['Stage0_asr_inference'].get('CONVERT_WORD')       

        # parameters of save parameter       
        self.save_path = config['Stage0_asr_inference'].get('SAVE_PATH') 

        # 
        self._init_params(asr_modes, model_paths, inference_files)
 
    def _init_params(self, asr_modes, model_paths, inference_files):     
        """
        處理 config data. 
        (str ---> list)        
        """
        asr_modes = re.sub(" ","",asr_modes)
        asr_modes = re.sub("\n","",asr_modes)
        self.asr_modes_list = asr_modes.split("|")    
       
        model_paths = re.sub(" ","",model_paths)
        model_paths = re.sub("\n","",model_paths)
        self.model_path_list = model_paths.split("|")    
        assert len(self.asr_modes_list) == len(self.model_path_list),"asr_mode 或 model_path 數量不對！！請修改 config_asr_inference.ini 之 [ASR_MODE] 或 [model_path]"

        inference_files = re.sub(" ","",inference_files)
        inference_files = re.sub("\n","",inference_files)
        self.inference_file_list = inference_files.split("|")    

    
    def Load_Model(self, model_path, device):
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

        if self.asr_mode == "MASR":
            self.masr_infer = MasrInference(model_path, device)            
            ASR_model = self.masr_infer.masr_model
            print("Load MASR model !")
            
        elif self.asr_mode == "Quartznet":
            self.quartznet_infer = QuartznetInference(model_path) 
            ASR_model =  self.quartznet_infer.quartznet_model
            print("Load Quartznet model !")
            
        elif self.asr_mode == "Espnet":
            self.espnet_infer = EspnetInference(model_path, device) 
            ASR_model =  self.espnet_infer.espnet_model
            print("Load Espnet model !") 
        
        elif self.asr_mode == "Google":
            self.google_infer = GoogleInference() 
            ASR_model =  self.google_infer.google_model
            print("Google Speech Recognition !")  

        else:            
            print(f"No {self.asr_mode} Exists !!")

        return ASR_model
    
    
    def inference_data(self, wav_folder, inference_file, convert_word):            
        """
        以4種 asr_mode(MASR、Quartznet、Espnet、Google) inference data.
        * Input : 
        	- wav_folder: 設定音檔所在資料夾位置
        	- inference_file: 要進行 inference 的檔案名 (csv檔)
        	- convert_word: (Trur/False) 是否要將Inference的結果文字簡轉繁
        * Output : 
        	- ASR_wer_avg: 語音辨識平均錯誤率
        	- asr_truth: (list) 每個音檔的正確答案
        	- asr_predict_result: (list) 每個音檔的語音辨識結果
        	- wer_list: 每個音檔的語音辨識錯誤率
        """
        wave_names, asr_truth, _ = read_mycsv_file(inference_file)  
        asr_truth = delete_blank(asr_truth)
        asr_truth = remove_punctuation(asr_truth)

        wave_paths = []
        if wav_folder != "None":
            for wav_path in wave_names:
                wave_paths.append(wav_folder + wav_path)
        else:
            wave_paths = wave_names

        wave_times = self.get_wav_duration(wave_paths)


        asr_predict_result = []
        reco_time = []

        if self.asr_mode == "MASR":
            print("Inference with MASR model !")
            for wav in wave_paths:  
                s_time = time.time()           
                text = self.masr_infer.masr_predict(wav, convert_word)
                e_time = time.time()   
                print(f"{wav}: {text}, reco_time: {str(e_time-s_time)}")
                          
                asr_predict_result.append(text)   
                reco_time.append(e_time-s_time)

        elif self.asr_mode == "Espnet":
            print("Inference with Espnet model !")
            for wav in wave_paths: 
                s_time = time.time()           
                text = self.espnet_infer.espnet_predict(wav, convert_word)
                e_time = time.time()         
                print(f"{wav}: {text}, reco_time: {str(e_time-s_time)}")

                asr_predict_result.append(text)
                reco_time.append(e_time-s_time)

        elif self.asr_mode == "Google":
            print("Inference with Google ASR !")
            for wav in wave_paths: 
                s_time = time.time()           
                text = self.google_infer.google_predict(wav)
                e_time = time.time()         
                print(f"{wav}: {text}, reco_time: {str(e_time-s_time)}")

                asr_predict_result.append(text)
                reco_time.append(e_time-s_time)  
        else:
            print(f"Can not inference this data !!")

        ASR_wer_avg, wer_list = eval_asr(asr_truth, asr_predict_result)
        
        self.uttarences = len(reco_time)        
        self.all_wavs_time = np.sum(reco_time)

        maxIndex, self.large_wav = max(enumerate(wave_times), key=lambda x: x[1])        
        minIndex, self.small_wav = min(enumerate(wave_times), key=lambda y: y[1])        

        self.small_wav_time = reco_time[minIndex]
        self.large_wav_time = reco_time[maxIndex]

        self.avg_recognition_time = np.sum(reco_time)/len(reco_time)

        return ASR_wer_avg, asr_truth, asr_predict_result, wer_list 


    def inference_flow(self):
        save_result = SaveResult(self.save_path)
        for asr_mode, model_path in zip(self.asr_modes_list, self.model_path_list):
            self.asr_mode = asr_mode
            for infer_file in self.inference_file_list:
                # Load ASR Model 
                self.ASR_model = self.Load_Model(model_path, self.device)
                ASR_wer_avg, asr_truth, asr_predict_result, wer_list = self.inference_data(self.wav_folder, infer_file, self.convert_word)
    
                kwargs = {'Inference_Mode':'ASR', 'Input_File': infer_file, 'Model_Path': model_path,
                          'ASR_wer_avg': ASR_wer_avg,
                          'Ground_Truth': asr_truth, 'Predict_Result': asr_predict_result, 'Wer_list': wer_list,
                          'wav_num': self.uttarences, 'all_wav_recognition_time': self.all_wavs_time,
                          'max_wav_time': self.large_wav, 'max_wav_recognition_time': self.large_wav_time, 
                          'min_wav_time': self.small_wav, 'min_wav_recognition_time': self.small_wav_time,
                          'Avg_recognition_time': self.avg_recognition_time,
                          'Py_file_name': os.path.basename(sys.argv[0])}
                save_result.save_follow(**kwargs)  
                print(f"ASR_Model-{asr_mode}, Model_name-{model_path}, 檔名:{infer_file}, 平均wer= {ASR_wer_avg}")          
    
    
    @staticmethod     
    def get_wav_duration(wave_pathes):
        """
        Get wave's duration by librosa.get_duration(wave_path).
        * Input :
            wave_pathes : waves'path (list).
        * Output : 
            wave_times : waves'duration (list).
        """
        wave_times = []
        for wav in wave_pathes:
            # wav_data, sample_rate = librosa.load(wav, sr=16000)
            wav_duration = librosa.get_duration(filename=wav) 
            wave_times.append(wav_duration)
        return wave_times                
                
if __name__ == "__main__":
    path = "config_asr_time.ini" #"config_asr_inference.ini"
    config = read_config(path)

    asr_inference = InferenceASRmodelsTime(config)
    asr_inference.inference_flow()    
 
