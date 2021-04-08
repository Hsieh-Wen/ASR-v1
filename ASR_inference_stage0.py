#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:53:48 2021

@author: c95hcw

整合4個模型(MASR、Quartznet、Espnet、Google) 的inference code:
"""

import re
import os
import configparser
import numpy as np

import sys
sys.path.append("./Utils")
sys.path.append("./Utils/masr_util")
sys.path.append("./Utils/quartznet_util")

from Save_Inference_Result import SaveResult

from MASR_inference import MasrInference
from Quartznet_inference import QuartznetInference
from Espnet_inference import EspnetInference
from Google_inference import GoogleInference


def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf


class InferenceASRmodels():
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
        if self.asr_mode == "MASR":
            print("Inference with MASR model !")
            ASR_wer_avg, asr_truth, asr_predict_result, wer_list = self.masr_infer.masr_recognition(wav_folder, inference_file, convert_word)
            
        elif self.asr_mode == "Quartznet":
            ASR_wer_avg, asr_truth, asr_predict_result, wer_list = self.quartznet_infer.quartznet_recognition(wav_folder, inference_file, convert_word)
            
        elif self.asr_mode == "Espnet":
            ASR_wer_avg, asr_truth, asr_predict_result, wer_list = self.espnet_infer.espnet_recognition(wav_folder, inference_file, convert_word)

        elif self.asr_mode == "Google":
            ASR_wer_avg, asr_truth, asr_predict_result, wer_list = self.google_infer.google_recognition(wav_folder, inference_file)
            
        else:
            print(f"Can not inference this data !!")

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
                          'Py_file_name': os.path.basename(sys.argv[0])}

                if "/" in infer_file:
                    infer_file_name = infer_file.split("/")[-1]
                    infer_folder = "/".join(infer_file.split("/")[0:-1])
                else:
                    infer_file_name = infer_file
                    infer_folder = "./"
                npz_name = re.sub(".csv",".npz",infer_file_name)
                npz_path = infer_folder + "/" + npz_name

                if os.path.isfile(infer_folder + "/" + npz_name):
                    print("Load orgiginal npz data !!")
                    npz_data = np.load(npz_path, allow_pickle=True)
                    kwargs.update(npz_data)
                save_result.save_follow(**kwargs)  
                print(f"ASR_Model-{asr_mode}, Model_name-{model_path}, 檔名:{infer_file}, 平均wer= {ASR_wer_avg}")          
                
                
if __name__ == "__main__":
    path = "config_asr_inference.ini" #"config_asr_inference.ini"
    config = read_config(path)

    asr_inference = InferenceASRmodels(config)
    asr_inference.inference_flow()    
 
