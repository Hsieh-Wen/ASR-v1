#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:53:48 2021

@author: c95hcw

整合4個模型(MASR、Quartznet、Espnet、Google) 的inference code:
"""

import re
import os
import numpy as np

import sys
sys.path.append("./Utils")
sys.path.append("./Utils/masr_util")
sys.path.append("./Utils/quartznet_util")

from load_config_args import load_config
from Save_Inference_Result import SaveResult

from MASR_inference import MasrInference
from Quartznet_inference import QuartznetInference
from Espnet_inference import EspnetInference
from Google_inference import GoogleInference


class InferenceASRmodels():
    def __init__(self, config_path):
        # Load config
        self.args = load_config(config_path=config_path)

        # parameters of initial function
        self.device = self.args.DEVICE
        self.convert_word = self.args.CONVERT_WORD

        self.asr_models = self.args.ASR_Models
        self.inference_files =  self.args.INFERENCE_FILEs

        # parameters of save parameter       
        self.save_path =  self.args.SAVE_PATH
    
    def Load_Model(self, model_path, device):
        """
        目前有四種 ASR Model(MASR、Quartznet、Espnet、Google_ASR).
        
        若一次 inference 同時須輸入2個模型，以 && 分割
            EX: Expnet 同時需要 ASR與LM
            `MODEL_PATH = ./Models/ASR_models/Espnet/espnet_v1/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.ave_10best.pth &&
                        ./Models/ASR_models/Espnet/espnet_v1/lm_train_lm_transformer_zh_char/valid.loss.ave_10best.pth`     
        """
        if "&&" in model_path:
            model_path = re.sub(" ","",model_path)
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
        if wav_folder == "None":
            wav_folder = ""
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
        Model_keys = self.asr_models.keys()
        inferfile_keys = self.inference_files.keys()
        for model_id in Model_keys:
            self.asr_mode = self.asr_models[model_id]['ASR_Mode']
            model_path = self.asr_models[model_id]['Model_Path']
            for inferfile_id in inferfile_keys:
                infer_file = self.inference_files[inferfile_id]['csv_file']
                wav_folder = self.inference_files[inferfile_id]['WAV_FOLDER']
                # Load ASR Model 
                self.ASR_model = self.Load_Model(model_path, self.device)
                ASR_wer_avg, asr_truth, asr_predict_result, wer_list = self.inference_data(wav_folder, infer_file, self.convert_word)

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
                print(f"ASR_Model-{self.asr_mode}, Model_name-{model_path}, 檔名:{infer_file}, 平均wer= {ASR_wer_avg}")          
                
                
if __name__ == "__main__":
    config_path = "/home/c95hcw/ASR/config_ASR_inference.yaml"
    asr_inference = InferenceASRmodels(config_path)
    asr_inference.inference_flow()    
 
