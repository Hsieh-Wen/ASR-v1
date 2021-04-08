#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:40:35 2021

@author: c95hcw
"""


import os
import re
import configparser

import sys
sys.path.append("./Utils")
from MASR_data_preprocess import MasrDataPreprocess
from Quartznet_data_preprocess import QuartznetDataPreprocess
from Espnet_data_preprocess import EspnetDataProcess
from ASR_inference_data_preprocess import ASRIDataProcess

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf

class ASRDatasetPreprocess():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        stage0_folder = data_folder + "Stage0/"   
        self.save_data_folder = data_folder + "Stage1/" 
        # print(stage0_folder)
        if not os.path.exists(stage0_folder + "ASR/"):
            sys.exit("Please run Data_preprocess_ASR_stage0.py")
        self.asr_train_data_path = stage0_folder + "ASR/combined_train_corpus.csv"
        self.asr_valid_data_path = stage0_folder + "ASR/combined_valid_corpus.csv"
 

        if not os.path.exists(self.save_data_folder):
            os.makedirs(self.save_data_folder)    
        else:
            print(f"資料夾已存在！！已有 Stage1 資料夾")
            
            

    def prepare_data(self, train_mode):
        """
        Data Preprocess.
        (from Stage0 folder-train_corpus and Stage0 folder-valid_corpus)
        (Output result to Stage1 folder)
        """
        if "&&" in train_mode:
            train_parameters = train_mode.split("&&")
            train_mode = train_parameters[0]
            train_parameter_data = train_parameters[1:]
        else:
            train_mode = train_mode

        
        if train_mode == "MASR":
            print("Prepare dataset for MASR model !")
            masr_data_preprocess = MasrDataPreprocess()
            masr_data_preprocess.masr_dataset(self.asr_train_data_path, self.asr_valid_data_path, self.save_data_folder)            
            
        elif train_mode == "Quartznet":
            print("Prepare dataset for Quartznet model !")
            quartznet_data_preprocess = QuartznetDataPreprocess()
            quartznet_data_preprocess.quartznet_dataset(self.asr_train_data_path, self.asr_valid_data_path, self.save_data_folder)
          
        elif train_mode == "Espnet":
            print("Prepare dataset for Espnet model !") 
            espnet_data_process = EspnetDataProcess(self.data_folder)
            espnet_data_process.espnet_dataset()

        elif train_mode == "ASR_Inference_mode":
            print("Prepare dataset for ASR inference !") 
            inference_data_process = ASRIDataProcess()
            inference_data_process.asr_inference_dataset(self.asr_train_data_path, self.save_data_folder)            

        else:
            print(f"沒有此 train_mode: {train_mode} ！！")
        print(f"-------------已完成 {train_mode} 資料前處理-------------")

  
        
    
if __name__ == "__main__":
    path = "config_ASR_data_process.ini"
    config = read_config(path)
 
    # parameters of saving folder name    # "data_v1/"
    data_folder = config['Stage1'].get('SAVE_FOLDER_NAME') 
         
    # preprocessing
    train_modes = config['Stage1'].get('TRAIN_MODES') 
    train_modes = re.sub(" ","",train_modes)
    train_modes = re.sub("\n","",train_modes)
    train_mode_list = train_modes.split("|")
  
    data_process = ASRDatasetPreprocess(data_folder)
    for train_mode in train_mode_list:
        data_process.prepare_data(train_mode)





