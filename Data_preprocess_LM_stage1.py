#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:40:35 2021

@author: c95hcw
"""


import os
import re

import sys
sys.path.append("./Utils")
from load_config_args import load_config
#from MASR_data_preprocess import MasrDataPreprocess
#from Quartznet_data_preprocess import QuartznetDataPreprocess
#from Espnet_data_preprocess import EspnetDataProcess
from BERT_data_preprocess import BertDataPreprocess

class LMDatasetPreprocess():
    def __init__(self, config_path):
        """
        Load config params.
        """
        args = load_config(config_path)
        stage1_args = args.Stage1

        self.process_modes = stage1_args['PROCESS_MODES']
        self.data_folder = stage1_args['SAVE_FOLDER_NAME']

        stage0_folder = self.data_folder + "Stage0/"    
        self.save_data_folder = self.data_folder + "Stage1/" 

        if not os.path.exists(stage0_folder + "LM/"):
            sys.exit("Please run Data_preprocess_LM_stage0.py")
        self.lm_train_data_path = stage0_folder + "LM/combined_train_corpus.csv"
        self.lm_valid_data_path = stage0_folder + "LM/combined_valid_corpus.csv"
 

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
            train_mode = re.sub(" ","",train_mode)
            train_parameters = train_mode.split("&&")
            train_mode = train_parameters[0]
            train_parameter_data = train_parameters[1:]
        else:
            train_mode = train_mode

        
        if train_mode == "BERT":
            print("Prepare dataset for BERT model !") 
            bert_data_process = BertDataPreprocess()
            bert_data_process.bert_dataset(self.lm_train_data_path, self.lm_valid_data_path, self.save_data_folder)
        
        elif train_mode == "BERT_wwm":
            print("Prepare dataset for BERT model !") 
            bert_data_process = BertDataPreprocess()
            special_word_file_path = "./Utils/LM_BERT/special_word.txt"
            pretrained_bert = train_parameter_data[0]
            bert_data_process.bert_wwm_dataset(self.lm_train_data_path, self.lm_valid_data_path,
                                            special_word_file_path, pretrained_bert, self.save_data_folder)

        else:
            print(f"沒有此 train_mode: {train_mode} ！！")


    def process_flow(self):
        for train_mode in self.process_modes:
            self.prepare_data(train_mode)
            print(f"-------------已完成 {train_mode} 資料前處理-------------")  
        
    
if __name__ == "__main__":
    config_path = "/home/c95hcw/ASR/config_LM_data_process.yaml"
    data_process = LMDatasetPreprocess(config_path)
    data_process.process_flow()





