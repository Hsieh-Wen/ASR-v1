#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:36:11 2021

@author: c95hcw
"""
import sys
sys.path.append("./Utils")
from load_config_args import load_config
from CSV_utils import read_lm_csv_file, combine_csv_files
from Bulid_Vocabulary import bulid_vocab

import json
import pandas as pd
import numpy as np
import random
from math import floor
import re
import os
import sys


class LmCsvDataPreporcessing():
    def __init__(self, config_path):
        # load config params
        args = load_config(config_path=config_path)
        stage0_args = args.Stage0

        self.input_files = stage0_args['INPUT_FILES']

        save_folder_name = stage0_args['SAVE_FOLDER_NAME']
        self.save_data_folder = save_folder_name + "Stage0/LM/" 
        if os.path.exists(self.save_data_folder):
            sys.exit("資料夾已存在！！請修改 config_data_process.ini 之 [SAVE_FOLDER_NAME]") 

    def data_preprocess_parameters(self, parameters_dict): 

        self.check_data_parameter_exist(parameters_dict)

        self.csv_name = parameters_dict['INTPUT_CSV_PATH']
        self.csv_column_label = parameters_dict['CSV_COLUMN_LABEL']

        self.word_len_threshold = parameters_dict['WORD_LEN_THRESHOLD']
        self.split_ratio =  parameters_dict['SPLIT_RATIO']
        self.split_mode =  parameters_dict['SPLIT_MODE']
        self.seed =  parameters_dict['SEED']
        
        self.print_paremeters()

    def check_data_parameter_exist(self, parameters_dict):
        # input csv         
        if 'INTPUT_CSV_PATH' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of INTPUT_CSV_PATH")   

        # Parameters of word         
        if 'WORD_LEN_THRESHOLD' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of WORD_LEN_THRESHOLD")      
            
        # parameters of split csv data            
        if 'SPLIT_RATIO' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SPLIT_RATIO")  
        if 'SPLIT_MODE' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SPLIT_MODE")  
        if 'SEED' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SEED")  

    def print_paremeters(self):
        print(f"{self.csv_name=}.")
        print(f"{self.word_len_threshold=}.")
        print(f"{self.split_ratio=}.")
        print(f"{self.split_mode=}.")
        print(f"{self.seed=}.")


    def _word_process(self, csv_name, split_ratio, split_mode):
        self._init_csv_cropus(csv_name,self.csv_column_label)
        
        # split csv data
        train_data, valid_data = self.get_training_and_testing_sets(split_ratio, split_mode, self.seed)

        # remove same data
        train_data = self.remove_same_data(train_data)
        valid_data = self.remove_same_data(valid_data)

        # remove small data(only one word)
        train_data = self.remove_small_data(train_data, self.word_len_threshold)
        valid_data = self.remove_small_data(valid_data, self.word_len_threshold)        

        self.save_train_valid_to_csvs(train_data, valid_data, self.save_data_folder)
          
        return {'csv_name': csv_name, 
                'word_len_threshold':self.word_len_threshold,
                'split_ratio': split_ratio,
                'split_mode': split_mode,
                'seed':self.seed,
                'save_folder_name': self.save_data_folder}


    def process_flow(self):
        
        data_parameter_dict = dict()    
        for i, input_file_id in enumerate(self.input_files.keys()):
            self.data_preprocess_parameters(self.input_files[input_file_id])
            print(f"\n--------------DATA PREPROCESSING - LM--{self.csv_name}----------------")

            # save setting data of data preprocessing
            data_parameter_dict['Corpus_' + str(i)] = self._word_process(self.csv_name, self.split_ratio, self.split_mode)
            
        # 合併 train/valid corpus 內所有 csv 為 train.csv / valid.csv
        self.combine_train_valid_corpus(self.save_data_folder, self.save_data_folder)        
        
        # save data process parameter to json
        json_path = self.save_data_folder + 'lm_data_processing_parameter.json'
        self.save_json(json_path, data_parameter_dict)
        

        
        
    @staticmethod
    def save_json(path, data_parameter_dict):
        with open(path, 'w') as fp:
            json.dump(data_parameter_dict, fp, indent=len(data_parameter_dict))
        print(f"save data_processing_paramete.json in path: {path}")
        
        
        
    def _init_csv_cropus(self, csv_name, csv_column_label):
        """Load text from csv.

        Args:
            csv_name (str): csv file's name.
        """
        self.csv_name = csv_name
        self.data_path = csv_name
        asr_truth = read_lm_csv_file(csv_name, csv_column_label)
        self.asr_truth = self.delete_blank(asr_truth) # 去除label檔中的空格

    
    def get_training_and_testing_sets(self, split_ratio, split_mode, seed=3):
        """
        將 csv file 分割成 train list 與 valid list，
        * Input :
            - split_ratio: 分割比例(default: train:valid = 8:2)
            - split_mode: 選擇分割方法
            - seed: 隨機排序的種子數
        
                兩種分割方法:
                    * split_mode = Vocab
                        根據字頻(每個字在資料集內出現的頻率)，
                        字頻較多之句子，並以 split_ratio 的比例作為 validation data.
                    * split_mode = Random
                        將資料集做隨機排序，
                        再以 split_ratio 的比例分割 train 和 validation data.  
        * Output :
            training_list: 分割好的 train list ; 若split_ratio=0.0, 則 train list ＝ [].
            validation_list:  分割好的 valid list ; 若split_ratio=1.0, 則 valid list ＝ [].       
        """


        if split_mode == "Vocab":
            
            vocabulary, sorted_vob, vocab_json_path = bulid_vocab(self.data_path, "no_save" , save_file = False, csv_column_label=self.csv_column_label)
            
            vocab = dict((x,y) for x,y in sorted_vob)
            counter_all = []
            for i in range(len(self.asr_truth)):
                idx_vec = list(map(lambda j: vocab[j],self.asr_truth[i]))    
                counter_all.append(int(np.sum(idx_vec)))
            
            all_data = list(zip(self.asr_truth,counter_all))
            vocab_sorted_list = sorted(all_data, key=lambda x: x[1])  
            file_o_list = []
            for item in vocab_sorted_list:
                file_o_list.append(item[0])
        
        elif split_mode == "Random":
            file_list = list(self.asr_truth)    
            random.seed(seed)
            random.shuffle(file_list)
            file_o_list = list(labels)

        elif split_mode == "Sequence":
            file_o_list = list(self.asr_truth) 
        else:
            sys.exit("No this split_mode !!!!!")

        split_index = floor(len(file_o_list) * split_ratio)
        training_list = file_o_list[:split_index]
        validation_list = file_o_list[split_index:]
        
        print("Split dataset !!")
        print(f"Training data = {split_ratio*100} %, Validation data = {(1-split_ratio)*100} %\n")
        print(f"Orginal data : {self.data_path}")
        
        return training_list, validation_list

    @staticmethod
    def remove_same_data(text_list):
        """去除list中重複之句子

        Args:
            text_list (list): label list.

        Returns:
            text_list(list): label list.
        """
        org_len = len(text_list)
        labels_set = set()
        for text in text_list:
            if text not in labels_set:
                labels_set.add(text)
        data_list = list(labels_set)
        out_len = len(data_list)
        print(f"已去除{str(org_len - out_len)}個相同句子[{org_len=} --> {out_len=}]")
        return data_list

    @staticmethod
    def remove_small_data(text_list, word_len_threshold):
        """去除text_list中字數過少之句子

        Args:
            text_list (list): 
            word_len_threshold (float): limit of word short length.

        Returns:
            text_list(list): label list.
        """
        delete_idx = []
        org_len = len(text_list)
        for i, txt in enumerate(text_list):
            if len(txt) <= word_len_threshold:
                delete_idx.append(i)
                text_list.pop(i)
                print(f"text's length too short. {txt=}")
        out_len = len(text_list) 
        print(f"已去除{str(org_len - out_len)}個字數過少的句子[{org_len=} --> {out_len=}]")
        return text_list


    def save_train_valid_to_csvs(self, training_list, validation_list, save_data_folder):
        """
        將 train list 與 valid list 存為 train.csv 與 valid.csv
        * Input :
            - training_list: train list
            - validation_list: valid list
            - save_data_folder: 最後存 train/valid csv 的資料夾路徑
        * Output :
            (default save folder  = csv file 的資料夾內)
            (default save name = *_train.csv or *_valid.csv)
        """    
        save_lm_folder = save_data_folder
        if not os.path.exists(save_lm_folder):
            os.makedirs(save_lm_folder)  

        if not os.path.exists(save_lm_folder + "train_corpus/"):
            os.makedirs(save_lm_folder + "train_corpus/")   
        if not os.path.exists(save_lm_folder + "valid_corpus/"):
            os.makedirs(save_lm_folder + "valid_corpus/") 
        
        if "/" in self.csv_name:
            save_csv_name = self.csv_name.split("/")[-1]  
        else:
            save_csv_name = self.csv_name 

        if training_list != []:
            train_data_name = re.sub(".csv","_train" + ".csv",save_csv_name)
            train_data_path = self.save_csv_file(save_lm_folder + "train_corpus/" + train_data_name, training_list)
            print(f"Training data path : {train_data_path}")          
        else: 
            print(f"{self.csv_name} is not split to train.csv") 
            
        if validation_list != []:   
            valid_data_name = re.sub(".csv","_valid" + ".csv",save_csv_name)  
                
            valid_data_path = self.save_csv_file(save_lm_folder + "valid_corpus/" + valid_data_name, validation_list)
            print(f"Validation data path : {valid_data_path}")  
        else: 
            print(f"{self.csv_name} is not split to valid.csv")
            
    @staticmethod             
    def combine_train_valid_corpus(input_csv_folder, output_folder): 
        """
        Combined csv files to train.csv and valid.csv.
        (from folder-train_corpus and folder-valid_corpus)
        """
        data_modes = ["train_corpus","valid_corpus"]
        for data_mode in data_modes:
            combine_csv_files(input_csv_folder, data_mode, output_folder)            
            
    @staticmethod        
    def delete_blank(asr_truth):
        """
        去除 csv 資料(asr_truth)中的空格、空白行或tab.
        """
        # pattern = re.compile(r'\s+')
        for x in asr_truth:
            re.sub("\s+","",x)
            re.sub("\n|\t","",x)
            re.sub(' ',"",x)
        return asr_truth      

    @staticmethod    
    def save_csv_file(save_path, csv_datalist):
        """
        將 data list 存為 csv 檔.
        * Intput:
            - save_path: 存 CSV 的路徑
            - csv_datalist: data list 
                includ three data : 
                    * 音檔內容(labels)
        """
        # 以 data list 存取 csv
        name=['labels']      
        csv_data = pd.DataFrame(columns=name,data=csv_datalist)
        csv_data.to_csv(save_path, encoding='utf-8', index=False)        
        return save_path 
# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    config_path = "/home/c95hcw/ASR/config_LM_data_process.yaml"
    csv_data_processing = LmCsvDataPreporcessing(config_path)
    csv_data_processing.process_flow()