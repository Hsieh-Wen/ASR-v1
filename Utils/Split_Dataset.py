#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:14:08 2021

@author: c95hcw
"""

from CSV_utils import read_csv_file
from Bulid_Vocabulary import bulid_vocab

import pandas as pd
import numpy as np
import random
from math import floor
import re

def save_csv_file(save_path, csv_datalist):
    # 以 data list 存取 csv
    name=['wave_name','labels','wave_times']      
    csv_data = pd.DataFrame(columns=name,data=csv_datalist)
    csv_data.to_csv(save_path, encoding='utf-8')        
    return save_path  

def get_training_and_testing_sets(data_path, split_ratio, vocab_to_split=False):
    """
    將 csv file 分割成 train.csv 與 valid.csv，
    * Input :
        - data_path: 要分割的 csv file
        - split_ratio: 分割比例(default= train:valid = 8:2)
        - vocab_to_split: 選擇分割方法
    
            兩種分割方法:
                * vocab_to_split = True
                    根據字頻(每個字在資料集內出現的頻率)，
                    字頻較多之句子，並以 split_ratio 的比例作為 validation data.
                * vocab_to_split = False
                    將資料集做隨機排序，
                    再以 split_ratio 的比例分割 train 和 validation data.  
    * Output :
        -  train_data_path: train.csv 的存檔路徑
        -  valid_data_path: valid.csv 的存檔路徑
        
        (default save folder  = csv file 的資料夾內)
        (default save name = *_train.csv or *_valid.csv)
    
    """

    folders = data_path.split("/")
    data_folder = "/".join(folders[0:-1])
    data_name = folders[-1]

    wave_names, asr_truth, wave_times = read_csv_file(data_path)
    file_list = list(zip(wave_names, asr_truth, wave_times))
      
    if vocab_to_split == True:
        save_file = False
        vocabulary, sorted_vob, vocab_json_path = bulid_vocab(data_path,data_folder, save_file)
        
        vocab = dict((x,y) for x,y in sorted_vob)
        counter_all = []
        for i in range(len(asr_truth)):
            idx_vec = list(map(lambda j: vocab[j],asr_truth[i]))    
            counter_all.append(int(np.sum(idx_vec)))
        
        all_data = list(zip(wave_names,asr_truth,wave_times,counter_all))
        vocab_sorted_list = sorted(all_data, key=lambda x: x[3])  
        file_list = []
        for item in vocab_sorted_list:
            file_list.append(item[0:3])
    
    else:
        random.seed(3)
        random.shuffle(file_list)
        wave_path = []
        labels = []
        wav_time = []        
        for t_data in file_list:
            wave_path.append(t_data[0])
            labels.append(t_data[1])
            wav_time.append(t_data[2])
        file_list = list(zip(wave_path,labels,wav_time))
        
    split_index = floor(len(wave_names) * split_ratio)
    training_list = file_list[:split_index]
    validation_list = file_list[split_index:]

    train_data_name = re.sub(".csv","_train.csv",data_name)
    valid_data_name = re.sub(".csv","_valid.csv",data_name)
  
    train_data_path = save_csv_file(data_folder + "/" + train_data_name, training_list)
    valid_data_path = save_csv_file(data_folder + "/" + valid_data_name, validation_list)

    print("Split dataset !!")
    print(f"Orginal data : {data_path}")
    print(f"Training data path : {train_data_path}")  
    print(f"Validation data path : {valid_data_path}")  
    
    return train_data_path, valid_data_path      

# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    split_ratio = 0.9
    vocab_to_split = True
    data_path = "/home/c95hcw/ASR/Dataset/data/raw_data/aishell_fan_time.csv" 
      
    train_org_data, valid_org_data = get_training_and_testing_sets(data_path, split_ratio, vocab_to_split)