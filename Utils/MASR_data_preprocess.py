#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASR Data Preprocess.
 * Main Function:
    - masr_dataset : 

Created on Thu Mar  4 14:08:18 2021
@author: c95hcw
"""
import os
import sys
from CSV_utils import read_csv_file, read_mycsv_file
from Bulid_Vocabulary import bulid_vocab

class MasrDataPreprocess():
    def __init__(self):
        pass
    
    @staticmethod    
    def csv_to_txt(input_data_path, save_txt_path):
        """
        Convert csv dataset to masr training data format.
        (csv --> txt)
        """
        # wave_names, asr_truth = read_csv_file(input_data_path)
        wave_names, asr_truth, wave_times = read_mycsv_file(input_data_path)
        txt_file = open(save_txt_path, 'a+')#M33_M0622_train_v1.txt
        for i in range(len(asr_truth)):    
            txt_file.write(wave_names[i] + "," + asr_truth[i] +"," + str(wave_times[i]) + "\n" )
            # txt_file.write(wave_names[i] + "," + asr_truth[i] + "\n" )
        txt_file.close()
#        print(f"已完成 MASR 的[csv-->txt]處理 : {save_txt_path}")
        return save_txt_path
    
    
    
    def masr_dataset(self, train_data_path, valid_data_path, save_txt_folder):    
        """
        Convert csv file to MASR training dataset.
            * Include:
                - train.txt
                - valid.txt
                - vocab.json        
        """
        save_folder = save_txt_folder + "MASR/"            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)        
        else:
            sys.exit("資料夾已存在！！已有 MASR/ 資料夾") 
        # train dataset
        train_path = self.csv_to_txt(train_data_path, save_folder + "train.txt")
        # valid dataset
        valid_path = self.csv_to_txt(valid_data_path, save_folder + "valid.txt")
        # vocabulary.json
        _, _, vocab_path = bulid_vocab(train_data_path, save_folder, save_file = True)
        print(f"已完成 MASR 的[csv-->txt]處理 : {train_path}")
        print(f"已完成 MASR 的[csv-->txt]處理 : {valid_path}")
        print(f"已完成 MASR 的[csv-->vocab.json]處理 : {vocab_path}")
        return train_path, valid_path, vocab_path

if __name__ == "__main__":
    
    train_data_path = "/home/c95hcw/ASR/Dataset/data_t3/Stage0/combined_train_corpus.csv"
    valid_data_path = "/home/c95hcw/ASR/Dataset/data_t3/Stage0/combined_valid_corpus.csv"
    save_txt_folder = "/home/c95hcw/ASR/Dataset/data_t3/Stage1/"
    
    masr_data_preprocess = MasrDataPreprocess()
    masr_data_preprocess.masr_dataset(train_data_path, valid_data_path, save_txt_folder)