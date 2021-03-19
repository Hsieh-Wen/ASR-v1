#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:06:24 2021

@author: c95hcw
"""
from Utils.CSV_utils import read_csv_file

def csv_to_txt(data_list, save_txt_path):
    """
    Convert csv dataset to masr training data format.
    (csv --> txt)
    """
    txt_file = open(save_txt_path, 'a+')#M33_M0622_train_v1.txt
    for i in range(len(data_list)):    
        txt_file.write( data_list[i] + "\n" )
    txt_file.close()

    return save_txt_path

def remove_same_sentence_in_train_valid(sent_list):
    """
    
    """
    save_labels = []
    completed_lines_hash = set()
    
    for line in sent_list:  
       if line not in completed_lines_hash:
            save_labels.append(line)
            completed_lines_hash.add(line)   
    
    return save_labels

def remove_duplicated_sentence(sent_list1, sent_list2):
    """
    
    """
    save_labels = []
    completed_lines_hash = set(sent_list1)
    
    for line in sent_list2:  
       if line not in completed_lines_hash:
            save_labels.append(line)
            completed_lines_hash.add(line)   
    
    return sent_list1, save_labels
        
# =============================================================================
#  Main Code
# =============================================================================        

data_folder = "/home/c95hcw/ASR/Dataset/LM_data/V2/raw_asr_data/"
save_data_folder = "/home/c95hcw/ASR/Dataset/LM_data/V2/"

train_path = data_folder + "train_quartznet_v1.csv"
valid_path = data_folder + "dev_quartznet_v1.csv"

_, train_asr_truth, _ = read_csv_file(train_path)       
_, valid_asr_truth, _ = read_csv_file(valid_path)        

train_LM_list, valid_LM_list = remove_duplicated_sentence(train_asr_truth, valid_asr_truth)

train_LM_list = remove_same_sentence_in_train_valid(train_LM_list)
valid_LM_list = remove_same_sentence_in_train_valid(valid_LM_list)


save_txt_train_path = save_data_folder + "train_LM.txt"
save_txt_valid_path = save_data_folder + "valid_LM.txt"

csv_to_txt(train_LM_list, save_txt_train_path)
csv_to_txt(valid_LM_list, save_txt_valid_path)