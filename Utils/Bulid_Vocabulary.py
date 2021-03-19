#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:55:05 2021

@author: c95hcw
"""
import re
from torchtext import data
import json

def bulid_vocab(csvfile_path, save_vcab_folder, save_file):
    """
    建立csv file 之 字典，
    * Input:
        - csvfile_path: 要建字典的 csv file 路徑
        - save_vcab_folder: 要存取字典檔的資料夾路徑
        - save_file: 是否將字典存檔(list 存為 vocab.txt )
    """
    # step 1: using csvfile list as Field tokenize function for train dataset
    LABEL = data.Field(sequential=True, tokenize=list)
    fields = {"labels":('labels',LABEL)}
    train = data.TabularDataset(path=csvfile_path,format='csv',fields=fields)
    # for i in range(0, len(train)):
        # print(vars(train[i]))
        
    # step 2: build vocabulary
    MER = data.Field(sequential=True, tokenize=list)
    MER.build_vocab(train.labels)
    MER.vocab.itos[1] = "_"
    vocabulary= MER.vocab.itos[1:] 
    
    # step 3: extract frequency part and sorting by counter
    sorted_vob = list(reversed(sorted(MER.vocab.freqs.items(), key=lambda kv: kv[1])))
    
    if save_file:
        # step 4: output text file and json file
        csvfile_name = csvfile_path.split("/")[-1]
        save_csv_txt_name = re.sub(".csv","_vocab.txt",csvfile_name)
        save_csv_json_name = re.sub(".csv","_vocab.json",csvfile_name)
        
        # save vocab.txt 
        vocab_txt_path = save_vcab_folder + save_csv_txt_name
        txt_file = open(vocab_txt_path,'w')
        txt_file.write(str(vocabulary))
        txt_file.close()
    
        # save vocab.json 
        vocab_json_path = save_vcab_folder + save_csv_json_name
        json_file = open(vocab_json_path,'w')
        json.dump(vocabulary,json_file,ensure_ascii=False)
        json_file.close()
    else:
        vocab_json_path = None
    
    return vocabulary, sorted_vob, vocab_json_path

# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":

    csvfile_path = "/home/c95hcw/ASR/Dataset/raw_data/csvs/combined_AI_morn_mozilla.csv"
    save_vcab_folder = "/home/c95hcw/ASR/Dataset/data/raw_data/vocab/"
    save_file=True
    
    vocabulary, sorted_vob, vocab_json_path = bulid_vocab(csvfile_path,save_vcab_folder, save_file)
