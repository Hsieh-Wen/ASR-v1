#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Data Preprocess.
 * 2 Main Functions:
    - bert_dataset : 
        Convert csv file to BERT training dataset.   
        * Input:
            - train_data_path
            - valid_data_path
            - save_txt_folder
        * Output:
            - train.txt
            - valid.txt
    - bert_wwm_dataset :
        Convert csv file to BERT training dataset.   
        * Input:
            - train_data_path
            - valid_data_path
            - special_word_file_path
            - save_folder_path
        * Output:
            - train.txt
            - train_ref.txt
            - valid.txt
            - valid_ref.txt  

Created on Thu Mar  4 14:56:28 2021
@author: c95hcw            
"""

import os
import sys
import json
import shutil 
from CSV_utils import read_lm_csv_file


from typing import List
from transformers import BertTokenizer
from ckiptagger import data_utils, WS
from ckiptagger import construct_dictionary
from LM_BERT.run_chinese_ref_with_chkip import add_word_to_dictionary, prepare_ref

class BertDataPreprocess():
    def __init__(self):
        pass

    @staticmethod    
    def label_to_txt(input_data_path, save_txt_path):
        """
        Extract labels from csv file,
            and remove same label.
        Then, output data to txt file.
        * Input:
            - input_data_path :  the path of csv file.(input file)
            - save_txt_path : the path of txt file.(save file)
        """
        asr_truth = read_lm_csv_file(input_data_path)

        labels_set = set()
        for text in asr_truth:
            if text not in labels_set:
                labels_set.add(text)
        txt_file = open(save_txt_path, 'a+')
        for label in labels_set:    
            txt_file.write(label + "\n" )
        txt_file.close()

        return save_txt_path


    def bert_dataset(self, train_data_path, valid_data_path, save_folder_path):
        """
        Convert csv file to BERT training dataset.   
        * Input:
            - train_data_path
            - valid_data_path
            - save_txt_folder
        * Output:
            - train.txt
            - valid.txt
        """
        save_folder = save_folder_path + "BERT/"            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)        
        else:
            sys.exit("資料夾已存在！！已有 BERT 資料夾") 
        # train dataset
        train_path = self.label_to_txt(train_data_path, save_folder + "train.txt")
        # valid dataset
        valid_path = self.label_to_txt(valid_data_path, save_folder + "valid.txt")

        print(f"已完成 BERT 的[csv.label-->txt]處理 : {train_path}")
        print(f"已完成 BERT 的[csv.label-->txt]處理 : {valid_path}")
        return train_path, valid_path

    @staticmethod
    def data_to_refid(input_data_path, 
                        special_data,ws_tokenizer,
                        bert_tokenizer, save_txt_path):
        with open(input_data_path, "r", encoding="utf-8") as f:
            input_data = f.readlines()
            input_data = [line.strip() for line in input_data if len(line) > 0 and not line.isspace()]

        ref_ids = prepare_ref(special_data, input_data, ws_tokenizer, bert_tokenizer)
        with open(save_txt_path, "w", encoding="utf-8") as fp:
            data = [json.dumps(ref) + "\n" for ref in ref_ids]
            fp.writelines(data)
        
        return save_txt_path


    def bert_wwm_dataset(self,
                         train_data_path, valid_data_path, 
                         special_word_file_path, pretrained_bert, save_folder_path):
        """
        Convert csv file to BERT training dataset.   
        * Input:
            - train_data_path
            - valid_data_path
            - special_word_file_path
            - save_folder_path
        * Output:
            - train.txt
            - train_ref.txt
            - valid.txt
            - valid_ref.txt  
        """
        save_folder = save_folder_path + "BERT_wwm/"            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)        
        else:
            sys.exit("資料夾已存在！！已有 BERT_wwm 資料夾") 
        # train dataset
        train_path = self.label_to_txt(train_data_path, save_folder + "train.txt")
        # valid dataset
        valid_path = self.label_to_txt(valid_data_path, save_folder + "valid.txt")

        # Load txt file(special_word_file)
        special_word_dictionary = add_word_to_dictionary(special_word_file_path)

        ckip_tagger_path = "./Utils/LM_BERT/Ckip_Tagger/data/"
        ws_tokenizer = WS(ckip_tagger_path)
    
        # pretrained_bert = "hfl/chinese-bert-wwm"
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert)

        # train_ref
        save_train_ref_path = save_folder + "train_ref.txt"
        train_ref_path = self.data_to_refid(train_path, special_word_dictionary,
                                         ws_tokenizer, bert_tokenizer, save_train_ref_path)

        # valid_ref
        save_valid_ref_path = save_folder + "valid_ref.txt"
        valid_ref_path = self.data_to_refid(valid_path, special_word_dictionary,
                                         ws_tokenizer, bert_tokenizer, save_valid_ref_path)

        # Copy special_data_file to Stage1 folder.
        shutil.copy2(special_word_file_path, save_folder) 

        print(f"已完成 BERT_wwm 的[csv.label-->txt]處理 : {train_path}")
        print(f"已完成 BERT_wwm 的[csv.label-->txt]處理 : {valid_path}")
        print(f"已完成 BERT_wwm 的[train_ref.txt]處理 : {train_ref_path}")
        print(f"已完成 BERT_wwm 的[valid_ref.txt]處理 : {valid_ref_path}")

# =============================================================================
#  Main Code
# =============================================================================
"""
 if test in this main code,
    please modify "ckip_tagger_path".
    ** ckip_tagger_path = "./LM_BERT/Ckip_Tagger/data/"
"""

if __name__ == "__main__":
    train_data_path = "/home/c95hcw/ASR/Dataset/data_t1/Stage0/combined_train_corpus.csv"
    valid_data_path = "/home/c95hcw/ASR/Dataset/data_t1/Stage0/combined_valid_corpus.csv"
    save_txt_folder = "/home/c95hcw/ASR/Dataset/data_t1/Stage1/"

    bert_data_preprocess = BertDataPreprocess()
    # train_path, valid_path = bert_data_preprocess.bert_dataset(train_data_path, valid_data_path, save_txt_folder)

    special_word_file_path = "./LM_BERT/special_word.txt"
    pretrained_bert = "hfl/chinese-bert-wwm"
    bert_data_preprocess.bert_wwm_dataset(train_data_path, valid_data_path,
                                            special_word_file_path, pretrained_bert, save_txt_folder)
