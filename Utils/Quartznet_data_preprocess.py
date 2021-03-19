#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quartznet Data Preprocess.
 * Main Function:
    - quartznet_dataset : 
        Convert csv file to Quartznet training dataset.
            * Include:
                - train.json
                - valid.json
                - quartznet15x5-Zh.yaml 

Created on Thu Mar  4 14:30:17 2021
@author: c95hcw
"""

import os
import json
from ruamel.yaml import YAML
from CSV_utils import read_mycsv_file
from Bulid_Vocabulary import bulid_vocab

class QuartznetDataPreprocess():
    def __init__(self):
        pass
    
    @staticmethod 
    def csv_to_manifest(input_data_path, manifest_path):
        """
        Convert csv dataset to Quartznet training data format.
        (csv --> json)
        """
        wave_names, asr_truth, wave_times = read_mycsv_file(input_data_path)     
        json_lines = []
        for num in range(len(wave_names)):
            json_lines.append(json.dumps({'audio_filepath':  wave_names[num],
                                          'duration': wave_times[num], 
                                          'text': asr_truth[num],},ensure_ascii=False))        
        with open(manifest_path, 'w', encoding='utf-8') as fout:
            for line in json_lines:
                fout.write(line + "\n")   
        return manifest_path


    def quartznet_dataset(self, train_data_path, valid_data_path, save_json_folder):     
        """
        Convert csv file to Quartznet training dataset.
            * Include:
                - train.json
                - valid.json
                - quartznet15x5-Zh.yaml        
        """        
        save_folder = save_json_folder + "Quartznet/"            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)        
        # train dataset
        train_path = self.csv_to_manifest(train_data_path, save_folder + "train.json")
        # valid dataset
        valid_path = self.csv_to_manifest(valid_data_path, save_folder + "valid.json")
        # vocabulary.ymal
        vocabulary, _, _ = bulid_vocab(train_data_path, save_folder, save_file = True)
        
        yaml = YAML(typ="safe")      
        yaml_path = "./Utils/quartznet_util/quartznet15x5-Zh.yaml"
        with open(yaml_path) as f:
            list_doc = yaml.load(f)        
        vocab_org = list_doc['init_params']['decoder_params']['init_params']['vocabulary']
        vocab = []
        vocab = [vocab_org[0],vocab_org[1]]
        for i in vocabulary[1:]:
            vocab.append(i)
        list_doc['init_params']['decoder_params']['init_params']['num_classes'] = len(vocab)
        list_doc['init_params']['decoder_params']['init_params']['vocabulary'] = vocab
    
        vocab_path = save_folder + "quartznet15x5-Zh.yaml"
        with open(vocab_path, "w", encoding='utf-8') as f:
            yaml.dump(list_doc, f)  
            # allow_unicode=True >>> output is utf-8  
            # yaml.dump(list_doc, f, allow_unicode=True, default_flow_style=True)       
                   
        print(f"已完成 Quartznet 的[csv-->json]處理 : {train_path}")        
        print(f"已完成 Quartznet 的[csv-->json]處理 : {valid_path}")        
        print(f"已完成 Quartznet 的[csv-->yaml]處理 : {vocab_path}")        
          
        return train_path, valid_path, vocab_path


# =============================================================================
#  Main Code
# =============================================================================
"""
 if test in this main code,
    please modify "yaml_path".
    ** yaml_path = "./quartznet_util/quartznet15x5-Zh.yaml"
"""
if __name__ == "__main__":
    
    train_data_path = "/home/c95hcw/ASR/Dataset/data_t3/Stage0/combined_train_corpus.csv"
    valid_data_path = "/home/c95hcw/ASR/Dataset/data_t3/Stage0/combined_valid_corpus.csv"
    save_json_folder = "/home/c95hcw/ASR/Dataset/data_t3/Stage1/"
    
    quartznet_data_preprocess = QuartznetDataPreprocess()
    quartznet_data_preprocess.quartznet_dataset(train_data_path, valid_data_path, save_json_folder)