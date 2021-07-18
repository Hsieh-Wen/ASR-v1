#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:30:59 2021

@author: c95hcw
"""

from load_config_args import load_config



#args = load_config(config_path="/home/c95hcw/ASR/config_ASR_inference.yaml")
#print(f"{args=}")


#args = load_config(config_path="/home/c95hcw/ASR/config_LM_inference.yaml")
#print(f"{args=}")
#LM_parameters = args.use_confusion_words


#config_path = "/home/c95hcw/ASR/config_ASR_data_process.yaml"
#args = load_config(config_path)
#stage0_args = args.Stage0
#print(f"{stage0_args=}")
#input_files = stage0_args['INPUT_FILES']


config_path = "/home/c95hcw/ASR/config_LM_data_process.yaml"
args = load_config(config_path)
stage0_args = args.Stage0
print(f"{stage0_args=}")
csv_name = stage0_args['INPUT_FILES']['Data_process1']['INTPUT_CSV_PATH']
csv_column_label = stage0_args['INPUT_FILES']['Data_process1']['CSV_COLUMN_LABEL']

from CSV_utils import read_csv_file
asr_truth = read_csv_file(csv_name,"Wave_path", "Labels")