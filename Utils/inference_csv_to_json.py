#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:16 2021
@author: c95hcw
"""

from  CSV_utils import read_inference_csv_file
import numpy as np
import json

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf

def read_csv_file(csv_path):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path), sep=r",|\t")
    wave_names = data.wave_name       
    asr_truth = data.labels   
    # wave_times = data.wave_times
    # return wave_names, asr_truth, wave_times
    return wave_names, asr_truth


# Main Code
input_csv = "../Inference_Result/test/result_id_0000.csv"
label_list, predict_list, wer_list = read_inference_csv_file(input_csv)

# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    input_csv = "/home/c95hcw/ASR_Data/Dataset/raw_data/waves/Training_dataset/original/All_data.csv"

    
