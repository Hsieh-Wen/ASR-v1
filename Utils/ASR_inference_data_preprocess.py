#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
import re
import numpy as np
import pandas as pd
from CSV_utils import read_mycsv_file
from shutil import copyfile


class ASRIDataProcess():
    def __init__(self):
        pass
      
    def asr_inference_dataset(self, input_data_path, save_inference_folder):    

        save_folder = save_inference_folder + "Inference_data/"            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)        

        # input dataset
        input_folder = "/".join(input_data_path.split("/")[0:-2])
        input_files = sorted(glob.glob(input_folder  + "/ASR/train_corpus/" + "*.csv"))

        for csv_file in input_files:
            file_name = csv_file.split("/")[-1]

            data_name = re.sub("_train","",file_name)
            npz_name = re.sub(".csv",".npz",data_name)
            npz_path = input_folder  + "/ASR/org_csvdata/" + npz_name
            copyfile(npz_path, save_folder + npz_name)

            csv_path = input_folder + "/ASR/train_corpus/" + file_name
            copyfile(csv_path, save_folder + data_name)

            print(f"已完成 ASR inference data 的處理 : {save_folder + data_name}")
        return save_folder

if __name__ == "__main__":
    
    input_data_path = "/home/c95hcw/ASR_Data/Dataset/speech_tw_test/Stage0/combined_train_corpus.csv"
    save_inference_folder = "/home/c95hcw/ASR_Data/Dataset/speech_tw_test/Stage1/"
    
    inference_data_process = ASRIDataProcess()
    inference_data_process.asr_inference_dataset(input_data_path, save_inference_folder)