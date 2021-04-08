#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:40:07 2021

@author: c95hcw
"""
import glob
import pandas as pd
import re
import numpy as np

def read_csv_file(csv_path):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path), sep=r",|\t")
    # wave_names = data.wave_name       
    # asr_truth = data.labels   
    wave_names = data.Wave_path       
    asr_truth = data.Labels
    return wave_names, asr_truth

def read_lm_csv_file(csv_path):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path))
    asr_truth = data.labels   

    return asr_truth

def read_mycsv_file(csv_path):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path))
    wave_names = data.wave_name       
    asr_truth = data.labels   
    wave_times = data.wave_times
    return wave_names, asr_truth, wave_times

def read_inference_csv_file(csv_path):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path))
    asr_truth = data.labels       
    predict_list = data.Predict   
    wer_list = data.WER
    return asr_truth, predict_list, wer_list

def save_dataframe_to_npz(csv_path, save_npz_path):
    """[save original csv data to npz]

    Args:
        csv_path ([str]): input csv path.
        save_npz_path ([str]): output npz path.
    """
    dataframe = pd.read_csv(open(csv_path), sep=r",|\t")
    dict_data = dataframe.to_dict('list')
    np.savez(save_npz_path, **dict_data)
    print(f"Already save npz file. {save_npz_path}")

def save_csv_file(save_path, wave_path,labels,wav_time):
    csv_datalist = list(zip(wave_path,labels,wav_time))     
    name=['wave_name','labels','wave_times']      
    csv_data = pd.DataFrame(columns=name,data=csv_datalist)
    csv_data.to_csv(save_path, encoding='utf-8')        
    return save_path      

def csv_data_expand(expand_value, csv_path):
    wave_names,asr_truth,wave_times = read_csv_file(csv_path)
    
    wavs = list(wave_names)*expand_value
    labels = list(asr_truth)*expand_value
    wav_times = list(wave_times)*expand_value
    
    folders = csv_path.split("/")
    csv_folder = "/".join(folders[0:-1])
    csv_name = folders[-1]
    save_csv_name = re.sub(".csv", "_expand"+str(expand_value)+".csv",  csv_name)
    save_path = csv_folder + "/" + save_csv_name
    save_csv_file(save_path, wavs,labels,wav_times)    
    print(f"{save_path} : 已完成{csv_path}的{expand_value}倍處理")
 

def combine_csv_files(input_csv_folder, data_mode, output_folder):
    filenames = sorted(glob.glob(input_csv_folder + data_mode + "/" + "*.csv"))
    print(filenames)
    if len(filenames) > 1 :
        combined_csv = pd.concat( [ pd.read_csv(f) for f in filenames ] )   
    elif len(filenames) == 1:
        combined_csv = pd.read_csv(filenames[0])
        print("Only one csv to combine !!")
    else:
        print("No csv to combine !!")
        combined_csv = pd.DataFrame(0, index=np.arange(10), columns=['wave_name','labels','wave_times'])
    combined_csv_name = output_folder + "combined_" + data_mode + ".csv"
    combined_csv.to_csv(combined_csv_name, index=False )
    print(f"{combined_csv_name} = 合併{data_mode}資料夾內的csv檔案")
    
def delete_blank(label_list):
    # pattern = re.compile(r'\s+')
    for x in label_list:
        re.sub("\s+","",x)
        re.sub("\n|\t","",x)
        re.sub(' ',"",x)
    return label_list


def remove_punctuation(asr_truth):
    """
    去除 csv 資料(asr_truth)中的標點符號.
    """
    puncs = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fa5]') 
    new_labels = []
    for text in asr_truth:
        text = puncs.sub("",text)
        if "。" in text:
            text = re.sub("。","",text)
        new_labels.append(text)
    return new_labels     
# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    
    csv_path = "/home/c95hcw/ASR/Dataset/raw_data/csvs/speech_Training_dataset.csv"
    wave_names, asr_truth = read_csv_file(csv_path)



#     data_expand = False
#     data_combined = True
    
#     if data_expand:
#         # Example of data expand    
#         csv_path = "/home/c95hcw/ASR/Dataset/raw_data/csvs/speech_Training_dataset.csv"
#         expand_value = 10
#         csv_data_expand(expand_value, csv_path)
    
#     if data_combined:
#         # Example of data combination
#         csv_folder = "/home/c95hcw/ASR/Dataset/data/prepare_data/"
        
#         data_mode1 = "train"
#         combine_csv_files(csv_folder, data_mode1)
        
#         data_mode2 = "valid"
#         combine_csv_files(csv_folder, data_mode2)    
    
# #    csv_folder = "/home/c95hcw/ASR/Dataset/data/prepare_data/"
# #    combine_csv_files(csv_folder, "train")    
