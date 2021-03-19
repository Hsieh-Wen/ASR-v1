#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:41:10 2021

@author: c95hcw
"""

import h5py  #匯入工具包
import numpy as np

asr_truth = ['首先肯定退輔會','呂玉林委員所提的','退役上校教補費的問題','退輔會馮主委李副主委','都能夠全力支持']
asr_predict_result = ['首先肯定退輔會','10委員所提的', '退役上校教補費的問題','退輔會馮主委李付租屋', '全力支持']
wer_list = [0.0, 0.375, 0.0, 0.3, 0.42857142857142855]

kwargs = {'Inference_Mode':'ASR', 'Input_File': "xxx.csv", 'Model_Path':"./xxx/xxx/xxx.pth",
                      'ASR_wer_avg':0.1,
                      'Ground_Truth':asr_truth, 'Predict_Result':asr_predict_result, 'Wer_list':wer_list,
                      'Py_file_name':"xxx.py", 'Test': '今天天氣好'}


# Write HDF5
f = h5py.File('HDF5_FILE.h5','w')
for key, value in kwargs.items():

    try:
        # write data to h5 file
        f[key] = value
        
        # h5py 只接受 ascii 的 data
        # 將 value 編碼 (ex:UTF-8)，轉為 Type bytes 格式
    except TypeError:
        if value == list or np.ndarray:
            v_list = []
            for v in value:
                new_v = v.encode() # 轉換編碼
                v_list.append(new_v)   
            f[key] = v_list
        else:
            new_v = v.encode()
            f[key] = new_v  
            
f.close()


# Load HDF5
fp = h5py.File('HDF5_FILE.h5','r')   #開啟h5檔案

group_key = list(fp.keys())

load_data_dict = dict()
# Get the data
for k in group_key:
#    print(f"{k=}")
    load_data_dict[k] = fp[k].value
fp.close()


# Load data from dict
wer_data = list(load_data_dict['Wer_list'])

# decoder data with UTF-8
ground_data  = []
for d in load_data_dict['Ground_Truth']:
    ground_data.append(d.decode('utf-8'))
    
predict_data = []
for d1 in load_data_dict['Predict_Result']:
    predict_data.append(d1.decode('utf-8'))