#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:34:08 2021

@author: c95hcw
"""
import numpy as np

asr_truth = ['首先肯定退輔會','呂玉林委員所提的','退役上校教補費的問題','退輔會馮主委李副主委','都能夠全力支持']
asr_predict_result = ['首先肯定退輔會','10委員所提的', '退役上校教補費的問題','退輔會馮主委李付租屋', '全力支持']
wer_list = [0.0, 0.375, 0.0, 0.3, 0.42857142857142855]

kwargs = {'Inference_Mode':'ASR', 'Input_File': "xxx.csv", 'Model_Path':"./xxx/xxx/xxx.pth",
                      'ASR_wer_avg':0.1,
                      'Ground_Truth':asr_truth, 'Predict_Result':asr_predict_result, 'Wer_list':wer_list,
                      'Py_file_name':"xxx.py", 'Test': '今天天氣好'}
np.savez('tt.npz', **kwargs)


npz_data = np.load('tt.npz', allow_pickle=True)
for key in npz_data:
    print(f"{key} : {npz_data[key]}")

#  Numpy 升版本之後將 “allow_pickle” 調整成了 False。
# 這導致了我們在讀取一些 pickle 封裝的 Training data 時遭受了阻礙 --> allow_pickle=True 。
    
    
#    exec("%s = %s" % (key, str(npz_data[key])))


np.save('tt2.npy', asr_truth)
npy_data = np.load('tt2.npy')
print(f"{npy_data=}")
