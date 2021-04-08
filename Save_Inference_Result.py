#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:26:22 2021

@author: c95hcw
"""

import os
import sys
import json
import pandas as pd
import configparser
import h5py 
import numpy as np


def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf

class SaveResult():
    def __init__(self, save_folder):
        self.save_folder = save_folder        

    def save_follow(self, **_settings):
        """
        Generate json/csv file from inference result.    
            - CSV file : 
                存取 inference 結果表格為 csv檔.
                    表格包括: labels, model_predict_result, wer
            - Json file :
                存取 輸入之參數數據(**_settings)為 json檔.    
            - Npz file :
                存取 輸入之參數數據(**_settings)為 npz檔.
        *Input:
            - **_settings : 參數數據
                其中必須存的變數:
                    - Inference_Mode : 
                        inference 模式. (EX: ASR, LM, ASR1....)
                    - Input_File : 
                        要 inference 的檔名路徑. (EX: xxx.csv)
                    - Model_Path : 
                        要 inference 的模型路徑.                         
                    - Ground_Truth : 
                        inference 的正確答案[ASR inference] 或 語音辨識結果[LM inference].
                    - Predict_Result : 
                        inference 後的辨識結果.
                    - Wer_list : 
                        inference 後的辨識錯誤率.
        """
        # load vars
        all_data = _settings.copy()
        parameters_dict = _settings 

        # need vars
        if 'Inference_Mode' in parameters_dict:
            print(f"{parameters_dict['Inference_Mode']=} Exist.")
        else:
            sys.exit("Please input data of Inference_Mode")               
        if 'Input_File' in parameters_dict:
            print(f"{parameters_dict['Input_File']=} Exist.")
        else:
            sys.exit("Please input data of Input_File")            
        if 'Model_Path' in parameters_dict:
            print(f"{parameters_dict['Model_Path']=} Exist.")
        else:
            sys.exit("Please input data of Model_Path")      
            
        if 'Ground_Truth' in parameters_dict:
            ground_truth = parameters_dict['Ground_Truth']
        else:
            sys.exit("Please input data of Ground_Truth")
        if 'Predict_Result' in parameters_dict:
            predict_result = parameters_dict['Predict_Result']
        else:
            sys.exit("Please input data of Predict_Result")           
        if 'Wer_list' in parameters_dict:
            wer_list = parameters_dict['Wer_list']            
        else:
            sys.exit("Please input data of Wer_list")

        # Some Key no need to write in json 
        parameters_dict2 = parameters_dict.copy()
        for key in parameters_dict2.keys():
            if np.size(parameters_dict2[key]) > 1:
                del parameters_dict[key]
                print(f"No save key:{key} in json file.")

        # del parameters_dict['Ground_Truth']
        # del parameters_dict['Predict_Result']
        # del parameters_dict['Wer_list']
        # if 'Error_word' in parameters_dict.keys():
        #     del parameters_dict['Error_word']


        # Setting of json/csv file path           
        save_json_path = self.save_folder + "result.json" 
        
        # Write save_csv_path in dict     
        if_save_result, save_index = self.check_if_save_data(save_json_path, **parameters_dict)   
        
        save_id = "%04d"%save_index 
        save_csv_name = "result_id_" +  str(save_id) + ".csv"
        
        save_csv_path = self.save_folder + save_csv_name            

        # Setting of  npz_file path 
        npz_folder = self.save_folder + "npz_files/"
        self.check_folder_exist(npz_folder)
        
        save_npz_path =  npz_folder + "result_id_" +  str(save_id)  + ".npz"
        
        if if_save_result:
            result_table = zip(ground_truth, predict_result, wer_list)
            self.json_save(save_json_path, save_csv_name, **parameters_dict)
            self.csv_save(save_csv_path, result_table)
            print(f"已存檔 {save_csv_path=}")       
            self.npz_file_save(save_npz_path, **all_data)
            print(f"已存檔 {save_npz_path=}")       

        else:
            print(f"已經 inference 過此檔案，檔名：{save_csv_path}！！")       


    def check_if_save_data(self, save_json_path, **kwargs):
        """
        判斷是否要存取 json/csv file.
        
        * Input:
            - save_json_path : json 檔路徑
            - **kwargs : 要存的資料 (dict格式)
        *Output:
            - save_result : 是否要存檔.
            - save_index : 檔名流水號.
                        (save_result=True --> save_index=存檔檔名流水號;
                         save_result=False --> save_index=已存檔檔名流水號)       
        
        * Follows:
        1. 確認 json_path 是否存在:
            False: 存取 json file.
                    (save_result = True) 
                    --> Function End
                    
            True : Load json file. (json_path 存在) 
                    2. 將要存的資料(kwargs)設為 Set() [kwargs(dict data) --> set data] 
                    3. 比對 要存的資料(kwargs) 是否 存在 json file 中.
                    True : 不存檔. (save_result = False)
                    False : 存檔.(save_result = True)                    
                    --> Function End      
        """
        save_result = True
        # Check json file is exist
        if not os.path.isfile(save_json_path):
            save_index = 0
            
        else:
            # Load json file.            
            with open(save_json_path, 'r') as f:
                asr_result_data = json.load(f)
                
            # kwargs(dict data) to set data
            save_data_set = set()
            for k, v in kwargs.items():      
                save_data_set.add(str(k) + "-" + str(v) )    
            
            # 比對 要存的資料(kwargs) 是否 存在 json file 中.
            
            for i, key in enumerate(asr_result_data): 
                # dict data to set data
                result_data_set = set()                
                for k, v in asr_result_data[key].items():
                    result_data_set.add(str(k) + "-" + str(v))

                if len(save_data_set) == len(result_data_set):
                    intersection_data = result_data_set & save_data_set
                    if len(intersection_data) == len(save_data_set):
                        save_result = False
                        break                      
            
            if not save_result:
                save_index = i
            else:
                save_index = len(asr_result_data)
                
        return save_result, save_index

    @staticmethod     
    def check_folder_exist(folder_path):
        """
        確認檔案路徑是否存在.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create csv Folder !!!, folder path : {folder_path}")
                
    @staticmethod        
    def csv_save(save_csv_path, table):  
        """
        存取 csv 檔案.
        (include: Answer, model_predict_result, wer)
        
        * Input:
            - save_csv_path : csv 檔路徑.
            - table : data list.
                    | Index | Type  | Size |               Value               |
                    |:-----:|:-----:|:----:|:---------------------------------:|
                    |   0   | tuple |  3   | (Answer,model_predict_result,wer) |        
        """
        name=['labels','Predict','WER']
        test=pd.DataFrame(columns=name,data=table)       
        test.to_csv(save_csv_path, encoding='utf-8', index=False)    
        
    @staticmethod
    def json_save(save_json_path, save_csv_name, **kwargs):   
        """
        存取 json 檔案.        
        * Input:
            - save_json_path : csv 檔路徑.
            - save_index : 檔名流水號.
            - **kwargs : 要存的資料 (dict格式).
       """ 
        
        if not os.path.isfile(save_json_path):            
            save_result_data = dict()
        else:
            with open(save_json_path, 'r') as f:
                save_result_data = json.load(f) 
                 
        save_result_data[save_csv_name] = kwargs           
        with open(save_json_path, 'w') as fp:
            json.dump(save_result_data, fp, indent=len(kwargs))
    
    @staticmethod
    def npz_file_save(save_npz_path, **_SaveData):
        """
        存取 .npz 檔案.        
        * Input:
            - save_npz_path : 存檔路徑.
            - **_SaveData : 要存的資料 (dict格式).
       """                 
        # Write npz
        np.savez(save_npz_path, **_SaveData)



if __name__ == "__main__":
    file_name =  os.path.basename(sys.argv[0])
    path = "config_asr_inference.ini"
    config = read_config(path)
    
    # parameters of save parameter    
    save_path = config['Stage0_asr_inference'].get('SAVE_PATH') 
    asr_model_path = "Google_ASR"
   
    asr_mode = "Google"    
    input_file = "TW_test.csv"

#    inference_ASR = InferenceASRmodels(asr_mode, asr_model_path, "gpu", False)
#    ASR_wer_avg, data_list = inference_ASR.inference_data("./Dataset/waves/", "./Dataset/data/raw_data/", inference_file)
    
    asr_truth = ['首先肯定退輔會','呂玉林委員所提的','退役上校教補費的問題','退輔會馮主委李副主委','都能夠全力支持']
    asr_predict_result = ['首先肯定退輔會','10委員所提的', '退役上校教補費的問題','退輔會馮主委李付租屋', '全力支持']
    wer_list = [0.0, 0.375, 0.0, 0.3, 0.42857142857142855]

    
    save_result = SaveResult(save_path)
    
    kwargs = {'Inference_Mode':'ASR', 'Input_File': input_file, 'Model_Path':asr_model_path,
                          'ASR_wer_avg':0.1,
                          'Ground_Truth':asr_truth, 'Predict_Result':asr_predict_result, 'Wer_list':wer_list,
                          'Py_file_name':file_name}
    save_result.save_follow(**kwargs)


#    kwargs_lm = {'Inference_Mode':'LM', 'Input_File': "tt.csv", 'Model_Path':asr_model_path, 
#                 'Ground_Truth':asr_truth, 'Predict_Result':asr_predict_result, 'Wer_list':wer_list,
#                 'ASR_wer_avg':0.1,'LM_wer_avg':0.1,                 
#                 'wwm':True,'use_confusion_word':False,'reverse':True,'token_replace':False}    
#    save_result.save_follow(**kwargs_lm) 
