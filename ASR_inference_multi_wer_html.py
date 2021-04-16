#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:13:09 2021

@author: c95hcw
"""



import configparser
from  Utils.CSV_utils import read_inference_csv_file, read_mycsv_file, remove_punctuation

import Levenshtein as Lev
import numpy as np
import re
import json
import os
import sys
import pandas as pd

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf

class HtmlResult():
    def __init__(self, config):
        # Inference result path
        save_data_folder = config['Html_Parameters'].get('SAVE_DATA_FOLDER')         
        html_name = config['Html_Parameters'].get('HTML_NAME') 

        self.save_path = save_data_folder + html_name
        if not os.path.exists(save_data_folder):
            os.makedirs(save_data_folder)

        # 要比較的模型之inference結果檔參數
        compare_data_pathes = config['Html_Parameters'].get('COLUMN_DATA_PATH')
        column_names = config['Html_Parameters'].get('COLUMN_NAME') 

        self.json_path = config['Html_Parameters'].get('JSON_PATH')
 
        self._init_params(compare_data_pathes, column_names)
    
    def _init_params(self, compare_data_pathes, column_names):
        compare_data_pathes = re.sub(" ","",compare_data_pathes)
        compare_data_pathes = re.sub("\n","",compare_data_pathes)
        self.compare_data_path_list = compare_data_pathes.split("|")    
    
        column_names = re.sub(" ","",column_names)
        column_names = re.sub("\n","",column_names)
        self.column_name_list = column_names.split("|")     
        assert len(self.column_name_list) == len(self.compare_data_path_list), "數量不對！！請修改 config_asr_inference.ini 之 [COLUMN_NAME] 或 [COLUMN_DATA_PATH]" 
        
        col_name_set = set()
        for col_name in self.column_name_list:
            if col_name not in col_name_set:
                col_name_set.add(col_name)
            else:
                sys.exit("不可有相同的欄位名,請修改 config_asr_inference.ini 之 [COLUMN_NAME]!!") 
                  

    def load_data_html_flow(self):
    	# load ground truth
        _, ground_truth,_ = list(read_mycsv_file(self.compare_data_path_list[0]))
     	
     	# load json data
        with open(self.json_path, 'r') as f:
            load_json_data = json.load(f) 
    	# 存取 Html 欄位名 與 模型路徑為字典
        compare_model_path_dict = dict()
        for i in range(1, len(self.compare_data_path_list)):
            compare_input_csv = self.compare_data_path_list[i].split("/")[-1]
            compare_model_path_dict[self.column_name_list[i]] = load_json_data[compare_input_csv]['Model_Path']          
            
        # 讀取已 inference 過的 csv 檔，並將資料存入字典.
        data_dict = self.load_models_data_to_dict()
        # 將字典資料中的 Predict 與 ground_truth 比對後，標記 Predict 中的錯誤字並存入字典資料中.
        html_dict = self.eval_data_to_html_dict(ground_truth, data_dict)
        # 字典資料以 Html 格式輸出.        
        self.dict_to_html(self.save_path , ground_truth, compare_model_path_dict, html_dict)        



    @staticmethod
    def mappingtxtans(txt_ans,speech_results):
        """
        以 Levenshtein 找出 辨識錯誤文字 的 index.
        * Input :
            txt_ans : 語音音檔之正確答案.
            speech_results : 語音模型預測結果 或 語言模型校正結果.
        * Output :
            index : speech_results 中所有 辨識錯誤文字 的 index.
        """
        # print("ans: ",txt_ans)
        index = []
        if type(speech_results) is str:
            # print("sr:",speech_results)
            e = Lev.editops(txt_ans, speech_results)
            e = list(filter(lambda x: x[0] != 'delete', e))
            for item in e:
                index.append(item[2])        
        if index == []:
            return "no different"
            print("no different")
        else:
            return index

    def maker_error_word(self, truth, predict):
        """
        將錯誤的文字以黃色標記.
        * Input :
            truth : 語音音檔之正確答案.
            predict : 語音模型預測結果 或 語言模型校正結果.
        * Output :
            maker_predict : 以標好顏色後的 Predict.
        """
        print(predict)
        red_index = self.mappingtxtans(truth,predict)
        
        if red_index != "no different":
            org_predict = list(predict)
            for idx in red_index:
                org_predict[idx] = '<font style="background-color:yellow;">' + org_predict[idx] + '</font>'        
            maker_predict = "".join(org_predict)   
        else:
            maker_predict = str(predict)
        return maker_predict

    def load_models_data_to_dict(self):
        """
        讀取已 inference 過的 csv 檔，並將資料存入字典.
        存入字典之資料: 
            - label_list : csv 檔之 labels 
                        (為 語音音檔之正確答案[ASR inference] 或 語音音檔之辨識結果[LM inference]) 
            - predict_list : csv 檔之 Predict  
                        (為 語音模型之預測結果[ASR inference] 或 語言模型之校正結果[LM inference])
            - wer_list :  csv 檔之 WER
                        (為 語音模型之預測結果的錯誤率[ASR inference] 或 語言模型之校正結果的錯誤率[LM inference])
            - wer_avg : wer_list 的 平均錯誤率
        """
        model_data_dict = dict()
        for i in range(1,len(self.compare_data_path_list)):
            # 讀取已 inference 過的 csv 檔
            label_list, predict_list, wer_list = read_inference_csv_file(self.compare_data_path_list[i])
            label_list = remove_punctuation(label_list)
            wer_avg = np.sum(wer_list)/len(wer_list)

            eng_idx, taiwan_idx, self.eng_num, self.taiwan_num, self.chinese_num = self.load_lang_data(self.compare_data_path_list[0])

            ch_ti_wer = wer_list.copy()
            if eng_idx != []:
                for e_idx in eng_idx:
                    ch_ti_wer.pop(e_idx)
                ch_ti_wer_avg = np.sum(ch_ti_wer)/len(ch_ti_wer)              
            else:
                ch_ti_wer_avg = wer_avg               

            ch_wer = wer_list.copy()
            if taiwan_idx != []:
                for t_idx in taiwan_idx:
                    ch_wer.pop(t_idx)
                ch_wer_avg = np.sum(ch_wer)/len(ch_wer)              
            else:
                ch_wer_avg = wer_avg
                
            
            # wer只顯示小數後兩位
            n_wer_list = []
            for wer in wer_list:
                n_wer = format(wer,'.2f') 
                n_wer_list.append(n_wer)
            # 將資料存入字典
            model_data_dict[self.column_name_list[i]] = {'label_list': list(label_list),
                                                         'predict_list': list(predict_list), 
                                                         'wer_list': n_wer_list,
                                                         'wer_avg': wer_avg,
                                                         'Chinese_Taiwan_wer_avg': ch_ti_wer_avg, 
                                                         'Chinese_wer_avg': ch_wer_avg}  
        return model_data_dict
    
    
    def eval_data_to_html_dict(self, ground_truth, model_data_dict):  
        """
        將經過 ground_truth 比對後，已標記 Predict 中的錯誤字存入字典資料中.
        """
        html_data_dict = model_data_dict.copy()
        for compare_data_num in range(1,len(self.compare_data_path_list)):
            #  print(f"{compare_data_num=}")
            assert len(ground_truth) == len(model_data_dict[self.column_name_list[compare_data_num]]['label_list']), "比較之資料集不同！！請確認" + str(self.compare_data_path_list[compare_data_num]) +  "是否正確！！"               
            for num in range(len(ground_truth)):
                predict = html_data_dict[self.column_name_list[compare_data_num]]['predict_list'][num]
                truth = ground_truth[num]        
                m_predict = self.maker_error_word(truth,predict)
                html_data_dict[self.column_name_list[compare_data_num]]['predict_list'][num] = m_predict
            
        return html_data_dict
    
    def load_lang_data(self, input_csv):
        # Open the CSV file for reading
        input_name = input_csv.split("/")[-1]
        npz_folder = "/".join(input_csv.split("/")[0:-2])
        npz_name = re.sub(".csv",".npz",input_name)
        npz_path = npz_folder + "/Inference_data/" + npz_name # + "/npz_files/"
        npz_data = np.load(npz_path)

        language_data = npz_data['Other_Lang'].tolist()


        # csv_data = pd.read_csv(open(input_csv), sep=r",|\t")
        # language_data = csv_data.Other_Lang

        # Get english index
        eng_idx = []
        taiwan_idx = []
        for i,lang in  enumerate(language_data):
            if lang == "English":
                eng_idx.append(i)
            if lang == "Taiwanese" or lang == "English":
                taiwan_idx.append(i)

        if eng_idx != []:
            eng_num = len(eng_idx)
        else:
            eng_num = 0

        if taiwan_idx != []:
            taiwan_num = len(taiwan_idx)-len(eng_idx) 
        else:
            taiwan_num = 0

        chinese_num = len(language_data) - taiwan_num - eng_num

        return eng_idx, taiwan_idx, eng_num, taiwan_num, chinese_num            

    # @staticmethod
    def dict_to_html(self, save_path, ground_truth, compare_model_path_dict, data_dict):      
        """
        字典資料以 Html 格式輸出.
        * Input:
            - save_path: 最後輸出的 Html 檔名.
            - ground_truth: 被比較的文字(list)
                (為 語音音檔之正確答案[ASR inference] 或 語音音檔之辨識結果[LM inference])
            - compare_model_path_dict : 要比較的所有模型之路徑(dict)
            - data_dict : 要以 Html 顯示的字典資料.
                包括:
                    * predict_list : (list) 語音模型之預測結果[ASR inference] 或 語言模型之校正結果[LM inference]
                    * wer_list : (list) 語音模型之預測結果的錯誤率[ASR inference] 或 語言模型之校正結果的錯誤率[LM inference] 
                    * wer_avg : wer_list 的 平均值
        """
        # Create the HTML file       
        f_html = open(save_path,"w")
        f_html.write("<html>")
        f_html.write("<header>")
        f_html.write('<title>inference</title>')
        f_html.write("</header>")
        f_html.write("<body>")
    
        f_html.write('<tr>')
        f_html.write('<br>') # 換行
        f_html.write('<font size="+3">')
        f_html.write('<td>' + " Compare " + str(len(compare_model_path_dict)) + " models\n" + '</td>')
        f_html.write('</br>')
        f_html.write('</font>')
        f_html.write('</tr>')    
    
        f_html.write('<tr>')
        f_html.write('<br>') # 換行
        f_html.write('<font size="+2">')
        # f_html.write('<td>' + " Inference file path: " + self.compare_data_path_list[0] + "\n\n" + '</td>')
        # f_html.write('</br>')
        # f_html.write('<br>') 
        # f_html.write('<font size="+1">')
        language_data = "[總句數: " + str(len(ground_truth)) + ", 中文句數: " + str(self.chinese_num) + ", 台語句數: " + str(self.taiwan_num) +", 英文句數: " + str(self.eng_num) + "]\n"
        f_html.write('<td>' + language_data + '</td>')
        f_html.write('</br>')
        f_html.write('</font>')
        f_html.write('</tr>')

        f_html.write('<tr>')
        for i in range(len(compare_model_path_dict)):
            input_csv_name = list(compare_model_path_dict.keys())
            chi_TW_avg_wer = format(data_dict[input_csv_name[i]]['Chinese_Taiwan_wer_avg'],'.2f')
            TW_avg_wer = format(data_dict[input_csv_name[i]]['Chinese_wer_avg'],'.2f')
            f_html.write('<br>')
            f_html.write('<font size="+1">')
            f_html.write('<td>' + input_csv_name[i] + "= " + compare_model_path_dict[input_csv_name[i]] + " [中文 平均錯誤率: " + TW_avg_wer + ", 中文+台語 平均錯誤率: "+  chi_TW_avg_wer +"]"+ "\n" + '</td>')
            f_html.write('</font>')
        f_html.write('</br>')
        f_html.write('</tr>')
        
        f_html.write('<table border=1>')
        f_html.write('<tr>')
        
        f_html.write('<td align="center">id</td>')
        f_html.write('<td align="center">labels</td>')    
        for i in range(len(compare_model_path_dict)):
            avg_wer = format(data_dict[input_csv_name[i]]['wer_avg'],'.2f')
            f_html.write('<td align="center">' + input_csv_name[i] + " (平均 WER = " + avg_wer + ")" + '</td>')
    
        for i2 in range(len(compare_model_path_dict)):
            f_html.write('<td align="center">' + "wer_" + input_csv_name[i2] + '</td>')
            
        f_html.write('</tr>')            
    
        for num2 in range(len(ground_truth)):         
            f_html.write('<tr>')
            f_html.write('<td align="center">' + str(num2) + '</td>')
            f_html.write('<td>' + ground_truth[num2]+ '</td>')
            
            for idx in range(len(compare_model_path_dict)): 
                f_html.write('<td>' + data_dict[input_csv_name[idx]]['predict_list'][num2] + '</td>')
            
            for idx2 in range(len(compare_model_path_dict)): 
                f_html.write('<td align="center">' + data_dict[input_csv_name[idx2]]['wer_list'][num2] + '</td>')          
            
        f_html.write('</tr>')        
        f_html.write('</table>')
        f_html.write('</body>')
        f_html.write('</html>')


# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    path = "config_html.ini"
    config = read_config(path)
   
    html_result = HtmlResult(config)
    html_result.load_data_html_flow()


#    compare_model_path_list = []
#    for compare_data_path in compare_data_path_list[1:]:
#        compare_input_csv = compare_data_path.split("/")[-1]
#        
#        inference_input = load_json_data[compare_input_csv]['Input_File']
#        inference_input_name = inference_input.split("/")[-1]
#        
#        if "result_id_" in inference_input_name:
#            compare_model_path_list.append(load_json_data[inference_input_name]['Input_File'])          
#        else:
#            compare_model_path_list.append(load_json_data[compare_input_csv]['Input_File'] )
