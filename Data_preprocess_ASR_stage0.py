#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:36:11 2021

@author: c95hcw
"""
from Utils.CSV_utils import read_csv_file, combine_csv_files, save_dataframe_to_npz
from Utils.Bulid_Vocabulary import bulid_vocab

import configparser
import json
import pandas as pd
import numpy as np
import random
import librosa
import soundfile as sf
from math import floor
import re
import os
import sys

from shutil import copyfile
from distutils.dir_util import copy_tree


def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf

class AsrCsvDataPreporcessing():
    def __init__(self, config):
        """
        Load config params.
        """
        self.seed = config['Stage0'].getint('SEED') 
        save_folder_name = config['Stage0'].get('SAVE_FOLDER_NAME')        
        data_folder = config['Stage0'].get('DATA_FOLDER')
        input_csvs = config['Stage0'].get('INTPUT_CSV_PATH') 

        self.duration_threshold = config['Stage0'].getfloat('WAVE_TIME_THRESHOLD') 
        self.sample_rate = config['Stage0'].getint('SAMPLERATE')  
        self.save_new_wav = config['Stage0'].getboolean('SAVE_NEW_WAVE')          
        if self.save_new_wav:
            self.wav_folder_new = save_folder_name + "waves/"
            if not os.path.exists(self.wav_folder_new):
                os.makedirs(self.wav_folder_new)

        split_ratios = config['Stage0'].get('SPLIT_RATIO')
        split_modes = config['Stage0'].get('SPLIT_MODE') 
        data_expands = config['Stage0'].get('DATA_EXPAND_value') 

        if os.path.exists(save_folder_name + "Stage0/ASR/"):
            sys.exit("資料夾已存在！！請修改 config_data_process.ini 之 [SAVE_FOLDER_NAME]") 
    
        # init params
        self._init_params(save_folder_name, input_csvs, data_folder, split_ratios, split_modes, data_expands)
        
    def _init_params(self, save_folder_name, input_csvs, data_folder, split_ratios, split_modes, data_expands):
        """
        config data(str to list).
        """

        # parameters of saving folder name    # "data_v1/"
        self.save_data_folder = save_folder_name + "Stage0/"
        
        # parameters of input path
        input_csvs = re.sub(" ","",input_csvs)
        input_csvs = re.sub("\n","",input_csvs)
        if "|" in input_csvs:
            self.csv_name_list = input_csvs.split("|")     
        else:
            self.csv_name_list = []
            self.csv_name_list.append(input_csvs)
        
        csv_num = len(self.csv_name_list)
        print(csv_num)
        self.csv_folder = data_folder + "csvs/"
        self.wav_folder = data_folder + "waves/" 
    
        # parameters of split csv data
        split_ratios = re.sub(" ","",split_ratios)
        split_ratios = re.sub("\n","",split_ratios)
        if "|" in split_ratios:
            self.split_ratio_list = split_ratios.split("|")     
        else:
            self.split_ratio_list = []
            self.split_ratio_list.append(split_ratios)
        assert csv_num == len(self.split_ratio_list),"split_ratio_list 數量不對！！請修改 config_data_process.ini 之 [SPLIT_RATIO]"
    
        split_modes = re.sub(" ","",split_modes)
        split_modes = re.sub("\n","",split_modes)
        if "|" in split_modes:
            self.split_mode_list = split_modes.split("|")     
        else:
            self.split_mode_list = []
            self.split_mode_list.append(split_modes)
        assert csv_num == len(self.split_mode_list),"split_modes 數量不對！！請修改 config_data_process.ini 之 [SPLIT_MODE]"
                
        # parameters of if expand data (default=10)
        data_expands = re.sub(" ","",data_expands)
        data_expands = re.sub("\n","",data_expands)
        if "|" in data_expands:
            self.expand_value_list = data_expands.split("|")     
        else:
            self.expand_value_list = []
            self.expand_value_list.append(data_expands)       
        assert csv_num == len(self.expand_value_list),"data_expands 數量不對！！請修改 config_data_process.ini 之 [DATA_EXPAND]"
    
    def _init_csv_cropus(self, csv_name):
        """
        Load csv data.
        * Input:
            csv_name.
        * Output:
            wave_names : wave's pathes (list). 
            asr_truth : wave's labels (list).
        """
        self.csv_name = csv_name
        self.data_path = self.csv_folder + csv_name
        wave_names, asr_truth = read_csv_file(self.csv_folder + csv_name)

        # save original csv data
        save_npz_folder = self.save_data_folder + "ASR/org_csvdata/"
        if not os.path.exists(save_npz_folder):
            os.makedirs(save_npz_folder)

        if "/" in csv_name:
            c_name = csv_name.split("/")[-1]
        else:
            c_name = csv_name
        npz_name = re.sub(".csv",".npz",c_name)
        
        save_npz_path = save_npz_folder + npz_name
        save_dataframe_to_npz(self.csv_folder + csv_name, save_npz_path)

        return wave_names, asr_truth

    def csv_data_preprocess(self, wave_names, asr_truth):
        """
        Data Preprocess.
            Include : 
             * labels (text data) :
                1. 去除label檔中的空格
                2. 去除 csv 資料(asr_truth)中的標點符號.
             * waves(audio data) : 
                1. 確認音檔是否存在或路徑是否正確.
                2. Get wave duration, and remove long wave.
                3. check samplerate (if save_new_wav = True)
                4. save new wave (if save_new_wav = True)
        """

        asr_truth = self.delete_blank(asr_truth) # 去除label檔中的空格
        asr_truth = self.remove_punctuation(asr_truth) # 去除 csv 資料(asr_truth)中的標點符號.

        # 加入音檔路徑 & new_wav_path
        wave_pathes = []
        wave_pathes_new = []
        dataset_folder = "corpus_" + str(self.corpus_num) + '_' + wave_names[0].split("/")[0]
        dataset_name =  wave_names[0].split("/")[0]
       
        for idx,wav in enumerate(wave_names):

            if self.save_new_wav:
                if not os.path.exists(self.wav_folder_new + dataset_folder):
                    os.makedirs(self.wav_folder_new + dataset_folder)

                wave_pathes.append(self.wav_folder + wav)
                new_wav_name = dataset_folder + "/"+ dataset_name + "_" + "%06d"%idx + ".wav"
                wave_pathes_new.append(self.wav_folder_new + new_wav_name)

            else: 
                for wav in wave_names:
                    wave_pathes.append(self.wav_folder + wav)
                    wave_pathes_new.append(self.wav_folder + wav)

        # 確認音檔是否存在
        wave_pathes, asr_truth = self.check_wav_isfile(wave_pathes, asr_truth)
        # Get wave duration
        wave_times = self.get_wav_duration(wave_pathes)

        # del long time wav
        wave_pathes, self.asr_truth, self.wave_times = self.remove_long_wave(self.duration_threshold, wave_pathes, asr_truth, wave_times)
        
        if self.save_new_wav:
            # check samplerate & save new wave
            self.wave_names = self.check_samplerate(wave_pathes, wave_pathes_new, self.sample_rate)
        else:
            self.wave_names = wave_pathes


    def data_process(self, csv_name, split_ratio, split_mode, expand_value):
        """
        Data Process.
            * Include:
                1. Load csv data.
                2. Get self.wave_names, self.asr_truth, self.wave_times.
                3. Split csv data to train/valid list.
                4. Save train/valid list to csv.
            * Input: 
                - csv_name
                - split_ratio
                - split_mode
                - expand_value
            * Output:
                dict() of config vars.
        """

        # Load csv data
        wave_names, asr_truth = self._init_csv_cropus(csv_name)

        # Get self.wave_names, self.asr_truth, self.wave_times
        self.csv_data_preprocess(wave_names, asr_truth)
        
        # Split csv data
        train_data, valid_data = self.get_training_and_testing_sets(split_ratio, split_mode, self.seed)

        # Save train/valid list to csv.
        self.save_train_valid_to_csvs(expand_value, train_data, valid_data, self.save_data_folder)

        return {'csv_name': csv_name,
                'save_folder_name': self.save_data_folder, 
                'save_new_wav' : self.save_new_wav,
                'wave_duration_threshold' : self.duration_threshold,
                'wave_sample_rate' : self.sample_rate,
                'split_ratio': split_ratio,
                'split_mode': split_mode,
                'seed':self.seed, 'expand_value':expand_value,
                }       
        
    def process_flow(self):
        """
        ASR data preprocess flow.
        """
        data_parameter_dict = dict()    
        for i, csv_name in enumerate(self.csv_name_list):
            print(f"\n--------------DATA PREPROCESSING--{csv_name}----------------")
    
            split_ratio = float(self.split_ratio_list[i])
            split_mode = self.split_mode_list[i]
            expand_value = float(self.expand_value_list[i])
            
            self.corpus_num = i
            # save setting data of data preprocessing
            data_parameter_dict['Corpus_' + str(i)] = self.data_process(csv_name, split_ratio, split_mode, expand_value)
            
        # 合併 train/valid corpus 內所有 csv 為 train.csv / valid.csv
        self.combine_train_valid_corpus(self.save_data_folder + "ASR/", self.save_data_folder + "ASR/")        
        
        # save data process parameter to json
        json_path = self.save_data_folder + "ASR/" + 'asr_data_processing_parameter.json'
        self.save_json(json_path, data_parameter_dict)
        

        
        
    @staticmethod
    def save_json(path, data_parameter_dict):
        with open(path, 'w') as fp:
            json.dump(data_parameter_dict, fp, indent=len(data_parameter_dict))
        print(f"save data_processing_paramete.json in path: {path}")


    def get_training_and_testing_sets(self, split_ratio, split_mode, seed=3):
        """
        將 csv file 分割成 train list 與 valid list，
        * Input :
            - split_ratio: 分割比例(default: train:valid = 8:2)
            - split_mode: 選擇分割方法
            - seed: 隨機排序的種子數
        
                兩種分割方法:
                    * split_mode = Vocab
                        根據字頻(每個字在資料集內出現的頻率)，
                        字頻較多之句子，並以 split_ratio 的比例作為 validation data.
                    * split_mode = Random
                        將資料集做隨機排序，
                        再以 split_ratio 的比例分割 train 和 validation data.  
        * Output :
            training_list: 分割好的 train list ; 若split_ratio=0.0, 則 train list ＝ [].
            validation_list:  分割好的 valid list ; 若split_ratio=1.0, 則 valid list ＝ [].       
        """


        if split_mode == "Vocab":
            
            vocabulary, sorted_vob, vocab_json_path = bulid_vocab(self.data_path, _ , save_file = False)
            
            vocab = dict((x,y) for x,y in sorted_vob)
            counter_all = []
            for i in range(len(self.asr_truth)):
                idx_vec = list(map(lambda j: vocab[j],self.asr_truth[i]))    
                counter_all.append(int(np.sum(idx_vec)))
            
            all_data = list(zip(self.wave_names,self.asr_truth,self.wave_times,counter_all))
            vocab_sorted_list = sorted(all_data, key=lambda x: x[3])  
            file_o_list = []
            for item in vocab_sorted_list:
                file_o_list.append(item[0:3])

        elif split_mode == "Random":
            file_list = list(zip(self.wave_names, self.asr_truth,self.wave_times))    
            random.seed(seed)
            random.shuffle(file_list)
            wave_path = []
            labels = []
            wav_times = []        
            for t_data in file_list:
                wave_path.append(t_data[0])
                labels.append(t_data[1])
                wav_times.append(t_data[2])
            file_o_list = list(zip(wave_path,labels,wav_times))
        elif split_mode == "Sequence":
            file_o_list = list(zip(self.wave_names, self.asr_truth,self.wave_times)) 
        else:
            sys.exit("No this split_mode !!!!!")

        split_index = floor(len(file_o_list) * split_ratio)
        training_list = file_o_list[:split_index]
        validation_list = file_o_list[split_index:]
        
        print("Split dataset !!")
        print(f"Training data = {split_ratio*100} %, Validation data = {(1-split_ratio)*100} %\n")
        print(f"Orginal data : {self.data_path}")
        
        return training_list, validation_list

    @staticmethod   
    def expand_train_data(expand_value, training_list, save_csv_name):
        """
        Expand / reduce data by expand_value.
            * expand_value > 1 : expand data.
            * expand_value = 1 : data no change.
            * expand_value < 1 : reduce data.
        """
        org_len = len(training_list)
        if expand_value == 0:
            expand_value = 1.0
            
        if expand_value >= 1:
            training_list = training_list * int(expand_value)
            rest_value = expand_value - int(expand_value)
            if rest_value != 0:
                value = int(rest_value * len(training_list))
                training_list = training_list + training_list[:value]
        else:
            value = int(expand_value * len(training_list))
            training_list = training_list[:value]                           
        after_len = len(training_list)   
        if expand_value != 1:
            train_data_name = re.sub(".csv", "_expand"+str(expand_value) + "_train.csv",save_csv_name)
            print(f"已完成 train data {expand_value}倍處理")
            print(f"org_len: {str(org_len)} --> after_len: {str(after_len)}")
        else:
            train_data_name = re.sub(".csv", "_train.csv",save_csv_name)
        return train_data_name, training_list
 
    def save_train_valid_to_csvs(self, expand_value, training_list, validation_list, save_data_folder):
        """
        將 train list 與 valid list 存為 train.csv 與 valid.csv
        * Input :
            - expand_value: 擴增資料的值
            - training_list: train list
            - validation_list: valid list
            - save_data_folder: 最後存 train/valid csv 的資料夾路徑
        * Output :
            (default save folder  = csv file 的資料夾內)
            (default save name = *_train.csv or *_valid.csv)
        """    
        save_asr_folder = save_data_folder + "ASR/"
        if not os.path.exists(save_asr_folder):
            os.makedirs(save_asr_folder)  

        if not os.path.exists(save_asr_folder + "train_corpus/"):
            os.makedirs(save_asr_folder + "train_corpus/")   
        if not os.path.exists(save_asr_folder + "valid_corpus/"):
            os.makedirs(save_asr_folder + "valid_corpus/") 
        
        if "/" in self.csv_name:
            save_csv_name = self.csv_name.split("/")[-1]  
        else:
            save_csv_name = self.csv_name 

        if training_list != []:

            train_data_name, training_list = self.expand_train_data(expand_value, training_list, save_csv_name)   
            train_data_path = self.save_csv_file(save_asr_folder + "train_corpus/" + train_data_name, training_list)
            print(f"Training data path : {train_data_path}")          
        else: 
            print(f"{self.csv_name} is not split to train.csv") 
            
        if validation_list != []:   
            valid_data_name = re.sub(".csv","_valid" + ".csv",save_csv_name)  
                
            valid_data_path = self.save_csv_file(save_asr_folder + "valid_corpus/" + valid_data_name, validation_list)
            print(f"Validation data path : {valid_data_path}")  
        else: 
            print(f"{self.csv_name} is not split to valid.csv")
            
    @staticmethod             
    def combine_train_valid_corpus(input_csv_folder, output_folder): 
        """
        Combined csv files to train.csv and valid.csv.
        (from folder-train_corpus and folder-valid_corpus)
        """
        data_modes = ["train_corpus","valid_corpus"]
        for data_mode in data_modes:
            combine_csv_files(input_csv_folder, data_mode, output_folder)            
            
    @staticmethod        
    def delete_blank(asr_truth):
        """
        去除 csv 資料(asr_truth)中的空格、空白行或tab.
        """
        # pattern = re.compile(r'\s+')
        for x in asr_truth:
            re.sub("\s+","",x)
            re.sub("\n|\t","",x)
            re.sub(' ',"",x)
        return asr_truth      

    @staticmethod        
    def remove_punctuation(asr_truth):
        """
        去除 csv 資料(asr_truth)中的標點符號.
        """
        puncs = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fa5]') 
        new_labels = []
        for text in asr_truth:
            text = puncs.sub("",text)
            new_labels.append(text)
        return new_labels  



    @staticmethod     
    def check_wav_isfile(wave_pathes, asr_truth):
        """
        確認音檔是否存在或路徑是否正確.
        """
        delete_wavs = []
        for i in range(len(wave_pathes)):
            if os.path.isfile(wave_pathes[i]) == False:
                print(f"{wave_pathes[i]} 檔案不存在!!")
                delete_wavs.append(i)
        if delete_wavs != []:
            print(f"{str(len(delete_wavs))} 個檔案不存在!!")
            for idx in delete_wavs:
                wave_pathes.pop(idx)
                asr_truth.pop(idx)
        else:
            print(f"Already check!! Waves are all exist.")
        return wave_pathes, asr_truth


    @staticmethod     
    def get_wav_duration(wave_pathes):
        """
        Get wave's duration by librosa.get_duration(wave_path).
        * Input :
            wave_pathes : waves'path (list).
        * Output : 
            wave_times : waves'duration (list).
        """
        wave_times = []
        for wav in wave_pathes:
            # wav_data, sample_rate = librosa.load(wav, sr=16000)
            wav_duration = librosa.get_duration(filename=wav) 
            wave_times.append(wav_duration)
        return wave_times

    @staticmethod 
    def remove_long_wave(duration_threshold, wave_pathes, asr_truth, wave_times):
        """
        Remove waves by duration_threshold.
        """
        org_len = len(wave_times)
        delete_id = []
        for i, wav_time in enumerate(wave_times):
            if wav_time > duration_threshold:
                delete_id.append(i)
        if delete_id != []:
            for idx in delete_id:
                wave_pathes.pop(idx)
                asr_truth.pop(idx)                
                wave_times.pop(idx)
            after_len = len(wave_times)
            print(f"Remove {str(org_len - after_len)} waves. {org_len=} --> {after_len=}")
        else:
            print(f"All waves' time small than {duration_threshold} sec.")
        return wave_pathes, asr_truth, wave_times

    @staticmethod 
    def check_samplerate(wave_pathes, wave_pathes_new, my_sample_rate):
        """
        Check samplerate to save new waves.
            - 若 samplerate 不等於16000,
                則 存新 wave 於 wave_pathes_new.
            - 若 samplerate 等於16000, 
                則 wave 複製到 wave_pathes_new.
        """
        new_wave_names = []
        wav_srs = []
        for org_wav_path, new_wav_path in zip(wave_pathes, wave_pathes_new):
            # get wave's samplerate.
            wav_sr = librosa.get_samplerate(org_wav_path)
            if wav_sr != my_sample_rate:
                wav_data, new_sr = librosa.load(org_wav_path, sr=my_sample_rate)
                sf.write(new_wav_path, wav_data, my_sample_rate)
                print(f"Save new wave file !!, {new_wav_path=}")
                new_wave_names.append(new_wav_path)
            else:
                copyfile(org_wav_path, new_wav_path)
                print(f"Copy wave in new path !!, {new_wav_path=}")
                new_wave_names.append(new_wav_path)
        return new_wave_names


    @staticmethod    
    def save_csv_file(save_path, csv_datalist):
        """
        將 data list 存為 csv 檔.
        * Intput:
            - save_path: 存 CSV 的路徑
            - csv_datalist: data list 
                includ three data : 
                    * 音檔名與路徑(wave_name)
                    * 音檔內容(labels)
                    * 音檔時間(wave_times)
        """
        # 以 data list 存取 csv
        name=['wave_name','labels','wave_times']      
        csv_data = pd.DataFrame(columns=name,data=csv_datalist)
        csv_data.to_csv(save_path, encoding='utf-8', index=False)        
        return save_path 
# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    path = "config_ASR_data_process.ini"
    config = read_config(path)
    
    csv_data_processing = AsrCsvDataPreporcessing(config)
    csv_data_processing.process_flow()