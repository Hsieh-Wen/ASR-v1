#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:36:11 2021

@author: c95hcw
"""
import sys
sys.path.append("./Utils")
from load_config_args import load_config
from CSV_utils import read_csv_file, combine_csv_files, save_dataframe_to_npz
from Bulid_Vocabulary import bulid_vocab

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



class AsrCsvDataPreporcessing():
    def __init__(self, config_path):
        """
        Load config params.
        """
        args = load_config(config_path=config_path)
        stage0_args = args.Stage0

        self.input_files = stage0_args['INPUT_FILES']

        self.save_folder_name = stage0_args['SAVE_FOLDER_NAME']
        self.save_data_folder = self.save_folder_name + "Stage0/ASR/"
        if os.path.exists(self.save_folder_name + "Stage0/ASR/"):
            sys.exit("資料夾已存在！！請修改 config_data_process.ini 之 [SAVE_FOLDER_NAME]") 
   
    def data_preprocess_parameters(self, parameters_dict): 

        self.check_data_parameter_exist(parameters_dict)

        data_folder = parameters_dict['DATA_FOLDER']
        self.csv_folder = data_folder + "csvs/"
        self.wav_folder = data_folder + "waves/" 

        self.csv_name = parameters_dict['INTPUT_CSV_PATH']
        self.csv_column_wav = parameters_dict['CSV_COLUMN_WAV']
        self.csv_column_label = parameters_dict['CSV_COLUMN_LABEL']

        self.duration_threshold = parameters_dict['WAVE_TIME_THRESHOLD']
        self.sample_rate = parameters_dict['SAMPLERATE']  
        self.save_new_wav = parameters_dict['SAVE_NEW_WAVE']         
        if self.save_new_wav:
            self.wav_folder_new = self.save_folder_name + "waves/"
            if not os.path.exists(self.wav_folder_new):
                os.makedirs(self.wav_folder_new)

        self.split_ratio =  parameters_dict['SPLIT_RATIO']
        self.split_mode =  parameters_dict['SPLIT_MODE']
        self.seed =  parameters_dict['SEED']
        self.expand_value =  parameters_dict['DATA_EXPAND_value']    
        
        self.print_paremeters()

    def check_data_parameter_exist(self, parameters_dict):
        # input csv
        if 'DATA_FOLDER' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of DATA_FOLDER")               
        if 'INTPUT_CSV_PATH' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of INTPUT_CSV_PATH")  

        if 'CSV_COLUMN_WAV' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of CSV_COLUMN_WAV")       
        if 'CSV_COLUMN_LABEL' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of CSV_COLUMN_LABEL")

        # Parameters of audio file         
        if 'SAVE_NEW_WAVE' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SAVE_NEW_WAVE")      
            
        if 'SAMPLERATE' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SAMPLERATE")
        if 'WAVE_TIME_THRESHOLD' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of WAVE_TIME_THRESHOLD")   

        # parameters of split csv data            
        if 'SPLIT_RATIO' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SPLIT_RATIO")  
        if 'SPLIT_MODE' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SPLIT_MODE")  
        if 'SEED' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of SEED")  
        if 'DATA_EXPAND_value' in parameters_dict:
            pass
        else:
            sys.exit("Please input data of DATA_EXPAND_value")  

    def print_paremeters(self):
        csv_path = self.csv_folder + self.csv_name
        print(f"{csv_path=}.")
        print(f"{self.save_new_wav}.")
        print(f"{self.sample_rate}.")
        print(f"{self.duration_threshold=}.")
        print(f"{self.split_ratio=}.")
        print(f"{self.split_mode=}.")
        print(f"{self.seed=}.")
        print(f"{self.expand_value=}.")

    def _init_csv_cropus(self, csv_name, csv_column_wav, csv_column_label):
        """
        Load csv data.
        * Input:
            csv_name.
        * Output:
            wave_names : wave's pathes (list). 
            asr_truth : wave's labels (list).
        """
        # self.csv_name = csv_name
        self.data_path = self.csv_folder + csv_name
        wave_names, asr_truth = read_csv_file(self.csv_folder + csv_name, csv_column_wav, csv_column_label)

        # save original csv data
        save_npz_folder = self.save_data_folder + "org_csvdata/"
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
        wave_names, asr_truth = self._init_csv_cropus(csv_name, self.csv_column_wav, self.csv_column_label)

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
        for i, input_file_id in enumerate(self.input_files.keys()):
            self.data_preprocess_parameters(self.input_files[input_file_id])           
            print(f"\n--------------DATA PREPROCESSING--{self.csv_name}----------------")
            self.corpus_num = i
            # save setting data of data preprocessing
            data_parameter_dict['Corpus_' + str(i)] = self.data_process(self.csv_name, self.split_ratio, self.split_mode, self.expand_value)
            
        # 合併 train/valid corpus 內所有 csv 為 train.csv / valid.csv
        self.combine_train_valid_corpus(self.save_data_folder , self.save_data_folder )        
        
        # save data process parameter to json
        json_path = self.save_data_folder + 'asr_data_processing_parameter.json'
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
            
            vocabulary, sorted_vob, vocab_json_path = bulid_vocab(self.data_path, "no_save" , save_file = False, csv_column_label=self.csv_column_label)
            
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
        save_asr_folder = save_data_folder 
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
    config_path = "/home/c95hcw/ASR/config_ASR_data_process.yaml"
    csv_data_processing = AsrCsvDataPreporcessing(config_path)
    csv_data_processing.process_flow()
            