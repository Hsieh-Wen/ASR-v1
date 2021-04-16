#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import configparser
from  Utils.CSV_utils import read_inference_csv_file

import Levenshtein as Lev
import numpy as np
import re
import json
import os
import sys

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf

class InferenceResultTOJson():
    def __init__(self, config):
        # Load config
        # parameters of initial function
        asr_folder = config['Parameters'].get('ASR_INPUT_FOLDER')
        self.asr_file = config['Parameters'].get('ASR_INPUT_FILE')
        self.google_file = config['Parameters'].get('Google_INPUT_FILE')

        self.asr_csv_path = asr_folder + self.asr_file
        self.google_csv_path = asr_folder + self.google_file
        self.asr_json_path = asr_folder + "result.json"

        npz_name = re.sub("csv","npz", self.asr_file)
        self.npz_file_path = asr_folder + "npz_files/" + npz_name
        
        speaker_folder = config['Parameters'].get('SPEAKER_INPUT_FOLDER')
        self.speaker_file = config['Parameters'].get('SPEAKER_INPUT_FILE')
        self.speaker_csv_path = speaker_folder + self.speaker_file    
        self.speaker_json_path = speaker_folder + "result.json"

        self.check_filepath_exist(self.asr_csv_path)
        self.check_filepath_exist(self.asr_csv_path)
        self.check_filepath_exist(self.npz_file_path)
        self.check_filepath_exist(self.speaker_csv_path)

        self.save_folder = config['Parameters'].get('SAVE_PATH')
        self.save_json_name = config['Parameters'].get('SAVE_NAME')
        self.check_folder_exist(self.save_folder)


    def output_json(self):

        # check same inference file(asr/google/speaker)
        asr_json = self.load_json_data(self.asr_json_path)
        speaker_json = self.load_json_data(self.speaker_json_path)

        if asr_json[self.asr_file]['Input_File'] != speaker_json[self.speaker_file]['Input_File']:
            raise ValueError("Please ckeck input files.")

        if asr_json[self.google_file]['Input_File'] != speaker_json[self.speaker_file]['Input_File']:
            raise ValueError("Please ckeck input files.")    


        # get avg. wer from asr/speaker json.
        asr_wer_avg = asr_json[self.asr_file]['ASR_wer_avg']
        speaker_wer_avg = speaker_json[self.speaker_file]['Speaker_wer_avg']

        # get data from csv files
        asr_truth, asr_result, _ = read_inference_csv_file(self.asr_csv_path)
        _ , google_result, _ = read_inference_csv_file(self.google_csv_path)
        speaker_truth, speaker_result, _ = read_inference_csv_file(self.speaker_csv_path)
        
        # npz - start_time/end_time
        npz_data = np.load(self.npz_file_path)
        wav_start = npz_data['Time_start'].tolist()
        wav_end = npz_data['Time_end'].tolist()

        save_result_data = []
        save_result_data.append({'speech_accuracy': int((1-asr_wer_avg)*100) , 'speaker_accuracy': int((1-speaker_wer_avg)*100)})
        for i in range(len(asr_truth)):
            kwargs = {'start_time':  wav_start[i], 'end_time':  wav_end[i], 
                    'MASR_results': speaker_result[i] + ":" + asr_result[i], 
                    'google_asr_results': google_result[i],
                    'labels': speaker_truth[i] + ":" + asr_truth[i]}
                    # 'asr_recognition_time': self.asr_recognition_time[i],
                    # 'google_recognition_time': self.google_recognition_time[i],
                    # 'speaker_recognition_time': self.speaker_recognition_time[i]}

            save_result_data.append(kwargs)
        
        with open(self.save_folder + self.save_json_name, 'w') as fp:
            json.dump(save_result_data, fp, indent=len(kwargs),ensure_ascii=False)        
        print("Already save data !!")


    @staticmethod     
    def check_filepath_exist(file_path):
        """Check file's path exist.
        Args:
            file_path (str): File's path.      
        Raises:
            IOError: check file exists. 
        """
        if not os.path.isfile(file_path):
            raise IOError("No such file or directory:" + file_path)

    @staticmethod  
    def check_folder_exist(folder_path):
        """
        確認檔案路徑是否存在.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create csv Folder !!!, folder path : {folder_path}")

    @staticmethod
    def load_json_data(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f) 
        return data


# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    path = "config_ui_json.ini" 
    config = read_config(path)
   
    ui_json_result = InferenceResultTOJson(config)
    ui_json_result.output_json()