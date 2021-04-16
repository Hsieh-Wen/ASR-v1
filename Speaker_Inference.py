#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import librosa
import requests
import configparser
import numpy as np

import sys
sys.path.append("./Utils")
from CSV_utils import read_mycsv_file
from Save_Inference_Result import SaveResult

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf

class SpeakerInference():
    def __init__(self, config):
        # Load config
        # parameters of initial function
        input_csvs = config['Speaker_predict'].get('INPUT_FILE')
        self.save_path = config['Speaker_predict'].get('OUTPUT_FOLDER')

        self.sample_rate = config['Speaker_predict'].getint('SAMPLE_RATE')
        self.url = config['Speaker_predict'].get('SPEAKER_URL')

        self._init_params(input_csvs)

    def _init_params(self, input_csvs):     
        """
        處理 config data. 
        (str ---> list)        
        """
        input_csvs = re.sub(" ","",input_csvs)
        input_csvs = re.sub("\n","",input_csvs)
        if "|" in input_csvs:
            self.input_csvs_list = input_csvs.split("|")
        else:
            self.input_csvs_list = [input_csvs]

    def _init_csv_data(self, input_csv):
        """Get originial data(wave_path and speaker_name) from csv/npz file.
                (Inference data in stage1 have csv and npz files at the same time.)
        Args:
            input_csv (str): input csv file's path.

        Returns:
            wav_path_list (list): list of waves'path. 
            speaker_truth_list (list): list of speakers' name.
            npz_data (dict): data from npz file.
        """
        self.check_filepath_exist(input_csv)
        wav_path_list = self.get_wav_path(input_csv)

        npz_folder = "/".join(input_csv.split("/")[0:-1])
        npz_name = re.sub(".csv",".npz",input_csv.split("/")[-1])
        npz_path = npz_folder + "/" + npz_name
        self.check_filepath_exist(npz_path)
        print("Load orgiginal npz data !!")
        npz_data = np.load(npz_path, allow_pickle=True)
        speaker_truth_list = npz_data['Speaker_name'].tolist()
        return wav_path_list, speaker_truth_list, npz_data

    def save_flow(self, infer_file, kwargs, npz_data):
        save_result = SaveResult(self.save_path)

        if "/" in infer_file:
            infer_file_name = infer_file.split("/")[-1]
            infer_folder = "/".join(infer_file.split("/")[0:-1])
        else:
            infer_file_name = infer_file
            infer_folder = "./"
        npz_name = re.sub(".csv",".npz",infer_file_name)
        npz_path = infer_folder + "/" + npz_name
        
        kwargs.update(npz_data)
        save_result.save_follow(**kwargs)  


    def inference_flow(self):
        for input_csv in self.input_csvs_list:
            wav_path_list, speaker_truth_list, npz_data = self._init_csv_data(input_csv)
            
            speaker_result = []
            speaker_wer_list = []
            speaker_list = []
            speaker_recognition_time =[]
            cal_wer_avg =[]

            for wav,spaeker_name in zip(wav_path_list, speaker_truth_list):
                s_time = time.time()
                speaker_predict_result = self.Speaker_Predict(self.url, wav, self.sample_rate)
                e_time = time.time()
                
                wer, speaker_truth = self.Speaker_Eval(speaker_predict_result, spaeker_name)

                speaker_result.append(speaker_predict_result)
                speaker_recognition_time.append(e_time-s_time)
                speaker_wer_list.append(wer)
                speaker_list.append(speaker_truth)

                if wer != "Unrecognizable":
                    cal_wer_avg.append(wer)

            # print(f"{speaker_wer_list=}")
            speaker_wer_avg = float((np.sum(cal_wer_avg))/len(cal_wer_avg))
            speaker_recotime_avg = float((np.sum(speaker_recognition_time))/len(speaker_recognition_time))  

            kwargs = {'Inference_Mode':'Spaker', 'Input_File': input_csv, 'Model_Path': self.url,
                        'Ground_Truth': speaker_list, 'Predict_Result': speaker_result, 'Wer_list': speaker_wer_list,
                        'Speaker_wer_avg': speaker_wer_avg,
                        'Avg_reco_time': speaker_recotime_avg, 'Recognition_time_list': speaker_recognition_time,
                        'Py_file_name': os.path.basename(sys.argv[0])}
    
            self.save_flow(input_csv, kwargs, npz_data)
            print(f"檔名:{input_csv}, 平均wer= {speaker_wer_avg}") 
            



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
    def get_wav_path(input_data_path):
        """Get wave path list from csv data.

        Args:
            input_data_path (str): Input csv name.

        Returns:
            wave_names (list): list of wave pathes.
        """
        wave_names, _, _ = read_mycsv_file(input_data_path)
        return wave_names
    @staticmethod
    def Speaker_Predict(url, wav_name, sample_rate):
        print(wav_name)
        wav, sr = librosa.load(wav_name, sr=sample_rate)
        text = {"speaker":wav.tolist()}
        a = requests.post(url ,json=text)
        
        predict_result = json.loads(a.text)["name"]    
        # predict_result = "None"
        return predict_result

    @staticmethod
    def Speaker_Eval(speaker_predict_result, speaker_truth):
        """Calculate wer.

        Args:
            speaker_predict_result (str): Result of speaker recognition.
            speaker_truth (str): Truth of speaker name.

        Returns:
            wer (float): wer of speaker recognition.
            speaker(str): Truth of speaker name(remove "#").

        """
        if "#" in speaker_truth:
            speakers = speaker_truth.split("#")[1:]
        else:
            print(f"Speaker label no # !!")
            speakers = speaker_truth
        if len(speakers) > 1 :
            print(f"more speaker !!")
            speaker = ','.join(speakers)
            wer = "Unrecognizable"
        else:
            speaker = speakers[-1]
            if speaker_predict_result == speaker:
                wer = 0.0
            else:
                wer = 1.0   
        return wer, speaker


# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    path = "config_speaker_inference.ini" 
    config = read_config(path)
   
    speaker_infer = SpeakerInference(config)
    speaker_infer.inference_flow()
