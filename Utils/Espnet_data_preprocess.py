#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:31:00 2021

@author: c95csy
"""

'''
espnet dataset process
    Step1 : icrd stage0 format -> aishell dataset format
    Step2 : aishell dataset format -> espnet traning file
    
data format
    icrd dataset stage0 format
    - data
        combined_train_corpus.csv
        :split: ","
        :param: wave_name	labels	wave_times
        
        combined_valid_corpus.csv
        
        - icrd_waves
            - cropus folder    # wavs path must find in csv, each cropus was difference
        ...
        
    aishell dataset format
    data_aishell
        - transcript
            aishell_transcript_v0.8.txt
            :split: " "
            :param: wav_name label
        - wav
            - dev
                - speaker name    # folder -> wavs
                ...
            - test
            - train
    
    generate file (for espnet train)
        spk2utt
        :split: " "
        :param: wav_folder wav_name(all in floder)
        
        wav.scp
        :split: " "
        :param: wav_name wav_name_path(absolute)
        
        utt2spk
        :split: " "
        :param: wav_name wav_folder
        
        text
        :split: " "
        :param: wav_name wav_text

'''

import os
import pandas as pd
import shutil

class EspnetDataFormat():
    """
    generate file (for espnet train)
        spk2utt
        :split: " "
        :param: wav_folder wav_name(all in floder)
        
        wav.scp
        :split: " "
        :param: wav_name wav_name_path(absolute)
        
        utt2spk
        :split: " "
        :param: wav_name wav_folder
        
        text
        :split: " "
        :param: wav_name wav_text
    
    """
    def __init__(self):
        pass
    
    def script2espnet_fromat(self, script_path, mode):
        with open(script_path) as f:
            self.texts = f.readlines()
            
#        self.save_path = os.path.join(self.wav_path, mode)
        self.save_path = os.path.join(self.data_path, mode)
        self.wave_name, self.wave_abs_path, self.labels, self.folders = self.generate(mode)
        
        # make each file
        self.make_wavscp_file()
        self.make_utt2spk_file()
        self.make_text_file()
        self.make_spk2utt()
        
    @staticmethod
    def extract_wavname(text):
        '''
        wave name without .wav
        '''
        return text[:-4]
    
    
    def extract_wave_abs_folder(self, text, mode):
        """
        extract wave folder
        extract wave abs path
        """
        folder = text.split(mode)[0]
        folder = folder + "_" + mode
        abs_path = os.path.join(self.wav_path, mode, folder, text)
#        return f"{self.wav_path}/{self.mode}/{folder}/{text}", folder
        return abs_path, folder
    
    def generate(self, mode):
        """
        generate wave_name, wave_abs_path, labels, folders
        """
        wave_name = []
        wave_abs_path = []
        labels = []
        folders = []
        for text in self.texts:
            text = text.strip().split(" ")
            wave_name.append(self.extract_wavname(text[0]))
            labels.append(text[1])
            abs_path, folder = self.extract_wave_abs_folder(text[0], mode)
            wave_abs_path.append(abs_path)
            folders.append(folder)
            
        return wave_name, wave_abs_path, labels, folders
        
        
    def make_wavscp_file(self):
        wav_scp = []
        
        for n, p in zip(self.wave_name, self.wave_abs_path):
            wav_scp.append(n + " " + p)
            
        wav_scp.sort()
        self.write_file(wav_scp, self.save_path + "/wav.scp")    

    
    def make_utt2spk_file(self):
        utt2spk = []
        
        for n, f in zip(self.wave_name, self.folders):
            utt2spk.append(n + " " + f)
            
        utt2spk.sort()
        self.write_file(utt2spk, self.save_path + "/utt2spk")
    
    def make_text_file(self):
        text = []
        
        for n, l in zip(self.wave_name, self.labels):
            text.append(n + " " + l)
        
        text.sort()
        self.write_file(text, self.save_path + "/text")
    
    def make_spk2utt(self):
        spk2utt = {}
        folder_list = list(set(self.folders))
        
        # init spk2utt dict
        for f in folder_list:
            spk2utt[f] = f
            
        for f, n in zip(self.folders, self.wave_name):
            spk2utt[f] += " " + n
      
        self.write_file(spk2utt.values(), self.save_path + "/spk2utt")
                    
    @staticmethod
    def write_file(file, save_path):
        """
        write list -> file
        """
        with open(save_path, "w") as f:
            l1 = map(lambda x : x+"\n", file)
            f.writelines(l1)
            
class EspnetDataProcess(EspnetDataFormat):
    def __init__(self, dataset_root):
        self.stage0 = os.path.join(dataset_root, "Stage0/ASR")
        self.save_dir = os.path.join(dataset_root, "Stage1", "Espent")
        self.modes = ["train", "dev", "test"]
        self.csv_names = ["combined_train_corpus.csv", "combined_valid_corpus.csv", "combined_valid_corpus.csv"]
        
        self.wav_path, self.transcript_path, self.data_path = self.make_folder()
        self.wav_abs_path = os.path.abspath(self.wav_path)
        
    def _init_params(self, csv_path):
        self.csv_path = csv_path
        self.wave_path, self.labels = self.load_csv(csv_path)
        
        self.wave_folder = []
        self.wave_name = []        
    
    
    def make_folder(self):
        self.check_dir(self.save_dir)
        wav_path = self.check_dir(os.path.join(self.save_dir, "wav"))
        transcript_path = self.check_dir(os.path.join(self.save_dir, "transcript"))
        data_path = self.check_dir(os.path.join(self.save_dir, "data"))
        for mode in self.modes:
            self.check_dir(os.path.join(wav_path, mode))
            self.check_dir(os.path.join(data_path, mode))
        
        return wav_path, transcript_path, data_path
    
    def extract_wave_foldername(self):
        # extract wave name
        """
        dataset path
        ./Dataset/raw_data/waves/ + cropus + ... + wav_name
        """
        for p in self.wave_path:
            p_split = p.split("/")
            self.wave_folder.append(p_split[4])
            self.wave_name.append(p_split[-1])
                
    def extract_speaker(self, mode):
        # extract speaker(cropus name = speaker)
        folders = list(set(self.wave_folder))
        wav_new_name_dict = {}
        for f in folders:
            # make speaker folder
            self.check_dir(os.path.join(self.wav_path, mode, f+"_"+mode))
            wav_new_name_dict[f] = 0
        return wav_new_name_dict, folders
    
    def process(self, csv_path, mode):
        
        self._init_params(csv_path)
        self.extract_wave_foldername()
        wav_new_name_dict, folders= self.extract_speaker(mode)
        
        '''
        - copy wav file
        - generate transcipt.txt
        '''
        count = 0
        transcript = []
        for i in range(len(self.wave_path)):
            if not os.path.isfile(self.wave_path[i]):
                print(f"{i=} - {self.wave_path[i]=}")
                count += 1
                continue
            folder = folders[folders.index(self.wave_folder[i])]     # this wav speaker
            new_wav_name = folder + mode + "W" + "%05d"%wav_new_name_dict[folder] + ".wav"
            shutil.copy(self.wave_path[i], os.path.join(self.wav_path, mode, folder+"_"+mode, new_wav_name)) 
            wav_new_name_dict[folder] += 1
            
            text = new_wav_name + " " + self.labels[i]
            transcript.append(text)
        
        script_path = os.path.join(self.transcript_path, mode+"transcript.text")
        self.write_file(transcript, script_path)
        
        self.script2espnet_fromat(script_path, mode)
        
    def espnet_dataset(self):
        for csv_name, mode in zip(self.csv_names, self.modes):
            csv_path = os.path.join(self.stage0, csv_name)
            self.process(csv_path, mode)
        
    @staticmethod
    def check_dir(path):
        """
        check dir is exit,
        if not mkdir
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        
        return path
    
            
    @staticmethod
    def load_csv(path):
        # load icrd cropus
        csv = pd.read_csv(open(path))
        wave_path = csv.wave_name.to_list()
        labels = csv.labels.to_list()
        
        return wave_path, labels
    


            
if __name__ == "__main__":
    dataset_root = "./Dataset/data_t1"
    espnet_data_process = EspnetDataProcess(dataset_root)
    espnet_data_process.espnet_dataset()
