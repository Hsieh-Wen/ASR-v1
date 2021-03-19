#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:26:06 2021

@author: c95hcw
"""
import sys
sys.path.append("./quartznet_util")
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import post_process_predictions

import os
import re
import json
from opencc import OpenCC
from CSV_utils import read_mycsv_file, delete_blank
from wer_eval import eval_asr

class QuartznetInference():
    def __init__(self, model_path):
        self.quartznet_model = self.quartznet_load_model(model_path)

    def quartznet_load_model(self, model_path):
        self.nf = nemo.core.NeuralModuleFactory()        
        quartznet_model = nemo_asr.models.ASRConvCTCModel.from_pretrained(model_info = model_path)
        quartznet_model.eval()
        return quartznet_model    

     
    def csv_to_manifest(self, wav_folder, inference_file):      
        """
        Generate Quartznet inference dataset.        
        (csv ->> json)
        
        csv data include: wave_path, labels, wave_time
        """
        wave_names, asr_truth, wave_times = read_mycsv_file(inference_file)     
        asr_truth = delete_blank(asr_truth)     
        json_lines = []
        for num in range(len(wave_names)):
            json_lines.append(json.dumps({'audio_filepath':  wav_folder + wave_names[num], \
                                          'duration': wave_times[num], 'text': asr_truth[num],},ensure_ascii=False,))        
        
        inference_file_name = inference_file.split("/")[-1]
    
        manifest_name = re.sub(".csv",".json",inference_file_name)

        save_manifest_folder = "./Inference_Result/Inference_manifest_folder/"
        
        if not os.path.exists(save_manifest_folder):
            os.makedirs(save_manifest_folder)             
 
        manifest_path = save_manifest_folder + manifest_name
        with open(manifest_path, 'w', encoding='utf-8') as fout:
            for line in json_lines:
                fout.write(line + "\n")       
        return manifest_path, asr_truth    
    
    
    def quartznet_recognition(self, wav_folder, inference_file, convert_word):
        # Load csv data and Generate manifest dataset
        dataset, asr_truth = self.csv_to_manifest(wav_folder, inference_file)
        # Load inference dataset and predict
        eval_data_layer = nemo_asr.AudioToTextDataLayer(
            manifest_filepath=dataset,
            labels=self.quartznet_model.vocabulary,
            batch_size=64,
            trim_silence=1.0,
            shuffle=False,
            normalize_transcripts=False,
        )
        greedy_decoder = nemo_asr.GreedyCTCDecoder()
        
        # data load
        audio_signal, audio_signal_len, transcript, transcript_len = eval_data_layer()
        
        # inference
        log_probs, encoded_len = self.quartznet_model(input_signal=audio_signal, length=audio_signal_len)
        predictions = greedy_decoder(log_probs=log_probs)
        
        eval_tensors = [log_probs, predictions, transcript, transcript_len, encoded_len]
        evaluated_tensors = self.nf.infer(tensors=eval_tensors)    
        asr_predict_result = post_process_predictions(evaluated_tensors[1], self.quartznet_model.vocabulary)   
        if convert_word:
            cc = OpenCC('s2tw')
            for i in range(len(asr_predict_result)):
                asr_predict_result[i] = cc.convert(asr_predict_result[i])
        ASR_wer_avg, wer_list = eval_asr(asr_truth, asr_predict_result)              
        return ASR_wer_avg, asr_truth, asr_predict_result, wer_list    
    
if __name__ == "__main__":
    model_path = "../Models/ASR_models/quar_model_v1.7.5/QuartzNet-EPOCH-n.pt.nemo"

    wav_folder = "../Dataset/raw_data/waves/"
    inference_file = "../Dataset/raw_data/csvs/TW_test.csv"
    convert_word = False
    
    quartznet_infer = QuartznetInference(model_path) 
    ASR_wer_avg, asr_truth, asr_predict_result, wer_list = quartznet_infer.quartznet_recognition(wav_folder, inference_file, convert_word)
    
    
    