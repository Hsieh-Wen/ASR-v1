#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:59:54 2021

@author: c95csy
"""

import sys
sys.path.append("./Utils/LM_BERT/")

import configparser
import bert_corrector
import os
import pandas as pd
import Levenshtein as Lev
import re

from Save_Inference_Result import SaveResult

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf


class InferenceBertLm():
    def __init__(self, ASR_csv_name, lm_path, device):
        
        self.ASR_csv_name = ASR_csv_name 
        
        self.lm_path = lm_path
        self.LM_model = bert_corrector.BertCorrector(bert_model_dir = self.lm_path, device=device)

        
        # Open the CSV file for reading
        # load asr predict result to inference mask LM model
        asr_data = pd.read_csv(open(self.ASR_csv_name))
        self.asr_truth = asr_data.labels
        self.asr_pred = asr_data.Predict
        self.asr_wer = asr_data.WER
        
        
        
    def print_args(self):
        print(f"{self.lm_path=}")
        print(f"{self.use_confusion_word=}")
        print(f"{self.wwm=}")
        print(f"{self.reverse=}")
        print(f"{self.token_replace=}")
        print(f"{self.muti_method=}")
        print(f"{self.replace=}")
        
    def set_predict_method(self, use_confusion_word, wwm, reverse, token_replace, muti_method, replace):
        
        self.use_confusion_word = use_confusion_word
        self.wwm = wwm
        self.reverse = reverse
        self.token_replace = token_replace
        self.muti_method = muti_method
        self.replace = replace
        
        
    def inference_test_data(self, use_confusion_word, wwm, reverse, token_replace, muti_method=False, replace="intersection"):
        """
        use Bert Mask LM to inference ASR predict result
        1. set predict method
        2. use bert predict
        3. calculate result wer
        4. save result to (csv & txt)
        
        :input:
            :param: method args.
        
        :return: ASR wer & LM correct wer
        """
        self.set_predict_method(use_confusion_word, wwm, reverse, token_replace, muti_method, replace)
        self.print_args()
        
     
        LM_correct = []
        LM_err = []
        LM_wer = []
        str
        lm_wers = 0
        asr_wers = 0
        
        for i in range(len(self.asr_pred)):
            # use muti method or not
            if self.muti_method:
                correct_sent, err = self.muti_method_predict(self.asr_pred[i])
                
            else:
                correct_sent, err = self.LM_model.bert_correct(self.asr_pred[i], 
                                                                   wwm=self.wwm, 
                                                                   reverse=self.reverse, 
                                                                   token_replace=self.token_replace, 
                                                                   use_confusion_word=self.use_confusion_word)
#            print(correct_sent)
            LM_correct.append(correct_sent)
            LM_err.append(err)   
            print("original sentence:{} => {}, err:{}".format(self.asr_pred[i], correct_sent, err))
            lm_wer = self.cul_wer(correct_sent, self.asr_truth[i]) / len(self.asr_truth[i])
            LM_wer.append(lm_wer)
            lm_wers += lm_wer            
            asr_wers += self.asr_wer[i]       
        LM_wer_avg = lm_wers/len(LM_wer)
        ASR_wer_avg = asr_wers/len(self.asr_wer)
        
        self.print_args()
        print("ASR - 平均 WER = ", ASR_wer_avg)
        print("LM - 平均 WER = ", LM_wer_avg)
        asr_predict =  self.asr_pred
        lm_predict_result = LM_correct
        lm_wer_list = LM_wer
        return LM_wer_avg, ASR_wer_avg, asr_predict, lm_predict_result, lm_wer_list, LM_err
            
            
    def args2list(self):
        """
        for loop must con iterable, so transform bool to list
        """
        if type(self.use_confusion_word) == bool:
            self.use_confusion_word = [self.use_confusion_word]
            
        if type(self.wwm) == bool:
            self.wwm = [self.wwm]
        
        if type(self.reverse) == bool:
            self.reverse = [self.reverse]
            
        if type(self.token_replace) == bool:
            self.token_replace = [self.token_replace]

    def muti_method_predict(self, text):
        """
        use muti method predict amd conbime result
        :param self.replace: "intersection or union"
        replace text token method
        """
        self.args2list()
        all_details = []
        for use_confusion_word in self.use_confusion_word:
            for wwm in self.wwm:
                for reverse in self.reverse:
                    for token_replace in self.token_replace:
                        _, details = self.LM_model.bert_correct(text, 
                                                                wwm=wwm, 
                                                                reverse=reverse, 
                                                                token_replace=token_replace, 
                                                                use_confusion_word=use_confusion_word)                        
                        all_details.append(details)            
        num_details = len(all_details)
#        print(all_details)
        if self.replace == "intersection":
            if num_details == 2:
                final_details = self.intersection_list(all_details[0], all_details[1])
            else:
                final_details = all_details[0]
                for d in all_details[1:]:
                    final_details = self.intersection_list(final_details, d)
      
        elif self.replace == "union":
            if num_details == 2:
                final_details = self.union_list(all_details[0], all_details[1])
            else:
                final_details = all_details[0]
                for d in all_details[1:]:
                    final_details = self.union_list(final_details, d)        
        result = self.LM_model.replace_predict_token(text, final_details)
            
        return result, final_details
            

    @staticmethod
    def cul_wer(s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.
    
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
        return Lev.distance(s1, s2)           
                
    @staticmethod
    def intersection_list(list1, list2):
        """
        list1, list2 -> intersection
        """
        result = []
        for l2 in list2:
            if l2 in list1:
                    result.append(l2)                   
        return result
    
    @staticmethod
    def union_list(list1, list2):
        """
        list1, list2 -> union
        """
        for l2 in list2:
            if not l2 in list1:
                list1.append(l2)
        return list1



if __name__ == "__main__":
    file_name =  os.path.basename(sys.argv[0])
    path = "config_asr_inference.ini"
    config = read_config(path)
# Parameters    
    # parameters of initial function
    device = config['Stage1_lm_inference'].get('DEVICE') 
        
    ASR_csv_names = config['Stage1_lm_inference'].get('INFERENCE_FILE')   
    ASR_csv_names = re.sub(" ","",ASR_csv_names)
    ASR_csv_names = re.sub("\n","",ASR_csv_names)
    ASR_csv_names_list = ASR_csv_names.split("|")   
    
    lm_paths = config['Stage1_lm_inference'].get('MODEL_PATH')
    lm_paths = re.sub(" ","",lm_paths)
    lm_paths = re.sub("\n","",lm_paths)
    lm_paths_list = lm_paths.split("|")  
    
    # inference parameter
    use_confusion_words = config['Stage1_lm_inference'].get('use_confusion_words')
    use_confusion_words = re.sub(" ","",use_confusion_words)
    use_confusion_words = re.sub("\n","",use_confusion_words)
    use_confusion_words_list = use_confusion_words.split("|")  
    
    wwms = config['Stage1_lm_inference'].get('wwms')
    wwms = re.sub(" ","",wwms)
    wwms = re.sub("\n","",wwms)
    wwms_list = wwms.split("|")     
    
    reverses = config['Stage1_lm_inference'].get('reverses')
    reverses = re.sub(" ","",reverses)
    reverses = re.sub("\n","",reverses)
    reverses_list = reverses.split("|")   
    
    token_replaces = config['Stage1_lm_inference'].get('token_replaces')
    token_replaces = re.sub(" ","",token_replaces)
    token_replaces = re.sub("\n","",token_replaces)
    token_replaces_list = token_replaces.split("|")      


    
    # parameters of save parameter       
    save_path = config['Stage1_lm_inference'].get('SAVE_PATH') 
    save_result = SaveResult(save_path)

    
    for csv_name in ASR_csv_names_list:
        for lm_path in lm_paths_list:
            inference_bert_lm = InferenceBertLm(csv_name, lm_path, device=device)
            
            for use_confusion_word in use_confusion_words_list[1:]:
                for wwm in wwms_list:
                    for reverse in reverses_list:
                        for token_replace in token_replaces_list:
                            LM_wer_avg, ASR_wer_avg, asr_predict, lm_predict_result, lm_wer_list, LM_err= inference_bert_lm.inference_test_data(use_confusion_word, wwm, reverse, token_replace)                            
                                                      
                            kwargs_lm = {'Inference_Mode':'LM', 'Input_File': csv_name, 'Model_Path':lm_path, 
                                         'Ground_Truth':asr_predict, 'Predict_Result':lm_predict_result, 'Wer_list':lm_wer_list, 'Error_word':LM_err, 
                                         'ASR_wer_avg':ASR_wer_avg,'LM_wer_avg':LM_wer_avg,                 
                                         'wwm':wwm,'use_confusion_word':use_confusion_word,'reverse':reverse,'token_replace':token_replace,
                                         'Py_file_name':file_name}    
                            save_result.save_follow(**kwargs_lm)             
            
            del inference_bert_lm
            
            
            
            
            