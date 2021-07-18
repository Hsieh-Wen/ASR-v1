#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:59:54 2021

@author: c95csy
"""

import sys
sys.path.append("./Utils")
sys.path.append("./Utils/LM_BERT/")

# import configparser
import bert_corrector
import os
import pandas as pd
import Levenshtein as Lev
import re

from load_config_args import load_config
from CSV_utils import read_inference_csv_file
from Save_Inference_Result import SaveResult

# def read_config(path):
#     conf = configparser.ConfigParser()
#     candidates = [path]
#     conf.read(candidates)
#     return conf

# def list2bool(data_list):
#     bool_list = []
#     for item in data_list:
#         if item == "True":
#             bool_list.append(True)
#         elif item == "False":
#             bool_list.append(False)
#         else:
#             print("Please input True/False.")
#     return bool_list


class InferenceBertLm():
    def __init__(self, config_path): #
        # Load config
        self.args = load_config(config_path=config_path)

        # parameters of initial function
        self.device = self.args.DEVICE

        self.LM_models = self.args.LM_Models
        self.inference_files =  self.args.INFERENCE_FILEs

        # parameters of save parameter       
        self.save_path =  self.args.SAVE_PATH

        self.use_confusion_words = self.args.use_confusion_words
        self.wwms = self.args.wwms
        self.reverses = self.args.reverses
        self.token_replaces = self.args.token_replaces

    def Load_LM_Model(self, model_path, device):
        if self.lm_mode == "BERT":
            LM_model = bert_corrector.BertCorrector(bert_model_dir = model_path, device=device)
        else:            
            print(f"No {self.LM_mode} Exists !!")
        return LM_model

    def inference_test_data(self,ASR_csv_name, use_confusion_word, wwm, reverse, token_replace, muti_method=False, replace="intersection"):
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
        # self.print_args()
        self.asr_truth, self.asr_pred, self.asr_wer = read_inference_csv_file(ASR_csv_name)
        
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
                print(f"Muti_method !!!!!!!!!!!!!!!!")
                
            else:
                print(f"{self.asr_pred[i]=}")
                if type(self.asr_pred[i]) != str:
                    self.asr_pred[i] = "None"
                    print(f"No ASR Predice !!! {type(self.asr_pred[i])=}")
                correct_sent, err = self.LM_model.bert_correct(self.asr_pred[i], 
                                                                   wwm=self.wwm, 
                                                                   reverse=self.reverse, 
                                                                   token_replace=self.token_replace, 
                                                                   use_confusion_word=self.use_confusion_word)
                # print(f" bert_correct !!!!!!!!!!!!!!!!")

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
        
        # self.print_args()
        print("ASR - 平均 WER = ", ASR_wer_avg)
        print("LM - 平均 WER = ", LM_wer_avg)
        asr_predict =  self.asr_pred
        lm_predict_result = LM_correct
        lm_wer_list = LM_wer
        return LM_wer_avg, ASR_wer_avg, asr_predict, lm_predict_result, lm_wer_list, LM_err


        
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

    def LM_inference_flow(self):
        save_result = SaveResult(self.save_path)
        file_name =  os.path.basename(sys.argv[0])
        Model_keys = self.LM_models.keys()
        inferfile_keys = self.inference_files.keys()
        for inferfile_id in inferfile_keys:
            infer_file = self.inference_files[inferfile_id]['csv_file']
            for model_id in Model_keys:
                self.lm_mode = self.LM_models[model_id]['LM_Mode']
                model_path = self.LM_models[model_id]['Model_Path']
                # Load LM Model 
                self.LM_model = self.Load_LM_Model(model_path, self.device)
                for use_confusion_word in self.use_confusion_words:
                    for wwm in self.wwms:
                        for reverse in self.reverses:
                            for token_replace in self.token_replaces:
                                LM_wer_avg, ASR_wer_avg, asr_predict, lm_predict_result, lm_wer_list, LM_err= self.inference_test_data(infer_file, use_confusion_word, wwm, reverse, token_replace)                            
                                                        
                                kwargs_lm = {'Inference_Mode':'LM', 'Input_File': csv_name, 'Model_Path':lm_path, 
                                            'Ground_Truth':asr_predict, 'Predict_Result':lm_predict_result, 'Wer_list':lm_wer_list, 'Error_word':LM_err, 
                                            'ASR_wer_avg':ASR_wer_avg,'LM_wer_avg':LM_wer_avg,                 
                                            'wwm':wwm,'use_confusion_word':use_confusion_word,'reverse':reverse,'token_replace':token_replace,
                                            'Py_file_name':file_name}    
                                save_result.save_follow(**kwargs_lm)             
                            
                            # del inference_bert_lm



if __name__ == "__main__":
    # path = "config_asr_inference.ini"
    # config = read_config(path)
    config_path = "/home/c95hcw/ASR/config_LM_inference.yaml"
    inference_bert_lm = InferenceBertLm(config_path)
    inference_bert_lm.LM_inference_flow()    
            
            
            
            
            