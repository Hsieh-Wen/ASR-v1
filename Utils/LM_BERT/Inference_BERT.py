#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:20:42 2020

@author: c95hcw
"""
import sys
#sys.path.append("./pycorrector/bert/")
import bert_corrector
import os
import pandas as pd
import Levenshtein as Lev

import re



class InferenceBertLm():
    def __init__(self,ASR_csv_name, lm_path, device):
        
        self.ASR_csv_name = ASR_csv_name
        self.result_path = "pycorrector/result/"
        
        if "/" in ASR_csv_name:
            folders = ASR_csv_name.split("/")
            save_txt_name = re.sub(".csv",".txt",folders[-1])
        else:
            save_txt_name = re.sub(".csv",".txt",ASR_csv_name)
        self.result_txt = "pycorrector/result/result_" + save_txt_name
        
        self.lm_path = lm_path
        self.LM_model = bert_corrector.BertCorrector(bert_model_dir = self.lm_path, device=device)
        
        # Open the CSV file for reading
        # load asr predict result to inference mask LM model
        asr_data = pd.read_csv(open(self.ASR_csv_name))
        self.asr_truth = asr_data.Truth
        self.asr_pred = asr_data.ASR_Predict
        self.asr_wer = asr_data.asr_wer
        
        
        
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
        
        if self.lm_path[:12] == "./LM_models/":
            save_csv_path = self.result_path + self.lm_path[12:]
        else:
            print("Please put LM in folder [LM_models] !!!")
            save_csv_path = self.result_path + self.lm_path + "/"
        if not os.path.exists(save_csv_path):
            os.makedirs(save_csv_path)
            print("Create Folder !!!")        
        
        self.LM_save_name = f"{save_csv_path}inference_TW_meeting_ASR_confuse-{use_confusion_word}_wwm-{wwm}_rever-{reverse}_rep-{token_replace}.csv"
        
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
        
        
        self.save_inference_result_txt(LM_wer_avg, ASR_wer_avg)
        
        tables = zip(self.asr_truth, self.asr_pred, LM_correct, LM_err, self.asr_wer, LM_wer)    
                    
        name=['Truth', 'ASR_Predict', 'LM_correct', 'LM_err', 'asr_wer', 'LM_wer']
        test=pd.DataFrame(columns=name, data=tables) #数据有4列，列名分别为'Truth','Predict','Acuracy','WER'
        test.to_csv(self.LM_save_name, encoding='utf-8')     
        
        return LM_wer_avg, ASR_wer_avg
    
    def save_inference_result_txt(self, LM_wer_avg, ASR_wer_avg):
        """
        save inference result to txt
        """
        with open(self.result_txt, "a") as f:
            f.write(f"{self.lm_path=}\t")
            f.write(f"{self.use_confusion_word=}\t")
            f.write(f"{self.wwm=}\t")
            f.write(f"{self.reverse=}\t")
            f.write(f"{self.token_replace=}\t")
            f.write(f"{self.muti_method=}\t")
            f.write(f"{self.replace=}\n")
            
            f.write(f"{ASR_wer_avg=:.4f}\t")
            f.write(f"{LM_wer_avg=:.4f}\n\n")
            
            
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
    
#    # single
#    lm_path = "./LM_models/V1_bert_base5/"
#    lm_path = "hfl/chinese-roberta-wwm-ext"
#    lm_path = "./LM_models/V1_roberta_wwm_ext2/checkpoint-869000/"
#    lm_path = "./LM_models/V1_roberta_wwm_ext2/"
    lm_path = "./LM_models/Kenlm/"
    
    use_confusion_word = True
    wwm = False
    reverse = False
    token_replace = False
    inference_bert_lm = InferenceBertLm(ASR_csv_name,lm_path, device="gpu")
    
    LM_wer_avg, ASR_wer_avg = inference_bert_lm.inference_test_data(use_confusion_word, wwm, reverse, token_replace)
    
    # muti
#    lm_path = "./LM_models/V1_bert_base5/"
#    use_confusion_word = False
#    wwm = [True, False]
#    reverse = False
#    token_replace = False
#    muti_method = True
#    replace = "intersection"
#    inference_bert_lm = InferenceBertLm(lm_path, device="gpu")
#    
#    LM_wer_avg, ASR_wer_avg = inference_bert_lm.inference_test_data(use_confusion_word, 
#                                                                    wwm, 
#                                                                    reverse, 
#                                                                    token_replace,
#                                                                    muti_method,
#                                                                    replace)
    
#lm_path = "bert-base-chinese"
#lm_path = "hfl/chinese-roberta-wwm-ext"

#lm_path = "/home/c95csy/csy/language_model/LM_inference/LM/Bert/V1_bert_base3/"
##
#use_confusion_word = False
#wwm = True
#reverse = False
#token_replace = False

#LM_model = bert_corrector.BertCorrector(bert_model_dir = lm_path)
#
#ASR_csv_name = "pycorrector/inference_Morning_ASR.csv"
#LM_save_name = "pycorrector/result/inference_Morning_ASR_LM.csv"
#save_csv_name = re.sub(".csv",LM_save_name,ASR_csv_name)


# Open the CSV file for reading
#asr_data =pd.read_csv(open(ASR_csv_name))
#asr_truth = asr_data.Truth
#asr_pred = asr_data.ASR_Predict
#asr_wer = asr_data.asr_wer

#LM_correct = []
#LM_err = []
#LM_wer = []
#
#lm_wers = 0
#asr_wers = 0
#
#
#print(f"{lm_path=}")
#print(f"{use_confusion_word=}")
#print(f"{wwm=}")
#print(f"{reverse=}")
#print(f"{token_replace=}")
     
#for i in range(len(asr_pred)):
#    correct_sent, err = LM_model.bert_correct_wwm(asr_pred[i], 
#                                                  wwm=wwm, 
#                                                  reverse=reverse, 
#                                                  token_replace=token_replace, 
#                                                  use_confusion_word=use_confusion_word)
#    
#    LM_correct.append(correct_sent)
#    LM_err.append(err)   
#    print("original sentence:{} => {}, err:{}".format(asr_pred[i], correct_sent, err))
#    lm_wer = cul_wer(correct_sent,asr_truth[i]) / len(asr_truth[i])
#    LM_wer.append(lm_wer)
#    lm_wers += lm_wer
#    
#    asr_wers += asr_wer[i]


#LM_wer_avg = lm_wers/len(LM_wer)
#ASR_wer_avg = asr_wers/len(asr_wer)
#
#
#tables = zip(asr_truth, asr_pred, LM_correct, LM_err, asr_wer, LM_wer)    
#            
#name=['Truth', 'ASR_Predict', 'LM_correct', 'LM_err', 'asr_wer', 'LM_wer']
#test=pd.DataFrame(columns=name, data=tables) #数据有4列，列名分别为'Truth','Predict','Acuracy','WER'
#test.to_csv(LM_save_name, encoding='utf-8')     


#print(f"{lm_path=}")
#print(f"{use_confusion_word=}")
#print(f"{wwm=}")
#print(f"{reverse=}")
#print(f"{token_replace=}")


#print("ASR - 平均 WER = ", ASR_wer_avg)
#print("LM - 平均 WER = ", LM_wer_avg)









