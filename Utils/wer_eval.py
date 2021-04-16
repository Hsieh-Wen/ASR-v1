#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:50:51 2021

@author: c95hcw
"""

import numpy as np
import Levenshtein as Lev

def eval_asr(asr_truth,asr_predict_result):
    wer_list = []
    for i in range(len(asr_truth)):
      # cul wer
       asr_predict_result[i], asr_truth[i] = asr_predict_result[i].replace(" ", ""), asr_truth[i].replace(" ", "")
       wer = Lev.distance(asr_predict_result[i], asr_truth[i]) / float(len(asr_truth[i]))
#       print(f"{wer}")
       wer_list.append(wer)
    ASR_wer_avg = float(np.sum(wer_list)/len(wer_list))        
    return ASR_wer_avg, wer_list


