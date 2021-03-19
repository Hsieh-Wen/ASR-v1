#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:55:06 2021

@author: c95csy
"""
import pandas as pd
from ckiptagger import data_utils, WS, construct_dictionary
from pycorrector.utils.text_utils import is_chinese_string, convert_to_unicode
import numpy as np

data_path = "pycorrector/inference_Morning_ASR.csv"

# Open the CSV file for reading
asr_data =pd.read_csv(open(data_path))
asr_truth = asr_data.Truth
asr_pred = asr_data.ASR_Predict
lm_correct = asr_data.LM_correct

#bert   bert-base-chinese
#RoBERTa-wwm-ext-large	hfl/chinese-roberta-wwm-ext-large
#RoBERTa-wwm-ext	hfl/chinese-roberta-wwm-ext
#BERT-wwm-ext	hfl/chinese-bert-wwm-ext
lm_path = "hfl/chinese-roberta-wwm-ext"

LM_model = bert_corrector.BertCorrector(bert_model_dir = lm_path)
#correct_sent, err = LM_model.bert_correct(asr_pred[i], use_confusion_word=False)

text = asr_pred[0]

from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained(lm_path)
model_lm = BertForMaskedLM.from_pretrained(lm_path)

def init_bertlm_model(lm_path="hfl/chinese-roberta-wwm-ext", method="eval"):
    tokenizer = BertTokenizer.from_pretrained(lm_path)
    model_lm = BertForMaskedLM.from_pretrained(lm_path)
    
    if method == "eval":
        model_lm.eval()
    
    return model_lm, tokenizer
    
def make_result(predict, index, mask_index):
    
    result = []
    
    predict = predict.tolist()
    index = index.tolist()
    mask_index = mask_index.tolist()
    
    predict = list(zip(*predict))
    index = list(zip(*index))
    
    for p, i in zip(predict, index):
        result.append({"score" : p, 
                       "token" : i, 
                       "token_str" : tokenizer.convert_ids_to_tokens(i),
                       "mask_index" : mask_index})
        
    return result
    
def lm_predict(sentence, topk=5, device="cuda"):
    if device == "cuda":
        token = tokenizer(sentence, return_tensors="pt")
    else:
        token = tokenizer(sentence)
    
    with torch.no_grad():
        input_ids = token["input_ids"]
        mask_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
        prediction = model_lm(**token)[0]
        
        predict, index = torch.topk(torch.softmax(prediction[0], dim=1)[mask_index], topk)
        
        result = make_result(predict, index, mask_index)        
        
        return result
    
    
sentence = "胖虎叫大雄[MASK]去買[MASK]畫"

token = tokenizer(sentence, return_tensors="pt")

model_lm.eval()
with torch.no_grad():
    input_ids = token["input_ids"]
    mask_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    prediction = model_lm(**token)[0]
    
    predict, index = torch.topk(torch.softmax(prediction[0], dim=1)[mask_index], 5)
    
    result = {""}
text_new = ''
details = []


#@staticmethod
def add_word_to_dictionary(path_of_dict_txt):
    d = {}
    with open(path_of_dict_txt) as f:
        for line in f:
            key = line.split("\n")[0]
            d[str(key)] = 1
    
    print(d)
    dictionary = construct_dictionary(d)
    return dictionary

def init_chkip(path_of_dict_txt):
    
    dictionary = add_word_to_dictionary(path_of_dict_txt)
    ws = WS("./Ckip_Tagger/data/")
    
    return ws, dictionary

ws, dictionary = init_chkip(path_of_dict_txt="special_word.txt")


blocks = LM_model.split_text_by_maxlen(text)
topk = 5
use_confusion_word = False
details = []

for blk, start_idx in blocks:
    
    word_sentence_list = LM_model.ws([blk], recommend_dictionary=LM_model.dictionary)[0]
    print(word_sentence_list)
    
    blk_new = ''
    sentence = blk
    sentence_lst = word_sentence_list.copy()
    
    for idx, s in enumerate(word_sentence_list):
        
        if is_chinese_string(s):
            sentence_lst[idx] = LM_model.mask*len(s)
            sentence_new = "".join(sentence_lst)
            
            # 预测，默认取top5
            predicts = LM_model.lm_predict(sentence_new, topk=topk)
            top_tokens = []
            
            for t in predicts:
                top_tokens.append(t["token_str"])
            top_tokens = list(zip(*top_tokens))
            
            for i, ss in enumerate(s):
                
                if top_tokens and (ss not in top_tokens[i]):
                    if use_confusion_word:
                        pass
                    
                    else:
                        token_str = top_tokens[i][0]
                        details.append([ss, token_str, start_idx + idx + i, start_idx + idx + i + 1])
                        print(s)
                        s = s[:i] + token_str + s[i+1:]
                        print(s, top_tokens, i)
                        
        blk_new += s
        sentence_lst[idx] = s
    
    text_new += blk_new
                    
for blk, start_idx in blocks:
    blk_new = ''
    for idx, s in enumerate(blk):
        # 处理中文错误
        if LM_model.is_chinese_string(s):
            sentence_lst = list(blk_new + blk[idx:])
            sentence_lst[idx] = self.mask
            sentence_new = ''.join(sentence_lst)
            # 预测，默认取top5
            predicts = self.model(sentence_new)
            top_tokens = []
            for p in predicts:
                token_id = p.get('token', 0)
                token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
                top_tokens.append(token_str)
                break

            if top_tokens and (s not in top_tokens):
                if use_confusion_word:
                    # 取得所有可能正确的词
                    candidates = self.generate_items(s)
                    if candidates:
                        for token_str in top_tokens:
                            if token_str in candidates:
                                details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                                s = token_str
                                break
                            
                else:
                    token_str = top_tokens[0]
                    details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                    s = token_str
                        
        blk_new += s
    text_new += blk_new
details = sorted(details, key=operator.itemgetter(2))



for idx, s in enumerate(aam):
    print(f"{idx=}  {s=}")
    
for idx, s in enumerate(aam[::-1]):
    print(f"{idx=}  {s=}")
    
    
ASR - 平均 WER =  0.1405994684032765
LM - 平均 WER =  0.13027066836159668
























