# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: use bert detect and correct chinese char error
"""

import operator
import os
import sys
import time
from transformers import pipeline, BertTokenizer, BertForMaskedLM
import torch
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#sys.path.append("./Utils/LM_BERT/")

from pycorrector.utils.text_utils import is_chinese_string, convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector

from ckiptagger import data_utils, WS, construct_dictionary

pwd_path = os.path.abspath(os.path.dirname(__file__))


import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2) #per_process_gpu_memory_fraction=0.2 --> 限定gpu使用量
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True # 不可佔滿 gpu，tf會預設先佔滿gpu ; True就不會佔滿
session = tf.compat.v1.Session(config=config)
# compat.v1:
# 因為tf有大改版過， tf=2.* 以去掉許多function ,所以 tf.compat.v1.連回初版來呼叫 ConfigProto 、 GPUOptions

class BertCorrector(Corrector):
    def __init__(self, bert_model_dir, topk=5, device="gpu"):
        super(BertCorrector, self).__init__()
        self.name = 'bert_corrector'
#        t1 = time.time()
#        self.model = pipeline('fill-mask',
#                              model=bert_model_dir,
#                              tokenizer=bert_model_dir,
#                              device=0)
#        if self.model:
#            self.mask = self.model.tokenizer.mask_token
#            logger.debug('Loaded bert model: %s, spend: %.3f s.' % (bert_model_dir, time.time() - t1))
            
        self.ws, self.dictionary = self.init_chkip(path_of_dict_txt="./Utils/LM_BERT/special_word.txt")
        
#        print(bert_model_dir)
        self.model_lm, self.tokenizer_lm = self.init_bertlm_model(bert_model_dir)
        
        self.mask = self.tokenizer_lm.mask_token
        self.topk = topk
        self.device = device
        self.gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.device == "gpu":
            self.model_lm.to(self.gpu_device)
        
#    def bert_correct_old(self, text, use_confusion_word=True):
#        """
#        句子纠错
#        :param text: 句子文本
#        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
#        """
#        text_new = ''
#        details = []
#        self.check_corrector_initialized()
#        # 编码统一，utf-8 to unicode
#        text = convert_to_unicode(text)
#        # 长句切分为短句
#        blocks = self.split_text_by_maxlen(text, maxlen=128)
#        for blk, start_idx in blocks:
#            blk_new = ''
#            for idx, s in enumerate(blk):
#                # 处理中文错误
#                if is_chinese_string(s):
#                    sentence_lst = list(blk_new + blk[idx:])
#                    sentence_lst[idx] = self.mask
#                    sentence_new = ''.join(sentence_lst)
#                    # 预测，默认取top5
#                    predicts = self.model(sentence_new)
#                    top_tokens = []
#                    for p in predicts:
#                        token_id = p.get('token', 0)
#                        token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
#                        top_tokens.append(token_str)
#                        break
#
#                    if top_tokens and (s not in top_tokens):
#                        if use_confusion_word:
#                            # 取得所有可能正确的词
#                            candidates = self.generate_items(s)
#                            if candidates:
#                                for token_str in top_tokens:
#                                    if token_str in candidates:
#                                        details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
#                                        s = token_str
#                                        break
#                                    
#                        else:
#                            token_str = top_tokens[0]
#                            details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
#                            s = token_str
#                                
#                blk_new += s
#            text_new += blk_new
#        details = sorted(details, key=operator.itemgetter(2))
#        return text_new, details
    
    @staticmethod
    def init_bertlm_model(lm_path="hfl/chinese-roberta-wwm-ext", method="eval"):
        """
        initial bert mask model by transformer library
        mothod: eval
        """
#        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer_lm = BertTokenizer.from_pretrained(lm_path)
        model_lm = BertForMaskedLM.from_pretrained(lm_path)
        
        if method == "eval":
            model_lm.eval()
            
#        model_lm.to(device)
        
        return model_lm, tokenizer_lm
    

    
    def make_result(self, predict, index, mask_index):
        """
        (predict, index) -> result dict
        format {"score", "token_id", "token_str", "mask_index"}
        :param predict: tensor -> [topk] , topk -> [mask predicts score]
        :param index: tensor -> [topk] , topk -> [mask indexs]
        return
        """
        result = []
        
        predict = predict.tolist()
        index = index.tolist()
        mask_index = mask_index.tolist()
        
        predict = list(zip(*predict))
        index = list(zip(*index))
        
        for p, i in zip(predict, index):
            result.append({"score" : p, 
                           "token" : i, 
                           "token_str" : self.tokenizer_lm.convert_ids_to_tokens(i),
                           "mask_index" : mask_index})
            
        return result
        
    
    def lm_predict(self, sentence):
        """
        used bert mask model predict sentence [mask]
        :param sentence: sentence
        :return: result[topk] -> {"score", "token_id", "token_str", "mask_index"}
        """
        
        # cpu or gpu
        token = self.tokenizer_lm(sentence, return_tensors="pt")
        
        if self.device == "gpu":
            token.to(self.gpu_device)

        
        # predict
        with torch.no_grad():
            input_ids = token["input_ids"]
                     
            mask_index = torch.where(input_ids == self.tokenizer_lm.mask_token_id)[1]
            prediction = self.model_lm(**token)[0]

            
            # tensor type
            predict, index = torch.topk(torch.softmax(prediction[0], dim=1)[mask_index], self.topk)
            
            result = self.make_result(predict, index, mask_index)        
            
            return result
    
    
    @staticmethod
    def _add_word_to_dictionary(path_of_dict_txt):
        """
        add customize dictionary
        """
        d = {}
        with open(path_of_dict_txt) as f:
            for line in f:
                key = line.split("\n")[0]
                d[str(key)] = 1
        
#        print(d)
        dictionary = construct_dictionary(d)
        return dictionary

    def init_chkip(self, path_of_dict_txt, ws_path="./Utils/LM_BERT/Ckip_Tagger/data/"):
        """
        initial chkip hyphenation model
        :param path_of_dict_txt: customize dictionary
        :param ws_path: chkip lm model
        """
        dictionary = self._add_word_to_dictionary(path_of_dict_txt)
        ws = WS(ws_path, disable_cuda=False) # 
        
        return ws, dictionary
    
    def bert_correct(self, text, wwm, reverse=False, token_replace=True, use_confusion_word=False, topk=5):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        self.topk = topk
        self.use_confusion_word = use_confusion_word
        self.reverse = reverse
        self.token_replace = token_replace
        self.wwm = wwm
        
        text_new = ''
        details = []
        self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
#         长句切分为短句
#        blocks = self.split_text_by_maxlen(text, maxlen=128)
        blk = text
        
#        for blk, start_idx in blocks:
            
        # 是否使用wwm
        if self.wwm:
            # 利用chkip做斷詞
            # str -> list 
            word_sentence_list = self.ws([blk], recommend_dictionary=self.dictionary)[0]
#                print(word_sentence_list)
        else:
            word_sentence_list = list(str(blk))            
            
        _, details = self.predict_fc(word_sentence_list, details)
        
        blk_new = self.replace_predict_token(blk, details)

        text_new += blk_new
        
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details
   
             
    @staticmethod
    def replace_predict_token(blk, details):
        
        details = sorted(details, key=operator.itemgetter(2))
                
        for d in details:
            blk = blk[:d[2]] + d[1] + blk[d[3]:]
            
        return blk
            
        
    def predict_fc(self, word_sentence_list, details):
        """
        predict type
        :reverse:
        :token replace:
        """
        blk_new = ''
        sentence_lst = word_sentence_list.copy()
        # reverse
        if self.reverse:
            idxs = list(range(len(word_sentence_list)))
            for idx, s in zip(idxs[::-1], word_sentence_list[::-1]):
                
                predict_s, details = self.predict_sentence_mask(sentence_lst, s, details, idx)
                
                blk_new += s
                
                # 是否將bert predict word replace on sentence
                if self.token_replace:
                    sentence_lst[idx] = predict_s
                else:
                    sentence_lst[idx] = s
                
            return blk_new[::-1], details
        
        
        else:
            for idx, s in enumerate(word_sentence_list):
                
                predict_s, details = self.predict_sentence_mask(sentence_lst, s, details, idx)
                    
                blk_new += s
                
                # 是否將bert predict word replace on sentence
                if self.token_replace:
                    sentence_lst[idx] = predict_s
                else:
                    sentence_lst[idx] = s
                
            return blk_new, details
        
    def predict_sentence_mask(self, sentence_lst, s, details, idx):
        """
        利用bert mask lm預測欲mask的文字
        1. judge chinese token
        2. generate mask
        3. predict [MASK] token
        """

        # 判斷是否為中文
        if is_chinese_string(s):
            # idx : sentence list index, index : text str index
            # because wwm, idx != sentence len
            index = len("".join(sentence_lst[:idx]))
            
            sentence_new = self.generate_mask(sentence_lst, idx, s)
            s, details = self.predict_word(sentence_new, s, details, index)
            
        return s, details
    
    
    def generate_mask(self,sentence_lst, idx, s):
        """
        create mask on predict word
        """
        sentence_lst[idx] = self.mask*len(s)
        sentence_new = "".join(sentence_lst)
            
        return sentence_new
            
    def predict_word(self, sentence_new, s, details, index):
        """
        bert mask LM predict
        :param sentence_new: must have [mask] on predict word
        """
        # 预测，默认取top5
        predicts = self.lm_predict(sentence_new)
        top_tokens = []
        
        for t in predicts:
            top_tokens.append(t["token_str"])
        # s1 - stopk
        # [[s1_token1, s2_token2,...], [s1_token2, s2_token2...], ...] -> [[s1_token1, s1_token2,...], [s2_token1, s2_token2,...],...]
        top_tokens = list(zip(*top_tokens))
        
        for i, ss in enumerate(s):
            # s是否在predict tokens
            if top_tokens and (ss not in top_tokens[i]):
                # 是否使用相近詞
                if self.use_confusion_word:
                    # 取得所有可能正确的词
                    candidates = self.generate_items(s)
                    if candidates:
                        token_str = top_tokens[i][0]
                        details.append([ss, token_str, index + i, index + i + 1])
                        s = s[:i] + token_str + s[i+1:]
                
                else:
                    token_str = top_tokens[i][0]
                    details.append([ss, token_str, index + i, index + i + 1])
                    s = s[:i] + token_str + s[i+1:]
                    
        
        return s, details
        
if __name__ == "__main__":
    d = BertCorrector()
    error_sentences = [
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
    ]
    for sent in error_sentences:
        corrected_sent, err = d.bert_correct(sent)
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))















































