#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:22:47 2021

@author: c95hcw
"""
import json

corpus1_path = "/home/c95hcw/ASR/Dataset/data/raw_data/vocab/aishell_fan_time_vocab.json"
corpus2_path = "/home/c95hcw/ASR/Dataset/data/raw_data/vocab/combined_AI_morn_mozilla_vocab.json"



with open(corpus1_path) as json_file:
   corpus1_data = json.load(json_file)
   corpus1_data = "".join(corpus1_data)
with open(corpus2_path) as json_file:
    corpus2_data = json.load(json_file)   
    corpus2_data = "".join(corpus2_data)
    
add_word = []
for word in corpus2_data:
    word_is_none=corpus1_data.find(word)
    if word_is_none==-1:
        add_word.append(word)
 
print(f"Copus 1 字數: {len(corpus1_data)}個字.")
print(f"Copus 2 字數: {len(corpus2_data)}個字.")
       
print(f"Copus 1 比 Corpus2 少了 {len(add_word)}個字.")
        