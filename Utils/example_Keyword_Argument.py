#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:04:45 2021

@author: c95hcw
"""
def fun(**_settings):
    print(_settings)




def inference_parameters(**kwargs):
#        print(kwargs)
        parameters_dict = kwargs
        return parameters_dict


def test_fun(**parameters):
        print(parameters)
#        data_dict = parameters
#        return data_dict

#fun(name='Sky', attack=100, hp=500)
parameters_dict = inference_parameters( wwm=True, use_confusion_word=True, reverse=True, token_replace=True)
test_fun(**parameters_dict)


kwargs = {'inference_mode':"xxx", 'inference_file': "xxxxx",
          'ASR_mode':"xxxxx", 'ASR_model_path':"xxxxx", 
          'ASR_wer_avg':"xxxxx", 'result_table':"xxxxx"}
    

# key,value to varabules
for k, v in  kwargs.items():
    print(k)
    print(v)
    exec("%s = %s" % (k, v))

    
