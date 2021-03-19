#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 9 14:41:15 2021
@author: c95hcw
"""

from  CSV_utils import read_inference_csv_file
import numpy as np
import json

def read_config(path):
    conf = configparser.ConfigParser()
    candidates = [path]
    conf.read(candidates)
    return conf


# Main Code
input_csv = "../Inference_Result/test/result_id_0000.csv"
label_list, predict_list, wer_list = read_inference_csv_file(input_csv)

# =============================================================================
#  Main Code
# =============================================================================
if __name__ == "__main__":
    path = "config_ui_json.ini"
    config = read_config(path)
   
    html_result = HtmlResult(config)
    html_result.load_data_html_flow()
