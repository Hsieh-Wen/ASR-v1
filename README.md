# ASR-v1
[TOC]
## 說明
此為用於 語音/語言模型訓練資料集前處理 與 語音/語言辨識 之程式.

* 主要程式：
  - 模型訓練資料集前處理
    * 語音模型訓練資料集前處理 : 
      - 程式名 : Data_preprocess_ASR_stage0.py , Data_preprocess_ASR_stage1.py
      - 程式說明文件 : [Data preprocessing Code - ASR](https://hackmd.io/XwNQ5Xd0SmSdpPk5lhTWAw) 
    * 語言模型訓練資料集前處理 : 
      - 程式名 : Data_preprocess_LM_stage0.py, Data_preprocess_LM_stage1.py
      - 程式說明文件 : [Data preprocessing Code - LM](https://hackmd.io/CNW_oBL9SAat6FhE4HCXqA)
  - 語音/語言辨識
    * 語音辨識 : ASR_inference_stage0.py
    * 語言校正 : ASR_inference_stage1_LM.py
    - 程式說明文件 : [Inference Code (ASR/LM)](https://hackmd.io/W6j35008Q7inhn3LKNvrDQ)

