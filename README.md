# ASR-v1
   * [ASR-v1](#asr-v1)
      * [說明](#說明)
      * [語音/語言模型訓練資料集前處理](#語音語言模型訓練資料集前處理)
         * [語音模型訓練資料集前處理](#語音模型訓練資料集前處理)
         * [語言模型訓練資料集前處理](#語言模型訓練資料集前處理)
         * [模型資料格式](#模型資料格式)
            * [MASR 資料格式](#masr-資料格式)
            * [Quartznet 資料格式](#quartznet-資料格式)
            * [Espnet 資料格式](#espnet-資料格式)
            * [BERT 資料格式](#bert-資料格式)
            * [BERT-wwm](#bert-wwm)
      * [語音/語言辨識](#語音語言辨識)

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

## 語音/語言模型訓練資料集前處理
### 語音模型訓練資料集前處理
* 語音模型(`MASR`、`Quartznet`、`Espnet` )的 資料前處理
    主要分為兩步驟: 
    - Step 0 : 將 csv 檔案資料作預處理
        包括:    
        - 確認音檔路徑是否存在
        - 計算音檔時間
        - 去除音檔內容之空格、空白行或TAB
        - 濾除音檔內容之標點符號
        - 分割音檔為 train / valid
        - 擴增/縮減資料集
        - 存取 train.csv / valid.csv 至 Stage 1 資料夾
    - Step 1 : 將 csv 檔案資料轉換為特定模型的訓練資料格式
        包括:
        - MASR 訓練資料格式
        - Quartznet 訓練資料格式
        - Espnet 訓練資料格式
### 語言模型訓練資料集前處理
* 兩種模型(`BERT` 、 `BERT-wwm`)的 資料前處理
    主要分為兩步驟: 
    - Step 0 : 將 csv 檔案資料作預處理
        包括:    
        - 去除檔案內容之空格、空白行或TAB
        - 去除檔案中相同之句子
        - 去除檔案中字數過少之句子
        - 分割檔案為 train / valid
        - 存取 train.csv / valid.csv 至 Stage 1 資料夾)
    - Step 1 : 將 csv 檔案資料轉換為特定模型的訓練資料格式
        包括:
        - BERT 訓練資料格式
        - BERT-wwm 訓練資料格式
### 模型資料格式
#### MASR 資料格式
詳細程式說明： [MASR Dataset](https://hackmd.io/ZVLiom3pQSaIoDs_Ff74fQ?view#MASR-Dataset)
* train.txt
```python
./Dataset/waves/masr_org_data/S0191/train/S0191/BAC009S0191W0322.wav,而自二零一四年九月開展機器人換人專項資金申請以來,6.4549375
./Dataset/waves/masr_org_data/S0157/train/S0157/BAC009S0157W0374.wav,這個事件被認為是組委會決定取消會徽的關鍵點,5.5690625
./Dataset/waves/masr_org_data/S0100/train/S0100/BAC009S0100W0201.wav,煤礦安全整治的力度加大,3.38
```
* valid.txt
```python
./Dataset/waves/Mozilla_1090622/wav/common_voice_zh-TW_20146021.wav,只是成就上位者的一點點薄弱的光環,4.224
./Dataset/waves/waves/Mozilla_1090622/wav/common_voice_zh-TW_21329105.wav,是不是很有可能是假的,3.696
```
* vocabulary.json
```python
["_", "的", "一", "是", "有", "在", "個", "這", "我", "不", "人", "了", "中", "國", "十", "們", "上", "會", "為", "大", "要", "二", "來", "年", "以", "到", "市", "業", "他", "三", "能", "出", "發", "時", "對", "分", "公", "就", "成", "行", "地", "也", "報", "家", "後", "下", "長", "現", "可", "那", "之", "多", "所", "五", "新", "和", "百", "部", "場", "資", "方", "好", "產", "開", "前", "機", "者", "生", "都", "經", "點", "事", "本", "房", "日", "還", "月", "於", "動", "過", "四", "將", "用", "面", "作", "你", "沒", "說", "比",,.....]
```
####  Quartznet 資料格式
詳細程式說明: [Quartznet Dataset](https://hackmd.io/ZVLiom3pQSaIoDs_Ff74fQ#Quartznet-Dataset)
* train.json
```python
{"audio_filepath": "./Dataset/waves/masr_org_data/S0191/train/S0191/BAC009S0191W0322.wav", "duration": 6.4549375, "text": "而自二零一四年九月開展機器人換人專項資金申請以來"}
{"audio_filepath": "./Dataset/waves/masr_org_data/S0157/train/S0157/BAC009S0157W0374.wav", "duration": 5.5690625, "text": "這個事件被認為是組委會決定取消會徽的關鍵點"}
{"audio_filepath": "./Dataset/waves/masr_org_data/S0100/train/S0100/BAC009S0100W0201.wav", "duration": 3.38, "text": "煤礦安全整治的力度加大"}
```
* valid.json
```python
{"audio_filepath": "./Dataset/waves/Mozilla_1090622/wav/common_voice_zh-TW_20146021.wav", "duration": 4.224, "text": "只是成就上位者的一點點薄弱的光環"}
{"audio_filepath": "./Dataset/waves/Mozilla_1090622/wav/common_voice_zh-TW_21329105.wav", "duration": 3.696, "text": "是不是很有可能是假的"}
```
* quartznet15x5-Zh.yaml
```python
vocabulary: [' ', '''',"的", "一", "是", "有", "在", "個", "這", "我", "不", "人", "了", "中", "國", "十", "們", "上", "會", "為", "大", "要", "二", "來", "年", "以", "到", "市", "業", "他", "三", "能", "出", "發", "時", "對", "分", "公", "就", "成", "行", "地", "也", "報", "家", "後", "下", "長", "現", "可", "那", "之", "多", "所", "五", "新", "和", "百", "部", "場", "資", "方", "好", "產", "開", "前", "機", "者", "生", "都", "經", "點", "事", "本", "房", "日", "還", "月", "於", "動", "過", "四", "將", "用", "面", "作", "你", "沒", "說", "比",.....]
```
#### Espnet 資料格式
espnet dataset主要分成兩個部份，
Step1 : icrd stage0 format -> aishell dataset format
Step2 : aishell dataset format -> espnet traning file
詳細程式說明： [Espnet dataset](https://hackmd.io/8cQFuHIoQLmz4GuXZi4LxA?view#Train)

- Input
經過data_preprocess生成的資料夾（必須包含stage0資料夾），
給定資料庫的位置，`dataset_root = "./Dataset/data_t1"`

- Output
```python=
dataset_root = "./Dataset/data_t1
espnet_data_process = EspnetDataProcess(dataset_root)
espnet_data_process.espnet_dataset()
```
會生成出
1. data
2. transcript
3. wav
三個資料夾，之後的train會用到data資料夾。
#### BERT 資料格式
詳細程式說明: [BERT Dataset](https://hackmd.io/We15sTLlTciUqQoB7_PRUQ#BERT-Dataset)
* train.txt
```python            
上半年上海商品住宅成交量
比去年同期水平上漲
南海網記者從澄邁縣公安局了解到
```                       
* valid.txt
```python            
她直覺認為別再待下去比較好
也太冷清了
買了串燒和鯛魚燒
```  
#### BERT-wwm
詳細程式說明:[BERT-wwm Dataset](https://hackmd.io/We15sTLlTciUqQoB7_PRUQ#BERT-wwm-Dataset)
* train.txt
```python            
銀行短期理財有風險嗎
就像在喀山世錦賽時一樣
你一定要把流裏流氣的毛病都改了
在父親的影響下十歲開始接受排球訓練
以鞏固如今世界排名第三的位置
他都感覺非常意外和高興
```     
* train_ref.txt (train.txt 的 分詞對照表)
```python            
[2, 4, 6, 9] --> 銀行 短期 理財 有風險嗎
[5, 7, 8, 11] --> 就像在喀山 世錦 賽 時一樣
[3, 9, 12] --> 你一定 要把流裏流氣 的毛病 都改了
[3, 6, 11, 13, 15, 17] --> 在父親 的影響 下十歲開始 接受 排球 訓練
[3, 5, 7, 9, 11, 14] --> 以鞏固 如今 世界 排名 第三 的位置
[4, 6, 8, 11] --> 他都感覺 非常 意外 和高興
```  
* valid.txt
```python            
是很迷人的所以
有有請主委
呃提出質詢哪因為當天喔
我們看一下當天的訊息齁從五號、六號、七號喔，有從幾百，後來到上千，將近有兩千
在我感覺起來非常準確的時間
我希望用嚴謹的態度來面對好不好
```  
* valid_ref.txt (valid.txt 的 分詞對照表)
```python            
[4, 7]  --> 是很迷人 的所以
[5] --> 有有請主委
[3, 5, 8, 10] --> 呃提出 質詢 哪因為 當天 喔
[2, 5, 7, 10, 12, 14, 17, 20, 26, 29, 32, 35, 38] 
    --> 我們 看一下 當天 的訊息 齁從 五號 、六號 、七號 喔，有從幾百 ，後來 到上千 ，將近 有兩千
[4, 6, 8, 10, 13] --> 在我感覺 起來 非常 準確 的時間
[3, 6, 9, 12, 14] --> 我希望 用嚴謹 的態度 來面對 好不好
```  
* special_word.txt (專有名詞)
```python            
日本東京大學
國立臺灣大學
本院
研製
監控系統
...
```  
## 語音/語言辨識
* 合併三種ASR模型(`MASR`、`Quartznet`、`Espnet`)與LM(`BERT`)的 inference.py
    主要分為兩步驟以及兩個工具: 
    - ASR Inference : `ASR_inference_stage0.py` 
       (Step 0)以 ASR 模型 inference 測試資料集(csv 檔案)
        ASR 模型包括:    
        - MASR 
        - Quartznet 
        - Espnet 
            * with LM
            * without LM
        - Google ASR
    - LM Infernce : `ASR_inference_stage1_LM.py`
       (Step 1)以 LM 模型 inference 測試資料集(csv 檔案)
        LM 模型包括:    
        - BERT
        - BERT - wwm
    - Save inference result tool : `Save_Inference_Result.py`
        (Save Tool)存取 ASR 或 LM 的 inference 結果
        - result_id_xxxx.csv : 存取每次 inference 的結果(包含 labels ,predict result, wer)
        - result.json : 存取每次執行 inference 的參數值(可視化檔案)
        - result_id_xxxx.npz :  存取每次執行 inference 的參數值(非可視化檔案)
    - Compare model iference result :`ASR_inference_html.py`
        (Html tool)比較不同模型、不同inference參數的 inference 結果，並輸出 Html 檔
