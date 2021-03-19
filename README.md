# ASR-v1

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
    ![](https://i.imgur.com/636ueOe.png)
####  Quartznet 資料格式
詳細程式說明: [Quartznet Dataset]()
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
![](https://i.imgur.com/CMjlIbd.png)

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
會將資料生成在Stage1/Espnet中
![](https://i.imgur.com/VEiILPH.png)
會生成出
1. data
2. transcript
3. wav
三個資料夾，之後的train會用到data資料夾。
* LM
#### BERT 資料格式
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
![](https://i.imgur.com/xPce3V8.png)


## 語音/語言辨識
