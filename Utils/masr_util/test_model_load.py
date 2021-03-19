#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:13:09 2021

@author: c95hcw
"""

import sys
sys.path.append("./model_utils/masr_util/")
import json
from models.conv import GatedConv

def path2vocabulary(json_path):
    with open(json_path) as f:
        vocabulary = json.load(f)
        vocabulary = "".join(vocabulary)
    return vocabulary




#model_path = "/home/c95hcw/ASR/ASR_models/MASR_v1.2.2/model_ma.pth"
model_path = "/home/c95hcw/ASR/ASR_models/asr_training_data_v2_Corpus1/Corpus1_aishell/model_epo159.pth"
#model_path = "/home/c95hcw/ASR/ASR_models/asr_training_data_v2_Corpus1/fine_tune_corpus2/model_epo165.pth"


new_voca_path = "/home/c95hcw/ASR/Dataset/data/training_data/asr_training_data_v2/Corpus2/MASR/combined_train_vocab.json"
new_vocabulary = path2vocabulary(new_voca_path)

save_org_model_path = "org_model_data.txt"
save_new_model_path = "new_model_data.txt"

original_model = GatedConv.load(model_path)
original_model.to("cuda")
new_model = GatedConv.load_with_new_vocabulary(model_path, new_vocabulary)
new_model.to("cuda")

#print(f"Original model structure: {original_model}")
#print(f"New model structure: {new_model}")

org_state_dict = original_model.state_dict()
org_model_data = org_state_dict.items()

new_state_dict = new_model.state_dict()
new_model_data = new_state_dict.items()

with open(save_org_model_path, "w") as f_org:
    f_org.write( str(org_model_data) )

with open(save_new_model_path, "w") as f_new:
    f_new.write( str(new_model_data) )
#
#list1 = new_model_data.readlines()
#list2 = str(new_model_data)

wav_path = "/home/c95hcw/ASR/Dataset/waves/masr_org_data/S0002/train/S0002/BAC009S0002W0122.wav"
text = "而對樓市成交抑制作用最大的限購"
text_org_model = original_model.predict(wav_path, "gpu") 
text_new_model = new_model.predict(wav_path, "gpu") 
print(f"{model_path= }")
print(f"{text= }")
print(f"{text_org_model= }")
print(f"{text_new_model=}")


print(f"-"*50)
corpus1_model_path = "/home/c95hcw/ASR/ASR_models/asr_training_data_v2_Corpus1/Corpus1_aishell/model_epo159.pth"
corpus2_model_path = "/home/c95hcw/ASR/ASR_models/asr_training_data_v2_Corpus1/fine_tune_corpus2/model_epo165.pth"

corpus1_model = GatedConv.load(corpus1_model_path)
corpus2_model = GatedConv.load(corpus2_model_path)
#all_vocabulary = "/home/c95hcw/ASR/ASR_models/MASR_v1.2.2/train_data_v2.json"
#corpus2_model = GatedConv.load_with_new_vocabulary(corpus2_model_path, all_vocabulary)

corpus1_model.to("cuda")
corpus2_model.to("cuda")


wav_path_c1 = "/home/c95hcw/ASR/Dataset/waves/masr_org_data/S0186/train/S0186/BAC009S0186W0397.wav"
text1 = "蘇炳添目前已經是中國接力隊的新靈魂人物"

wav_path_c2 = "/home/c95hcw/ASR/Dataset/waves/morning_conference/1_1080114audio/wav/M000000.wav"
text2 = "承辦單位報告所長好報告所長回報人數"

wav_path_c3 = "/home/c95hcw/ASR/Dataset/waves/y_speech_label/waves/clip6/ytb_000175.wav"
text3 = "我們研究完以後我們也會跟委員報告"


corpus1_model_pred_text1 = corpus1_model.predict(wav_path_c1, "gpu") 
corpus1_model_pred_text2 = corpus1_model.predict(wav_path_c2, "gpu") 
corpus1_model_pred_text3 = corpus1_model.predict(wav_path_c3, "gpu") 

print(f"training_data = corpus1")
print(f"{text1= }")
print(f"{corpus1_model_pred_text1= }")
print(f"{text2= }")
print(f"{corpus1_model_pred_text2= }")
print(f"{text3= }")
print(f"{corpus1_model_pred_text3= }")


print(f"-"*50)

corpus2_model_pred_text1 = corpus2_model.predict(wav_path_c1, "gpu") 
corpus2_model_pred_text2 = corpus2_model.predict(wav_path_c2, "gpu") 
corpus2_model_pred_text3 = corpus2_model.predict(wav_path_c3, "gpu") 

print(f"training_data = corpus2")
print(f"{text1= }")
print(f"{corpus2_model_pred_text1= }")
print(f"{text2= }")
print(f"{corpus2_model_pred_text2= }")
print(f"{text3= }")
print(f"{corpus2_model_pred_text3= }")