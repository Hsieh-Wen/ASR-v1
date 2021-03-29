#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:55:59 2020

@author: c95hcw
"""
import numpy as np
from matplotlib import pyplot as plt 
import re

def accuracy_plot(epoch_data,learning_rate_data,loss_data,result_path):

    x = np.zeros(len(epoch_data)) 
    y = np.zeros((2,len(epoch_data))) 
    for idx in range(len(epoch_data)):
        x[idx] = epoch_data[idx]
        y[0,idx]= float(learning_rate_data[idx])
        y[1,idx]= float(loss_data[idx]) 
        
    plt.title("Model loss and lr") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
#    plt.plot(x, y[0,:], 'b', x, y[1,:], 'r', x, y[2,:], 'g')
#    plt.plot(x, y[0,:], 'b', x, y[1,:], 'r')
    
    l1 = plt.plot(x, y[0,:], color='blue', label='learning_rate_data')
    l2 = plt.plot(x, y[1,:], color='red', label='loss_data')
    #設置圖例
    plt.legend(loc='lower left')   
    plt.savefig(result_path)
    plt.show()



# read txt 
log_path = "/home/c95hcw/"
txt_name = "data_tw_v1_bert.txt"#"ft_V1_roberta_wwm_ext2.txt"#"ft_V1_roberta_wwm_ext2.txt"    
fp = open(log_path + txt_name , "r")
#lines=[line.decode('utf-8') for line in fp.readlines()]#, encoding='big5'
#for i in fp:
#    print(i)
lines = fp.readlines()
#fp.close() 

training_data = []

for context in lines:
    if context[0:8] == "{'loss':":
        training_data.append(context)
        
training_data = sorted(training_data,reverse=True ) 

loss_data = []
learning_rate_data = []
epoch_data = []

# 逐行讀取檔案內容，直至檔案結尾
for data in training_data:    
    all_data=data.split(",",2) # separet wave_name and label
    loss_data.append(all_data[0].split(":",1)[1])
    learning_rate_data.append(all_data[1].split(":",1)[1])
    epoch_data.append(all_data[2].split(":",1)[1][0:-2])    
    
save_fig_name = re.sub(".txt",".png",txt_name)   
result_path = log_path + save_fig_name

accuracy_plot(epoch_data,learning_rate_data,loss_data,result_path)