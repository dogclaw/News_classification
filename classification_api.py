# -*- coding:utf-8 -*-
import  numpy as np
import tensorflow as tf
from read_config import config
import pandas as pd
from classification_utils import text_filter

def classification_api(one_text_input):
    """
    :param text_input: 一条数据库新闻
    :return: tuple,[tag,y_score,y_con,y_normal_con]
    y_con = (y_score - thresh)/thresh ---tag's conffidence
    y_normal_con = (ycon - ycon.min )/ (ycon.max - ycon.min)
    """
    tags = np.load('./data/tag_list.npy','r')
    auc= pd.read_pickle('./AUC_PR_Thresholds/auc_threshold.pkl')
    theta = np.load('./theta_weight/epoch_theta29.npy')
    threshold = auc['auc_threshold']
    text = text_filter(one_text_input)
    vector = np.load('./data/tfidf.pkl')
    text_array = vector.transform(text).todense()
    text_array = np.c_[[1],text_array]
    y_score = 1/(1+np.exp(-np.matmul(text_array,theta)))
    #y_score = y_score.flatten()
    tags_score = []
    #取出来大于阈值的标签和score，排序后返回前三个
    y_diff = np.array(y_score )-np.array(threshold)
    y_con = y_diff/np.array(threshold)
    y_normal_con = (y_con-y_con.min())/(y_con.max() - y_con.min())
    for k,v in enumerate(tags):
        if y_diff[0,k]>0:
            tags_score.append((v,y_score[0,k],y_con[0,k],y_normal_con[0,k]))
    if tags_score:
        tags_score = sorted(tags_score,key=lambda x:x[3],reverse=True)
    return tags_score
f = open('./news.txt','r')
text=f.read()
tags = classification_api(text)
print(text)
print(tags)



