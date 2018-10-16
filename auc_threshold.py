# -*- coding:utf-8 -*-
import  numpy as np
import pandas as pd
from sklearn import  metrics
import tensorflow as tf
from datetime import  datetime
from classification_utils import *
from mini_batch import mini_batch
import  matplotlib.pyplot as plt
import  os
from sklearn.utils import compute_sample_weight,compute_class_weight

tag_list = np.load('data/tag_list.npy')

def parse_csv(line):
    example_defaults = [['']] + [[0.]] * len(tag_list)
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[0], shape=(1,))
    labels = tf.reshape(parsed_line[1:], shape=(len(tag_list),))
    return features, labels
def auc_pr_threshold():
    news = pd.read_csv('./data/news.train',encoding='utf-8',header=None)
    x_test_df = news.ix[:,0]
    y_test_df = news.ix[:,1:]
    tags_num = len(tag_list)
    tfidf_vectorizer= pd.read_pickle('./data/tfidf.pkl')
    theta = np.load('./theta_weight/epoch_theta29.npy')
    x_test = tfidf_vectorizer.transform(x_test_df)
    x_test = x_test.todense()
    x_test = np.c_[np.ones(len(x_test)), x_test]
    y_predict_logit = np.matmul(x_test,theta)
    y_score = 1/(1+np.exp(-y_predict_logit))
    auc_threshold_df = pd.DataFrame()
    pr_threshold_df = pd.DataFrame()
    for i in range(tags_num):
        tag_score = y_score[:,i]
        #auc and threshold
        fpr, tpr, thresholds = metrics.roc_curve(y_test_df.ix[:,i+1].values, y_score[:,i],pos_label=1,drop_intermediate=False)
        roc_auc_score = metrics.auc(fpr, tpr)
        tag_threshold_auc = get_auc_threshold(fpr,tpr,thresholds)
        auc_dict = pd.DataFrame({'AUC':roc_auc_score,'auc_threshold':tag_threshold_auc,'tag':tag_list[i]},index=[tag_list[i]])
        auc_threshold_df = pd.concat([auc_threshold_df,auc_dict],ignore_index=True)
        #precision and threshold
        # p, r, pr_thresholds = metrics.precision_recall_curve(y_test_df.ix[:, i + 1].values, y_score[:, i], pos_label=1)
        # tag_threshold_pr = get_pr_threshold(p, r, pr_thresholds)
        # y_pred = tag_score - tag_threshold_pr
        # y_pred[y_pred>0] = 1
        # y_pred[y_pred<0] = 0
        # tag_precision = metrics.precision_score(y_test_df.ix[:,i+1].values,y_pred,pos_label=1)
        # pr_dict = pd.DataFrame({'pr_precision':tag_precision,'pr_threshold':tag_threshold_pr,'tag':tag_list[i]},index=[tag_list[i]])
        # pr_threshold_df = pd.concat([pr_threshold_df,pr_dict],ignore_index=True)
    if not os.path.exists('./AUC_PR_Thresholds'):
        os.makedirs('./AUC_PR_Thresholds')
    pd.to_pickle(auc_threshold_df, './AUC_PR_Thresholds/auc_threshold.pkl')
    # pd.to_pickle(pr_threshold_df, './AUC/pr_threshold.pkl')
# auc_pr_threshold()

