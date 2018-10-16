# -*- coding:utf8 -*-
import  numpy as np
import  re
import tensorflow as tf
from sklearn.metrics import  hamming_loss,accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
import pymysql
from read_config import config
import pandas as pd
from mini_batch import mini_batch
from  collections import  Counter
from sklearn import metrics
import pickle
from sklearn.utils import compute_sample_weight,compute_class_weight
import matplotlib.pyplot as plt
# def get_metrics(true_labels,predicted_labels):
#     #输出评价指标


def compute_metric(y_true,y_predict):
    columns = y_true.shape[1]
    accuracy = []
    precision = []
    recall = []
    for i in range(columns):
        accuracy.append(metrics.accuracy_score(y_true[:,i],y_predict[:,i]))
        precision.append(metrics.precision_score(y_true[:,i],y_predict[:,i]))
        recall.append(metrics.recall_score(y_true[:,i],y_predict[:,i]))
    accuracy = np.mean(accuracy)
    precision = np.mean(precision)
    recall = np.mean(recall)
    return accuracy,precision,recall

def text_filter(text):
    """
    process one news text
    :param text: content of one news
    :return: [text]
    """
    if isinstance(text,str):
        pass
    else:
        # text = ' '.join(text)
        text = ' '.join([str(each) for each in text if each])
    pattern_map = re.compile(r'<div class="cmsMap".*?</div>|<div class="cmsMapCaption".*?</div>')
    text = re.sub(pattern_map, '', text)
    pattern_others = re.compile(r'<.*?>|&nbsp|&amp|&quot|&lt|&gt')
    text = re.sub(pattern_others, ' ', text)
    return [text]

def tags_filter():
    conn = pymysql.connect(host=config.get('write_mysql', 'host'),
                                        user=config.get('write_mysql', 'user'),
                                        passwd=config.get('write_mysql', 'passwd'),
                                        port=int(config.get('write_mysql', 'port')),
                                        charset="utf8",
                                        binary_prefix=True)
    conn.select_db(config.get('write_mysql', 'db'))
    query_tags = 'SELECT tags.tag ' \
                'FROM tags'
    cur = conn.cursor()
    cur.execute(query_tags)
    tags_all = cur.fetchall()
    tags = []
    for i in range(len(tags_all)):
        tags_tmp = tags_all[i][0]
        tags.append(tags_tmp)
    tags_count =dict(Counter(tags))
    conn.close()
    tags_train = []
    for k,v in tags_count.items():
        if v>25:
            tags_train.append(k)
    np.save('./tags.npy',tags_train)
    return tags_train,tags_count

def get_auc_threshold(fpr,tpr,thresholds):
    """
    :param fpr:
    :param tpr:
    :param thresholds:
    :return:
    """
    th = tpr - fpr
    best_threshold = thresholds[list(th).index(max(th))]
    return best_threshold

def get_pr_threshold(p,r,threshold):
    """
    use p-r to get theshold
    :param p:
    :param r:
    :param threshold:
    :return:best_threshold
    """
    th = p-r
    best_threshold = threshold[list(th).index(th == 0)]
    return best_threshold
#----shujuyuan minibatch de classweight---
def get_weight_matrix(batch_size=500):
    tags = np.load('./tags.npy', 'r')
    with mini_batch(tags=tags) as mb:
        length = int(np.ceil(len(mb.id_list) / batch_size))
        y_data= pd.DataFrame()
        classes = 2
        for i in range(length):
            mb.get_ordered_ids(batch_size, i * batch_size)
            x_test_batch, y_test_batch = mb.load_mini_batch()
            y_data = pd.concat([y_data, pd.DataFrame(y_test_batch)], ignore_index=True)
            print('batch:',i)
        # tags_num = y_data.shape[1]
        tags_num = len(tags)
        # sample_num = y_data.shape[0]
        weight_matrix = np.zeros(shape=(2,tags_num))
        recip_freq = compute_sample_weight('balanced', y_data)
        recip_freq = np.vstack(recip_freq).T
        for i in range(tags_num):
            recip_freq = np.bincount(y_data.ix[:,i])/len(y_data.ix[:,i].values)

            # if recip_freq.shape[1] == classes:
            #     weight_matrix_column = y_data.ix[:,i].replace({0:recip_freq[1],1:recip_freq[0]}).tolist()
            # else:
            #     weight_matrix_column = 1
            # weight_matrix .append(weight_matrix_column)
            weight_matrix[:,i] = recip_freq
            print('tag:',i)
        return weight_matrix
#----shujuyuan minibatch de sampleweight---
def get_sample_matrix(batch_size = 500):
    tags = np.load('./tags.npy', 'r')
    with mini_batch(tags=tags) as mb:
        length = int(np.ceil(len(mb.id_list) / batch_size))
        y_data = pd.DataFrame()
        classes = 2
        for i in range(length):
            mb.get_ordered_ids(batch_size, i * batch_size)
            x_test_batch, y_test_batch = mb.load_mini_batch()
            y_data = pd.concat([y_data, pd.DataFrame(y_test_batch)], ignore_index=True)
            print('batch:', i)
        # tags_num = y_data.shape[1]
        tags_num = len(tags)
        # sample_num = y_data.shape[0]
        sample_weight = compute_sample_weight('balanced',y_data)
        sample_weight = sample_weight[:,np.newaxis]
        class_weight = []
        for i in range(tags_num):
            class_weight_tag = compute_class_weight('balanced',[0,1],y_data.ix[:,i])
            class_weight.append(class_weight_tag)
        class_weight = np.vstack(class_weight).T
        return class_weight,sample_weight
# ----shujuyuan dataset de classweight---
def get_class_weight(y_output):
    #news = pd.read_csv(file,encoding='utf-8',header=None)
    #tags = news.ix[:,1:].as_matrix()
    class_weight = []
    tags_num = y_output.shape[1]
    for i in range(tags_num):
        sample_weight = compute_class_weight('balanced',[0,1],y_output[:,i])
        class_weight.append(sample_weight)
    class_weight = np.vstack(class_weight).T
    return class_weight
def get_sample_weight(y_output):
    #news = pd.read_csv(file,encoding='utf-8',header=None)
    #tags = news.ix[:,1:].as_matrix()
    class_weight = []
    sample_weight = compute_sample_weight('balanced',y_output)
    sample_weight = sample_weight[:,np.newaxis]
    return sample_weight
def split_shuffle_csv(file,ratio):
    df = pd.read_csv(file, encoding='utf-8',header=None)
    # df.drop_duplicates(keep='first', inplace=True)
    df = df.sample(frac=1.0,axis=0)
    cut_idx = int(round(ratio * df.shape[0]))
    df_train, df_test = df.iloc[:cut_idx], df.iloc[cut_idx:]
    return df_train,df_test
def evaluate_model(x_test,y_test,tags_name,theta,threshold,tfidfvector):
    #text = text_filter(x_test)
    text_array = tfidfvector.transform(x_test).todense()
    text_array = np.c_[np.ones(len(x_test)), text_array]
    y_score = 1 / (1 + np.exp(-np.matmul(text_array, theta)))
    # y_score = y_score.flatten()
    y_con = np.array(y_score) - np.array(threshold).reshape((1,292))
    y_con[y_con>0]=1
    y_con[y_con<0]=0
    #metric
    accuracy, precision ,recall = compute_metric(y_test.values,y_con)
    return accuracy,precision,recall
# weight_matrix= get_weight_matrix()
# pd.to_pickle(weight_matrix,'./weight_matrix.pkl')
# a = pd.read_pickle('./class_weight.pkl')
# print(a)
# class_weight,sample_weight = get_sample_weight()
# pd.to_pickle(sample_weight,'./sample_weight.pkl')
#------compute class weight ----
# news = pd.read_csv('./data/news.train',encoding='utf-8',header=None)
# tags = news.ix[:,1:].as_matrix()
# class_weight = get_class_weight(tags)
# pd.to_pickle(class_weight,'./data/class_weight.pkl')
#----Done----
#------compute sample weight ----
# news = pd.read_csv('./data/news.train',encoding='utf-8',header=None)
# tags = news.ix[:,1:].as_matrix()
# sample_weight = get_sample_weight(tags)
# pd.to_pickle(sample_weight,'./data/sample_weight.pkl')
#----Done----

# tags_train ,tags_count = tags_filter()
# tag_label = list(tags_count)
# tag_num = list(tags_count.values())
# idx =len(tag_label)
# plt.bar(idx,tag_num)
# plt.show()
# news = pd.read_csv('./data/all_news.csv',encoding='utf-8',header=None)
# print(news)
#-----spilit csv --
# df_train,df_test = split_shuffle_csv('./data/all_news.csv',0.8)
# df_train.to_csv('./data/news_train.train',encoding='utf-8',index=False,header=False)
# df_test.to_csv('./data/news_train.test',encoding='utf-8',index=False,header=False)
#----done--------
# da = pd.read_csv('./data/news_train.train',encoding='utf-8',header=None)
# print('a')
#------evaluate model------
# tags = np.load('./data/tag_list.npy','r')
# theta = np.load('./theta_weight/epoch_theta29.npy')
# auc= pd.read_pickle('./AUC_PR_Thresholds/auc_threshold.pkl')
# threshold = auc['auc_threshold']
# tfidfvector = np.load('./data/tfidf.pkl')
# news = pd.read_csv('./data/news.test',encoding='utf-8',header=None)
# x_test = news.ix[:,0]
# y_test = news.ix[:,1:]
# a = evaluate_model(x_test,y_test,tags,theta,threshold,tfidfvector)
# print(a)
#----Done------
