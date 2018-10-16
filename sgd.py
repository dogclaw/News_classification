# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import  datetime
import pandas as pd
import  csv
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  metrics
# import matplotlib.pyplot as plt
tag_list = np.load('data/tag_list.npy')

def parse_csv(line):
    example_defaults = [['']] + [[0.]] * len(tag_list)
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[0], shape=(1,))
    labels = tf.reshape(parsed_line[1:], shape=(len(tag_list),))
    return features, labels

if __name__ == '__main__':
    #tags = ['China']
    # load file and theta etc
    class_weight = pd.read_pickle('./data/class_weight.pkl')
    sample_weight = pd.read_pickle('./data/sample_weight.pkl')
    news = pd.read_csv('./data/all_news.csv',encoding='utf-8',header=None)
    m = len(tag_list)
    learning_rate = 0.01
    n_epochs = 100
    batch_size = 500
    regular_scaler = 0.5
    #tags_num = len(tags)
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=5,
                                 token_pattern=r"(?x)(?:[A-Z]\.)+|\d+(?:\.\d+)?%?|\w+(?:[-']\w+)*",
                                 stop_words=stopwords.words('english'), decode_error='ignore')
    if not os.path.exists('./data/tfidf.pkl'):
        tfidf = vectorizer.fit(news[0])
        pd.to_pickle(tfidf,'./data/tfidf.pkl')
    else:
        tfidf = pd.read_pickle('./data/tfidf.pkl')
    n_batches = round(news[0].shape[0] / batch_size)
    n = len(tfidf.idf_)
    print('x_clunms length:',n)
    #------Generate graph-------
    with tf.name_scope('placeholder'):
        x = tf.placeholder(tf.float32, shape=(None,n+1), name='x')
        y = tf.placeholder(tf.float32, shape=(None, m), name='y')
        weights = tf.placeholder(tf.float32,shape=(None,m),name='weights')
    with tf.name_scope('Variable'):
        theta = tf.Variable(tf.random_uniform([n+1, m], -1.0, 1.0), name='theta')
    #cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=y,logits=tf.matmul(x,theta),weights=weight)
    with tf.name_scope('Loss'):
        logits = tf.matmul(x,theta)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits = logits)
        #loss = tf.reduce_mean(cross_entropy)
        regular_loss = tf.reduce_mean(tf.abs(theta)**2)
        loss = tf.reduce_mean(tf.multiply(weights,cross_entropy) + regular_loss * regular_scaler)
    with tf.name_scope('training_op'):
        training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss',loss)
        #tf.summary.scalar('max_logits',tf.reduce_max(cross_entropy))
        #tf.summary.histogram('weights',weights)
        tf.summary.histogram('theta',theta)
        tf.summary.histogram('cross_entropy',cross_entropy)
        merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log',tf.get_default_graph())
    # 设置存储模型路径
    # ckpt_dir = './ckpt_dir'
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    # saver = tf.train.Saver()
    # non_storable_variable = tf.Variable(777)
    #---------use dataset import data---------
    dataset = tf.data.TextLineDataset('data/news.train')
    dataset = dataset.map(parse_csv)
    # shuffle data ,batch
    dataset = dataset.shuffle(10000).repeat().batch(500)
    iter = dataset.make_one_shot_iterator()
    data_iter = iter.get_next()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #重新加载已训练的模型，继续
        # ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     print('loading model',ckpt.model_checkpoint_path)
        #     saver.restore(sess,ckpt.model_checkpoint_path)
        #     print('load over')
        print(datetime.datetime.now())
        batch_step = 0
        epoch_step = 0
        for epoch in range(n_epochs):
            print('into epoch %d' % epoch)
            print(datetime.datetime.now())
            # ========mini batch=========指定迭代次数，利用minibatch进行梯度下降训练
            for batch_index in range(n_batches):
                #read data
                data = (sess.run(data_iter))
                content = data[0]
                x_batch = []
                n_content = len(content)
                for i in range(n_content):
                    x_tfidf = tfidf.transform(content[i])
                    x_batch.append(x_tfidf.todense())
                x_batch = np.c_[np.ones(len(x_batch)), np.vstack(x_batch)]
                y_batch = data[1]
                #caculate class_weight
                y_train_dt = pd.DataFrame(y_batch)
                y_batch_row = y_batch.shape[0]
                y_batch_column = y_batch.shape[1]
                weight_dt = pd.DataFrame()
                for j in range(y_batch_column):
                    weight_tag = y_train_dt.ix[:,j].replace({0:class_weight[0,j],1:class_weight[1,j]})
                    weight_dt = pd.concat([weight_dt,weight_tag],axis = 1)
                weight = weight_dt.values*sample_weight
                # #weight = 1.0
                _, summary, c = sess.run([training_op, merged, loss], feed_dict={x: x_batch, y: y_batch,weights:weight})
                if batch_step % 10 == 0:
                    train_writer.add_summary(summary, batch_step)
                    print('epoch: %d, batch_index: %d, loss:%.9f' % (epoch, batch_index, c))
                batch_step += 1
            epoch_step +=1
            if epoch_step % 10 == 0:
                epoch_theta = theta.eval()
                if not os.path.exists('./theta_weight1'):
                    os.makedirs('./theta_weight1')
                np.save('./theta_weight1/epoch_theta{0}'.format(epoch), epoch_theta)
            print(datetime.datetime.now())
        print('All Done!',datetime.datetime.now())




