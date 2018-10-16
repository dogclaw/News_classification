# -*- coding: utf-8 -*-
import re
import csv
import pymysql
import numpy as np


def text_filter(text_list):
    """
    CMS文本过滤
    :param text_list:输入文本列表
    :return: 过滤后文本
    """
    text = ' '.join(text_list)
    pattern_map = re.compile(r'<div class="cmsMap".*?</div>|<div class="cmsMapCaption".*?</div>')
    text = re.sub(pattern_map, '', text)
    pattern_others = re.compile(r'<.*?>|&nbsp|&amp|&quot|&lt|&gt')
    text = re.sub(pattern_others, ' ', text)
    pattern_line = re.compile(r'\n')
    text = re.sub(pattern_line, ' ', text)
    return text


def read_mysql():
    """
    从生产环境读取数据
    """
    conn = pymysql.connect(host='slave-cgtn-app-prod.cqhyewxpye7e.rds.cn-north-1.amazonaws.com.cn',
                           user='prodselect',
                           passwd='prodselect!',
                           port=3306,
                           charset="utf8")
    conn.select_db('prodgateway')
    cur = conn.cursor()
    query_tags = 'SELECT resource_tag_info.tag_name ' \
                 'FROM prodgateway.publish_news_label, ' \
                 'prodgateway.resource_tag_info ' \
                 'WHERE publish_news_label.lable_id = resource_tag_info.id ' \
                 'AND resource_tag_info.status = 0 ' \
                 'AND resource_tag_info.topic_type != 7 ' \
                 'GROUP BY resource_tag_info.tag_name ' \
                 'HAVING COUNT(*) > 50'
    cur.execute(query_tags)
    result_tags = cur.fetchall()
    tag_list = [each[0] for each in result_tags]
    np.save('data/tag_list.npy', tag_list)
    tag_list_str = '(%s)' % ','.join('"%s"' % each[0] for each in result_tags)
    query_relations = 'SELECT publish_news_label.news_id, ' \
                      'resource_tag_info.tag_name ' \
                      'FROM prodgateway.publish_news_label, ' \
                      'prodgateway.resource_tag_info ' \
                      'WHERE publish_news_label.lable_id = resource_tag_info.id ' \
                      'AND resource_tag_info.tag_name IN %s' % tag_list_str
    cur.execute(query_relations)
    result_relations = cur.fetchall()
    query_news = 'SELECT publish_news.id, ' \
                 'publish_news.headline, ' \
                 'publish_news.news_detail, ' \
                 'publish_news_data.content ' \
                 'FROM prodgateway.publish_news, ' \
                 'prodgateway.publish_news_data ' \
                 'WHERE publish_news.id = publish_news_data.news_id ' \
                 'AND publish_news_data.type = 0 ' \
                 'AND publish_news.news_type = 2 '
    cur.execute(query_news)
    result_news = cur.fetchall()
    id_list = [each[0] for each in result_news]
    data_matrix = np.zeros((len(id_list), len(result_tags)), dtype=np.int32)
    for news_id, tag_name in result_relations:
        try:
            data_matrix[id_list.index(news_id), tag_list.index(tag_name)] = 1
        except:
            pass
    out_csv_file = csv.writer(
        open('/home/ec2-user/data_loader_temp/data/all.csv', mode='a', encoding='utf-8', newline=''))
    for line in range(len(result_news)):
        item = result_news[line]
        x_ = text_filter([item[1], item[2], item[3].decode('utf-8')])
        y_ = data_matrix[line, :]
        out_csv_file.writerow([x_] + list(y_))
    conn.close()


if __name__ == '__main__':
    read_mysql()
