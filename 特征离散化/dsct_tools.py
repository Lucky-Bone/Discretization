#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2019/9/5
@Author   :  yangjing gan
@File     :  dsct_tools.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''

import numpy as np
import pandas as pd

def data_describe(data, var_name_bf, var_name_target, feature_type):
    """
    统计各取值的正负样本分布 [累计样本个数，正例样本个数，负例样本个数] 并排序
    :param data: DataFrame 输入数据
    :param var_name_bf: str 待分箱变量
    :param var_name_target: str 标签变量（y)
    :param feature_type: 特征的类型：0（连续） 1（离散）
    :return: DataFrame 排好序的各组中正负样本分布 count
    """
    # 统计待离散化变量的取值类型（string or digits)
    data_type = data[var_name_bf].apply(lambda x: type(x)).unique()
    var_type = True if str in data_type else False # 实际取值的类型：false(数字） true(字符）
    
    # 是否需要根据正例样本比重编码，True：需要，False：不需要
    #                   0（连续）    1（离散）
    #     false（数字）    0              0（离散有序）
    #     true（字符）     ×             1（离散无序）
    if feature_type == var_type:
        ratio_indicator = var_type
    elif feature_type == 1:
        ratio_indicator = 0
        print("特征%s为离散有序数据，按照取值大小排序！" % (var_name_bf))
    elif feature_type == 0:
        exit(code="特征%s的类型为连续型，与其实际取值（%s）型不一致，请重新定义特征类型！！！" % (var_name_bf, data_type))

    # 统计各分箱（group）内正负样本分布[累计样本个数，正例样本个数，负例样本个数]
    count = pd.crosstab(data[var_name_bf], data[var_name_target])
    total = count.sum(axis=1)
    
    # 排序：离散变量按照pos_ratio排序，连续变量按照index排序
    if ratio_indicator:
        count['pos_ratio'] = count[count.columns[count.columns.values>0]].sum(axis=1) * 1.0 / total#？？？
        count = count.sort_values('pos_ratio') #离散变量按照pos_ratio排序
        count = count.drop(columns = ['pos_ratio'])
    else:
        count = count.sort_index() # 连续变量按照index排序
    return count, ratio_indicator


def calc_IV(count):
    """
    计算各分组的WOE值以及IV值
    :param count: DataFrame 排好序的各组中正负样本分布
    :return: 各分箱的woe和iv值
    
    计算公式：WOE_i = ln{(sum_i / sum_T) / [(size_i - sum_i) / (size_T - sum_T)]}
    计算公式：IV_i = [sum_i / sum_T - (size_i - sum_i) / (size_T - sum_T)] * WOE_i
    """
    # 计算全体样本中好坏样本的比重
    good = (count[count.columns[count.columns.values>0]].sum(axis=1) / count[count.columns[count.columns.values>0]].values.sum()).values # ？？？
    # good = (count[1] / count[1].sum()).values
    bad = (count[0] / count[0].sum()).values
    
    woe = np.log(good / bad)
    if 0 in bad:
        ind = np.where(bad == 0)[0][0]
        woe[ind] = 0
        print('第%s类负例样本个数为0！！！' % ind)
    if 0 in good:
        ind = np.where(good == 0)[0][0]
        woe[ind] = 0
        print('第%s类正例样本个数为0！！！' % ind)
    iv = (good - bad) * woe
    return woe, iv