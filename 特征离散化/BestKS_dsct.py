#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2019/9/10
@Author   :  yangjing gan
@File     :  BestKS_desc.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dsct_tools import data_describe


def calc_ks(count, idx):
    """
    计算各分组的KS值
    :param count: DataFrame 待分箱变量各取值的正负样本数
    :param group: list 单个分组信息
    :return: 该分箱的ks值

    计算公式：KS_i = |sum_i / sum_T - (size_i - sum_i)/ (size_T - sum_T)|
    """
    # 计算每个评分区间的好坏账户数。
    # 计算各每个评分区间的累计好账户数占总好账户数比率（good %)和累计坏账户数占总坏账户数比率（bad %）。
    # 计算每个评分区间累计坏账户比与累计好账户占比差的绝对值（累计good % -累计bad %），然后对这些绝对值取最大值记得到KS值
    ks = 0.0
    # 计算全体样本中好坏样本的比重(左开右闭区间)
    good = count[1].iloc[0:idx + 1].sum() / count[1].sum() if count[1].sum()!=0 else 1
    bad = count[0].iloc[0:idx + 1].sum() / count[0].sum() if count[0].sum()!=0 else 1
    ks += abs(good - bad)
    
    good = count[1].iloc[idx + 1:].sum() / count[1].sum() if count[1].sum()!=0 else 1
    bad = count[0].iloc[idx + 1:].sum() / count[0].sum() if count[0].sum()!=0 else 1
    ks += abs(good - bad)
    return ks


def calc_ks2(count):
    # 方法二：
    a = count.cumsum(axis=0) / count.sum(axis=0)
    a = a.fillna(1)
    a = abs(a[0] - a[1])
    
    count = count.sort_index(ascending=False)
    b = count.cumsum(axis=0) / count.sum(axis=0)
    b = b.fillna(1)
    b = abs(b[0] - b[1])
    ks = [a.values[idx] + b.values[len(a.index) - 2 - idx] for idx in range(len(a.index) - 1)]
    return ks


def get_best_cutpoint(count, group):
    """
        根据指标计算最佳分割点
        :param count:
        :param group: list 待划分的取值
        :return:
        """
    # entropy_list = [calc_ks(count, idx) for idx in range(0, len(group) - 1)]  # 左开右闭区间
    entropy_list = calc_ks2(count)
    
    intv = entropy_list.index(max(entropy_list))
    return intv


def BestKS_dsct(count, max_interval):
    """
    基于BestKS的特征离散化方法
    :param count: DataFrame 待分箱变量的分布统计
    :param max_interval: int 最大分箱数量
    :return: 分组信息（group）
    """
    group = count.index.values.reshape(1, -1).tolist()  # 初始分箱:所有取值视为一个分箱
    # 重复划分，直到KS的箱体数达到预设阈值。
    while len(group) < max_interval:
        group_intv = group[0] # 先进先出
        if len(group_intv) == 1:
            group.append(group[0])
            group.pop(0)
            continue
        
        # 选择最佳分箱点。
        count_intv = count[count.index.isin(group_intv)]
        intv = get_best_cutpoint(count_intv, group_intv)
        cut_point = group_intv[intv]
        # print(cut_point)
        
        # 状态更新
        group.append(group_intv[0:intv + 1])
        group.append(group_intv[intv + 1:])
        group.pop(0)
    return group


def BestKS_Discretization(data, var_name, var_name_target, max_interval=6, binning_method='bestKS', feature_type=0):
    """
    基于BestKS的离散化方法
    :param data: DataFrame 原始输入数据
    :param var_name: str 待离散化变量
    :param var_name_target: str 离散化后的变量
    :param max_interval: int 最大分箱数量
    :param binning_method: string 分箱方法
    :param var_type: bool 待分箱变量的类型（0: 连续型变量  1：离散型变量）
    :return: 分组信息（group）
    """
    # 1. 初始化：将每个值视为一个箱体统计各取值的正负样本分布 & 从小到大排序
    print("分箱初始化开始：")
    count, var_type = data_describe(data, var_name, var_name_target, feature_type)
    print("分箱初始化完成！！！")
    
    # 2. BestKS分箱
    if binning_method == 'bestKS':
        group = BestKS_dsct(count, max_interval)
    else:
        exit(code='无法识别分箱方法')
    group.sort()

    # 根据var_type修改返回的group样式(var_type=0: 返回分割点列表；var_typ=1：返回分箱成员列表）
    if not feature_type:
        group = [ele[-1] for ele in group] if len(group[0]) == 1 else [group[0][0]] + [ele[-1] for ele in group]
        group[0] = group[0] - 0.001 if group[0] == 0 else group[0] * (1 - 0.001)  # 包含最小值
        group[-1] = group[-1] + 0.001 if group[-1] == 0 else group[-1] * (1 + 0.001)  # 包含最大值
    return group


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    
    data = load_iris()
    x = data.data
    y = data.target
    y = np.where(y>1, 1, y)
    data = pd.DataFrame(x, columns=['A', 'B', 'C', 'D'])
    data['E'] = y
    group = BestKS_Discretization(data, 'A', 'E', max_interval=4, binning_method='bestKS', feature_type=0)
    print(group)
    
