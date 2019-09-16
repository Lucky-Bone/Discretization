#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2019/9/10
@Author   :  yangjing gan
@File     :  Entropy_dsct.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dsct_tools import data_describe

def optimal_binning_boundary(x: pd.Series, y: pd.Series, max_interval: int = 6, nan: float = -999.) -> list:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表
    
    x = x.fillna(nan).values  # 填充缺失值
    y = y.values
    
    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=max_interval,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
    
    clf.fit(x.reshape(-1, 1), y)  # 训练决策树
    
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])
    
    boundary.sort()
    
    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]
    
    return boundary


def calc_entropy(count):
    """
    计算分组的熵值
    :param count: DataFrame 分组的分布统计
    :return: float 该分组的熵值
    """
    # print(count.index.tolist())
    entropy = count.sum().values / count.values.sum()
    entropy = entropy[entropy!=0]
    entropy = -entropy * np.log2(entropy)
    return entropy.sum()


def calc_entropy2(count):
    p_clockwise = count.cumsum(axis=0).div(count.sum(axis=1).cumsum(axis=0), axis=0)
    entropy_clockwise = -p_clockwise * np.log2(p_clockwise)
    entropy_clockwise = entropy_clockwise.fillna(0)
    entropy_clockwise = entropy_clockwise.sum(axis=1)
    
    count = count.sort_index(ascending=False)
    p_anticlockwise = count.cumsum(axis=0).div(count.sum(axis=1).cumsum(axis=0), axis=0)
    entropy_anticlockwise = -p_anticlockwise * np.log2(p_anticlockwise)
    entropy_anticlockwise = entropy_anticlockwise.fillna(0)
    entropy_anticlockwise = entropy_anticlockwise.sum(axis=1)
    entropy = [entropy_clockwise.values[idx] + entropy_anticlockwise.values[len(entropy_clockwise.index) - 2 - idx] for idx in range(len(entropy_clockwise.index) - 1)]
    return entropy


def calc_ks(count, idx):
    """
    计算以idx作为分割点，分组的KS值
    :param count: DataFrame 待分箱变量各取值的正负样本数
    :param idx: list 单个分组信息
    :return: 该分箱的ks值

    计算公式：KS_i = |sum_i / sum_T - (size_i - sum_i)/ (size_T - sum_T)|
    """
    ks = 0.0
    # 计算左评分区间的累计好账户数占总好账户数比率（good %)和累计坏账户数占总坏账户数比率（bad %）。
    good_left = count[1].iloc[0:idx + 1].sum() / count[1].sum() if count[1].sum()!=0 else 1  # 左区间
    bad_left = count[0].iloc[0:idx + 1].sum() / count[0].sum() if count[0].sum()!=0 else 1
    # 计算左区间累计坏账户比与累计好账户占比差的绝对值（累计good % -累计bad %）
    ks += abs(good_left - bad_left)
    
    # 计算右评分区间的累计好账户数占总好账户数比率（good %)和累计坏账户数占总坏账户数比率（bad %）。
    good_right = count[1].iloc[idx + 1:].sum() / count[1].sum() if count[1].sum()!=0 else 1 # 右区间
    bad_right = count[0].iloc[idx + 1:].sum() / count[0].sum() if count[0].sum()!=0 else 1  #
    # 计算右区间累计坏账户比与累计好账户占比差的绝对值（累计good % -累计bad %）
    ks += abs(good_right - bad_right)
    return ks


def get_best_cutpoint(count, group, binning_method):
    """
        根据指标计算最佳分割点
        :param count: 待分箱区间
        :param group: # 待分箱区间内值的数值分布统计
        :return: 分割点的下标
        """
    # 以每个点作为分箱点（左开右闭区间），以此计算分箱后的指标（熵，信息增益，KS）值=左区间+右区间
    if binning_method == 'entropy':
        # entropy_list = [calc_entropy(count.iloc[0:idx + 1]) + calc_entropy(count.iloc[idx + 1:]) for idx in
        #                 range(0, len(group) - 1)]
        entropy_list = calc_entropy2(count)
        
    elif binning_method == 'bestKS':
        entropy_list = [calc_ks(count, idx) for idx in
                        range(0, len(group) - 1)]

    else:
        exit(code='无法识别分箱方法')
    
    # 最大指标值对应的分割点即为最佳分割点
    intv = entropy_list.index(max(entropy_list))
    return intv


def BestKS_dsct(count, max_interval, binning_method):
    """
    基于BestKS的特征离散化方法
    :param count: DataFrame 待分箱变量的分布统计
    :param max_interval: int 最大分箱数量
    :return: 分组信息（group）
    """
    group = count.index.values.reshape(1,-1).tolist() # 初始分箱:所有取值视为一个分箱
    # 重复划分，直到KS的箱体数达到预设阈值。
    while len(group) < max_interval:
        group_intv = group[0] # 待分箱区间
        if len(group_intv) == 1:
            group.append(group[0])
            group.pop(0)
            continue
            
        # 选择最佳分箱点。
        count_intv = count[count.index.isin(group_intv)] # 待分箱区间内值的数值分布统计
        intv = get_best_cutpoint(count_intv, group_intv, binning_method) # 分箱点的下标
        cut_point = group_intv[intv]
        print("cut_point:%s" % (cut_point))
        
        # 状态更新
        group.append(group_intv[0:intv+1])
        group.append(group_intv[intv+1:])
        group.pop(0)
    return group


def Entropy_Discretization(data, var_name, var_name_target, max_interval=6, binning_method='entropy', feature_type=0):
    """
    基于熵的离散化方法
    :param data: DataFrame 原始输入数据
    :param var_name: str 待离散化变量
    :param var_name_target: str 离散化后的变量
    :param max_interval: int 最大分箱数量
    :param binning_method: string 分箱方法
    :param var_type: bool 待分箱变量的类型（0: 连续型变量  1：离散型变量）
    :return: 分组信息（group）
    """
    
    # 1. 初始化：将每个值视为一个箱体 & 统计各取值的正负样本分布并排序
    print("分箱初始化开始：")
    count, var_type = data_describe(data, var_name, var_name_target, feature_type)
    print("分箱初始化完成！！！")
    
    # 2. 决策树分箱
    if binning_method in ['entropy', 'bestKS']:
        group = BestKS_dsct(count, max_interval, binning_method)
        # group = optimal_binning_boundary(data['A'], data['E'], 4)
    else:
        exit(code='无法识别分箱方法')
    group.sort()
    
    # 3. 根据var_type修改返回的group样式(var_type=0: 返回分割点列表；var_typ=1：返回分箱成员列表）
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
    group = Entropy_Discretization(data, 'A', 'E', max_interval=4, binning_method='entropy', feature_type=0)
    print(group)
    
    # print(optimal_binning_boundary(data['A'], data['E'], 4))
