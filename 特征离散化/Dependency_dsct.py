#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2019/9/5
@Author   :  yangjing gan
@File     :  Dependency_dsct.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import chi2
from dsct_tools import data_describe

def calc_chi2(count, group1, group2):
    """
    根据分组信息（group）计算各分组的卡方值
    :param count: DataFrame 待分箱变量各取值的正负样本数
    :param group1: list 单个分组信息
    :param group2: list 单个分组信息
    :return: 该分组的卡方值
    """
    count_intv1 = count.loc[count.index.isin(group1)].sum(axis=0).values
    count_intv2 = count.loc[count.index.isin(group2)].sum(axis=0).values
    count_intv = np.vstack((count_intv1, count_intv2))
    # 计算四联表
    row_sum = count_intv.sum(axis=1)
    col_sum = count_intv.sum(axis=0)
    total_sum = count_intv.sum()

    # 计算期望样本数
    count_exp = np.ones(count_intv.shape) * col_sum / total_sum
    count_exp = (count_exp.T * row_sum).T

    # 计算卡方值
    chi2 = (count_intv - count_exp) ** 2 / count_exp
    chi2[count_exp == 0] = 0
    return chi2.sum()
    

def merge_adjacent_intervals(count, chi2_list, group):
    """
    根据卡方值合并卡方值最小的相邻分组并更新卡方值
    :param count: DataFrame 待分箱变量的
    :param chi2_list: list 每个分组的卡方值
    :param group: list 分组信息
    :return: 合并后的分组信息及卡方值
    """
    min_idx = chi2_list.index(min(chi2_list))
    # 根据卡方值合并卡方值最小的相邻分组
    group[min_idx] = group[min_idx] + group[min_idx+1]
    group.remove(group[min_idx+1])
    
    # 更新卡方值
    if min_idx == 0:
        chi2_list.pop(min_idx)
        chi2_list[min_idx] = calc_chi2(count, group[min_idx], group[min_idx+1])
    elif min_idx == len(group)-1:
        chi2_list[min_idx-1] = calc_chi2(count, group[min_idx-1], group[min_idx])
        chi2_list.pop(min_idx)
    else:
        chi2_list[min_idx-1] = calc_chi2(count, group[min_idx-1], group[min_idx])
        chi2_list.pop(min_idx)
        chi2_list[min_idx] = calc_chi2(count, group[min_idx], group[min_idx+1])
    return chi2_list, group


def calc_inconsistency_rate(count, group):
    """
    计算分组的不一致性，参考论文《Feature Selection via Discretizations》
    :param count: DataFrame 待分箱变量的分布统计
    :param group: list 分组信息
    :return: float 该分组的不一致性
    """
    inconsistency_rate = 0.0
    for intv in group:
        count_intv = count.loc[count.index.isin(intv)].sum(axis = 0)
        inconsistency_rate += count_intv.sum() - max(count_intv)
    inconsistency_rate = inconsistency_rate / count.sum().sum()
    # print(inconsistency_rate)
    return inconsistency_rate


def Chi_Merge(count, max_interval=6, sig_level=0.05):
    """
    基于ChiMerge的卡方离散化方法
    :param count: DataFrame 待分箱变量各取值的正负样本数
    :param max_interval: int 最大分箱数量
    :param sig_level: 显著性水平(significance level) = 1 - 置信度
    :return: 分组信息（group）
    """
    print("ChiMerge分箱开始：")
    deg_freedom = len(count.columns) - 1 # 自由度(degree of freedom)= y类别数-1
    chi2_threshold = chi2.ppf(1 - sig_level, deg_freedom)  # 卡方阈值
    group = np.array(count.index).reshape(-1, 1).tolist()  # 分组信息
    
    while len(group) > max_interval:
        # 2. 计算相邻分组的卡方值
        chi2_list = [calc_chi2(count, group[idx], group[idx + 1]) for idx in range(len(group) - 1)]
        print(chi2_list)
    
        # 3. 合并相似分组
        if min(chi2_list) >= chi2_threshold:
            print("最小卡方值%.3f大于卡方阈值%.3f，分箱合并结束！！！" % (min(chi2_list), chi2_threshold))
            break
        _, group = merge_adjacent_intervals(count, chi2_list, group)
    print("ChiMerge分箱完成！！！")
    return group


def Chi_Merge1(count, max_interval=6, sig_level=0.05):
    """
    基于ChiMerge的卡方离散化方法
    :param count: DataFrame 待分箱变量各取值的正负样本数
    :param max_interval: int 最大分箱数量
    :param sig_level: 显著性水平(significance level) = 1 - 置信度
    :return: 分组信息（group）
    """
    print("ChiMerge分箱开始：")
    deg_freedom = len(count.columns) - 1  # 自由度(degree of freedom)= y类别数-1
    chi2_threshold = chi2.ppf(1 - sig_level, deg_freedom)  # 卡方阈值
    group = np.array(count.index).reshape(-1, 1).tolist()  # 分组信息
    
    # 2. 计算相邻分组的卡方值
    chi2_list = [calc_chi2(count, group[idx], group[idx + 1]) for idx in range(len(group) - 1)]
    
    # 3. 合并相似分组并更新卡方值
    while 1:
        if min(chi2_list) >= chi2_threshold:
            print("最小卡方值%.3f大于卡方阈值%.3f，分箱合并结束！！！" % (min(chi2_list), chi2_threshold))
            break
        if len(group) <= max_interval:
            print("分组长度%s等于指定分组数%s" % (len(group), max_interval))
            break
        chi2_list, group = merge_adjacent_intervals(count, chi2_list, group)
        # print(chi2_list)
    print("ChiMerge分箱完成！！！")
    return group


def Chi2(count, max_interval=6, sig_level=0.5, inconsistency_rate_thresh = 0.05, sig_level_desc = 0.1):
    """
    基于Chi2的卡方离散化方法
    :param count: DataFrame 待分箱变量的分布统计
    :param max_interval: int 最大分箱数量
    :param sig_level: 显著性水平(significance level) = 1 - 置信度
    :param inconsistency_rate_thresh: 不一致性阈值
    :return: 分组信息（group）
    """
    print("Chi2分箱开始：")
    deg_freedom = len(count.columns) - 1 # 自由度(degree of freedom)= y类别数-1

    group = np.array(count.index).reshape(-1, 1).tolist()  # 分组信息
    # 2. 阶段1：
    print("Chi2分箱第一阶段开始：")
    while calc_inconsistency_rate(count, group) < inconsistency_rate_thresh: # 不一致性检验
        chi2_threshold = chi2.ppf(1-sig_level, deg_freedom) # 卡方阈值
        while len(group) > max_interval:
            # 2. 计算相邻分组的卡方值
            chi2_list = [calc_chi2(count, group[idx], group[idx + 1]) for idx in range(len(group) - 1)]
            # 3. 合并相似分组
            if min(chi2_list) >= chi2_threshold:
                print("最小卡方值%.3f大于卡方阈值%.3f，分箱合并结束！！！" % (min(chi2_list), chi2_threshold))
                break
            group = merge_adjacent_intervals(chi2_list, group)
        if len(group) <= max_interval:
            break
        # 阈值更新
        sig_level = sig_level - sig_level_desc # 降低显著性水平，提高卡方阈值
    print("Chi2分箱第一阶段完成！！！")
    
    # 3. 阶段2：
    print("Chi2分箱第二阶段开始：")
    sig_level = sig_level + sig_level_desc  # 回到上一次的值
    while True:
        chi2_threshold = chi2.ppf(1 - sig_level, deg_freedom)  # 卡方阈值
        while len(group) > max_interval:
            # 2. 计算相邻分组的卡方值
            chi2_list = [calc_chi2(count, group[idx], group[idx + 1]) for idx in range(len(group) - 1)]
            # 3. 合并相似分组
            if min(chi2_list) >= chi2_threshold:
                print("最小卡方值%.3f大于卡方阈值%.3f，合并分箱结束！！！" % (min(chi2_list), chi2_threshold))
                break
            group = merge_adjacent_intervals(chi2_list, group)
        if len(group) <= max_interval:
            break
        in_consis_rate = calc_inconsistency_rate(count, group)
        if in_consis_rate < inconsistency_rate_thresh:  # 不一致性检验
            sig_level = sig_level - sig_level_desc  # 降低显著性水平，提高卡方阈值
        else:
            print("分组的不一致性(%.3f)大于阈值(%.3f)，无法继续合并分箱！！！" % (in_consis_rate, inconsistency_rate_thresh))
            break
    print("Chi2分箱第二阶段完成！！！")
    
    return group


def Chi_Discretization(data, var_name, var_name_target, max_interval=6, binning_method = 'chi2', feature_type = 0):
    """
    基于卡方的离散化方法
    :param data: DataFrame 原始输入数据
    :param var_name: str 待离散化变量
    :param var_name_target: str 标签变量（y)
    :param max_interval: int 最大分箱数量
    :param binning_method: string 分箱方法
    :param feature_type: bool 待分箱变量的类型（0: 连续型变量  1：离散型变量）
    :return: 分组信息（group）
    """
    
    # 1. 初始化：将每个值视为一个箱体 & 统计各取值的正负样本分布并排序
    print("分箱初始化开始：")
    count, var_type = data_describe(data, var_name, var_name_target, feature_type)
    print("分箱初始化完成！！！")
    
    # 2. 卡方分箱
    if binning_method == 'chiMerge':
        # group = Chi_Merge(count,max_interval)
        group = Chi_Merge1(count,max_interval)
    elif binning_method == 'chi2':
        group = Chi2(count,max_interval)
    else:
        exit(code='无法识别分箱方法')
        
    # 后处理
    if not feature_type:
        group = [sorted(ele) for ele in group]
    group.sort()
    
    if len(group) > max_interval:
        print("warning: 分箱后，%s的箱体个数（%s）与您输入的分箱数量（%s）不符，这是由分组间的相似性太低导致的。如对分箱效果不满意，请更换其他分箱方法" % (
        var_name, len(group), max_interval))

    # 3. 根据var_type修改返回的group样式(var_type=0: 返回分割点列表；var_typ=1：返回分箱成员列表）
    if not feature_type:
        group = [ele[-1] for ele in group] if len(group[0])==1 else [group[0][0]] + [ele[-1] for ele in group]
        group[0] = group[0]-0.001 if group[0]==0 else group[0] * (1-0.001) # 包含最小值
        group[-1] = group[-1]+0.001 if group[-1]==0 else group[-1] * (1+0.001) # 包含最大值
    return group

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data = load_iris()
    x = data.data
    y = data.target
    # y = np.where(y>1, 1, y)
    data = pd.DataFrame(x, columns=['A','B','C','D'])
    data['E'] = y
    group = Chi_Discretization(data, 'A', 'E', max_interval = 4, binning_method='chiMerge', feature_type= 0)
    print(group)
