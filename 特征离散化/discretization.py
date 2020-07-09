#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2020/5/11
@Author   :  yangjing gan
@File     :  discretization.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans


def calc_IV(data, var_name, var_name_target):
    """
    计算各分组的WOE值以及IV值
    :param data: DataFrame 输入数据
    :param var_name: str 分箱后的变量
    :param var_name_target: str 标签变量
    :return: 各分箱的woe和iv值

    计算公式：WOE_i = ln{(sum_i / sum_T) / [(size_i - sum_i) / (size_T - sum_T)]}
    计算公式：IV_i = [sum_i / sum_T - (size_i - sum_i) / (size_T - sum_T)] * WOE_i
    """
    # 计算全体样本中好坏样本的比重
    count = pd.crosstab(data[var_name], data[var_name_target])
    try:
        # good = (count[count.columns[count.columns.values > 0]].sum(axis=1) / count[
        # count.columns[count.columns.values > 0]].values.sum()).values  # ？？？
        good = (count[1] / count[1].sum()).values
        bad = (count[0] / count[0].sum()).values
    except Exception:
        exit('请确认标签变量(y)的取值为且仅为[0，1]')
    
    woe = np.log(good / bad)
    if 0 in bad:
        ind = np.where(bad == 0)
        woe[ind] = 0
        print('第%s类负例样本个数为0！！！' % ind)
    if 0 in good:
        ind = np.where(good == 0)
        woe[ind] = 0
        print('第%s类正例样本个数为0！！！' % ind)
    iv = (good - bad) * woe
    return woe, iv


def group_to_col(data, group, labels):
    """
    将特征的分组信息映射至原始数据列
    :param data: DataFrame 原始数据列
    :param group: list 分组信息
    :param labels: list 分组代号
    :return:
    """
    assert len(group) == len(labels), "分组个数与分组编号数目不一致"
    col = data.name
    data = pd.DataFrame(
        {col: data, "tmp": pd.Categorical(data.values, categories=list(set(unique_nan(data)).union(labels)))})
    for i in range(len(group)):
        data.loc[data[col].isin(group[i]), 'tmp'] = labels[i]
    data['tmp'] = data['tmp'].cat.set_categories(unique_nan(data['tmp']).tolist())  # 保持categories参数与数据一致
    return data['tmp']


def unique_nan(data):  # 存在缺失值时，unique会保留nan，crosstab之类不会保留缺失值，统一不处理缺失值
    return np.array(list(filter(lambda ele: ele == ele, data.unique()))) # 内存&速度优化：dropna
    return np.array(data.dropna().unique())


class Discretization(object):
    """
    离散化基类
    """
    
    def __init__(self, max_interval=6, feature_type=0):
        """
        初始化参数
        :param max_interval: int 最大分箱数量
        :param feature_type:  bool 待分箱变量的类型（0：连续  1：离散）
        """
        self.max_interval = max_interval
        self.feature_type = feature_type
    
    def dsct_pipeline(self, data, var_name, var_name_target=None):
        """
        离散化处理流程
        :param data: DataFrame 原始输入数据
        :param var_name: str 待离散化变量
        :param var_name_target: 目标变量
        :return: 有序的分割点列表或者分组列表
        """
        assert var_name in data.columns, "数据中不包含变量%s，请检查数据" % (var_name)
        if len(unique_nan(data[var_name])) <= self.max_interval:
            print('warning：变量%s的分箱数量大于或等于其不同取值个数，无需进行分箱。\n变量的取值如下：%s' % (var_name, unique_nan(data[var_name])))
            group = self.processing_when_not_binning(data, var_name, var_name_target)
            group = self.postprocessing(group)
        else:
            group = self.dsct_general_method(data, var_name, var_name_target)
        
        print("分箱结束！")
        return group
    
    def processing_when_not_binning(self, data, var_name, var_name_target=None):
        group = unique_nan(data[var_name]).reshape(-1, 1).tolist()  # 分组信息
        return group
    
    def postprocessing(self, group):
        """
        后处理：根据feature_type修改返回的group样式（feature_type=0：返回分割点列表；feature_type=1：返回分箱成员列表)
        :param group:
        :return:
        """
        group = [sorted(ele) for ele in group]
        if not self.feature_type:
            group.sort()
            group = self.group_to_cutpoints(group)
        return group
    
    def cut_points_expand(self, cut_points, precision=3):
        """
        扩展分割点的边界，以包含最大最小值
        :param cut_points: list 分组信息
        :param precision:
        :return: list 扩展后的魅族分割点（长度比cut_points大1）
        """
        cut_points[0] = cut_points[0] - 0.001 if cut_points[0] == 0 else cut_points[0] - abs(
            cut_points[0]) * 0.001  # 包含最小值
        cut_points[-1] = cut_points[-1] + 0.001 if cut_points[-1] == 0 else cut_points[-1] + abs(
            cut_points[-1]) * 0.001  # 包含最大值
        
        ## 保留指定小数位，有可能会导致cut_points部分重合
        # cut_points[0] = floor(cut_points[0] * pow(10, precision)) / pow(10, precision) # 包含最小值
        # cut_points[1:-1] = [round(ele, precision) for ele in cut_points[1:-1]]
        # cut_points[-1] = ceil(cut_points[-1] * pow(10, precision)) / pow(10, precision) # 包含最大值
        
        return cut_points
    
    def group_to_cutpoints(self, group):
        """
        将group转换为cut_points（仅使用于连续型变量）
        :param group: list 分组信息
        :return: list 每组的分割点（长度比cut_points大1）
        """
        cut_points = [group[0][0]] + [ele[-1] for ele in group]
        cut_points = self.cut_points_expand(cut_points)
        return cut_points
    
    def encode_by_mapdict(self, srs, map_dict):
        if srs in map_dict.keys():
            return map_dict[srs]
        else:
            return srs


class equalWide(Discretization):
    """
    等宽分箱
    """
    def dsct_general_method(self, data, var_name, var_name_target=None):
        var_name_af = var_name + '_BIN'
        data[var_name_af] = data[var_name]
        if self.feature_type:
            # 1. 统计各分箱（group）内正负样本分布[累计样本个数，正例样本个数，负例样本个数]
            count = pd.crosstab(data[var_name], data[var_name_target])
            count['ratio'] = (count.iloc[:, count.columns.values > 0].sum(axis=1) * 1.0 / count.sum(
                axis=1)).values  # 正例样本占该取值总样本的比值
            
            # 映射ratio至原始列
            data[var_name_af] = pd.Categorical(data[var_name_af], categories=list(set(unique_nan(data[var_name])).union(unique_nan(count['ratio']))))
            if len(count.index) > 50:
                map_dict = dict(zip(count.index, count['ratio']))
                data[var_name_af] = data.apply(lambda ele: self.encode_by_mapdict(ele[var_name], map_dict),
                                               axis=1)  # id列
            else:
                for ele in count.index:
                    data[var_name_af][data[var_name] == ele] = count['ratio'][ele]  # 非id列
            
            data[var_name_af], group = pd.cut(data[var_name_af], bins=self.max_interval, retbins=True,
                                              duplicates='drop', include_lowest=True, labels=False)
            group = group.tolist()
            if len(group) - 1 != self.max_interval:
                print('warning：分箱后，%s的箱体个数%s与您输入的箱体数量%s不符，可能是变量%s的相同取值过多，导致样本分布不均衡，请检查数据集。' % (
                    var_name, len(group) - 1, self.max_interval, var_name))
            group = self.cut_points_expand(group)
            
            if self.feature_type:
                group = [list(set(data[var_name][data[var_name_af] == ele])) for ele in unique_nan(data[var_name_af])]
            return group


class equalFreq(Discretization):
    """
    等频分箱
    """
    
    def dsct_general_method(self, data, var_name, var_name_target=None):
        var_name_af = var_name + '_BIN'
        data[var_name_af] = data[var_name]
        if self.feature_type:
            # 1. 统计各分箱（group）内正负样本分布[累计样本个数，正例样本个数，负例样本个数]
            count = pd.crosstab(data[var_name], data[var_name_target])
            count['ratio'] = (count.iloc[:, count.columns.values > 0].sum(axis=1) * 1.0 / count.sum(
                axis=1)).values  # 正例样本占该取值总样本的比值
            
            # 映射ratio至原始列
            data[var_name_af] = pd.Categorical(data[var_name_af], categories=list(set(unique_nan(data[var_name])).union(unique_nan(count['ratio']))))
            if len(count.index) > 50:
                map_dict = dict(zip(count.index, count['ratio']))
                data[var_name_af] = data.apply(lambda ele: self.encode_by_mapdict(ele[var_name], map_dict),
                                               axis=1)  # id列
            else:
                for ele in count.index:
                    data[var_name_af][data[var_name] == ele] = count['ratio'][ele]  # 非id列
            
            data[var_name_af], group = pd.qcut(data[var_name_af], q=self.max_interval, retbins=True, precision=3,
                                               duplicates='drop', labels=False)
            group = group.tolist()
            
            if len(group) - 1 != self.max_interval:
                print('warning：分箱后，%s的箱体个数%s与您输入的箱体数量%s不符，可能是变量%s的相同取值过多，导致样本分布不均衡，请检查数据集。' % (
                var_name, len(group) - 1, self.max_interval, var_name))
            group = self.cut_points_expand(group)
            
            if self.feature_type:
                group = [list(set(data[var_name][data[var_name_af] == ele])) for ele in
                         unique_nan(data[var_name_af])]
            return group


class SuperDiscretization(Discretization):
    """
    监督离散化基类
    """
    
    def dsct_pipeline(self, data, var_name, var_name_target):
        """
        离散化处理流程
        :param data: DataFrame 输入数据
        :param var_name: str 待离散化变量
        :param var_name_targe: str 标签变量（y）
        :return:
        """
        assert var_name_target in data.columns, "数据中不包含类别变量%s，请检查数据！" % (var_name_target)
        group = super(SuperDiscretization, self).dsct_pipeline(data, var_name, var_name_target)
        return group
    
    def dsct_general_method(self, data, var_name, var_name_target):
        """
        离散化通用功能
        """
        # 1. 初始化：将每个值视为一个箱体 & 统计各取值的正负样本分布并排序
        count, var_type = self.dsct_init(data, var_name, var_name_target, self.feature_type)
        group = self.group_init(count)  # 分组信息
        print("分箱初始化完成！")
        
        # 2. 分箱主体
        group = self.dsct(count, group, self.max_interval)
        print("分箱主体功能完成")
        
        # 3. 后处理
        group = self.postprocessing(group)
        return group
    
    def dsct_init(self, data, var_name_bf, var_name_target, feature_type):
        """
        特征离散化节点初始化：统计各取值的正负样本分布[正例样本个数，负例样本个数]
        :param data: DataFrame 输入数据
        :param var_name_bf: str 待分箱变量
        :param var_name_target: str 标签变量（y）
        :param feature_type: 特征类型：0（连续） 1（离散）
        :return: DataFrame 排好序的各组中正负样本分布 count
        """
        # 统计待离散化变量的取值类型（string or digits)
        data_type = data[var_name_bf].apply(lambda x: type(x)).unique()
        var_type = True if str in data_type else False  # 实际取值的类型：false(数字） true(字符）
        
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
            count['pos_ratio'] = count[1].sum(axis=1) * 1.0 / total  # 计算正例比例
            count = count.sort_values('pos_ratio')  # 离散变量按照pos_ratio排序
            count = count.drop(columns=['pos_ratio'])
        else:
            count = count.sort_index()  # 连续变量按照index排序
        return count, ratio_indicator


class clusterDsct(Discretization):
    """
    聚类分箱：仅针对离散变量
    先对var_name变量中的各个取值，计算正例样本占该取值总样本的比例（ratio），然后根据ratio值聚类分组，最后计算WOE和IV值，用于评价该分组的合理性
    """
    
    def dsct_general_method(self, data, var_name, var_name_target):
        if self.feature_type:
            # 1. 初始化：将每个值视为一个箱体 & 统计各取值的正负样本分布并排序
            count = pd.crosstab(data[var_name], data[var_name_target])
            group = np.array(count.index)
            
            # 2. 聚类分组
            ratio = (count.iloc[:, count.columns.values > 0].sum(axis=1) * 1.0 / count.sum(
                axis=1)).values  # 正例样本占该取值总样本比值
            model = KMeans(n_clusters=self.max_interval, init='k-means++')
            label_pred = model.fit_predict(ratio.reshape(-1, 1))
            
            # 3. 更新分组信息
            group_update = [list(group[label_pred == i]) for i in range(label_pred.max() + 1)]
            group = [tmp for tmp in group_update if tmp]
        else:
            raise ValueError("聚类分箱暂不支持连续特征！")
        return group


class DependencyDsct(SuperDiscretization):
    """
    基于卡方的离散化方法
    """
    
    def __init__(self, max_interval, feature_type, sig_leval=0.05, n_chi2_dsct_thresh=100):
        super(DependencyDsct, self).__init__(max_interval, feature_type)
        self.sig_level = sig_leval
        self.n_chi2_dsct_thresh = n_chi2_dsct_thresh
    
    def dsct_general_method(self, data, var_name, var_name_target):
        """
        离散化处理流程
        """
        # 类别数太多时
        status = 0
        if len(unique_nan(data[var_name])) > self.n_chi2_dsct_thresh:
            # 粗分组
            status = 1
            data_bf, var_name_bf, group_bf = self.data_split(data, var_name, var_name_target, self.n_chi2_dsct_thresh,
                                                             self.feature_type)
        else:
            data_bf = data
            var_name_bf = var_name
        
        # 1. 初始化：将每个值视为一个箱体 & 统计各取值的正负样本分布并排序
        count, var_type = self.dsct_init(data_bf, var_name_bf, var_name_target, self.feature_type)
        group = self.group_init(count)
        print("分组初始化完成！")
        
        # 2. 分箱主体
        group = self.dsct(count, group, self.max_interval)
        print("分箱主体功能完成！")
        
        # 映射至粗分组区间
        if status:
            if not self.feature_type:
                group = [(group_bf[list(map(int, np.array(ele) + 1))]).tolist() for ele in group]
                group[0].append(group_bf[0])
            else:
                group = [sum(np.array(group_bf)[list(map(int, np.array(ele)))], []) for ele in group]
        
        # 后处理
        group = self.postprocessing(group)
        return group
    
    def dsct(self, count, group, max_interval):
        """
        离散化主体方法
        :param count: DataFrame 待分箱变量的分布统计
        :param max_interval: int 最大分箱数量
        :return: 分组信息（group）
        """
        self.deg_freedom = len(count.columns) - 1  # 自由度(degree of freedom)= y类别数-1
        self.chi2_threshold = chi2.ppf(1 - self.sig_level, self.deg_freedom)  # 卡方阈值
        return group
    
    def group_init(self, count):
        # 获取初始分组
        group = np.array(count.index).reshape(-1, 1).tolist()  # 分组信息
        return group
    
    def calc_chi2(self, count, group1, group2):
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
    
    def merge_adjacent_intervals(self, count, chi2_list, group):
        """
        根据卡方值合并卡方值最小的相邻分组并更新卡方值
        :param count: DataFrame 待分箱变量的分布统计
        :param chi2_list: list 每个分组的卡方值
        :param group: list 分组信息
        :return: 合并后的分组信息及卡方值
        """
        min_idx = chi2_list.index(min(chi2_list))
        # 根据卡方值合并卡方值最小的相邻分组
        group[min_idx] = group[min_idx] + group[min_idx + 1]
        group.remove(group[min_idx + 1])
        
        # 更新卡方值
        if min_idx == 0:
            chi2_list.pop(min_idx)
            chi2_list[min_idx] = self.calc_chi2(count, group[min_idx], group[min_idx + 1])
        elif min_idx == len(group) - 1:
            chi2_list[min_idx - 1] = self.calc_chi2(count, group[min_idx - 1], group[min_idx])
            chi2_list.pop(min_idx)
        else:
            chi2_list[min_idx - 1] = self.calc_chi2(count, group[min_idx - 1], group[min_idx])
            chi2_list.pop(min_idx)
            chi2_list[min_idx] = self.calc_chi2(count, group[min_idx], group[min_idx + 1])
        return chi2_list, group
    
    def update_group_by_chi2(self, count, group, idx):
        if id == 0:
            merge_idx = idx + 1
        elif idx == len(group) - 1:
            merge_idx = idx - 1
        else:
            # 根据卡方值合并卡方值最小的相邻分组
            merge_idx = idx + 1 if self.calc_chi2(count, group[idx - 1], group[idx]) > self.calc_chi2(count, group[idx],
                                                                                                      group[
                                                                                                          idx + 1]) else idx - 1
        group[idx] = group[idx] + group[merge_idx]
        group.remove(group[merge_idx])
        return group
    
    def guarantee_completeness(self, count, group):
        """
        检查每个箱体中是否都有正负样本。如果没有，则需要跟相邻的箱体合并，直至每个箱体都包含正负样本
        :param count: DataFrame 待分箱变量的分布统计
        :param group: list 分组信息
        :return: list 分组信息
        """
        while True:
            # 计算pos_ratio
            count_update = [count[count.index.isin(ele)].sum(axis=0).tolist() for ele in group]
            count_update = pd.DataFrame(count_update, columns=count.columns)
            count_update['pos_ratio'] = count_update[0] / count_update.sum(axis=1)
            # 合并分组
            if min(count_update['pos_ratio'] == 0) or max(count_update['pos_ratio']) == 1:
                idx = count_update[count_update['pos_ratio'].isin([0, 1])].index[0]
                group = self.update_group_by_chi2(count_update, group, idx)
            else:
                break
        return group
    
    def data_split(self, data, var_name, var_name_target, n_chi2_dsct_thresh, feature_type):
        # 粗分组
        var_name_bf = var_name + '_coarse'
        if not feature_type:  # 连续型
            print("特征%s的取值数目%s超过100个，先采用等频分箱将其粗分为%s组，然后采用卡方分箱" % (
            var_name, len(unique_nan(data[var_name])), n_chi2_dsct_thresh))  # 等频分箱
            data[var_name_bf], cut_points = pd.qcut(data[var_name], q=n_chi2_dsct_thresh, labels=False,
                                                    duplicates='drop', retbins=True)  # 等频分箱
            return data, var_name_bf, cut_points
        else:  # 离散型
            print("特征%s的取值数目%s超过100个，先采用聚类分箱将其粗分为%s组，然后采用卡方分箱" % (
            var_name, len(unique_nan(data[var_name])), n_chi2_dsct_thresh))  # 等频分箱
            group_bf = clusterDsct(n_chi2_dsct_thresh, 1).dsct_pipeline(data[[var_name, var_name_target]], var_name,
                                                                        var_name_target)  # 聚类分箱
            group_bf = [ele for ele in group_bf if ele != []]
            data[var_name_bf] = group_to_col(data[var_name], group_bf, range(len(group_bf)))
            return data, var_name_bf, group_bf


class chiMerge(DependencyDsct):
    def dsct(self, count, group, max_interval):
        group = super(chiMerge, self).dsct(count, group, max_interval)
        # 2. 计算相邻分组的卡方值
        chi2_list = [self.calc_chi2(count, group[idx], group[idx + 1]) for idx in range(len(group) - 1)]
        
        # 3. 合并相似分组并更新卡方值
        while 1:
            # if min(chi2_list) >= self.chi2_threshold:
            #     print("最小卡方值%.3f大于卡方阈值%.3f，分箱合并结束！！！" % (min(chi2_list), chi2_threshold))
            #     break
            if len(group) <= max_interval:
                print("分组长度%s等于指定分组数%s" % (len(group), max_interval))
                break
            chi2_list, group = self.merge_adjacent_intervals(count, chi2_list, group)
            # print(chi2_list)
        return group


class chi2Merge(DependencyDsct):
    def __init__(self, max_interval, feature_type, sig_level=0.5, inconsistency_rate_thresh=0.05, sig_level_desc=0.1):
        super(chi2Merge, self).__init__(max_interval, feature_type, sig_level)
        self.inconsistency_rate_thresh = inconsistency_rate_thresh
        self.sig_level_desc = sig_level_desc
    
    def dsct(self, count, group, max_interval):
        group = super(chi2Merge, self).dsct(count, group, max_interval)
        sig_level = self.sig_level
        # 2. 阶段1：
        while self.calc_inconsistency_rate(count, group) < self.inconsistency_rate_thresh:  # 不一致性检验
            # 2. 计算相邻分组的卡方值
            chi2_threshold = chi2.ppf(1 - sig_level, self.deg_freedom)  # 卡方阈值
            chi2_list = [self.calc_chi2(count, group[idx], group[idx + 1]) for idx in range(len(group) - 1)]
            
            # 3. 合并相似分组并更新卡方值
            while 1:
                if min(chi2_list) >= chi2_threshold:
                    print("最小卡方值%.3f大于卡方阈值%.3f，分箱合并结束！！！" % (min(chi2_list), chi2_threshold))
                    break
                if len(group) <= max_interval:
                    print("分组长度%s等于指定分组数%s" % (len(group), max_interval))
                    break
                chi2_list, group = self.merge_adjacent_intervals(count, chi2_list, group)
            
            # 阈值更新
            sig_level = sig_level - self.sig_level_desc  # 降低显著性水平，提高卡方阈值
        print("Chi2分箱第一阶段完成！！！")
        
        # 3. 阶段2：
        print("Chi2分箱第二阶段开始：")
        sig_level = sig_level + self.sig_level_desc  # 回到上一次的值
        while True:
            # 2. 计算相邻分组的卡方值
            chi2_threshold = chi2.ppf(1 - sig_level, self.deg_freedom)  # 卡方阈值
            chi2_list = [self.calc_chi2(count, group[idx], group[idx + 1]) for idx in range(len(group) - 1)]
            
            # 3. 合并相似分组并更新卡方值
            while 1:
                if min(chi2_list) >= chi2_threshold:
                    print("最小卡方值%.3f大于卡方阈值%.3f，分箱合并结束！！！" % (min(chi2_list), chi2_threshold))
                    break
                if len(group) <= max_interval:
                    print("分组长度%s等于指定分组数%s" % (len(group), max_interval))
                    break
                chi2_list, group = self.merge_adjacent_intervals(count, chi2_list, group)
            
            # 阈值更新
            in_consis_rate = self.calc_inconsistency_rate(count, group)
            if in_consis_rate < self.inconsistency_rate_thresh:  # 不一致性检验
                sig_level = sig_level - self.sig_level_desc  # 降低显著性水平，提高卡方阈值
            else:
                print("分组的不一致性(%.3f)大于阈值(%.3f)，无法继续合并分箱！！！" % (in_consis_rate, self.inconsistency_rate_thresh))
                break
        print("Chi2分箱第二阶段完成！！！")
        return group
    
    def calc_inconsistency_rate(self, count, group):
        """
        计算分组的不一致性，参考论文《Feature Selection via Discretizations》
        :param count: DataFrame 待分箱变量的分布统计
        :param group: list 分组信息
        :return: float 该分组的不一致性
        """
        inconsistency_rate = 0.0
        for intv in group:
            count_intv = count.loc[count.index.isin(intv)].sum(axis=0)
            inconsistency_rate += count_intv.sum() - max(count_intv)
        inconsistency_rate = inconsistency_rate / count.sum().sum()
        return inconsistency_rate


class bestChi(chiMerge):
    def dsct(self, count, group, max_interval):
        group = super(bestChi, self).dsct(count, group, max_interval)
        # 检查每个箱体是否都有好坏样本
        group = self.guarantee_completeness(count, group)
        print("各分组好坏样本分布检验完成！")
        # 单项占比检验
        group = self.guarantee_proportion(count, group)
        print("各分组单项占比检验完成！")
        while not self.check_posRate_monotone(count, group):  # 单调性检验
            # 合并分组
            max_interval -= 1
            group = super(bestChi, self).dsct(count, group, max_interval)
        print("单调性检验完成")
        if len(group) < max_interval:
            print("分箱后的箱体个数（%s)与输入的箱体个数（%s)不符，因为评分阿卡最优分箱需要优先确保单调性！" % (len(group), max_interval))
        return group
    
    def check_posRate_monotone(self, count, group):
        if len(group) <= 2:
            return True
        count_update = [count[count.index.isin(ele)].sum(axis=0).tolist() for ele in group]
        count_update = pd.DataFrame(count_update, columns=count.columns)
        count_update['pos_ratio'] = count_update[0] / count_update.sum(axis=1)
        posRate_not_mono = count_update['pos_ratio'].diff()[1:]
        if sum(posRate_not_mono >= 0) == len(posRate_not_mono) or sum(posRate_not_mono <= 0) == len(posRate_not_mono):
            return True
        else:
            return False
    
    def guarantee_proportion(self, count, group, thresh=0.05):
        """
        检查单箱占比
        :param count: DataFrame 待分箱变量的分布统计
        :param group: list 分组信息
        :return: 分组信息
        """
        while True:
            # 计算pos_ratio
            count_update = [count[count.index.isin(ele)].sum(axis=0).tolist() for ele in group]
            count_update = pd.DataFrame(count_update, columns=count.columns)
            count_update['sample_ratio'] = count_update.sum(axis=1) / count_update.values.sum()
            # 合并分组
            if sum(count_update['sample_ratio'] < thresh) > 0:
                idx = count_update[count_update['sample_ratio'] < thresh].index[0]
                group = self.update_group_by_chi2(count_update, group, idx)
            else:
                break
        return group
    
    def processing_when_not_binning(self, data, var_name, var_name_target):
        count, var_type = self.dsct_init(data, var_name, var_name_target, self.feature_type)
        group = np.array(count.index).reshape(-1, 1).tolist()
        # 检查每个箱体是否都有好坏样本
        group = self.guarantee_completeness(count, group)
        print("各分组好坏样本分布检验完成！")
        # 单项占比检验
        group = self.guarantee_proportion(count, group)
        print("各分组单项占比检验完成！")
        return group


class SplitDsct(SuperDiscretization):
    """
    基于split的离散化方法
    """
    
    def group_init(self, count):
        # 获取初始分组
        group = np.array(count.index).reshape(1, -1).tolist()  # 分组信息
        return group
    
    def dsct(self, count, group, max_interval):
        # 重复划分，直到箱体数达到预定阈值
        while len(group) < max_interval:
            group_intv = group[0]  # 待分箱区间
            if len(group_intv) == 1:
                group.append(group[0])
                group.pop(0)
                continue
            
            # 选择最佳分箱点。
            count_intv = count[count.index.isin(group_intv)]  # 待分箱区间内值的数值分布统计
            intv = self.get_best_cutpoint(count_intv)  # 分箱点的下标
            cut_point = group_intv[intv]
            # print("cut_point:%s" % (cut_point))
            
            # 状态更新
            group.append(group_intv[0:intv + 1])
            group.append(group_intv[intv + 1:])
            group.pop(0)
        return group
    
    def get_best_cutpoint(self, count):
        """
        根据指标计算最佳分割点
        :param count: 待分箱区间
        :param group: # 待分箱区间内值的数值分布统计
        :return: 分割点的下标
        """
        # 以每个点作为分箱点（左开右闭区间），以此计算分箱后的指标（熵，信息增益，KS）值=左区间+右区间
        indices_list = self.calc_split_indices(count)
        
        # 最大指标值对应的分割点即为最佳分割点
        intv = indices_list.index(max(indices_list))
        return intv
    
    def calc_split_indices(self, count):
        return np.zeros(len(count))


class entropyDsct(SplitDsct):
    """
    基于最小熵的离散化
    """
    
    def get_best_cutpoint(self, count):
        """
        根据指标计算最佳分割点
        :param count: 待分箱区间
        :param group: # 待分箱区间内值的数值分布统计
        :return: 分割点的下标
        """
        # 以每个点作为分箱点（左开右闭区间），以此计算分箱后的指标（熵，信息增益，KS）值=左区间+右区间
        indices_list = self.calc_split_indices(count)
        
        # 最大指标值对应的分割点即为最佳分割点
        intv = indices_list.index(min(indices_list))
        return intv
    
    def calc_split_indices(self, count):
        """
        根据指标计算最佳分割点
        :param count: 待分箱区间
        :return: 分割点的下标
        """
        entropy_list = self.calc_entropy(count)
        # entropy_list = [calc_entropy(count.iloc[0:idx + 1]) + calc_entropy(count.iloc[idx + 1:]) for idx in
        #                 range(0, len(group) - 1)]
        return entropy_list
    
    def calc_entropy1(self, count):
        """
        计算分组的熵值
        :param count: DataFrame 分组的分布统计
        :return: float 该分组的熵值
        """
        # print(count.index.tolist())
        entropy = count.sum().values / count.values.sum()
        entropy = entropy[entropy != 0]
        entropy = -entropy * np.log2(entropy)
        return entropy.sum()
    
    def calc_entropy(self, count):
        p_clockwise = count.cumsum(axis=0).div(count.sum(axis=1).cumsum(axis=0), axis=0)
        entropy_clockwise = -p_clockwise * np.log2(p_clockwise)
        entropy_clockwise = entropy_clockwise.fillna(0)
        entropy_clockwise = entropy_clockwise.sum(axis=1)
        
        count = count.sort_index(ascending=False)
        p_anticlockwise = count.cumsum(axis=0).div(count.sum(axis=1).cumsum(axis=0), axis=0)
        entropy_anticlockwise = -p_anticlockwise * np.log2(p_anticlockwise)
        entropy_anticlockwise = entropy_anticlockwise.fillna(0)
        entropy_anticlockwise = entropy_anticlockwise.sum(axis=1)
        entropy = [entropy_clockwise.values[idx] + entropy_anticlockwise.values[len(entropy_clockwise.index) - 2 - idx]
                   for idx in range(len(entropy_clockwise.index) - 1)]
        return entropy


class bestKSDsct(SplitDsct):
    """
    基于bestKS的离散化
    """
    
    def calc_split_indices(self, count):
        # 以每个点作为分箱点（左开右闭区间），以此计算分箱后的指标（熵，信息增益，KS）值=左区间+右区间
        ks_list = self.calc_ks(count)
        # ks_list = [self.calc_ks1(count, idx) for idx in range(0, len(count) - 1)]
        return ks_list
    
    def calc_ks1(self, count, idx):
        """
        计算以idx作为分割点，分组的KS值
        :param count: DataFrame 待分箱变量各取值的正负样本数
        :param idx: list 单个分组信息
        :return: 该分箱的ks值

        计算公式：KS_i = |sum_i / sum_T - (size_i - sum_i)/ (size_T - sum_T)|
        """
        ks = 0.0
        # 计算左评分区间的累计好账户数占总好账户数比率（good %)和累计坏账户数占总坏账户数比率（bad %）。
        good_left = count[1].iloc[0:idx + 1].sum() / count[1].sum() if count[1].sum() != 0 else 1  # 左区间
        bad_left = count[0].iloc[0:idx + 1].sum() / count[0].sum() if count[0].sum() != 0 else 1
        # 计算左区间累计坏账户比与累计好账户占比差的绝对值（累计good % -累计bad %）
        ks += abs(good_left - bad_left)
        
        # 计算右评分区间的累计好账户数占总好账户数比率（good %)和累计坏账户数占总坏账户数比率（bad %）。
        good_right = count[1].iloc[idx + 1:].sum() / count[1].sum() if count[1].sum() != 0 else 1  # 右区间
        bad_right = count[0].iloc[idx + 1:].sum() / count[0].sum() if count[0].sum() != 0 else 1  #
        # 计算右区间累计坏账户比与累计好账户占比差的绝对值（累计good % -累计bad %）
        ks += abs(good_right - bad_right)
        return ks
    
    def calc_ks(self, count):
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


if __name__ == '__main__':
    # 数据导入
    from sklearn.datasets import load_iris
    
    data = load_iris()
    x = data.data
    y = data.target
    # y = np.where(y > 1, 1, y)
    data = pd.DataFrame(x, columns=['A', 'B', 'C', 'D'])
    data['E'] = y
    
    # 离散化
    dsct_method_mapping = {'chiMerge': chiMerge, 'chi2': chi2Merge, 'bestChi': bestChi, 'entropy': entropyDsct,
                           'bestKS': bestKSDsct, 'cluster': clusterDsct, 'equal_wide': equalWide,
                           'equal_freq': equalFreq}
    
    # test1
    dsct_method = 'equal_wide'
    binning_estimator = dsct_method_mapping[dsct_method](max_interval=12, feature_type=0)
    group = binning_estimator.dsct_pipeline(data=data, var_name='A', var_name_target='E')
    print(group)
    
    # test2
    import traceback
    dsct_methods = dsct_method_mapping.keys()
    for dsct_method in dsct_methods:
        print(dsct_method, ':')
        binning_estimator = dsct_method_mapping[dsct_method](max_interval=12, feature_type=1)
        try:
            group = binning_estimator.dsct_pipeline(data=data, var_name='A', var_name_target='E')
            print(group)
        except Exception as e:
            print(traceback.format_exc())
        
        print('*' * 100)
        binning_estimator = dsct_method_mapping[dsct_method](max_interval=12, feature_type=0)
        try:
            if dsct_method in ['equal_wide', 'equal_freq']:
                group = binning_estimator.dsct_pipeline(data=data, var_name='A')
            else:
                group = binning_estimator.dsct_pipeline(data=data, var_name='A', var_name_target='E')
            print(group)
        except Exception as e:
            print(traceback.format_exc())
        print('*' * 100)

