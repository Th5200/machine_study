'''
encoding:   -*- coding: utf-8 -*-
@Time           :  2024/3/27 12:45
@Project_Name   :  python_project
@Author         :  lhw
@File_Name      :  trees.py

功能描述

实现步骤

'''
import math
import operator


def calculate_entropy(data_set):
    """
    计算给定数据集的信息熵
    :param data_set: 数据集
    :return: 信息熵
    """
    num_entries = len(data_set)  # 计算数据集的实例总数(行数),即所有实例的数量,后面用于计算每个实例在数据集中出现的概率
    label_counts = {}  # 创建数据字典,键值为数据集最后一列的数值[标签]出现的次数,记录当前类别出现的次数
    for data in data_set:
        current_label = data[-1]  # 获取当前实例的标签
        if current_label not in label_counts.keys():  # 如果标签不在字典中,则添加
            label_counts[current_label] = 0
        label_counts[current_label] += 1  # 如果当前标签在字典中,则将标签出现的次数加1
    shannon_entropy = 0.0  # 初始化信息熵为0
    for key in label_counts:  # 遍历字典,计算信息熵
        prob = float(label_counts[key]) / num_entries  # 计算当前标签在数据集中出现的概率
        shannon_entropy -= prob * math.log(prob, 2)  # 计算信息熵
    return shannon_entropy  # 返回信息熵


def split_data_set(data_set, axis, value):
    """
    按照给定的特征划分数据集
    :param data_set: 数据集
    :param axis: 划分的特征
    :param value: 特征的值
    :return: 划分后的数据集
    """
    ret_data_set = []  # 创建一个空列表,用于存储划分后的数据集
    for data in data_set:  # 遍历数据集
        if data[axis] == value:  # 如果当前实例的特征值等于给定的特征值
            reduced_data = data[:axis]  # 创建一个空列表,用于存储当前实例的特征值
            reduced_data.extend(data[axis + 1:])  # 将当前实例的特征值后面的特征值添加到列表中
            ret_data_set.append(reduced_data)  # 将列表添加到数据集中
    return ret_data_set  # 返回划分后的数据集


def choose_best_feature_to_split(data_set):
    """
    选择最好的特征进行划分
    :param data_set: 数据集
    :return:
    """
    num_features = len(data_set[0]) - 1  # 计算数据集的列数,即特征数,减去标签列
    best_info_gain = 0.0  # 初始化信息增益为0
    best_feature = -1  # 初始化最佳特征为-1
    base_entropy = calculate_entropy(data_set)  # 计算数据集的信息熵
    for i in range(num_features):  # 遍历数据集的列,即特征
        feature_list = [data[i] for data in data_set]  # 创建一个空列表,用于存储当前特征列的值
        unique_values = set(feature_list)  # 将列表内的特征值唯一化
        new_entropy = 0.0  # 初始化新的信息熵为0
        for value in unique_values:  # 遍历特征列的值
            sub_data_set = split_data_set(data_set, i, value)  # 划分数据集
            prob = len(sub_data_set) / float(len(data_set))  # 计算当前特征列的值在数据集中出现的概率
            new_entropy += prob * calculate_entropy(sub_data_set)  # 计算新的信息熵
        info_gain = base_entropy - new_entropy  # 计算信息增益
        if info_gain > best_info_gain:  # 如果信息增益大于最佳信息增益
            best_info_gain = info_gain  # 更新最佳信息增益
            best_feature = i  # 更新最佳特征
    return best_feature  # 返回最佳特征


def select_labels(data_list):
    '''
    挑选出出现最多的标签
    :param data_list: 列表形数据
    :return:
    '''
    class_count = {}  # 设置字典来接收标签极其出现的次数
    for vote in data_list:
        if vote not in class_count.keys(): class_count = 0  # 如果该标签未出现,则设置该标签为0
        class_count[vote] += 1  # 如果该标签已经存在,则+1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # 降序排序,key=operator.itemgetter(1)是比较字典的第二项
    return sorted_class_count[0][0]  # 返回出现最多的标签



data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'maybe']]
labels = ['no surfacing', 'flippers']
# print(choose_best_feature_to_split(data_set))
print(data_set)
print(split_data_set(data_set, 1, 1))
