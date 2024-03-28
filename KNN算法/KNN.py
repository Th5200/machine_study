'''
encoding:   -*- coding: utf-8 -*-
@Time           :  2024/3/24 0:46
@Project_Name   :  python_project
@Author         :  lhw
@File_Name      :  KNN.py

功能描述

实现步骤
                KNN算法：
                    对未知类别属性的数据集中的每个点依次执行以下操作：
                    (1) 计算已知类别数据集中的点与当前点之间的距离；
                    (2) 按照距离递增次序排序；
                    (3) 选取与当前点距离最小的k个点；
                    (4) 确定前k个点所在类别的出现频率；
                    (5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
'''
import numpy as np  # 科学计算库
import operator  # 运算符模块


def create_dataset():
    '''
    创建数据集及标签
    '''
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = create_dataset()


def classify0(inX, dataSet, labels, k):
    '''
    分类函数
    :param inX: 测试集
    :param dataSet: 训练集
    :param labels: 训练集标签
    :param k: 选择最近的k个点
    :return: 分类结果
    '''
    dataSetSize = dataSet.shape[0]  # 1.获取训练集的行数,行数为训练集的多少,列数为训练集的特征数
    # 2-5为求欧式距离的步骤
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 2.求差值
    sqDiffMat = diffMat ** 2  # 3.求平方
    sqDistances = sqDiffMat.sum(axis=1)  # 4.求平方和,将四行两列数组转化为一行四列数组
    distance = sqDistances ** 0.5  # 5.开方
    sortedDisIndices = distance.argsort()  # 6.返回distance排序后的各元素在distance中的索引位置
    classCount = {}  # 7.设置字典来存放训练集标签出现的频率
    for i in range(k):
        voteIlabel = labels[sortedDisIndices[i]]  # 8.设置varIlabel来存储排序后各元素在distance的索引对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 9.将varIlabel对应的标签放入字典,在下一次出现的时候对其进行累加
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 10.将字典的键值对分解为元组形式并对其进行降序的排序
    return sortedClassCount[0][0]  # 返回排序后的出现频率最高的标签

def file_to_np(filename):
    '''
    该函数是将训练集数据转化为矩阵数据
    :param filename: 文件名
    :return:
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    length_lines = len(lines)  # 获取训练集行数
    zeros_mat = np.zeros((length_lines, 3))  # 以训练集行数为行数,特征值个数为列数,创建0矩阵[几个特征值就把3改成几]
    train_labels = []  # 设置列表(train_labels)接收训练集每个数据的标签
    index = 0  # 设置索引,用来通过索引逐行将训练集数据写入零矩阵
    for line in lines:
        line = line.strip()  # 去除每行字符串中的空格
        list_from_line = line.split('\t')  # 按照制表符'\t'分割每一行数据
        zeros_mat[index, :] = list_from_line[0:3]  # 将每行数据写入零矩阵
        train_labels.append(list_from_line[-1])  # 将每行数据的标签写入列表
        index += 1
    return zeros_mat, train_labels  # 返回矩阵化后的训练集数据和每个数据对应的标签

def auto_norm(dataSet):
    '''
    该函数将数据归一化
    :param dataSet: 训练集
    :return:
    '''
    min_vals = dataSet.min(0)  # 获取训练集每一列的最小值
    max_vals = dataSet.max(0)  # 获取训练集每一列的最大值
    ranges = max_vals - min_vals  # 获取训练集每一列的最大值和最小值的差值
    norm_dataSet = np.zeros(np.shape(dataSet))  # 创建与训练集相同大小的零矩阵
    m = dataSet.shape[0]  # 获取训练集行数
    norm_dataSet = dataSet - np.tile(min_vals, (m, 1))  # 将训练集每一行数据减去训练集每一列的最小值
    norm_dataSet = norm_dataSet / np.tile(ranges, (m, 1))  # 将训练集每一行数据除以训练集每一列的最大值和最小值的差值
    return norm_dataSet, ranges, min_vals  # 返回归一化后的训练集数据,训练集每一列的最大值和最小值的差值,训练集每一列的最小值


def test_classify():
    '''
    测试分类器classify的函数
        (1)设置数据中用于测试的数据比例p
        (2)设置参数接收矩阵化数据及标签[file_to_np函数],再通过auto_norm()函数实现归一化
        (3)获取行数m,test_data = int(m*p)为测试集
        (4)设置error_count累加测试错误数据
        (5)遍历测试集,通过calssify0函数返回测试结果,与测试集的标签进行比对
        (6)如果对,则输出训练正确及结果;不对则使error_count+1,在测试完后(error_count / float(test_data))作为错误率
    :return:
    '''
    proportion = 0.1  # 设置测试集比例
    tran_data, train_labels = file_to_np('datingTestSet.txt')  # 获取训练集数据及标签
    norm_dataSet, ranges, min_vals = auto_norm(tran_data)  # 将训练集数据归一化处理
    m = norm_dataSet.shape[0]  # 获取归一化后的训练集数据行数
    test_data = int(m * proportion)  # 获取测试集数据行数
    error_count = 0  # 创建错误计数器
    for i in range(test_data):
        classifier_result = classify0(norm_dataSet[i, :], norm_dataSet[test_data:m, :], train_labels[test_data:m], 3)
        print(f"分类器返回结果: {classifier_result}, 真正的结果: {train_labels[i]}")
        if classifier_result != train_labels[i]:
            error_count += 1
    print(f"错误率: {error_count / float(test_data)}")


def main():
    filename = input('请输入训练集文件名:')  # 输入测试集文件名
    print('训练开始...')
    print('正在将文件中的训练集数据转化为矩阵数据...')
    train_data, train_labels = file_to_np(filename)
    print('正在将训练集数据归一化...')
    norm_dataSet, ranges, min_vals = auto_norm(train_data)
    print('正在测试分类器...')
    test_classify()
    print('训练结束...')
    inX = input('请输入需要分类的信息')
    print('正在分类...')
    result = classify0(inX, norm_dataSet, train_labels, 3)
    print(f'分类完成,结果为:{result}')


if __name__ == '__main__':
    test_classify()

