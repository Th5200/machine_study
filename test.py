'''
encoding:   -*- coding: utf-8 -*-
@Time           :  2024/3/24 1:15
@Project_Name   :  python_project
@Author         :  lhw
@File_Name      :  test.py

功能描述

实现步骤

'''
import numpy as np
import operator


A = np.array([[1, 2], [3, 4]])
B = np.tile(5, (4, 2))
print(f'原数组:\n{A}\n复制后数组:\n{B}')

arr = np.array([3, 1, 4, 2, 5])
sort_arr = arr.argsort()
print(f'原数组:\n{arr}\n排序后各数在原数组中的索引:\n[1 2 3 4 5]\n{sort_arr}')

dic = {'A':1, 'B':2}
print(f'原字典:\n{dic}')
print(f'分解后字典:\n{dic.items()}')
print(f'降序排序后:{sorted(dic.items(), reverse=True, key=operator.itemgetter(1))}')