'''
encoding:   -*- coding: utf-8 -*-
@Time           :  2023/10/26 12:55
@Project_Name   :  机器学习
@Author         :  lhw
@File_Name      :  main.py

功能描述

实现步骤

'''
# from numpy import *
import numpy as np
from KNN算法 import KNN

result = np.random.rand(4, 4)
print(f'随机数组:\n{result}')
randMat = np.mat(result)
print(f'矩阵化:\n{randMat}')
invRandMat = randMat.I
print(f'矩阵的逆:\n{invRandMat}')
content = randMat.I * randMat
print(f'矩阵乘法:\n{content}')

zero = np.zeros((1000, 3))
print(zero)

