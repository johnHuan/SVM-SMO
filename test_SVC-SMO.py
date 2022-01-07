# -*- coding: utf-8 -*-
# @Time    : 2022/1/7 14:22
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : test_SVC-SMO.py
from __future__ import division, print_function
import csv, os, sys
import numpy as np
from SVCSMO import SVCSMO

filepath = os.path.dirname(os.path.abspath(__file__))
filepath = filepath.replace('\\', '/')


def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN) / len(y)


def calc_mse(y, y_hat):
    return np.nanmean(((y - y_hat) ** 2))


def test_main(filename, C=1.0, kernel_type='linear', epsilon=0.001):
    # Load data
    data = np.loadtxt('%s/%s' % (filepath, filename), delimiter=',', dtype=np.float32)
    data = data.astype(float)

    # Split data
    X, y = data[:, 0:-1], data[:, -1].astype(int)

    # Initialize model
    model = SVCSMO()

    # Fit model
    support_vectors, iterations = model.fit(X, y)

    # Support vector count
    sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(X)

    # Calculate accuracy
    acc = calc_acc(y, y_hat)
    mse = calc_mse(y, y_hat)

    print("Support vector count: %d" % (sv_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))
    print("mse:\t%.3f" % (mse))
    print("Converged after %d iterations" % (iterations))


if __name__ == '__main__':
    param = {}
    # param['filename'] = 'small_data/iris-slwc.txt'
    # param['filename'] = 'small_data/iris-versicolor.txt'
    param['filename'] = 'small_data/iris-virginica.txt'
    param['C'] = 0.1
    # param['kernel_type'] = 'linear'
    # param['kernel_type'] = 'quadratic'
    param['kernel_type'] = 'gaussian'
    param['epsilon'] = 0.001

    test_main(**param)
