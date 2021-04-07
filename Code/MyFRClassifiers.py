from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from Code.Message import create_dir

import random
import numpy as np
import matplotlib.pyplot as plt

from Code.MyConfusionMatrixPrinter import MyPlot_Confusion_Matrix as plot_confusion_matrix

import Code.ClassificationReportPrinter as crPrint


def split_data_labels(data_list):
    # extract data and labels
    data, label = [], []
    for row in data_list:
        data.append(row[:-1])
        label.append(row[-1])

    return data, label


def prepare_data(data, label, n_img_person, n_img_train):
    x_train, x_test, y_train, y_test = [], [], [], []
    L = n_img_person  # number of images per person (10)
    n = n_img_train

    for i in range(0, len(label), L):
        x = data[i:i + L]
        y = label[i:i + L]
        random.shuffle(x)

        x_train.extend(x[L - n:])
        x_test.extend(x[:L - n])
        y_train.extend(y[L - n:])
        y_test.extend(y[:L - n])

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def run_svm(x_train, x_test, y_train, y_test, dir=None, iterations=0, jiterations=0, cfa=False):
    if dir is None:
        if not cfa:
            dir = 'CM_noCFA'
        else:
            dir = 'CM_cfa'

    print('--------------- Begin SVM ------------------')
    print('Iteration: ', iterations, jiterations)
    # Train
    svm_classifier = svm.SVC()
    svm_classifier.fit(x_train, y_train)

    # Test
    y_pred = svm_classifier.predict(x_test)

    # Results
    acc = metrics.accuracy_score(y_test, y_pred)
    macroAV = metrics.precision_score(y_test, y_pred, average='macro')
    weightedAV = metrics.precision_score(y_test, y_pred, average='weighted')

    cm = confusion_matrix(y_test, y_pred)

    # Print
    h = (8 + (8 / 6) * 6)
    w = (6 + (8 / 6) * 6)
    fig, ax = plt.subplots(figsize=(h, w))

    # CONFUSION MATRIX
    plot_confusion_matrix(cm, y_pred, y_test, cmap=plt.cm.Blues, ax=ax)

    create_dir(dir)
    plt.savefig(dir+'/SVM_c_matrix_iteration_'+str(iterations)+str(jiterations)+'.png', dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()

    # CLASSIFICATION REPORT
    clfreport = classification_report(y_test, y_pred, zero_division=0)
    crPrint.plot_classification_report(clfreport)

    create_dir(dir+'ClsReport/')
    plt.savefig(dir+'ClsReport/' + 'SVM_c_report_iteration_' +str(iterations)+str(jiterations)+'.png', dpi=200, format='png',
                bbox_inches='tight')
    # plt.show()
    plt.close()

    # # Print CM to console screen
    # for _l in cm:
    #     print()
    #     for i in _l:
    #         print(i, end=' ')
    #

    print('\n', classification_report(y_test, y_pred, zero_division=0))
    print('--------------- END SVM ------------------')
    return acc, macroAV, weightedAV, clfreport, cm


def run_knn(x_train, x_test, y_train, y_test, dir=None, iterations=0, jiterations=0, cfa=False, n_neighbors=5):
    if dir is None:
        if not cfa:
                dir = 'CM_noCFA'
        else:
                dir = 'CM_cfa'
    print('--------------- Begin KNN ------------------')
    print('Iteration: ', iterations, jiterations)
    # Train
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(x_train, y_train)

    # Test
    y_pred = classifier.predict(x_test)

    # Results
    acc = metrics.accuracy_score(y_test, y_pred)
    macroAV = metrics.precision_score(y_test, y_pred, average='macro')
    weightedAV = metrics.precision_score(y_test, y_pred, average='weighted')

    cm = confusion_matrix(y_test, y_pred)

    # Print
    h = (8 + (8 / 6) * 6)
    w = (6 + (8 / 6) * 6)
    fig, ax = plt.subplots(figsize=(h, w))

    # CONFUSION MATRIX
    plot_confusion_matrix(cm, y_pred, y_test, cmap=plt.cm.Blues, ax=ax)

    create_dir(dir)
    plt.savefig(dir+'/KNN_c_matrix_iteration_'+str(iterations)+str(jiterations)+'.png', dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()

    # CLASSIFICATION REPORT
    clfreport = classification_report(y_test, y_pred, zero_division=0)
    crPrint.plot_classification_report(clfreport)

    create_dir(dir+'ClsReport/')
    plt.savefig(dir+'ClsReport/' + 'KNN_c_report_iteration_' +str(iterations)+str(jiterations)+'.png', dpi=200, format='png',
                bbox_inches='tight')
    # plt.show()
    plt.close()

    # # Print CM to console screen
    # for _l in cm:
    #     print()
    #     for i in _l:
    #         print(i, end=' ')
    #

    print('\n', classification_report(y_test, y_pred, zero_division=0))
    print('--------------- END KNN ------------------')
    return acc, macroAV, weightedAV, clfreport, cm


# # old
# def run_knn(x_train, x_test, y_train, y_test):
#     # Train
#     classifier = KNeighborsClassifier(n_neighbors=5)
#     classifier.fit(x_train, y_train)
#
#     # Test
#     y_pred = classifier.predict(x_test)
#
#     # Results
#     acc = metrics.accuracy_score(y_test, y_pred)
#
#     for _l in confusion_matrix(y_test, y_pred):
#         print()
#         for i in _l:
#             print(i, end=' ')
#
#     print(classification_report(y_test, y_pred, zero_division=0))
#
#     return acc
