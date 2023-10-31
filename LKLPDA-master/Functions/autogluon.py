# -*- coding: utf-8 -*

from sklearn.ensemble import RandomForestClassifier

from numpy import *
from sklearn.metrics import roc_curve, auc, precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
import  csv
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularDataset, TabularPredictor
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#random.seed ( 8 )

import utils
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
def StorFile1(data,fileName):
    data = list(map(lambda x:[x],data))
    with open(fileName,'w',newline ='') as f:
        mywrite = csv.writer(f)
        for i in data:
            mywrite.writerow(i)

if __name__ == "__main__":

    num = 0
    result = []
    " for train, test in cv.split(SampleFeature, SampleLabel):"
    while num < 5:

            X_train = []
            ReadMyCsv(X_train, "../Datasets/five-fold/train_feature" + str(num + 1) + ".csv")
            Y_train = []
            ReadMyCsv(Y_train, "../Datasets/five-fold/train_label" + str(num + 1) + ".csv")
            X_test = []
            ReadMyCsv(X_test, "../Datasets/five-fold/test_feature" + str(num + 1) + ".csv")
            Y_test = []
            ReadMyCsv(Y_test, "../Datasets/five-fold/test_label" + str(num + 1) + ".csv")

            label = 523

            save_path = './predict_models/' + str(num)  # where to save trained models
            # save_path = './LKL_models1/' + str(0)

            Train = np.hstack((X_train, Y_train))

            train_data = pd.DataFrame(Train)
            X_test = pd.DataFrame(X_test)

            # predicted = model.fit(SampleFeature[train], SampleLabel[train]).predict_proba(SampleFeature[test])
            # Y_prob = predicted[:,1]
            # Y_pred = model.predict(X_test)

            metric = 'roc_auc'
            presets = 'best_quality'
            #redictor = TabularPredictor(label=label, path=save_path).fit(train_data)

            predictor = TabularPredictor.load(save_path)
            Y_prob = predictor.predict_proba(X_test)
            Y_pred = predictor.predict(X_test)

            Y_prob = [i[1] for i in Y_prob.values.tolist()]
            label = utils.list_toColumn(Y_prob)
            # StorFile(label, './Case_add_data/' +'prob.csv')
            Y_pred = Y_pred.values.tolist()
            Y_pred = [float(i) for i in Y_pred]

            print(dir(predictor))

            temporary = []
            # Tags are encoded as integer types
            le = LabelEncoder()
            Y_test_encoded = le.fit_transform(Y_test)
            Y_pred_encoded = le.transform(Y_pred)
            temporary = utils.save_result(Y_test_encoded, Y_prob, Y_pred_encoded, num)
            result.extend(temporary)
            num = num + 1

    acc_values, std_values = utils.save_fullreports(result)
    mean_auc, std_auc = utils.AUC()
    utils.AUPR()
    print("---------------roc_auc-------------------")
    print("---mean_auc---", mean_auc)
    print("--std_auc--", std_auc)
    print("--acc_values--", acc_values)
    print("--std_values--", std_values)

