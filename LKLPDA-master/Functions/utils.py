# -*- coding: utf-8 -*
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import tensorflow

from sklearn.metrics import roc_curve,auc,precision_recall_curve
import os
import joblib
RESULT='../Datasets/result/'


def prework():

    try:
        os.mkdir('./result')
    except:
        pass
    try:
        os.mkdir('./AUC')
    except:
        pass
    try:
        os.mkdir('./AUPR')
    except:
        pass
    try:
        os.mkdir('./casestudy')
    except:
        pass



def ReadMyCsv(SaveList, fileName):

    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except:
                pass

        SaveList.append(row)
    return
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
def list_toRow(x):
    y=[]
    for i in x:

        y.append(i[0])
    return y
def list_toColumn(x):

    y=[]
    for i in x:

        y.append([i])

    return y
def save_model_keras(model1,path):


    model1.save(path)

def Density_Plot(x,y): #密度图
    plt.figure(figsize=(10, 5), dpi=300)
    sns.kdeplot(x, shade=True, color="black", label="Cyl=5", alpha=.7)
    sns.kdeplot(y, shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
    # sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)
    #plt.ylim(0, 5)
    # Decoration
    plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=22)
    plt.legend()
    plt.savefig('./Density_Plot/Density Plot.tif')


def AUPR():
    # 用于保存混淆矩阵
    AllResult = []
    Ps = []
    Rs = []
    Pre =[]
    RPs = []
    mean_R = np.linspace(0, 1, 1000)
    # 使用一个灰色的背景图
    plt.figure(figsize=(5.4, 4.5), facecolor='w')
    ax = plt.axes(facecolor='#FFFFFF')
    ax.set_axisbelow(True)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.125))
    # 加上白色实心的网格线
    plt.xticks(np.linspace(0, 1, 5))
    plt.yticks(np.linspace(0, 1, 5))
    plt.grid(color='#CCCCCC', linestyle='solid', linewidth=1.5)
    plt.grid(which='minor', linewidth=1, linestyle='--', color='#CCCCCC')

    plt.grid(which='minor', linewidth=1, linestyle='--', color='#CCCCCC')
    # 隐藏坐标系的外围框线

    # 隐藏上方与右侧的坐标轴刻度
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    counter0 = 0
    i=0
    while counter0 < 5:
        #print(i)
        # 读取文件
        RealAndPrediction = []
        RealAndPredictionProb = []
        RAPName = RESULT+'/'+str(counter0) + '.csv'
        RAPNameProb = RESULT+'/'+str(counter0) + '_prob.csv'
        ReadMyCsv(RealAndPrediction, RAPName)
        ReadMyCsv(RealAndPredictionProb, RAPNameProb)
        # 生成Real和Prediction
        Real = []
        Prediction = []
        PredictionProb = []
        counter = 0
        while counter < len(RealAndPrediction):
            Real.append(int(RealAndPrediction[counter][0]))
            Prediction.append(RealAndPrediction[counter])
            PredictionProb.append(RealAndPredictionProb[counter])
            counter = counter + 1

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)

        Pre.append(np.mean(precision, axis=0))
        # 阶梯状
        # plt.step(recall, precision, color=colorlist[i], alpha=0.4, where='post')
        # 弧线
        plt.plot(recall, precision, lw=2, alpha=0.5,
                 label='fold %d (AUPR = %0.4f)' % (i + 1, average_precision))



        print('-------------AUPR--------------', average_precision)
        i += 1
        counter0 = counter0 + 1

    # # 画均值
    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    std_RPs = np.std(RPs)
    # plt.plot(recall, precision, lw=2, alpha=0.5,
    #          label='Mean ( AUPR = %0.4f±%0.4f)' % (mean_RPs, std_RPs))
    print('-------------meanprc--------------',mean_RPs,std_RPs)
    a = mean_P
    b = mean_R
    c = []
    for i in a:
        c.append([i])
    a = c
    c = []
    for i in b:
        c.append([i])
    b = c
    StorFile(a, './AUPR/AUPR_mean_P.csv')
    StorFile(b, './AUPR/AUPR_mean_R.csv')
    plt.plot(mean_P, mean_R, color='b',
             label=r'Mean (AUPR = %0.4f $\pm$ %0.4f)' % (mean_RPs, std_RPs),
             lw=2.5, alpha=0.8)
    # MyEnlarge(0, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_P, mean_R, 2, colorlist[5])

    PAndR = []
    counter = 0
    while counter < len(mean_P):
        pair = []
        pair.append(mean_P[counter])
        pair.append(mean_R[counter])
        PAndR.append(pair)
        counter = counter + 1
    StorFile(PAndR, './AUPR/PAndRAttribute.csv')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    #plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(mean_RPs))
    # 画网格

    # 画对角线
    plt.plot([-0.05, 1.05], [1.05, -0.05], linestyle='--', lw=2, color='#999999', alpha=.6)
    # plt.legend(bbox_to_anchor=(0.65, 0.40))
    plt.legend(loc="lower left")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    #plt.savefig('./AUPR/PR-5fold.svg', dpi=300)
    plt.savefig('./AUPR/PR-5fold.tif', dpi=300)
    #plt.show()
def AUC( ):

    # 用于保存混淆矩阵
    AllResult = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # 使用一个灰色的背景图
    plt.figure(figsize=(5.4, 4.5), facecolor='w')
    ax = plt.axes(facecolor='#FFFFFF')
    ax.set_axisbelow(True)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.125))
    # 加上白色实心的网格线
    plt.xticks(np.linspace(0, 1, 5))
    plt.yticks(np.linspace(0, 1, 5))
    plt.grid(color='#CCCCCC', linestyle='solid', linewidth=1.5)
    plt.grid(which='minor', linewidth=1, linestyle='--', color='#CCCCCC')

    plt.grid(which='minor', linewidth=1, linestyle='--', color='#CCCCCC')
    # 隐藏坐标系的外围框线

    # 隐藏上方与右侧的坐标轴刻度
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    counter0 = 0
    i=0
    while counter0 < 5:
        #print(i)
        # 读取文件
        RealAndPrediction = []
        RealAndPredictionProb = []
        RealAndPredictionPred=[]

        RAPName = RESULT+'/'+str(counter0) + '.csv'
        RAPNameProb = RESULT+'/'+str(counter0) + '_prob.csv'
        RAPNamePred = RESULT+'/'+str(counter0) + '_pred.csv'
        ReadMyCsv(RealAndPrediction, RAPName)
        ReadMyCsv(RealAndPredictionProb, RAPNameProb)
        ReadMyCsv(RealAndPredictionPred, RAPNamePred)
        y_test = []
        for iq in RealAndPrediction:
            y_test.append(iq[0])
        predicted1 = []
        for iq in RealAndPredictionPred:
            predicted1.append(iq[0])

        # print("==================", counter0 + 1, "fold", "==================")
        # print('Test accuracy: ', accuracy_score(y_test, predicted1))
        # print(classification_report(y_test, predicted1, digits=4))
        # print(confusion_matrix(y_test, predicted1))
        # 生成Real和Prediction
        Real = []
        Prediction = []
        PredictionProb = []
        counter = 0
        while counter < len(RealAndPrediction):
            Real.append(int(RealAndPrediction[counter][0]))
            Prediction.append(RealAndPrediction[counter])
            PredictionProb.append(RealAndPredictionProb[counter])
            counter = counter + 1
        fpr, tpr, thresholds = roc_curve(Real, PredictionProb)
        #average_precision = average_precision_score(Real, PredictionProb)
        #precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 阶梯状
        # plt.step(recall, precision, color=colorlist[i], alpha=0.4, where='post')
        # 弧线
        plt.plot(fpr, tpr, lw=2, alpha=0.5,
                 label='fold %d (AUC = %0.4f)' % (i + 1, roc_auc))

        print('---------auc---------------',roc_auc)

        i += 1
        counter0 = counter0 + 1

    # # 画均值
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    part = []
    for i in mean_fpr:
        part.append([i])
    StorFile(part, './AUC/mean_fpr.csv')
    part = []
    for i in mean_tpr:
        part.append([i])
    StorFile(part, './AUC/mean_tpr.csv')
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2.5, alpha=0.8)
    # MyEnlarge(0, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_P, mean_R, 2, colorlist[5])
    # print('---------auc---------------',mean_auc,std_auc)
    # plt.plot(fpr, tpr, lw=2, alpha=0.8,
    #          label='Mean (AUC = %0.4f±%0.4f)' % ( mean_auc, std_auc))


    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    #plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(mean_RPs))
    # 画网格

    # 画对角线
    plt.plot([-0.05, 1.05],[-0.05, 1.05], linestyle='--', lw=2, color='#999999', alpha=.6)
    # plt.legend(bbox_to_anchor=(0.65, 0.40))
    plt.legend(loc="lower right")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    #plt.savefig('./AUC/AUC-5fold.svg', dpi=300)
    plt.savefig('./AUC/AUC-5fold.tif', dpi=300)
    #plt.show()
    return mean_auc, std_auc

def NegativeGenerate(LncDisease, AllRNA,AllDisease):
    # 负样本为全部的disease-rna（328*881）中随机抽取，未在内LncDisease即为负样本
    import random
    NegativeSample = []
    counterN = 0
    while counterN < len(LncDisease):  # 随机选出一个疾病rna对
        counterR = random.randint(0, len(AllRNA) - 1)
        counterD = random.randint(0, len(AllDisease) - 1)
        DiseaseAndRnaPair = []
        DiseaseAndRnaPair.append(AllRNA[counterR][0])
        DiseaseAndRnaPair.append(AllDisease[counterD][0])
        flag1 = 0
        counter = 0
        while counter < len(LncDisease):
            if DiseaseAndRnaPair == LncDisease[counter]:
                flag1 = 1
                #print('1!')
                break
            counter = counter + 1
        if flag1 == 1:
            continue
        flag2 = 0
        counter1 = 0
        while counter1 < len(NegativeSample):  # 在已选的负样本中没有，防止重复
            if DiseaseAndRnaPair == NegativeSample[counter1]:
                flag2 = 1
                #print('2!')
                break
            counter1 = counter1 + 1
        if flag2 == 1:
            continue
        if (flag1 == 0 & flag2 == 0):
            NamePair = []  # 生成对
            NamePair.append(AllRNA[counterR][0])
            NamePair.append(AllDisease[counterD][0])
            NegativeSample.append(NamePair)

            counterN = counterN + 1

    return NegativeSample

def save_result(label,prob,pred,n):
    report = classification_report(label, pred,digits=4, output_dict=True)
    acc=accuracy_score(label, pred)
    label = list_toColumn(label)
    prob = list_toColumn(prob)
    pred = list_toColumn(pred)
    StorFile(label, '../Datasets/result/'+ str(n)+'.csv')
    StorFile(prob, '../Datasets/result/' + str(n) + '_prob.csv')
    StorFile(pred, '../Datasets/result/' + str(n) + '_pred.csv')
    #print("Save Model!!!")

    report_data = []
    df = pd.DataFrame(report).transpose()
    row_data = df.loc['macro avg'].values
    row = {}

    row['acc'] = acc
    row['precision'] = float(row_data[0])
    row['recall'] = float(row_data[1])
    row['f1_score'] = float(row_data[2])

    report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('../Datasets/result/classification_report'+str(n)+'.csv', index=False)
    return report_data

def casestudy(diseaseName):
    HMDD = []
    ReadMyCsv(HMDD, "MDCuiMiDisease.csv")

    HMDDnum = []
    ReadMyCsv(HMDDnum, "MDCuiMiDiseaseNum.csv")

    feature = []
    ReadMyCsv3(feature, "casestudyAllNodeAttributeMannerNum.csv")

    miRNA=[]
    miRNAandnum = []
    for i in range(len(HMDD)):

        if HMDD[i][0] not in miRNA:
            miRNA_part=[]
            miRNA.append(HMDD[i][0])
            miRNA_part.append(HMDD[i][0])
            miRNA_part.append(HMDDnum[i][0])
            miRNAandnum.append(miRNA_part)

    disease = []
    diseaseandnum = []
    for i in range(len(HMDD)):

        if HMDD[i][1] not in disease:
            disease_part = []
            disease.append(HMDD[i][1])
            disease_part.append(HMDD[i][1])
            disease_part.append(HMDDnum[i][1])
            diseaseandnum.append(disease_part)





    casestudylist=[]
    diseaseName_list=[]



    for i in miRNA:

        part=[]

        part.append(i)
        part.append(diseaseName)
        if part not in HMDD:


            casestudylist.append(part)





    casestudydiseasenum=[]
    casestudymiRNAnum = []
    for i in casestudylist:

        for j in diseaseandnum:

            if i[1]==j[0]:

                part=[]
                part.append(j[1])
                casestudydiseasenum.append(part)




        for k in miRNAandnum:

            if i[0] == k[0]:
                part = []
                part.append(k[1])
                casestudymiRNAnum.append(part)



    casestudynum = np.hstack((casestudymiRNAnum, casestudydiseasenum))

    test=[]


    casestudydiseasefeature = []
    for i in casestudynum:

        casestudydiseasefeature_part=[]

        casestudydiseasefeature_part.extend(feature[int(i[0])])
        casestudydiseasefeature_part.extend(feature[int(i[1])])

        casestudydiseasefeature.append(casestudydiseasefeature_part)



    model = joblib.load("casestudy.model")

    y_score=model.predict_proba(casestudydiseasefeature)

    y_score=np.hstack((casestudylist, y_score))

    workbook = Workbook()
    booksheet = workbook.active

    for i in y_score.tolist():
        booksheet.append(i)




    workbook.save("./casestudy/caststudy"+str(diseaseName)+".xlsx")

def save_fullreports(reports):
    reports = pd.DataFrame.from_dict(reports)
    reports.to_csv('../Datasets/result/classification_report.csv', index=False)
    reports=[]
    ReadMyCsv(reports, '../Datasets/result/classification_report.csv')
    reports_values=np.array(reports[1:])
    Means=[]
    for i in range(0,len(reports_values[0])):
        Mean=np.mean(reports_values[:,i])
        std = np.std(reports_values[:,i])
        if i==0:
            acc_values=Mean
            std_values=std
        Mean=round(Mean, 4)
        std = round(std, 4)
        Means.extend([str(Mean)+'±'+str(std)])
    reports.extend([Means])
    StorFile(reports, '../Datasets/result/classification_report.csv')
    return acc_values,std_values

def AE(layer,epochs,batch_size,mat_contents):
    encoding_dim=600
    layer=int(layer)
    epochs=int(epochs)
    batch_size=int(batch_size)
    L=[int(len(mat_contents[0])/(2**i)) for i in range(layer)]

    # print(L[0])
    x_train_test = []
    x_test_test = []
    i = 0
    for row in mat_contents:  # 接口
        a = row
        x_train_test.append(a)
        i = i + 1

    x_train_test = np.array(x_train_test)
    x_train_test = x_train_test.reshape((x_train_test.shape[0], -1))
    x_train = x_train_test
    # in order to plot in a 2D figure
    # this is our input placeholder
    input_img = tensorflow.keras.layers.Input(shape=(L[0],))

    # encoder layers
    for i in range(layer-1):
        # print(L[i])
        if i==0:
            encoded = tensorflow.keras.layers.Dense(L[1], activation='relu')(input_img)
        else:    
            encoded = tensorflow.keras.layers.Dense(L[i+1], activation='relu')(encoded)
    
    encoder_output = tensorflow.keras.layers.Dense(encoding_dim)(encoded)
    
    # decoder layers
    decoded = tensorflow.keras.layers.Dense(L[-1], activation='relu')(encoder_output)
    for i in range(layer-1):               
        decoded = tensorflow.keras.layers.Dense(L[layer-i-1], activation='relu')(decoded)
    
    decoded = tensorflow.keras.layers.Dense(L[0], activation='tanh')(decoded)
    # construct the autoencoder model
    autoencoder = tensorflow.keras.models.Model(input_img, decoded)
    encoder = tensorflow.keras.models.Model(input_img, encoder_output)
    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    # training
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=0)
    # plotting
    encoded_imgs = encoder.predict(x_train)
    # to xlsx
    return encoded_imgs