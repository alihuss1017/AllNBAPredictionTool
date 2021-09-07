import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot
from mpl_toolkits import mplot3d
import sklearn
from sklearn import *
from sklearn.metrics import *
import pickle
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
trueNegative = 0
truePositive = 0
falseNegative = 0
falsePositive = 0
playerList =[]
statList=[[],
      [],
      [],
      [],
      [],
      [],
      [],
      []]
predList=[]
actList=[]
dataSet = pd.read_csv("DataSet.csv")
dataSet = dataSet[['Player','PTS', 'TRB', 'AST', 'GPnSround%', 'PER', 'WS', 'BPM', 'VORP', 'All-NBA?']]
a=dataSet.pop('Player')
A=np.array(a)

dictNba = {'Prediction': [],'PTS':[],'TRB':[],'AST':[],'GPnSround%':[], 'PER':[],'WS':[],'BPM':[],'VORP':[], 'Actual':[]}
predict = 'All-NBA?'
X = np.array(dataSet.drop([predict], 1))
trueX = X
Y = np.array(dataSet[predict])
LogisticRegression(solver='lbfgs', max_iter=1000)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

best=0
for _ in range(100):

    trainTrueX, testTrueX, trainA, testA, trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(trueX,A, X, Y, test_size=0.3)
    logistic = linear_model.LogisticRegression()
    logistic.fit(trainX, trainY)
    acc = logistic.score(testX,testY)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(logistic, f)

pickle_in = open("studentmodel.pickle", "rb")
logistic = pickle.load(pickle_in)
logPredY= logistic.predict(testX)

stats = 'PTS', 'TRB', 'AST', 'GPnSround%', 'PER', 'WS', 'BPM', 'VORP'
for x in range(len(logPredY)):

    if logPredY[x]==0 and logPredY[x]==testY[x]:
        trueNegative = trueNegative + 1

    if logPredY[x]==1 and logPredY[x]==testY[x]:
        truePositive = truePositive + 1

    if logPredY[x] == 0 and logPredY[x] != testY[x]:
        falseNegative = falseNegative + 1

    if logPredY[x] == 1 and logPredY[x] != testY[x]:
        falsePositive = falsePositive + 1

    if logPredY[x] == 1 or testY[x] == 1:
        playerList.append(testA[x])
        predList.append(logPredY[x])
        actList.append(testY[x])
        for y in range(len(stats)):
            statList[y].append(testTrueX[x][y])
            dictNba[stats[y]] = statList[y]


dictNba['Prediction'] = predList
dictNba['Actual']= actList

Precision = float(precision_score(testY, logPredY))
Recall = float(recall_score(testY, logPredY))
f1Score = float((2*Precision*Recall)/(Precision+Recall))

print(f'True Negatives: {trueNegative}')
print(f'True Positives: {truePositive}')
print(f'False Negatives: {falseNegative}')
print(f'False Positives: {falsePositive}')

print(f'Precision: {Precision}')
print(f'Recall: {Recall}')
print(f'f1Score: {f1Score}')
dictNbaFormat = pd.DataFrame(dictNba)
dictNbaFormat.index=playerList
print(f'{dictNbaFormat}')
x1='PTS'
y1='GPnSround%'
ax = pyplot.axes(projection='3d')
x=dataSet['PTS']*2+dataSet['TRB']+dataSet['AST']+dataSet['PER']+dataSet['WS']*2+dataSet['VORP']*2+dataSet['BPM']*2
y=dataSet[y1]
z=dataSet[predict]

ax.set_xlabel('Stats Combined')
ax.set_ylabel('% of Games Played and Started')
ax.set_zlabel(predict)
ax.scatter(x,y,z)
ax.set_title('Correlation of Stats and Games Played with All-NBA Teams')
pyplot.show()

tnfLabels = ['TP', 'FN', 'FP']
tnfResult = [truePositive, falseNegative, falsePositive]
pyplot.bar(tnfLabels, tnfResult)
pyplot.title('Results of Logistic Regression Algorithm')
pyplot.show()

pnrLabel = ['Precision', 'Recall']
pnrResult = [Precision, Recall]
pyplot.bar(pnrLabel, pnrResult, width = 0.1)
pyplot.title('Precision and Recall of Logistic Regression Algorithm')
pyplot.show()


