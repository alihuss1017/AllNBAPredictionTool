import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
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


best=0
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

dictNba = {'Prediction': [],'PTS':[],'TRB':[],'AST':[],'GPnSround%':[], 'PER':[],'WS':[],'BPM':[],'VORP':[], 'Actual':[]}
stats = 'PTS', 'TRB', 'AST', 'GPnSround%', 'PER', 'WS', 'BPM', 'VORP'
predict = 'All-NBA?'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

dataSet = pd.read_csv("DataSet.csv")
dataSet = dataSet[['Player','PTS', 'TRB', 'AST', 'GPnSround%', 'PER', 'WS', 'BPM', 'VORP', 'All-NBA?']]
a=dataSet.pop('Player')
A=np.array(a)

X = np.array(dataSet.drop([predict], 1))
trueX = X
Y = np.array(dataSet[predict])

LogisticRegression(solver='lbfgs', max_iter=1000)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


for _ in range(50):

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
dictNbaFrame = pd.DataFrame(dictNba)
dictNbaFrame.index=playerList

def printResults():
    print(f'True Negatives: {trueNegative}')
    print(f'True Positives: {truePositive}')
    print(f'False Negatives: {falseNegative}')
    print(f'False Positives: {falsePositive}')
    print(f'Precision: {Precision}')
    print(f'Recall: {Recall}')
    print(f'{dictNbaFrame}')

def statRelationPlot():
    plotStats = ['PTS','PER','WS','BPM','VORP']
    for stat in plotStats:
        x1= stat
        y1='GPnSround%'

        ax = plt.axes(projection='3d')
        x=dictNbaFrame[stat]
        y=dictNbaFrame[y1]
        z=dictNbaFrame['Prediction']
        z2=dictNbaFrame['Actual']

        ax.set_xlabel(f'{stat}')
        ax.set_ylabel('% of Games Played and Started')
        ax.set_zlabel(predict)

        ax.scatter(x,y,z, label='Prediction')
        ax.scatter(x,y,z2, label='Actual')
        ax.set_title(f'Correlation of {stat} and Games Played with All-NBA Teams')
        plt.show()

def tnfPlot():
    tnfLabels = ['TP', 'FN', 'FP']
    tnfResult = np.array([truePositive, falseNegative, falsePositive])

    plt.bar(tnfLabels, tnfResult, width=0.3)
    plt.yticks(np.arange(0, tnfResult.max()+1, 1))
    plt.title('Results of Logistic Regression Algorithm')
    plt.show()

def pnrPlot():
    pnrLabel = ['Precision', 'Recall']
    pnrResult = np.array([Precision, Recall])

    plt.bar(pnrLabel, pnrResult, width = 0.5)
    plt.yticks(np.arange(0,1.05,.05))
    plt.title('Precision and Recall of Logistic Regression Algorithm')
    plt.show()

def f1Score():
    f1ScoreLR = float((2*Precision*Recall)/(Precision+Recall))
    print(f'f1Score of Logistic Regression: {f1ScoreLR}')
    return f1ScoreLR

def main():
    printResults()
    statRelationPlot()
    tnfPlot()
    pnrPlot()
    f1Score()

if __name__ =="__main__":
    main()
