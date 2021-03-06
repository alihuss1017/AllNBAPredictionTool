import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sklearn
from sklearn import *
from sklearn.metrics import *
import pickle
import LogRegModel
from LogRegModel import *

if __name__ == "__LogRegModel__":
    main()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

trueNegative = 0
truePositive = 0
falseNegative = 0
falsePositive = 0

playerList=[]
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
dataSet = pd.read_csv("DataSet.csv")
dataSet = dataSet[['Player','PTS', 'TRB', 'AST', 'GPnSround%', 'PER', 'WS', 'BPM', 'VORP', 'All-NBA?']]

a=dataSet.pop('Player')
A=np.array(a)

predict = 'All-NBA?'
X = np.array(dataSet.drop([predict], 1))
Y = np.array(dataSet[predict])


best=0
for _ in range(50):

    trainA, testA, trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(A, X, Y, test_size=0.3)
    clf = svm.SVC(kernel="linear")
    clf.fit(trainX, trainY)
    acc= clf.score(testX, testY)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(clf, f)

pickle_in = open("studentmodel.pickle", "rb")
clf = pickle.load(pickle_in)
y_pred = clf.predict(testX)

stats = 'PTS', 'TRB', 'AST', 'GPnSround%', 'PER', 'WS', 'BPM', 'VORP'

for x in range(len(y_pred)):

    if y_pred[x]==0 and y_pred[x]==testY[x]:
        trueNegative = trueNegative + 1

    if y_pred[x]==1 and y_pred[x]==testY[x]:
        truePositive = truePositive + 1

    if y_pred[x] == 0 and y_pred[x] != testY[x]:
            falseNegative = falseNegative + 1

    if y_pred[x] == 1 and y_pred[x] != testY[x]:
                falsePositive = falsePositive + 1

    if y_pred[x] == 1 or testY[x] ==1:
        playerList.append(testA[x])
        predList.append(y_pred[x])
        actList.append(testY[x])

        for y in range(len(stats)):
            statList[y].append(testX[x][y])
            dictNba[stats[y]]=statList[y]

dictNba['Prediction']=predList
dictNba['Actual']=actList

Precision = float(precision_score(testY, y_pred))
Recall = float(recall_score(testY, y_pred))
f1Score = float((2*Precision*Recall)/(Precision+Recall))

print(f'True Negatives: {trueNegative}')
print(f'True Positives: {truePositive}')
print(f'False Negatives: {falseNegative}')
print(f'False Positives: {falsePositive}')

print(f'Precison: {Precision}')
print(f'Recall: {Recall}')
print(f'f1Score: {f1Score}')

dictNbaFrame = pd.DataFrame(dictNba)
dictNbaFrame.index=playerList
print(dictNbaFrame)

x1='PTS'
y1='GPnSround%'

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

    ax.scatter(x,y,z, label = 'Prediction')
    ax.scatter(x,y,z2, label = 'Actual')
    ax.legend()
    ax.set_title(f'Correlation of {stat} and Games Played with All-NBA Teams')
    plt.show()


tnfLabels = ['TP', 'FN', 'FP']
tnfResult = np.array([truePositive, falseNegative, falsePositive])

plt.bar(tnfLabels, tnfResult, width=0.3)
plt.yticks(np.arange(0,tnfResult.max()+1,1))
plt.title('Results of SVM Algorithm')
plt.show()

pnrLabel = ['Precision', 'Recall']
pnrResult = np.array([Precision, Recall])

plt.bar(pnrLabel, pnrResult, width = 0.5)
plt.yticks(np.arange(0,1.05,.05))
plt.title('Precision and Recall of SVM Algorithm')
plt.show()

f1Label = ['SVM', 'LR']
f1ScoreLR=LogRegModel.f1Score()
f1Result = np.array([f1Score, f1ScoreLR])
plt.bar(f1Label, f1Result, width = 0.5)
plt.yticks(np.arange(0,1.05,.05))
plt.title('f1Scores of SVM and LogReg')
plt.show()
