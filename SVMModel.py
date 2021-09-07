import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot
from mpl_toolkits import mplot3d
import sklearn
from sklearn import *
from sklearn.metrics import *
import pickle


trueNegative = 0
truePositive = 0
falseNegative = 0
falsePositive = 0

dataSet = pd.read_csv("DataSet.csv")
dataSet = dataSet[['Player','PTS', 'TRB', 'AST', 'GPnSround%', 'PER', 'WS', 'BPM', 'VORP', 'All-NBA?']]

a=dataSet.pop('Player')
A=np.array(a)

predict = 'All-NBA?'
X = np.array(dataSet.drop([predict], 1))
Y = np.array(dataSet[predict])


best=0
for _ in range(100):

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
        print(testA[x], y_pred[x], testX[x], testY[x])


Precision = float(precision_score(testY, y_pred))
Recall = float(recall_score(testY, y_pred))
f1Score = float((2*Precision*Recall)/(Precision+Recall))

print(trueNegative)
print(truePositive)
print(falseNegative)
print(falsePositive)

print(Precision)
print(Recall)
print(f1Score)

x1='PTS'
y1='GPnSround%'

x=dataSet['PTS']*2+dataSet['TRB']+dataSet['AST']+dataSet['PER']+dataSet['WS']*2+dataSet['VORP']*2+dataSet['BPM']*2
y=dataSet[y1]
z=dataSet[predict]

ax = pyplot.axes(projection='3d')
ax.set_xlabel(x1)
ax.set_ylabel(y1)
ax.set_zlabel(predict)
ax.scatter(x,y,z, color='red')
ax.set_title('Correlation of %s and %s with %s' % (x1, y1, predict))
pyplot.show()

tnfLabels = ['TP', 'FN', 'FP']
tnfResult = [truePositive, falseNegative, falsePositive]
pyplot.bar(tnfLabels, tnfResult)
pyplot.title('Results of SVM Algorithm')
pyplot.show()

pnrLabel = ['Precision', 'Recall']
pnrResult = [Precision, Recall]
pyplot.bar(pnrLabel, pnrResult, width = 0.1)
pyplot.title('Precision and Recall of SVM Algorithm')
pyplot.show()

