import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as pyplot
from mpl_toolkits import mplot3d

import sklearn
from sklearn import *
from sklearn.metrics import *
import pickle

dataSet = pd.read_csv("DataSet.csv")

dataSet = dataSet[['Player','PTS', 'TRB', 'AST', 'G', 'GS', 'PER', 'WS', 'BPM', 'VORP', 'All-NBA?']]

a=dataSet.pop('Player')
predict = 'All-NBA?'

a=np.array(a)
X = np.array(dataSet.drop([predict], 1))
Y = np.array(dataSet[predict])
best=0

for _ in range(100):

    trainA, testA, trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(a, X, Y, test_size=0.4)

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
    if y_pred[x] == 1 or testY[x] ==1:
        print(testA[x], y_pred[x], testX[x], testY[x])


Precision = float(precision_score(testY, y_pred))
Recall = float(recall_score(testY, y_pred))
f1Score = float((2*Precision*Recall)/(Precision+Recall))
print(Precision)
print(Recall)
print(f1Score)

x1='PTS'
y1='GS'
ax = pyplot.axes(projection='3d')
x=dataSet['PTS']*2+dataSet['TRB']+dataSet['AST']+dataSet['PER']*2+dataSet['WS']*2+dataSet['VORP']+dataSet['BPM']
y=dataSet['GS']
z=dataSet[predict]

ax.set_xlabel(x1)
ax.set_ylabel(y1)
ax.set_zlabel(predict)
ax.scatter(x,y,z)
ax.set_title('Correlation of %s and %s with %s' % (x1, y1, predict))
pyplot.show()

'''
train_input_fn = make_input_fn(trainX, trainY)   
eval_input_fn = make_input_fn(testX, testY, num_epochs=1, shuffle=False)
'''