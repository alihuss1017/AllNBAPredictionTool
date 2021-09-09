# SYNOPSIS
Developed using Python and multiple data science libraries (SciKit-Learn, NumPy, Pandas, TensorFlow, MatPlotLib), this program predicts whether or not an NBA player would make an all-NBA team based on multiple per-game stats, such as points, rebounds, and assists, as well as advanced stats including PER (Player Efficiency Rating) and VORP (Value Over Replacement Player). The training and testing data is derived from the 2015-2021 NBA seasons. The algorithm used to train the data is an SVM algorithm with a linear kernel.


##  SVM ALGORITHM PERFORMANCE
![svm](https://user-images.githubusercontent.com/83521645/132613508-42635e19-cc21-4ae3-a3ba-193c716a627c.jpg)

### PRECISION AND RECALL RESULTS
True Negatives: 308

True Positives: 22

False Negatives: 2

False Positives: 3

Precison: 0.88

Recall: 0.9166666666666666

f1Score: 0.8979591836734694

#### POINTS AND ALL-NBA CORRELATION
![ptsSVM](https://user-images.githubusercontent.com/83521645/132613641-b40eab49-c255-4cb8-91f5-650b75d427e2.png)
#### PLAYER EFFICIENCY RATING AND ALL-NBA CORRELATION
![perSVM](https://user-images.githubusercontent.com/83521645/132613715-6363d2c6-112a-4fa0-817c-08c52984a0ae.png)
#### WIN SHARES AND ALL-NBA CORRELATION
![wsSVM](https://user-images.githubusercontent.com/83521645/132613748-d1626f0a-623c-4d9f-bb3c-804661dcc496.png)
#### BOX PLUS/MINUS AND ALL-NBA CORRELATION
![bpmSVM](https://user-images.githubusercontent.com/83521645/132613754-50794b21-dd28-4791-9f1b-e13bf95068cf.png)
#### VALUE OVER REPLACEMENT PLAYER AND ALL-NBA CORRELATION
![vorpSVM](https://user-images.githubusercontent.com/83521645/132613763-63540d32-eb4e-4e0b-a1e6-da0b7c47e509.png)
#### DATA ACCURACY RESULTS
![resultSVM](https://user-images.githubusercontent.com/83521645/132613773-837dbf35-cfeb-4d00-9ed5-055480b8dc89.png)
#### PRECISION AND RECALL RESULTS
![pnrSVM](https://user-images.githubusercontent.com/83521645/132613782-3ba00019-4062-45cc-a5ec-885a1f1e7a14.png)






## LOGISTIC REGRESSION ALGORITHM PERFORMANCE
![logregData](https://user-images.githubusercontent.com/83521645/132617448-2d25c1c8-5545-4bc5-af25-bf30fa4012ca.jpg)

### PRECISION AND RECALL RESULTS
True Negatives: 314

True Positives: 13

False Negatives: 8

False Positives: 0

Precision: 1.0

Recall: 0.6190476190476191

f1Score: 0.7647058823529412

#### POINTS AND ALL-NBA CORRELATION
![ptsLR](https://user-images.githubusercontent.com/83521645/132617480-53a3c66c-5cfd-47cd-af45-c80d6aebffad.png)
#### PLAYER EFFICIENCY RATING AND ALL-NBA CORRELATION
![perLR](https://user-images.githubusercontent.com/83521645/132617488-ff60ac4d-479f-4712-b9ab-693644c5305a.png)
#### WIN SHARES AND ALL-NBA CORRELATION
![wsLR](https://user-images.githubusercontent.com/83521645/132617498-efa68682-5970-4e79-9466-c9beb8ed6b00.png)
#### BOX PLUS/MINUS AND ALL-NBA CORRELATION
![bpmLR](https://user-images.githubusercontent.com/83521645/132617503-9c91e675-957f-4c58-9f0c-7ec1e369ba5c.png)
#### VALUE OVER REPLACEMENT PLAYER AND ALL-NBA CORRELATION
![vorpLR](https://user-images.githubusercontent.com/83521645/132617510-c1000cbb-84db-4529-b036-9f76850eaaa4.png)
#### DATA ACCURACY RESULTS
![resultLR](https://user-images.githubusercontent.com/83521645/132617532-33602f3d-f03c-4862-82ce-4455ee0b6e16.png)
#### PRECISION AND RECALL RESULTS
![pnrLR](https://user-images.githubusercontent.com/83521645/132617538-02444d02-77dd-4892-82f3-5d4870b6930d.png)

## ANALYSIS AND CONCLUSION

![f1Score](https://user-images.githubusercontent.com/83521645/132617934-8d36a230-b4dc-458b-8d16-9441e3b2fe17.png)
