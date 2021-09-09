# SUMMARY
The program aims to predict whether or not an NBA player would make an All-NBA team(1st, 2nd, or 3rd) dependent on their stats in the regular season. Implements the use of an SVM and logistic regression algorithm, comparing the results between the two using evaluation metrics such as precision and recall. 

## STATS USED 

PTS (Points)

TRB (Total Rebounds)

AST (Assists)

%GPnSround (Percentage of Games Played and Started in Decimal Form)

PER (Player Efficiency Rating)

WS (Win Shares)

BPM (Box Plus/Minus)

VORP (Value Over Replacement Player)


## PREDICTION METRICS USED

TN(True Negative): How many NBA players were correctly predicted to not make an all-NBA team(y=0)?

TP(True Positive): How many NBA players were correctly predicted to make an all-NBA team(y=1)?

FN(False Negative): How many NBA players were incorrectly predicted to not make an all-NBA team?

FP(False Positive): How many NBA players were incorrectly predicted to not make an all-NBA team?

## EVALUATION METRICS USED

Precision: TP/(TP+FP). Of all NBA players predicted to make an all-NBA team, what fraction actually made an all-NBA team?

Recall: TP/(TP+FN). Of all NBA players predicted to make an all-NBA team, what fraction was correctly predicted to make an all-NBA team?

f1Score: 2(Precision*Recall)/(Precision+Recall). An f1Score is more indicative of the algorithm's performance rather than the accuracy score in this situation since the number of players correctly predicted to not make an all-NBA team(true negatives) is by far the greatest value amongst the other prediction metrics(true positives, false negatives, false positives), and thus every accuracy score would be in the high 90's, not reflective of the algorithm's performance.

Accuracy Score:(TP+TN)/(TP+TN+FP+FN). Of all NBA players in the testing data, how many were correctly predicted to make/not make an all-NBA team?

###  SVM ALGORITHM PERFORMANCE
![svm](https://user-images.githubusercontent.com/83521645/132613508-42635e19-cc21-4ae3-a3ba-193c716a627c.jpg)

Shown above is the result of the testing data(performed on SVM algorithm) that includes a player who was either predicted to make an all-NBA team or actually made an all-NBA team. NOTE: Player names may appear more than once, this means that the testing data contained multiple seasons from a single player.

#### EVALUATION RESULTS
True Negatives: 308

True Positives: 22

False Negatives: 2 (DeMarcus Cousins, Joel Embiid)

False Positives: 3 (Bradley Beal, Zion Williamson, Anthony Davis)

Precison: 0.88

Recall: 0.92

f1Score: 0.90

Accuracy Score: 0.99

###### POINTS AND ALL-NBA CORRELATION
![ptsSVM](https://user-images.githubusercontent.com/83521645/132613641-b40eab49-c255-4cb8-91f5-650b75d427e2.png)
###### PLAYER EFFICIENCY RATING AND ALL-NBA CORRELATION
![perSVM](https://user-images.githubusercontent.com/83521645/132613715-6363d2c6-112a-4fa0-817c-08c52984a0ae.png)
###### WIN SHARES AND ALL-NBA CORRELATION
![wsSVM](https://user-images.githubusercontent.com/83521645/132613748-d1626f0a-623c-4d9f-bb3c-804661dcc496.png)
###### BOX PLUS/MINUS AND ALL-NBA CORRELATION
![bpmSVM](https://user-images.githubusercontent.com/83521645/132613754-50794b21-dd28-4791-9f1b-e13bf95068cf.png)
###### VALUE OVER REPLACEMENT PLAYER AND ALL-NBA CORRELATION
![vorpSVM](https://user-images.githubusercontent.com/83521645/132613763-63540d32-eb4e-4e0b-a1e6-da0b7c47e509.png)
###### DATA ACCURACY RESULTS
![resultSVM](https://user-images.githubusercontent.com/83521645/132613773-837dbf35-cfeb-4d00-9ed5-055480b8dc89.png)
#### PRECISION AND RECALL RESULTS
![pnrSVM](https://user-images.githubusercontent.com/83521645/132613782-3ba00019-4062-45cc-a5ec-885a1f1e7a14.png)






### LOGISTIC REGRESSION ALGORITHM PERFORMANCE
![logregData](https://user-images.githubusercontent.com/83521645/132617448-2d25c1c8-5545-4bc5-af25-bf30fa4012ca.jpg)

Shown above is the result of the testing data(performed on logistic regression algorithm) that includes a player who was either predicted to make an all-NBA team or actually made an all-NBA team. NOTE: Player names may appear more than once, this means that the testing data contained multiple seasons from a single player.

#### EVALUATION RESULTS
True Negatives: 314

True Positives: 13

False Negatives: 8 (Paul George, Chris Paul, Rudy Gobert, Draymond Green, Kawhi Leonard, DeMarcus Cousins, Jimmy Butler, Joel Embiid)

False Positives: 0

Precision: 1.0

Recall: 0.62

f1Score: 0.76

Accuracy Score: 0.98

###### POINTS AND ALL-NBA CORRELATION
![ptsLR](https://user-images.githubusercontent.com/83521645/132617480-53a3c66c-5cfd-47cd-af45-c80d6aebffad.png)
###### PLAYER EFFICIENCY RATING AND ALL-NBA CORRELATION
![perLR](https://user-images.githubusercontent.com/83521645/132617488-ff60ac4d-479f-4712-b9ab-693644c5305a.png)
###### WIN SHARES AND ALL-NBA CORRELATION
![wsLR](https://user-images.githubusercontent.com/83521645/132617498-efa68682-5970-4e79-9466-c9beb8ed6b00.png)
###### BOX PLUS/MINUS AND ALL-NBA CORRELATION
![bpmLR](https://user-images.githubusercontent.com/83521645/132617503-9c91e675-957f-4c58-9f0c-7ec1e369ba5c.png)
###### VALUE OVER REPLACEMENT PLAYER AND ALL-NBA CORRELATION
![vorpLR](https://user-images.githubusercontent.com/83521645/132617510-c1000cbb-84db-4529-b036-9f76850eaaa4.png)
###### DATA ACCURACY RESULTS
![resultLR](https://user-images.githubusercontent.com/83521645/132617532-33602f3d-f03c-4862-82ce-4455ee0b6e16.png)
###### PRECISION AND RECALL RESULTS
![pnrLR](https://user-images.githubusercontent.com/83521645/132617538-02444d02-77dd-4892-82f3-5d4870b6930d.png)

## ANALYSIS AND CONCLUSION

![f1Score](https://user-images.githubusercontent.com/83521645/132617934-8d36a230-b4dc-458b-8d16-9441e3b2fe17.png)
