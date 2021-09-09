Developed using Python and multiple data science libraries (SciKit-Learn, NumPy, Pandas, TensorFlow, MatPlotLib), this program predicts whether or not an NBA player would make an all-NBA team based on multiple per-game stats, such as points, rebounds, and assists, as well as advanced stats including PER (Player Efficiency Rating) and VORP (Value Over Replacement Player). The training and testing data is derived from the 2015-2021 NBA seasons. The algorithm used to train the data is an SVM algorithm with a linear kernel.


SVM ALGORITHM:
True Negatives: 308

True Positives: 22

False Negatives: 2

False Positives: 3

Precison: 0.88

Recall: 0.9166666666666666

f1Score: 0.8979591836734694

Player                 Prediction   PTS   TRB   AST  GPnSround%   PER    WS   BPM  VORP  Actual
Stephen Curry                   1  26.4   5.1   6.1      0.6220  28.2   9.1   7.7   4.0       1
Kemba Walker                    1  25.6   4.4   5.9      1.0000  21.7   7.4   4.2   4.4       1
Stephen Curry                   1  25.3   4.5   6.6      0.9634  24.6  12.6   6.9   5.9       1
Kawhi Leonard                   1  26.6   7.3   3.3      0.7317  25.8   9.5   7.2   4.7       1
Giannis Antetokounmpo           1  26.9  10.0   4.8      0.9146  27.3  11.9   6.2   5.7       1
Giannis Antetokounmpo           1  28.1  11.0   5.9      0.8472  29.2  10.2   8.8   5.5       1
Russell Westbrook               1  31.6  10.7  10.4      0.9878  30.6  13.1  11.1   9.3       1
Joel Embiid                     0  22.9  11.0   3.2      0.7683  22.9   6.2   3.3   2.6       1
Julius Randle                   1  24.1  10.2   6.0      0.9861  19.7   7.8   3.7   3.8       1
Bradley Beal                    1  25.6   5.0   5.5      1.0000  20.8   7.6   2.9   3.7       0
Zion Williamson                 1  27.0   7.2   3.7      0.8472  27.1   8.7   5.4   3.8       0
Jimmy Butler                    1  21.5   6.9   7.1      0.7222  26.5   9.3   7.5   4.2       1
Anthony Davis                   1  28.0  11.8   2.1      0.9146  27.5  11.0   5.9   5.4       1
Damian Lillard                  1  25.1   4.0   6.8      0.9146  22.2   9.2   4.4   4.3       1
DeAndre Jordan                  1  12.7  13.8   1.2      0.9878  21.8  11.8   3.3   3.5       1
Anthony Davis                   1  25.9  12.0   3.9      0.6829  30.3   9.5   9.4   5.3       0
DeMarcus Cousins                0  26.9  11.5   3.3      0.7927  23.6   5.7   3.3   3.0       1
Kawhi Leonard                   1  21.2   6.8   2.6      0.8780  26.0  13.7   9.1   6.7       1
Kyrie Irving                    1  26.9   4.8   6.0      0.7500  24.4   7.4   5.3   3.5       1
Isaiah Thomas                   1  28.9   2.7   5.9      0.9268  26.5  12.5   6.7   5.6       1
Joel Embiid                     1  28.5  10.6   2.8      0.7083  30.3   8.8   7.2   3.7       1
Russell Westbrook               1  25.4  10.1  10.3      0.9756  24.7  10.1   6.3   6.1       1
Chris Paul                      1  16.4   4.5   8.9      0.9722  21.4   9.2   4.7   3.7       1
Stephen Curry                   1  27.3   5.3   5.2      0.8415  24.4   9.7   6.6   5.1       1
Kevin Durant                    1  26.4   6.8   5.4      0.8293  26.0  10.4   7.3   5.5       1
DeMar DeRozan                   1  23.0   3.9   5.2      0.9756  21.0   9.6   2.8   3.2       1
James Harden                    1  29.1   8.1  11.2      0.9878  27.4  15.0   8.7   8.0       1
