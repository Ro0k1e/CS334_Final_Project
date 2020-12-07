import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
import talib
from sklearn.decomposition import PCA

# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)
pd.set_option('display.max_columns', None)
stock_data = pd.read_csv('BIDU.csv')


# value measures the increase/decrease of a stock, computed by subtracting closing price of previous day from the
# closing price of the current day
value = pd.Series(stock_data['Close']-stock_data['Close'].shift(1),index=stock_data.index)
value = value.bfill()

# convert to a binary variable with 1 standing for increase and 0 standing for decrease
value[value >= 0] = 1
value[value < 0] = 0
stock_data['Value'] = value

# dealing with missing values
stock_data = stock_data.fillna(method='bfill')
stock_data.set_index('Date')
stock_data.sort_index()
stock_data.drop(columns='Date',inplace=True)
stock_data = stock_data.astype('float64')

# closing price
closed = stock_data['Close'].values

# moving averages of 5, 10, 20 days
ma5 = talib.SMA(closed,timeperiod=5)
ma10 = talib.SMA(closed,timeperiod=10)
ma20 = talib.SMA(closed,timeperiod=20)


stock_data['ma5'] = ma5
stock_data['ma10'] = ma10
stock_data['ma20'] = ma20
stock_data = stock_data.fillna(method='bfill')


# visualization
# Data = stock_data[['Open','High','Low','Close','Value','ma5','ma10','ma20']]
# Data=Data.astype(float)
# Data.plot()
# Volume = stock_data[['Volume']]
# Volume=Volume.astype(float)
# Volume.plot()
# # Data.ix[100:130].plot()
# plt.grid()
# plt.show()

# PCA
data_Y = stock_data['Value']
data_X = stock_data.drop(columns=['Value'],axis=1)
pca = PCA(n_components=5)
reduced_x = pca.fit_transform(data_X)
print(reduced_x)
data_X = preprocessing.scale(reduced_x)
print(data_X)

# grid search start
i_list = []
score_list =[]
# for c in [0.01,0.1,0.5,1,1.1,1.2,1.3,1.4,1.5]:
# for g in [0.001,0.01,0.1,1,10,100]:

# 80% of the dataset are chosen as the training set, and the rest are the testing set
L = len(stock_data)
train = int(L * 0.8)
total_predict_data = L - train

# cyclical prediction
correct = 0
train_original = train

y_test = value[train:]
y_pred = []

while train < L:
    Data_train = data_X[train - train_original:train]
    value_train = value[train - train_original:train]
    Data_predict = data_X[train:train + 1]
    value_real = value[train:train + 1]
    classifier = svm.SVC(C=1.3,kernel='rbf',gamma=10)
    classifier.fit(Data_train, value_train)
    value_predict = classifier.predict(Data_predict)
    y_pred.append(value_predict)
    if (value_real.values[0] == int(value_predict)):
        correct = correct + 1
    train = train + 1



# accuracy
correct = correct * 100 / total_predict_data
print("Correct=%.2f%%" % correct)

from sklearn.metrics import roc_curve, auc
# ROC AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
auc = auc(fpr,tpr)
print("AUC : ", auc)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# i_list.append(c)
# score_list.append(correct)

#
# results = pd.DataFrame({'i_list':i_list, 'score_list':score_list})
# results.plot(x='i_list' ,y='score_list', color='red', title='score change with c')
# plt.grid()
# plt.show()


# score_list_df = pd.DataFrame({'i_list':i_list, 'score_list':score_list})
# score_list_df.plot(x='i_list', y='score_list', title='score change with c')
# plt.grid()
# plt.show()


