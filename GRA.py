import pandas as pd
import matplotlib.pyplot as plt

# talib is a package with chinese stock data
import talib
path = 'BIDU.csv'

data = pd.read_csv(path, encoding='utf-8')


# closing price
closed = data['Close'].values

# moving averages of 5, 10, 20 days
ma5 = talib.SMA(closed, timeperiod=5)
ma10 = talib.SMA(closed, timeperiod=10)
ma20 = talib.SMA(closed, timeperiod=20)
data['ma5'] = ma5
data['ma10'] = ma10
data['ma20'] = ma20
data = data.fillna(method='bfill')

pd.set_option('display.max_columns', 20)

n_data = data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume', 'ma5', 'ma10', 'ma20']]

mean_data = n_data.mean(axis=0)

r, c = n_data.shape
#print(r,c)

# normalization
for item in range(c):
    n_data.iloc[:, item] = n_data.iloc[:, item] / mean_data[item]
print(n_data.head())


ck = n_data.loc[:, ['Open', 'High', 'Low', 'Volume', 'ma5', 'ma10', 'ma20']]
cp = n_data.loc[:, ['Close']]


t = pd.DataFrame()
print("====================================================")
for index, row in ck.iteritems():
    c_data = ck[index]-cp['Close']
    t[index] = c_data.abs()


print(t.head())
a = t.min().min()
b = t.max().max()


res = (a + 0.5 * b) / (t + 0.5 * b)
print(res)
ret = res.sum(axis=0)/r

# find the most relevant features
print(ret.sort_values())

###########################################

# # visualization
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['axes.unicode_minus'] = False
#
# y = cp.iloc[0:100,:]
#
# ydata = n_data.index.to_list()
# ydata = ydata[0:100]
# xdata1 = res.loc[:, ['Volume']]
# xdata1 = xdata1.iloc[0:100,:]
# xdata2 = res.loc[:, ['ma20']]
# xdata2 = xdata2.iloc[0:100,:]
# xdata3 = res.loc[:, ['ma10']]
# xdata3 = xdata3.iloc[0:100,:]
# xdata4 = res.loc[:, ['ma5']]
# xdata4 = xdata4.iloc[0:100,:]
# xdata5 = res.loc[:, ['Open']]
# xdata5 = xdata5.iloc[0:100,:]
# xdata6 = res.loc[:, ['Low']]
# xdata6 = xdata6.iloc[0:100,:]
# xdata7 = res.loc[:, ['High']]
# xdata7 = xdata7.iloc[0:100,:]
#
# plt.figure(1)
#
#
# # plt.plot(ydata, xdata1, color='green', label='Volume')
# # plt.plot(ydata, xdata2, color='red', label='ma20')
# # plt.plot(ydata, xdata3, color='yellow', label='ma10')
# plt.plot(ydata, xdata4, color='blue', label='ma5')
# plt.plot(ydata, xdata5, color='orange', label='Open')
# plt.plot(ydata, xdata6, color='gray', label='Low')
# plt.plot(ydata, xdata6, color='pink', label='High')
# # plt.plot(ydata, y, color='black', label='Close')
#
# plt.title(u"CTE error", size=10)
# plt.legend()
# plt.xlabel(u'daily(GRA)', size=10)
# plt.ylabel(u'days', size=10)
#
# # save the graph
# #plt.savefig()
# plt.show()
