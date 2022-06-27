from jinja2 import ModuleLoader
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("\Semester 2, 21-22\DATN\Code\TSLA.csv") # doc file csv
#print(data)

dataf = np.log(data.Close.pct_change() + 1) 
#print(dataf)

data.insert(7, "Return", dataf) 

data['Direction'] = [1 if i > 0 else 0 for i in data.Return]

def lagit(data, lags):
    names = []
    for i in range(1,lags + 1):
        data['Lag '+str(i)] = data['Return'].shift(i)
        data['Dr Lag '+str(i)] = [1 if j > 0 else 0 for j in data['Lag '+str(i)]]
        names.append('Dr Lag '+str(i))
    return names

dirname = lagit(data, 1)
data.dropna(inplace=True)
print(data)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(data[dirname], data['Direction'])
data['Prediction'] = model.predict(data[dirname])

data['strategy'] = data['Prediction'] * data['Return']

print(np.exp(data[['Return','strategy']].sum()))

np.exp(data[['Return','strategy']].cumsum()).plot()
plt.show()
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, shuffle=False, test_size=0.37, random_state=0)

train = train.copy()
test = test.copy()

model = LogisticRegression()
model.fit(train[dirname], train['Direction'])

test['prediction_Logit'] = model.predict(test[dirname])
test['strategy_logit'] = test['prediction_Logit'] * test['Return']

print(np.exp(test[['Return','strategy_logit']].sum()))

np.exp(test[['Return','strategy_logit']].cumsum()).plot()
plt.show()

from sklearn import metrics

print(metrics.confusion_matrix(test['Direction'], test['prediction_Logit']))

print(metrics.classification_report(test['Direction'], test['prediction_Logit']))

