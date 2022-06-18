import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("\Semester 2, 21-22\DATN\Code\TSLA.csv") # doc file csv
#loi suat % trong ngay hien tai
dataf = data['Adj Close'].pct_change() * 100
#print(dataf)
# them mot col Today vao data vi tri so 7
data.insert(7, "Today", dataf) 

#lay ra 2 cot
df = data[['Date','Today']] 

# Lag la tinh loi suat % cua cac ngay truoc ngay hien tai
def lag(df, lags):
    names = []
    for i in range(1,lags +1):
        df['Lag' + str(i)] = df['Today'].shift(i) 
        names.append('Lag' + str(i))
    return names
name = lag(df,5)

#tinh khoi luong co phieu giao dich
vol = data.Volume.shift(1).values / 100000000
df.insert(7, "Vol", vol)

# xac dinh chieu tang giam cua co phieu
df = df.dropna()
df['Direction'] = [1 if  i > 0 else 0 for i in df['Today']] 

name.append('Vol')


#print(name)
#print(df)

x = df[name]
y = df.Direction

md = sm.Logit(y,x)
result = md.fit()
#print(result.summary())

prediction = result.predict(x)

#print(prediction)

def confusion_matrix(real, pred):
    predtrans = ['Up' if i > 0.5 else 'Down' for i in pred]
    rl = ['Up' if i > 0 else 'Down' for i in real]
    confusion_matrix = pd.crosstab(pd.Series(rl),pd.Series(predtrans), rownames=['Real'],colnames=['Predicted'])
    return confusion_matrix

#print(confusion_matrix(y,prediction))

df['year'] = pd.DatetimeIndex(df['Date']).year  

#print(df.year)


x_train = df[df.year < 2020][name]
y_train = df[df.year < 2020]['Direction']

x_test = df[df.year >= 2020][name]
y_test = df[df.year >= 2020]['Direction']

model = sm.Logit(y_train,x_train)
rs = model.fit()

predictions = rs.predict(x_test)
print(confusion_matrix(y_test, predictions))


