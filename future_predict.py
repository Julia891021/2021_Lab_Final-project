import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import walk
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import SimpleRNN, LSTM, GRU
from numpy.linalg import inv

path = "./future/"#csv資料夾路徑，py檔放在外面一層
df = pd.DataFrame()#先開一個空的DataFrame(總表)
from os import walk

drop_columns=['Open','High','Low','Adj Close','Volume']
for root, dirs, files in walk(path):
    for f in files:
        print(f)
        #將讀入的檔案concat到總表
        #drop掉那幾個column
        dfnew = pd.read_csv(path+f).drop(columns=drop_columns)
        df = pd.concat([df,dfnew],axis=1) 
#將Date1設為index
df = df.set_index(df['Date1'].values,drop= True)
drop_columns=['Date1','Date']
df = df.drop(columns=drop_columns)
#df = np.log(df)

# df = df.dropna(axis = 'index')

#%%
# del nan

df = df.dropna(axis = 'index')

#%%
# open an array
SP = df.iloc[:,0].values
Gold = df.iloc[:,1].values
Treasury = df.iloc[:,2].values
Corn = df.iloc[:,3].values

# In[]
# model construction
model1 = Sequential()

model1.add(LSTM(10,input_shape=(30, 1),return_sequences=True))    # 10*(10+1)+10
model1.add(Dropout(0.2))

model1.add(LSTM(10,return_sequences=True))    # 10*(10+1)+10
model1.add(Dropout(0.2))

model1.add(LSTM(10,return_sequences=False))    # 10*(10+1)+10
model1.add(Dropout(0.2))

model1.add(Dense(units=5, activation='relu'))
model1.add(Dense(units=1, activation='sigmoid'))

model1.compile(loss='mse', optimizer='adam')

# In[]
# model construction
model2 = Sequential()

model2.add(LSTM(10,input_shape=(30, 1),return_sequences=True))    # 10*(10+1)+10
model2.add(Dropout(0.2))

model2.add(LSTM(10,return_sequences=True))    # 10*(10+1)+10
model2.add(Dropout(0.2))

model2.add(LSTM(10,return_sequences=False))    # 10*(10+1)+10
model2.add(Dropout(0.2))

model2.add(Dense(units=5, activation='relu'))
model2.add(Dense(units=1, activation='sigmoid'))

model2.compile(loss='mse', optimizer='adam')

# In[]
# model construction
model3 = Sequential()

model3.add(LSTM(10,input_shape=(30, 1),return_sequences=True))    # 10*(10+1)+10
model3.add(Dropout(0.2))

model3.add(LSTM(10,return_sequences=True))    # 10*(10+1)+10
model3.add(Dropout(0.2))

model3.add(LSTM(10,return_sequences=False))    # 10*(10+1)+10
model3.add(Dropout(0.2))

model3.add(Dense(units=5, activation='relu'))
model3.add(Dense(units=1, activation='sigmoid'))

model3.compile(loss='mse', optimizer='adam')
# In[]
# model construction
model4 = Sequential()

model4.add(LSTM(10,input_shape=(30, 1),return_sequences=True))    # 10*(10+1)+10
model4.add(Dropout(0.2))

model4.add(LSTM(10,return_sequences=True))    # 10*(10+1)+10
model4.add(Dropout(0.2))

model4.add(LSTM(10,return_sequences=False))    # 10*(10+1)+10
model4.add(Dropout(0.2))

model4.add(Dense(units=5, activation='relu'))
model4.add(Dense(units=1, activation='sigmoid'))

model4.compile(loss='mse', optimizer='adam')

#%%

def minmax(d):
    return (d-d.min())/(d.max()-d.min()) ,d.min(), d.max()

def price_predict(data, assetname):
    # split
    train = data[1:993][:,np.newaxis]
    test = data[993:][:,np.newaxis]
    # normalize
    
    plt.title(assetname+' Futures Price in training set')
    plt.plot(train)
    plt.xlabel('day')
    plt.ylabel('close')
    plt.show()
    
    window_size = int(train.shape[0]/4)
    
    for di in range(0, train.shape[0], window_size):
        train[di:di+window_size] = minmax(train[di:di+window_size])[0]
    
    plt.title('Normalized '+assetname+' Futures Price in training set')
    plt.xlabel('day')
    plt.ylabel('close')
    plt.plot(train)
    plt.show()
    
    test, test_min, test_max = minmax(test)
    
    # train test   
    x_train = np.zeros((train.shape[0]-30,30,1))
    x_test = np.zeros((test.shape[0]-30,30,1))
    y_train = np.zeros((train.shape[0]-30,1))
    y_test = np.zeros((test.shape[0]-30,1))
    
    for i in range(x_train.shape[0]):
        x_train[i] = train[i:i+30]
        y_train[i] = train[i+30]
    
    for i in range(x_test.shape[0]):
        x_test[i] = test[i:i+30]
        y_test[i] = test[i+30]
    
    if assetname=='S&P':
        h = model1.fit(x_train, y_train, batch_size=120, epochs=100, validation_split=0.2)
    if assetname=='Gold':
        h = model2.fit(x_train, y_train, batch_size=120, epochs=100, validation_split=0.2)
    if assetname=='Treasury':
        h = model3.fit(x_train, y_train, batch_size=120, epochs=100, validation_split=0.2)
    if assetname=='Corn':
        h = model4.fit(x_train, y_train, batch_size = 120, epochs=100, validation_split=0.2)
    
    plt.plot(h.history['loss'], label='training loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

    # predict
    if assetname=='S&P':
        pred = model1.predict(x_test)
    if assetname=='Gold':
        pred = model2.predict(x_test)
    if assetname=='Treasury':
        pred = model3.predict(x_test)
    if assetname=='Corn':
        pred = model4.predict(x_test)
    unadj_pred = pred
    unadj_pred = unadj_pred*(test_max-test_min) + test_min
    
    plt.title(assetname+' futures Price Prediction')
    plt.plot(pred, label='pred')
    plt.plot(y_test, label='true')
    plt.xlabel('day')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    
    for i in reversed(range(pred.shape[0])):
        if i != 0:
            pred[i] -= pred[i-1] - y_test[i-1]
    
    plt.title(assetname+' futures Price Prediction (adjusted)')
    plt.plot(pred, label='pred')
    plt.plot(y_test, label='true')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

    adj_pred = pred*(test_max-test_min) + test_min
    return unadj_pred, adj_pred

SP_price_pred, SP_price_pred_adj = price_predict(SP,'S&P')
Gold_price_pred, Gold_price_pred_adj = price_predict(Gold,'Gold')
Treasury_price_pred, Treasury_price_pred_adj = price_predict(Treasury,'Treasury')
Corn_price_pred, Corn_price_pred_adj = price_predict(Corn,'Corn')

pred = np.hstack((SP_price_pred, Gold_price_pred, Treasury_price_pred, Corn_price_pred))
adj_pred = np.hstack((SP_price_pred_adj, Gold_price_pred_adj, Treasury_price_pred_adj, Corn_price_pred_adj))


# In[]
# rolling window's utils
price = df.values[993:]

def return_array(data):
    log = np.log(data)
    rt = np.zeros(data.shape)
    for i in range(1,data.shape[0]):
        rt[i] = log[i]-log[i-1]
    return rt[1:]

price_return = return_array(price)

def mvpweight(sigma):
    one = np.full((sigma.shape[0],1),1)
    invsigma = inv(sigma)
    return (np.dot(invsigma,one))/np.dot(one.T,np.dot(invsigma,one))

# In[]
# rolling window

def RW(price, pred, method):
    initial_value = 100000000
    rolling_P = list(np.repeat(0, pred.shape[0])) # Portfolio value
    rolling_w = list(np.repeat(0, pred.shape[0])) # funds' weight
    rolling_s = list(np.repeat(0, pred.shape[0])) # number of shares
    for i in range(pred.shape[0]):
        if method == 'history':
            window = return_array(price[i:i+30])
        if method == 'lstm':
            window = return_array(np.append(price[i:i+30], pred[i].reshape(1, 4), axis=0))   # pred.shape==(220,4)
        sigma = np.cov(window.T)
        rolling_w[i] = mvpweight(sigma)
        rolling_price_today = price[i+30]
        rolling_price_yesterday = price[i+29]
        
        if i==0:
            rolling_s[i] = (initial_value * rolling_w[i]) / rolling_price_yesterday[:,np.newaxis] 
            rolling_P[i] = np.dot(np.squeeze(rolling_s[i]), rolling_price_today)  
        else:
            rolling_s[i] = (rolling_P[i-1] * rolling_w[i]) / rolling_price_yesterday[:,np.newaxis] 
            rolling_P[i] = np.dot(np.squeeze(rolling_s[i]), rolling_price_today) 
    return rolling_P, rolling_w

hp, hw = RW(price, pred, 'history')
lp, lw = RW(price, pred, 'lstm')
alp, alw = RW(price, adj_pred, 'lstm')

plt.title('Portfolio value')
plt.plot(hp, label='benchmark', lw=1)
plt.plot(lp,label='lstm', lw=1)
plt.plot(alp,label='lstm (adjusted prediction)', lw=1)
plt.xlabel('day')
plt.ylabel('value')
plt.legend()