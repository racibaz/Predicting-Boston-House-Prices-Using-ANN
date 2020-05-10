# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:51:48 2020

@author: Recai Cansız
"""

#Kütüphaneler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#Veriyi dataframe e alıyoruz.
from sklearn.datasets import load_boston
df = load_boston()

#1. Veri Setini analiz edip verileri normalize ediniz. 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(df.data)

scaled_train_df = pd.DataFrame(scaled_train, columns=df.feature_names)

#Konsoldan ziyade değişken üzerinden görmek daha kolay
describe = scaled_train_df.describe()

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

#2. Bir FFNN (Feed Forward Neural Network) modeli tanımlayınız.
model = Sequential()

model.add(Dense(104, input_dim=13, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

#Bağımlı ve bağımsı değikenleri ayırıyoruz.
#X = scaled_train_df.drop(["crim", "zn", "indus", "chas", "nox", "age", "dis", "rad", "tax", "black", "medv"], axis=1).values
X = scaled_train_df.values
Y = df.target

#Veri setinin 2/3’ ünü Train/Eğitim 1/3 ünü Test verisi olarak kullanınız.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=10)

#Model train işlemi
model.fit(x_train, y_train, epochs=50,  batch_size=10, verbose=2)

#Tahminleme yapıyoruz.
y_pred = model.predict(x_test)

#Vektörü matrix olarak görmek istediği için "reshape" işlemi yapıyoruz.
y_test = y_test.reshape(167,1)

#3. Kurduğunuz modelin >%85 oranında doğru tahmin yapabilmesi için gerekli iyileştirmeleri yapınız. 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)