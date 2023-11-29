import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler ,StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import feature_selection
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.metrics import r2_score

A = {'age' : [23, 24, 27, 28, 29, 32, 33, 32, 35, 35],
'pre_pay' : [160, 165, 180, 198, 200, 250, 255, 268, 275, 300],
'pay' : [190, 200, 230, 240, 300, 310, 320, 345, 350, 370]}


df = pd.DataFrame(A)
X = df.drop('pay', axis=1)
y = df.pay

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape) 
print(y_test.shape)

lr = LinearRegression()
model = lr.fit(X_train,y_train)

yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)

score = r2_score(y_test, yhat_test)

st.write(score)

a_slider = st.sidebar.slider("Select age", 18, 60,18)
pre_slider = st.sidebar.slider("Select Previous Pay", 100, 400,100)

df_pred = pd.DataFrame({'age':a_slider,
                        'pre_pay':pre_slider}, index=[0])

print(a_slider, pre_slider)
st.write("Pay is ", model.predict(df_pred))

