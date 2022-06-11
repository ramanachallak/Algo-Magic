'''
Import statements

'''

import requests
import json
import csv

import pandas as pd
import numpy as np

import datetime
import re

import os
from dotenv import load_dotenv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeavePOut,cross_val_score
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

'''
Function definitions

'''


def validate_ticker(symbol):
    
    '''Function to validate stock symbol to ensure 
    user input is a valid text and not numbers or special characters'''
    
    pattern = re.compile(r"[A-Za-z]+")
    
    if re.fullmatch(pattern, symbol):
        
        return True
    else:
        
        return False
    
def get_ticker_data(ticker_symbol):
    
    
        load_dotenv()
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        
    
        api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker_symbol}&outputsize=full&apikey={api_key}&datatype=csv"
        
        
    
        ticker_data = requests.get(api_url)
        
        decoded_ticker_data = ticker_data.content.decode('utf-8')
        
        
        if decoded_ticker_data.find("Error Message")>=0:
            return False
        else:
            return decoded_ticker_data
        
        
def create_df(ticker):
    ticker_df = pd.read_csv(f"Data/{ticker}_data.csv")
    
    ticker_df['timestamp']= pd.to_datetime(ticker_df['timestamp'])
    ticker_df.set_index("timestamp", inplace=True)
    
    ticker_df = ticker_df.drop(columns=["open", "high", "low", "volume"])
    ticker_df = ticker_df.sort_index(ascending=False)
    #timeframe start must be from today to 2 years back
    ticker_df = ticker_df.loc[pd.Timestamp('2022-06-10'):pd.Timestamp('2020-06-10')]
    
    return ticker_df



def window_data(df, window, feature_col_number, target_col_number):
    '''
    This function accepts the column number for the features (X) and the target (y)
    It chunks the data up with a rolling window of Xt-n to predict Xt
    It returns a numpy array of X any y
    '''
    
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)

    
    return np.array(X), np.array(y).reshape(-1, 1)

def cross_validate(data, target):
    labels = preprocessing.LabelEncoder()
    X=data
    y=target.ravel()
    y_cat = labels.fit_transform(y)
   # print(y_cat)
    logreg=LogisticRegression(max_iter=4000, solver='lbfgs')
    shuffle_split=ShuffleSplit(test_size=0.2,train_size=0.7,n_splits=20)
    scores=cross_val_score(logreg,X,y_cat,cv=shuffle_split)
    print("cross Validation scores:n {}".format(scores))
    print("Average Cross Validation score :{}".format(scores.mean()))
    
    
def model_predictions(ticker_df, window_size, ticker):
    global data 
    global target 
    # Column index 0 is the `close` column
    feature_column = 0
    target_column = 0
    X, y = window_data(ticker_df, window_size, feature_column, target_column)
    
    # Using 70% of the data for training and the remainder for testing
    split = int(0.7 * len(X))
    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]
    
    from sklearn.preprocessing import MinMaxScaler
    # Use the MinMaxScaler to scale data between 0 and 1.
    scaler = MinMaxScaler()

    scaler.fit(X_train)

    # Scale the features training and testing sets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the MinMaxScaler object with the target data Y
    scaler.fit(y_train)

    # Scale the target training and testing sets
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    model = Sequential()

    # Initial model setup
    number_units = 30
    dropout_fraction = 0.2

    # Layer 1
    model.add(
        LSTM(
            units=number_units,
            return_sequences=True,
            input_shape=(X_train.shape[1], 1)
        )
    )
    model.add(
        Dropout(
            dropout_fraction
        )
    )

    # Layer 2
    model.add(
        LSTM(
            units=number_units, 
            return_sequences=True
        )
    )

    model.add(
        Dropout(
            dropout_fraction
        )
    )

    # Layer 3
    model.add(
        LSTM(
            units=number_units
        )
    )

    model.add(
        Dropout(
            dropout_fraction
        )
    )

    # Output layer
    model.add(Dense(1))
    
    model.compile(
        optimizer="adam", 
        loss="mean_squared_error"
    )
    
    #model.summary()
    model.fit(
        X_train,
        y_train, 
        epochs=150, 
        shuffle=False, 
        batch_size=35, 
        verbose=0
    )
    
    test_loss =  model.evaluate(X_test, y_test, verbose=0)
    print(f"For window size={window_size} and ticker - {ticker} the loss is {test_loss}")
    
    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    ticker_predictions = pd.DataFrame({
        "Predicted": predicted_prices.ravel(),
        "Real Unraveled": real_prices.ravel(),
    }, index = ticker_df.index[:len(real_prices)]) 
    cross_validate(X_test, y_test)
    return ticker_predictions
    


        
def model_forecasts(ticker_df, window_size, ticker):
    
    y = ticker_df['close']
    y = y.values.reshape(-1, 1)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 90  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=30))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=150, batch_size=35, verbose=0)
    
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = ticker_df['close'].reset_index()
    df_past.rename(columns={'timestamp': 'Date', 'close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    #df_past['Forecast'] = np.nan
    #df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast_future'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast_future'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('Date')
    
    return results
    

