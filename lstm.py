import numpy as np
import pandas as pd
import itertools

# import hvplot.pandas
# import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# from pandas.plotting import lag_plot
# from datetime import datetime

from sklearn.metrics import mean_squared_error
from pandas_datareader import data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from numpy.random import seed

seed(1)
from tensorflow import random

random.set_seed(2)

START_DATE = '2020-01-01'
END_DATE = '2022-05-29'
# ASSETS = ['FB', 'TSLA', 'MSFT', 'AAPL', 'GOOGL']
ASSETS = ['GOOGL']
n_assets = len(ASSETS)

df = data.DataReader(ASSETS,
                     'yahoo',
                     start=START_DATE,
                     end=END_DATE
                     )

tesla = df['Adj Close']


# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)


feature_column = 0  # should be 1 if there is only one column in df
target_column = 0  # should be 1 if there is only one column in df

windows = np.arange(3, 21)
units = np.arange(20, 80, 10)
# units = [20, 50]

test_losses_df = pd.DataFrame(columns=['window',
                                       'units',
                                       'loss',
                                       'MSE'
                                       ]
                              )

for window_size, number_units in itertools.product(windows, units):
    X, y = window_data(
        tesla,
        window_size,
        feature_column,
        target_column
    )

    # Use 70% of the data for training and the remaineder for testing
    split = int(0.7 * len(X))

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    from sklearn.preprocessing import MinMaxScaler

    # Use the MinMaxScaler to scale data between 0 and 1.
    scaler = MinMaxScaler()

    scaler = scaler.fit(X_train)

    # Scale the features training and testing sets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the MinMaxScaler object with the target data Y
    y_test_scaler = MinMaxScaler()
    y_test_scaler = y_test_scaler.fit(y_train)

    # Scale the target training and testing sets
    y_train = y_test_scaler.transform(y_train)
    y_test = y_test_scaler.transform(y_test)

    # Build the LSTM model.
    # The return sequences need to be set to True if you are adding
    # additional LSTM layers, but
    # You don't have to do this for the final layer.
    # Note: The dropouts help prevent overfitting
    # Note: The input shape is the number of time steps and the number of indicators
    # Note: Batching inputs has a different input shape of Samples/TimeSteps/Features

    # Define the LSTM RNN model.
    model = Sequential()

    # Initial model setup
    # number_units = 30
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

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # Train the model
    # Use at least 10 epochs
    # Do not shuffle the data
    # Experiement with the batch size, but a smaller batch size is
    # recommended

    model.fit(
        X_train,
        y_train,
        epochs=20,
        shuffle=False,
        batch_size=64,
        verbose=0
    )
    # Make some predictions
    predicted = model.predict(X_test)

    # Recover the original prices instead of the scaled version
    predicted_prices = y_test_scaler.inverse_transform(predicted)
    real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Create a DataFrame of Real and Predicted values
    stocks = pd.DataFrame({
        "Real": real_prices.ravel(),
        "Predicted": predicted_prices.ravel()
    }, index=df.index[-len(real_prices):])

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    test_mse = mean_squared_error(real_prices, predicted_prices)
    print(f"For window size={window_size} and {number_units} units "
          f"\tthe loss is {test_loss:0.5f}"
          f"\tMSE = {test_mse}")
    # idx = f"w:{window_size}, n:{number_units}"
    tmp_dict = {
        'window': [window_size],
        'units': [number_units],
        'loss': [test_loss],
        'MSE': [test_mse]
    }
    tmp_df = pd.DataFrame(tmp_dict)
    test_losses_df = pd.concat([test_losses_df, tmp_df], ignore_index=True)


w, n, min_loss, min_mse = test_losses_df[test_losses_df.loss == test_losses_df.loss.min()].values[0]
print()
print(f"Minimal loss is for window size={w} and {n} units\n"
      f"\tthe loss is {min_loss:0.5f}\n"
      f"\tthe MSE is {min_mse:0.5f}")

w2, n2, min_loss2, min_mse2 = test_losses_df[test_losses_df.MSE == test_losses_df.MSE.min()].values[0]

if (w != w2) or (n != n2):

    print(f"Minimal loss is for window size={w2} and {n2} units\n"
        f"\tthe loss is {min_loss2:0.5f}\n"
        f"\tthe MSE is {min_mse2:0.5f}")



test_losses_df.to_csv(f"lstm_losses_{ASSETS[0]}.csv")
