import numpy as np
import pandas as pd
import itertools

import hvplot.pandas
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#from pandas.plotting import lag_plot
# from datetime import datetime


import pandas_ta
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")



def prepare_data_method_all(df, close_name):
    """
    Takes dataframe with closing prices for set of tickers and creates an array
    with technical indicators together as feature vector and target vector.
    """
    
    my_df = df.ta.macd(close=close_name,fast=8, slow=21)
    my_df = my_df[['MACD_8_21_9']]
    my_df = pd.concat([my_df, df.ta.hma(close=close_name)],  axis=1)
    my_df = pd.concat([my_df, df.ta.rsi(close=close_name, length=30)],  axis=1)
    my_df = pd.concat([my_df, df.ta.rsi(close=close_name, length=200)],  axis=1)
    my_df = my_df.dropna()


    # df['signal'] = (df.loc[:,'close'] < df.loc[:, 'close'].shift(-1)).astype(int)
    y_signal = (df.loc[my_df.index, close_name] < df.loc[my_df.index, close_name].shift(-1)).astype(int)
    y_signal = y_signal[:-1]

    X = my_df.values[:-1]
    y = df.loc[my_df.index, close_name].values[1:].reshape((-1,1))
    
    return (X, y, y_signal)

def prepare_data_method_1(df, close_name="close"):
    """
    Takes dataframe with closing price for a tickers and creates an array
    with technical indicators together as feature vector and target vector.
    """
    
    
    df.ta.rsi(close=close_name, length=30, append=True)
    df.ta.rsi(close=close_name, length=200, append=True)
    df.ta.macd(close=close_name,fast=8, slow=21, append=True)
    df.ta.hma(close=close_name, append=True)
    
    df = df.dropna()
    df = df.copy()
    # df['signal'] = (df.loc[:,'close'] < df.loc[:, 'close'].shift(-1)).astype(int)
    y_signal = (df[close_name] < df[close_name].shift(-1)).astype(int)
    y_signal = y_signal[:-1]
    
    X = df.drop(columns=[close_name, 'MACDs_8_21_9','MACDh_8_21_9']).values[:-1]
    y = df[close_name].values[1:].reshape((-1,1))
    
    # print(df.drop(columns=[close_name, 'MACDs_8_21_9','MACDh_8_21_9']).head())
    return (X, y, y_signal)


def train_test_split_ts(X, y, y_signal, alpha=0.7):
    """
    Creates train-test split for Timeseries
    """
    split = int(alpha * len(X))

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    y_signal_train = y_signal[:split].values.reshape((-1,1))
    y_signal_test = y_signal[split:].values.reshape((-1,1))
    
    idx_train = y_signal.index[:split]
    idx_test = y_signal.index[split:]
    
    return (X_train, X_test, y_train, y_test,  y_signal_train, y_signal_test, idx_train, idx_test)


def ols_predict_price(X_train, y_train, X_test, y_test, idx_test):
    """
    Predict prices for OLS model based on technical indicators
    """
    
    model = sm.OLS(y_train, sm.add_constant(X_train))
    results = model.fit()

    y_pred = results.predict(sm.add_constant(X_test))
    
    df_pred = pd.DataFrame(index=idx_test)
    
    df_pred['ols_pred'] = y_pred
    df_pred['true'] = y_test

    print(results.summary())
    
    return df_pred
    
    
    
    
    
def ols_regularized_predict_price(X_train, y_train, X_test, y_test, idx_test, alpha=1, l1_wt=1):
    """
    Predict prices for regularized OLS model based on technical indicators
    """
    
    model = sm.OLS(y_train, sm.add_constant(X_train))

    results = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1_wt)

    y_pred = results.predict(sm.add_constant(X_test))
    
    df_pred = pd.DataFrame(index=idx_test)
    
    df_pred['ols_regularized_pred'] = y_pred
    df_pred['true'] = y_test

    
    return df_pred
    
    
def compute_signal_strategy_1(y):
    """
    Compute signal for a strategy 1: If our prediction is that the price is goint to increase next day, 
    we buy stock in the morning and sell it at the end fo the day.
    """
    y_next_day = y.shift(-1)

    y_signal = (y < y_next_day).astype(int)
    y_signal = y_signal[:-1]

    return y_signal

    
    
def compute_profit_strategy_1(y_true, y_pred):
    """
    Compute profit for a strategy 1: If our prediction is that the price is goint to increase next day, 
    we buy stock in the morning and sell it at the end fo the day.
    """

    y_true = y_true.ravel()
    y_next_day = y_true[1:]
    y_signal  = compute_signal_strategy_1(y_pred)

    profit = (y_next_day - y_true[:-1]) * y_signal
    
    return profit

    
    
def sk_regularizer_predict(X_train, y_train, X_test, y_test, idx_test, alpha=1, l1_wt=1):

    """
    get prediction for regularized model.
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_wt)

    # Train the model
    model.fit(X_train, y_train)
    
    # Use model to make predictions
    y_reg_pred = model.predict(X_test)
    

    df_reg_pred = pd.DataFrame(index=idx_test)
    
    df_reg_pred['regularized_pred'] = y_reg_pred
    df_reg_pred['true'] = y_test
    
    # compute signals
    y_true_signal = compute_signal_strategy_1(df_reg_pred['true'])
    
    y_reg_signal = compute_signal_strategy_1(df_reg_pred['regularized_pred'])

    # compute profit
    
    profit = compute_profit_strategy_1(df_reg_pred['true'], df_reg_pred['regularized_pred'])
    cum_profit = profit.sum()
    
    #compute accuracy
    acc_score = accuracy_score(y_true_signal, y_reg_signal)

    return (acc_score, cum_profit, df_reg_pred, profit)




def get_max_regularizer_predict(X_train, y_train, X_test, y_test, idx_test, alphas, l1s):
    """
    Get the best set of parameters for regularized model.
    """
    test_acc_df = pd.DataFrame(columns=['alpha',
                                    'l1wt',
                                    'accuracy',
                                    'profit'
                                   ]
                          )

    for alpha, l1_wt in  itertools.product(alphas, l1s):
    
        acc_score, current_profit, _, _ = sk_regularizer_predict(X_train, y_train, X_test, y_test, idx_test, alpha=alpha, l1_wt=l1_wt)
    
        # Evaluate the model
        tmp_dict = {
            'alpha': [alpha],
            'l1wt': [l1_wt],
            'accuracy': [round(acc_score,5)],
            'profit':[current_profit]
        }
        tmp_df = pd.DataFrame(tmp_dict)
        test_acc_df = pd.concat([test_acc_df, tmp_df], ignore_index=True)
    
    alpha_max, l1wt_max, max_accuracy, max_profit = test_acc_df[test_acc_df.accuracy == test_acc_df.accuracy.max()].values[0]

    max_accuracy, max_cum_profit, df_reg_pred_max, max_profit = sk_regularizer_predict(X_train, y_train, X_test, y_test, idx_test, alpha=alpha_max, l1_wt=l1wt_max)
    
    return alpha_max, l1wt_max, max_accuracy, max_cum_profit, df_reg_pred_max, max_profit
    
    
    
    
    
def compute_weights(df, num_portfolios = 10 ** 5):
    '''
	compute weights for simulated portfolios
    '''    
    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    portf_sharpe_ratio = []

    # Log of percentage change
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()

    # Yearly returns for individual companies
    ind_er = df.resample('M').last().pct_change().mean()
    
    num_assets = len(df.columns)
    
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd)

        portf_sharpe_ratio.append(returns / sd) # Sharpe ratio
        
        
    portf_data = {
    'Returns':p_ret,
    'Volatility':p_vol,
    'sharpe_ratio':portf_sharpe_ratio
    }

    for counter, symbol in enumerate(df.columns.tolist()):
        #print(counter, symbol)
        portf_data[symbol+'_weight'] = [w[counter] for w in p_weights]

    portfolios  = pd.DataFrame(portf_data)
    
    
    max_sharpe_port = portfolios.iloc[portfolios['sharpe_ratio'].idxmax()]
    # idxmax() gives us the max value in the column specified.                               
    
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    # idxmin() gives us the minimum value in the column specified.                               

    
    return max_sharpe_port, min_vol_port, portfolios


def plot_portfolios(portfolios, max_sharpe_port, min_vol_port):
    
    """
    Plot the best portfolio according to the Sharpe ratio value
    """
    
    fig, ax = plt.subplots(figsize=[20,10])
    portfolios.plot(kind='scatter',
                x='Volatility',
                y=['Returns'],
                marker='o',
                s=20,
                c='sharpe_ratio',
                cmap='RdYlGn',
                alpha=1,
                edgecolors='black',
                grid=True,
                ax=ax,
                title='Simulated Portfolio Performance'
                )
    plt.scatter(min_vol_port[1], 
            min_vol_port[0], 
            color='r', 
            marker='*', 
            s=500,
            label='Min Volatility'
           )
    plt.scatter(max_sharpe_port[1], 
            max_sharpe_port[0], 
            color='white', 
            edgecolors='black',
            marker='*', 
            s=500,
            label='Max Sharpe Ratio'
           )
    plt.legend(loc='best')
   
