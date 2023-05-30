from binance.client  import Client
import pandas as pd
import numpy as np
import ta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


#from ta import add_all_ta_features

#API Key
#978BKbPQBTuvgxCpuYXaCCdnrobuil9KCOqaSrlySX83expjS7xOrSePvi80DGU7
#Secret Key
#ezYPqXPvRDK9g3MhYQUF7GVejiq5PYtZ2ZWrmnXnKZORDdwSGpcF1ZlckyODkHN8

api_key = '978BKbPQBTuvgxCpuYXaCCdnrobuil9KCOqaSrlySX83expjS7xOrSePvi80DGU7'
api_secret = 'ezYPqXPvRDK9g3MhYQUF7GVejiq5PYtZ2ZWrmnXnKZORDdwSGpcF1ZlckyODkHN8'

client = Client(api_key, api_secret,False)

tickers = client.get_all_tickers()
df1 = pd.DataFrame(tickers)
mask = df1['symbol'].str.endswith('USDT')
usdt_tickers = df1.loc[mask]

symbols = usdt_tickers['symbol'].tolist()

def calculate_cmf(df):
    cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20)
    df['CMF'] = cmf.chaikin_money_flow()
    return df
def calculate_rsi(df):
    rsi = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=False)
    df['RSI'] = rsi.rsi()
    return df
def calculate_macd(df,window_fast, window_slow,window_sign):
    macd = ta.trend.MACD(df['close'], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
    df['MACD'] = macd.macd()
    return df

def turtle_trading(df, short_window=20, long_window=50):
    """
    Generate buy/sell signals based on Turtle Trading strategy.
    Returns a DataFrame with columns 'Signal' and 'Position'.
    """

    # Calculate simple moving averages
    df[f'SMA_{short_window}'] = df['Close'].rolling(window=short_window).mean()
    df[f'SMA_{long_window}'] = df['Close'].rolling(window=long_window).mean()

    # Create signals based on SMA crossovers
    df['Signal'] = np.where(
        df[f'SMA_{short_window}'] > df[f'SMA_{long_window}'], 1, 0)
    df['Signal'] = df['Signal'].shift(1)

    # Generate position changes
    df['Position'] = df['Signal'].diff()

    return df

for symbol in symbols:
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "100 day ago UTC")

    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'timestamp_close',
                                       'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                       'Taker buy quote asset volume', 'Ignore'])
    # df.drop(['timestamp_close', 'Quote asset volume', 'Number of trades',
    #          'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], axis=1, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
  
   
 
 # Add technical indicators to the dataframe
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['momentum_rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=False)
    df['trend_macd'], df['trend_macd_signal'], _ = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)

    # Compute volume-related features
    df['volume_ratio'] = df['Volume']/df['Volume'].rolling(window=10).mean()
    df['buy_sell_ratio'] = df['Taker buy base asset volume']/df['Taker buy quote asset volume']
    df['trade_volume_ratio'] = df['Number of trades']/df['Volume']
    
    
    # df = turtle_trading(df)
    # if df.size > 0:
    #     if df['Position'].iloc[-1] == 1:
    #         print(f"Potential trading opportunity found for {symbol}.")
    # # dfrsi = calculate_rsi(df)
    # dfcmf = calculate_cmf(df)

    # #mask1 = (df['RSI'] < 30) & (df['CMF'] > 0)
    # mask1 = (dfcmf['CMF']>0) 
    # filtered_df = dfcmf.loc[mask1]
    
    # total_rows = len(filtered_df)
    # up_rows = len(filtered_df[filtered_df['Close'] > filtered_df['Close'].shift(1)])

    # if total_rows > 0:
    #     probability = up_rows / total_rows
    #     if probability >= 0.8:
    #         print(f'{symbol} has opportunity in CMF')
           


    
# ma = df['Close'].rolling(window=20).mean()
# std = df['Close'].rolling(window=20).std()
# upper_bollinger_band = ma + (std * 2)

# periods = 10
# average_volume = df['Volume'].rolling(window=periods, min_periods=periods).mean().shift(1)
# current_volume = df['Volume'].iloc[-1]

# if price > upper_bollinger_band.iloc[-1] and current_volume >= average_volume.iloc[-1] * 1.8:
#     print(f'Potential scalp trading opportunity identified. in {symbol}')
# else:
#     print("No scalp trading opportunity found.") 

        
    #else:
    #    print("Probability of price increase is low.")

   