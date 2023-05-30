from binance.client  import Client
import pandas as pd
import numpy as np
import ta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#sky learn model test

api_key = '978BKbPQBTuvgxCpuYXaCCdnrobuil9KCOqaSrlySX83expjS7xOrSePvi80DGU7'
api_secret = 'ezYPqXPvRDK9g3MhYQUF7GVejiq5PYtZ2ZWrmnXnKZORDdwSGpcF1ZlckyODkHN8'

client = Client(api_key, api_secret)

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2023", "30 May, 2023")

df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
# df.drop(['close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)
df = df.astype(float)

train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

from statsmodels.tsa.arima.model import ARIMA

# Choose an ARIMA model and train it using the training set
model = ARIMA(train['close'], order=(5,1,0))
model_fit = model.fit()

from sklearn.metrics import mean_squared_error

# Make predictions on the test set
predictions = model_fit.forecast(steps=len(test))

# Calculate the mean squared error between the actual values and the predicted values
mse = mean_squared_error(test['close'],predictions)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse}")


