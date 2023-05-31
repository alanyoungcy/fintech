import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client

# Step 1: Retrieve historical price data

api_key = '978BKbPQBTuvgxCpuYXaCCdnrobuil9KCOqaSrlySX83expjS7xOrSePvi80DGU7'
api_secret = 'ezYPqXPvRDK9g3MhYQUF7GVejiq5PYtZ2ZWrmnXnKZORDdwSGpcF1ZlckyODkHN8'
client = Client(api_key, api_secret)

symbol1 = 'ETHUSDT'
symbol2 = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1DAY
start_date = '2022-05-31'
end_date = '2023-05-31'

klines1 = client.get_historical_klines(symbol1, interval, start_date, end_date)
klines2 = client.get_historical_klines(symbol2, interval, start_date, end_date)

df1 = pd.DataFrame(klines1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df2 = pd.DataFrame(klines2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')

df1.set_index('timestamp', inplace=True)
df2.set_index('timestamp', inplace=True)

df1['close'] = df1['close'].astype(float)
df2['close'] = df2['close'].astype(float)

# Step 2: Define the pair trading strategy
def pair_trading_strategy(df1, df2):
    # Calculate the spread between the two assets
    spread = df1['close'] - df2['close']

    # Calculate the mean and standard deviation of the spread
    spread_mean = spread.mean()
    spread_std = spread.std()

    # Generate trading signals based on the spread
    df1['signal'] = np.where(spread > spread_mean + spread_std, -1, np.where(spread < spread_mean - spread_std, 1, 0))
    df2['signal'] = -df1['signal']

    # Calculate daily returns
    df1['returns'] = df1['close'].pct_change()
    df2['returns'] = df2['close'].pct_change()

    # Apply trading signals to generate positions
    df1['position'] = df1['signal'].shift()
    df2['position'] = df2['signal'].shift()
    df1['position'].fillna(0, inplace=True)
    df2['position'].fillna(0, inplace=True)

    # Calculate strategy returns
    df1['strategy_returns'] = df1['position'] * df1['returns']
    df2['strategy_returns'] = df2['position'] * df2['returns']

    # Calculate cumulative returns
    df1['cumulative_returns'] = (1 + df1['strategy_returns']).cumprod()
    df2['cumulative_returns'] = (1 + df2['strategy_returns']).cumprod()

    return df1, df2

# Step 3: Backtest the strategy
backtest_df1, backtest_df2 = pair_trading_strategy(df1, df2)

# Step 4: Evaluate performance
total_return = backtest_df1['cumulative_returns'][-1] - 1
annualized_return = (1 + total_return) ** (365 / len(backtest_df1)) - 1
max_drawdown = (backtest_df1['cumulative_returns'].max() - backtest_df1['cumulative_returns']) / backtest_df1['cumulative_returns'].max()
max_drawdown = max_drawdown.max()

print(f'Total return: {total_return:.2%}')
print(f'Annualized return: {annualized_return:.2%}')
print(f'Max drawdown: {max_drawdown:.2%}')

# Step 5: Optimize the strategy
# You can use optimization techniques to improve the strategy, such as adjusting the threshold for generating trading signals.

# Step 6: Validate the strategy
# Split the historical data into training and testing sets. Apply the strategy to the testing set.

# Step 7: Iterate and refine
# If necessary, iterate and refine your strategy by adjusting the parameters or exploring different indicators.

# Plotting the strategy results
# plt.figure(figsize=(10, 6))
plt.plot(backtest_df1['cumulative_returns'], label='ETH')
plt.plot(backtest_df2['cumulative_returns'], label='BTC')
plt.title('Pair Trading Strategy - Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
