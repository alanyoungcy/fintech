import talib as talb
import matplotlib.pyplot as plt 
import yfinance as yf
import pandas as pd
import numpy as np



# plt.style.use('bmh')
aapl = yf.download('AAPL', '2012-1-1','2012-12-27')
aapl['Simple MA'] = talb.SMA(aapl['Close'],14)
aapl['EMA'] = talb.EMA(aapl['Close'], timeperiod = 14)

aapl['SMA_20'] = aapl['Close'].rolling(window=20).mean()
aapl['SMA_50'] = aapl['Close'].rolling(window=50).mean()
aapl['momentum_rsi'] = talb.RSI(aapl['Close'], timeperiod=14)
aapl['trend_macd'], aapl['trend_macd_signal'], _ = talb.MACD(aapl['Close'], 12, 26, 9)

# aapl['volume_ratio'] = aapl['Volume']/aapl['Volume'].rolling(window=10).mean()
# aapl['buy_sell_ratio'] = aapl['Taker buy base asset volume']/aapl['Taker buy quote asset volume']
# aapl['trade_volume_ratio'] = aapl['Number of trades']/aapl['Volume']
    

print(aapl)
# # Plot aapl
# # [['Close','Simple MA','EMA']].plot(figsize=(15,15)) 
# plt.show()