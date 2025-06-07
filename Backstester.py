import asyncio
import pandas as pd
import logging
import ta
from binance import AsyncClient
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# SuperTrend fonksiyonu
def supertrend(df, period=10, multiplier=3):
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    trend = []
    in_uptrend = True

    for current in range(1, len(df.index)):
        if df['close'][current] > upperband[current - 1]:
            in_uptrend = True
        elif df['close'][current] < lowerband[current - 1]:
            in_uptrend = False
        trend.append(1 if in_uptrend else -1)

    df['SuperTrend_Up'] = pd.Series([None] + trend)
    return df

# Binance verisi çekme (kendi fetch fonksiyonu)
async def fetch_data(symbol="BTCUSDT", interval="1h", limit=200):
    client = await AsyncClient.create()
    klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
    await client.close_connection()

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'close_time', 'quote_asset_volume',
        'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Sadece ihtiyacımız olan sütunları al ve türlerini doğru çevir
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    return df


# Bağımsız test fonksiyonu
async def backtest_direction_signals(symbol="BTCUSDT", interval="1h", lookback_hours=200):
    df = await fetch_data(symbol, interval, lookback_hours)
    if df.empty or len(df) < 30:
        logging.warning(f"[{symbol}] Yeterli backtest verisi yok.")
        return

    results = []
    for i in range(20, len(df) - 1):
        sliced_df = df.iloc[:i+1].copy()
        next_close = df.iloc[i+1]['close']

        # SuperTrend yönü
        st = supertrend(sliced_df.copy())
        direction_slow = 'NEUTRAL'
        if st['SuperTrend_Up'].iloc[-1] == 1:
            direction_slow = 'LONG'
        elif st['SuperTrend_Up'].iloc[-1] == -1:
            direction_slow = 'SHORT'

        # Hızlı yön (EMA/RSI/MACD)
        try:
            close = sliced_df['close']
            ema5 = close.ewm(span=5).mean()
            ema20 = close.ewm(span=20).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            rsi = ta.momentum.rsi(close, window=7)
            macd = ta.trend.macd_diff(close)
            adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14)


            score = 0
            if close.iloc[-1] > ema50.iloc[-1]: score += 1
            if ema5.iloc[-1] > ema20.iloc[-1]: score += 1
            if rsi.iloc[-1] > 50: score += 1
            if macd.iloc[-1] > 0: score += 1
            if adx.iloc[-1] > 20: score += 1

            direction_fast = 'LONG' if score >= 2 else 'SHORT' if score <= 1 else 'NEUTRAL'
        except:
            direction_fast = 'NEUTRAL'

        entry = sliced_df['close'].iloc[-1]
        pnl_fast = (next_close - entry) / entry * 100 if direction_fast == 'LONG' else (entry - next_close) / entry * 100
        pnl_slow = (next_close - entry) / entry * 100 if direction_slow == 'LONG' else (entry - next_close) / entry * 100

        results.append({
            'timestamp': sliced_df['timestamp'].iloc[-1],
            'entry': entry,
            'next_close': next_close,
            'dir_fast': direction_fast,
            'pnl_fast': pnl_fast,
            'dir_slow': direction_slow,
            'pnl_slow': pnl_slow
        })

    df_result = pd.DataFrame(results)
    fast_winrate = (df_result['pnl_fast'] > 0).mean() * 100
    slow_winrate = (df_result['pnl_slow'] > 0).mean() * 100
    avg_pnl_fast = df_result['pnl_fast'].mean()
    avg_pnl_slow = df_result['pnl_slow'].mean()

    # Kümülatif PnL hesapla
    df_result['cum_fast'] = df_result['pnl_fast'].cumsum()
    df_result['cum_slow'] = df_result['pnl_slow'].cumsum()

    # Grafik çiz
    plt.figure(figsize=(10, 5))
    plt.plot(df_result['cum_fast'], label='FAST Yön', color='blue')
    plt.plot(df_result['cum_slow'], label='SLOW Yön', color='orange')
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f"{symbol} Backtest - Kümülatif PnL")
    plt.xlabel("İşlem Sırası")
    plt.ylabel("Toplam Kâr/Zarar (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n📊 Backtest sonucu: {symbol}")
    print(f"⚡ FAST  Yön | Win Rate: {fast_winrate:.1f}% | Avg PnL: {avg_pnl_fast:.2f}%")
    print(f"🐢 SLOW  Yön | Win Rate: {slow_winrate:.1f}% | Avg PnL: {avg_pnl_slow:.2f}%")

# Ana çalıştırıcı
if __name__ == "__main__":
    asyncio.run(backtest_direction_signals())

