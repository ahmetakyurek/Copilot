import time
import numpy as np
from binance.client import Client
from binance.enums import *
from datetime import datetime, timedelta
import pandas as pd
import requests

# Binance API keys
BINANCE_API_KEY = ""
BINANCE_SECRET_KEY = ""

# Telegram Bot Token and Chat ID
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# Binance Futures client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# RSI ayarları
RSI_PERIOD = 500
RSI_OVERBOUGHT = int(input("RSI Aşırı Alım Değerini Giriniz (Örnek:70): "))
RSI_OVERSOLD = int(input("RSI Aşırı Satış Değerini Giriniz (Örnek:30): "))
interval = input("Mum Periyodunu Giriniz (Örnek:4h-1d): ")

# Telegram mesajı gönderme fonksiyonu
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram mesajı gönderilemedi:", str(e))

# Binance'teki USDT çiftlerini al
def get_binance_futures_coins():
    info = client.futures_exchange_info()
    symbols = [s['symbol'] for s in info['symbols']
               if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
    return symbols[:400]

# UTC+3 saat dilimi
def get_utc3_timestamp():
    return datetime.utcnow() + timedelta(hours=3)

# RSI hesaplama
def calculate_rsi(symbol, interval=interval, period=RSI_PERIOD):
    candles = client.futures_klines(symbol=symbol, interval=interval, limit=RSI_PERIOD)
    closes1 = [float(candle[4]) for candle in candles]
    close_array = np.asarray(closes1)
    closes = close_array[:-1]

    diff = np.diff(closes)
    up_chg = 0 * diff
    down_chg = 0 * diff

    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]

    up_chg = pd.DataFrame(up_chg)
    down_chg = pd.DataFrame(down_chg)

    up_chg_avg = up_chg.ewm(com=13, min_periods=14).mean()
    down_chg_avg = down_chg.ewm(com=13, min_periods=14).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return int(rsi[0].iloc[-1])

# Ana işlem fonksiyonu
# Long/Short oranı alma fonksiyonu
def get_long_short_ratio(symbol):
    url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
    try:
        response = requests.get(url)
        data = response.json()
        long_ratio = float(data[0]['longShortRatio'])
        return long_ratio
    except Exception as e:
        print(f"{symbol} için long/short oranı alınamadı: {e}")
        return None

def main():
    coins = get_binance_futures_coins()
    message = f"Tarih: {get_utc3_timestamp().strftime('%Y-%m-%d %H:%M:%S')} UTC+3\n\n"

    for coin in coins:
        try:
            rsi = calculate_rsi(coin)
            print(f"{coin}: RSI {rsi}")

            ls_ratio = get_long_short_ratio(coin)
            if ls_ratio is None:
                continue

            if rsi <= RSI_OVERSOLD and ls_ratio < 1:
                message += f"\U0001F7E2 AL - {coin} RSI: {rsi} | LS Oranı: {ls_ratio:.2f}\n"
            elif rsi >= RSI_OVERBOUGHT and ls_ratio > 1:
                message += f"\U0001F534 SAT - {coin} RSI: {rsi} | LS Oranı: {ls_ratio:.2f}\n"

        except Exception as e:
            print(f"Hata ({coin}): {str(e)}")

    if message.strip():
        send_telegram_message(message)
    else:
        send_telegram_message("Mevcut sinyal yok.")

# Hedef saatler (UTC+3)
target_hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
last_run_hour = None

# Sürekli saat kontrolü
while True:
    now = get_utc3_timestamp()
    print(now)
    current_hour = now.hour
    current_minute = now.minute
    print(current_hour)
    print(current_minute)

    if current_hour in target_hours and current_hour != last_run_hour and current_minute == 0:
        print(f"⏰ Uygun saat geldi: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC+3")
        main()
        last_run_hour = current_hour
        print("✅ Taramalar tamamlandı.\n")

    time.sleep(30)  # Her 30 saniyede bir kontrol