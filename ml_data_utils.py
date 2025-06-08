# ml_data_utils.py

import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone
import time
import logging
import ta
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_klines_iterative(client, symbol, interval, start_str_utc, end_str_utc=None, limit=1000):
    """
    Belirtilen başlangıç ve bitiş tarihleri arasında (veya başlangıçtan itibaren 'limit' kadar)
    veriyi parça parça çeker. Binance Futures için.
    start_str_utc ve end_str_utc 'YYYY-MM-DD HH:MM:SS' formatında UTC olmalı.
    """
    df_list = []
    current_start_time = pd.to_datetime(start_str_utc, utc=True)
    end_time_dt = pd.to_datetime(end_str_utc, utc=True) if end_str_utc else None

    while True:
        start_ts_ms = int(current_start_time.timestamp() * 1000)
        logging.info(f"{symbol} için {current_start_time} itibariyle {limit} kline çekiliyor...")
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_ts_ms, limit=limit)
        except Exception as e:
            logging.error(f"futures_klines API çağrısında hata: {e}")
            # Belirli hata kodlarına göre farklı bekleme süreleri veya çıkış stratejileri eklenebilir
            time.sleep(5) # Genel bir hata durumunda 5 saniye bekle ve tekrar dene
            continue # Döngünün başına dön

        if not klines:
            logging.info("Daha fazla veri bulunamadı veya API'den boş liste döndü.")
            break

        df_chunk = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df_chunk = df_chunk[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df_list.append(df_chunk)

        last_timestamp_ms = klines[-1][0]
        current_start_time = pd.to_datetime(last_timestamp_ms, unit='ms', utc=True) + pd.Timedelta(milliseconds=1)

        if end_time_dt and current_start_time > end_time_dt:
            logging.info(f"Belirtilen bitiş zamanına ({end_str_utc}) ulaşıldı.")
            break
        
        if len(klines) < limit:
            logging.info(f"API istenen limitten ({limit}) daha az veri ({len(klines)}) döndürdü. Veri çekme tamamlandı.")
            break
        time.sleep(0.2)

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    
    full_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    full_df.set_index('timestamp', inplace=True)
    full_df.sort_index(inplace=True)
    full_df.dropna(inplace=True) # Tüm temel sütunlarda NaN varsa at
    return full_df

def create_market_direction_target(df_close_series, look_forward_periods, price_change_threshold_pct):
    if not isinstance(df_close_series, pd.Series):
        raise ValueError("df_close_series bir Pandas Serisi olmalıdır.")

    n = len(df_close_series)
    targets = pd.Series(np.nan, index=df_close_series.index)

    for i in range(n - look_forward_periods):
        current_price = df_close_series.iloc[i]
        future_price = df_close_series.iloc[i + look_forward_periods]
        
        if pd.isna(current_price) or pd.isna(future_price) or current_price == 0:
            targets.iloc[i] = np.nan
            continue

        price_change = (future_price - current_price) / current_price

        if price_change > price_change_threshold_pct:
            targets.iloc[i] = 1  # LONG
        elif price_change < -price_change_threshold_pct:
            targets.iloc[i] = 0  # SHORT
        else:
            targets.iloc[i] = 2  # NEUTRAL
    
    return targets.astype('category')

def bullish_engulfing(df_column_open, df_column_close):
    # ... (bu fonksiyonun tanımı aynı) ...
    prev_open = df_column_open.shift(1)
    prev_close = df_column_close.shift(1)
    condition1 = prev_close < prev_open
    condition2 = df_column_close > df_column_open
    condition3 = (df_column_close > prev_open) & (df_column_open < prev_close)
    return condition1 & condition2 & condition3

def bearish_engulfing(df_column_open, df_column_close):
    # ... (bu fonksiyonun tanımı aynı) ...
    prev_open = df_column_open.shift(1)
    prev_close = df_column_close.shift(1)
    condition1 = prev_close > prev_open
    condition2 = df_column_close < df_column_open
    condition3 = (df_column_open > prev_close) & (df_column_close < prev_open)
    return condition1 & condition2 & condition3

def create_features(df_klines):
    df = df_klines.copy()

    # Optimizer'dan gelen veya manuel olarak belirlenen en iyi parametreler
    EMA_SHORT_PERIOD = 12
    EMA_LONG_PERIOD = 30
    RSI_PERIOD = 14
    # RSI_OB_THRESHOLD = 70 # Bu eşikler artık doğrudan özellik üretmeyecek
    # RSI_OS_THRESHOLD = 30 # Bu eşikler artık doğrudan özellik üretmeyecek
    # RSI_MODE = 'momentum' # rsi_is_... özellikleri kaldırıldığı için bu da gereksiz olabilir
                            # ama rsi değeri hala hesaplanıyor, o yüzden kalsın.
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGN = 9
    USE_VWAP = True
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_STD_DEV = 2.0
    USE_OBV_SLOPE = False 
    OBV_SLOPE_WINDOW = 5
    ADX_PERIOD = 20
    USE_ADX_FILTER = True # ADX özelliklerini ekleyip eklememeyi kontrol eder

    # EMA
    df['ema_short'] = ta.trend.ema_indicator(df['close'], window=EMA_SHORT_PERIOD, fillna=False)
    df['ema_long'] = ta.trend.ema_indicator(df['close'], window=EMA_LONG_PERIOD, fillna=False)
    df['ema_diff_abs'] = df['ema_short'] - df['ema_long']
    df['ema_diff_pct'] = np.where(df['ema_long'] != 0, (df['ema_short'] - df['ema_long']) / df['ema_long'] * 100, 0)

    # RSI (Ham RSI değeri hala bir özellik)
    df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD, fillna=False)
    # rsi_is_bullish_momentum ve rsi_is_bearish_momentum özellikleri KALDIRILDI

    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGN, fillna=False)
    df['macd_line'] = macd.macd()
    df['macd_signal_line'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # VWAP
    if USE_VWAP and 'volume' in df.columns and not df['volume'].isnull().all() and df['volume'].sum() > 0 :
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], fillna=False)
        df['price_vs_vwap_pct'] = np.where(df['vwap'].notnull() & (df['vwap'] != 0), (df['close'] - df['vwap']) / df['vwap'] * 100, 0)
    else:
        df['vwap'] = np.nan
        df['price_vs_vwap_pct'] = np.nan
        
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_PERIOD, fillna=False)
    df['atr_pct'] = np.where(df['close'] != 0, (df['atr'] / df['close']) * 100, 0)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'], window=BB_PERIOD, window_dev=BB_STD_DEV, fillna=False)
    df['bb_hband'] = bb.bollinger_hband() # Bunları da özellik olarak ekleyebiliriz
    df['bb_lband'] = bb.bollinger_lband() # Bunları da özellik olarak ekleyebiliriz
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_pband'] = bb.bollinger_pband() 
    df['bb_wband'] = np.where(df['bb_mavg'].notnull() & (df['bb_mavg'] != 0), (df['bb_hband'] - df['bb_lband']) / df['bb_mavg'] * 100, 0)

    # OBV
    if USE_OBV_SLOPE and 'volume' in df.columns and not df['volume'].isnull().all():
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'], fillna=False)
        df['obv_slope'] = df['obv'].diff(OBV_SLOPE_WINDOW)
    else:
        df['obv_slope'] = np.nan

    # ADX
    if USE_ADX_FILTER:
        adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=ADX_PERIOD, fillna=False)
        df['adx'] = adx_ind.adx()
        df['adx_pos'] = adx_ind.adx_pos()
        df['adx_neg'] = adx_ind.adx_neg()
    else:
        df['adx'] = np.nan
        df['adx_pos'] = np.nan
        df['adx_neg'] = np.nan

    # Ek Basit Özellikler
    df['price_change_1p'] = df['close'].pct_change(periods=1)
    df['price_change_5p'] = df['close'].pct_change(periods=5)
    
    # volume_change_1p KALDIRILDI
    # if 'volume' in df.columns:
    #     prev_volume = df['volume'].shift(1)
    #     df['volume_change_1p'] = np.where(
    #         prev_volume.notnull() & (prev_volume != 0),
    #         (df['volume'] - prev_volume) / prev_volume,
    #         0 
    #     )
    # else:
    #     df['volume_change_1p'] = np.nan

    # Engulfing Özellikleri
    ADD_ENGULFING_FEATURES = True # Şimdilik False, bir sonraki adımda True yapacağız
    if ADD_ENGULFING_FEATURES:
        df['bullish_engulfing'] = bullish_engulfing(df['open'], df['close']).astype(int)
        df['bearish_engulfing'] = bearish_engulfing(df['open'], df['close']).astype(int)

    # Özellik olarak kullanılacak sütunları tanımla
    feature_columns = [
        'ema_diff_abs', 'ema_diff_pct', 'rsi', # Ham RSI değeri kaldı
        'macd_line', 'macd_signal_line', 'macd_diff',
        'price_vs_vwap_pct', 'atr_pct', 
        'bb_hband', 'bb_lband', 'bb_mavg', # BB bantlarını da ekledim
        'bb_pband', 'bb_wband',
        'price_change_1p', 'price_change_5p'
        # 'volume_change_1p' KALDIRILDI
    ]
    # Koşullu eklenenler (RSI momentum/reversal özellikleri KALDIRILDI)
    
    if USE_OBV_SLOPE: feature_columns.append('obv_slope')
    if USE_ADX_FILTER: feature_columns.extend(['adx', 'adx_pos', 'adx_neg'])
    
    if ADD_ENGULFING_FEATURES:
        feature_columns.extend(['bullish_engulfing', 'bearish_engulfing'])

    existing_feature_columns = [col for col in feature_columns if col in df.columns]
    X = df[existing_feature_columns].copy()
        
    return X



# Örnek Kullanım (test için):
# if __name__ == "__main__":
#     client = Client() # API anahtarına gerek yok public veri için
#     start_date = "2023-01-01 00:00:00" # UTC
#     # end_date = "2023-01-10 00:00:00" # UTC (opsiyonel)
#     # df_eth = get_klines_iterative(client, "ETHUSDT", Client.KLINE_INTERVAL_15MINUTE, start_date, end_date)
#     # Sadece başlangıç verip son N mumu çekmek için limitli bir yapıya da dönüştürülebilir
#     # veya start_date'i (now - X gün) olarak ayarlayabiliriz.
#     # Şimdilik belirli bir aralık çekmek daha kontrollü.
    
#     # Son 2 yılın verisini çekmek için:
#     two_years_ago = (datetime.now(timezone.utc) - timedelta(days=2*365)).strftime('%Y-%m-%d %H:%M:%S')
#     now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
#     df_eth_long = get_klines_iterative(client, "ETHUSDT", Client.KLINE_INTERVAL_15MINUTE, two_years_ago, now_utc_str)
#     if not df_eth_long.empty:
#         print(f"Toplam {len(df_eth_long)} adet 15dk'lık ETHUSDT mumu çekildi.")
#         print(df_eth_long.head())
#         print(df_eth_long.tail())
#         # df_eth_long.to_csv("ethusdt_15m_last_2_years.csv")
#     else:
#         print("Veri çekilemedi.")