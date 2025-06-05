# -*- coding: utf-8 -*-
# AI-Powered QuantumTrade Bot (TEMA+SuperTrend, ML Destekli, Full Çalışır)

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import ta
import tensorflow as tf
import pytz
import logging
from datetime import datetime, timedelta
import time
import hmac
import hashlib
import urllib.parse
import requests
import warnings
import joblib # Model kaydetme/yükleme için eklendi

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn') # Sklearn UserWarning'lerini gizle

# ===== KONFİGÜRASYON =====
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
IST_TIMEZONE = pytz.timezone("Europe/Istanbul")
COOLDOWN_MINUTES = 60 # Stop loss sonrası cooldown süresi 1 saat yapıldı
MODEL_UPDATE_INTERVAL = timedelta(hours=6) # Modellerin kaç saatte bir yeniden eğitileceği
LSTM_WINDOW_SIZE = 30 # LSTM için geçmiş veri penceresi (şu an kullanılmıyor, sadece hedef için N_FUTURE_CANDLES kullanılıyor)
N_FUTURE_CANDLES = 3 # Hedef değişken oluştururken gelecekteki kaç mum çubuğuna bakılacağı
BASE_URL = "https://fapi.binance.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


MARKET_DIRECTION_SYMBOLS = ['BTCUSDT', 'ETHUSDT'] # Piyasa yönünü belirlemek için kullanılacak semboller
MARKET_DIRECTION_INTERVAL = '4h' # Piyasa yönünü belirlemek için kullanılacak zaman aralığı (örn. 4 saatlik)
MARKET_DIRECTION_LOOKBACK = 50 # Piyasa yönü için çekilecek mum sayısı

# ========== YARDIMCI FONKSİYONLAR ==========

SYMBOLS = [
    'AAVEUSDT', 'ALGOUSDT', 'ALICEUSDT', 'APEUSDT', 'APTUSDT', 'ARUSDT',
    'ATAUSDT', 'AUCTIONUSDT', 'AVAXUSDT', 'AXSUSDT', 'BCHUSDT', 'BLURUSDT',
    'BNBUSDT', 'BTCUSDT', 'BTCDOMUSDT', 'CELOUSDT', 'COMPUSDT', 'CRVUSDT',
    'CYBERUSDT', 'DOGEUSDT', 'DOTUSDT', 'DYDXUSDT', 'EGLDUSDT', 'ENAUSDT',
    'ENJUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'FLOWUSDT', 'GRTUSDT',
    'INJUSDT', 'IOTAUSDT', 'JTOUSDT', 'KAVAUSDT', 'LINKUSDT', 'LDOUSDT',
    'LPTUSDT', 'LTCUSDT', 'MANAUSDT', 'MKRUSDT', 'NEARUSDT', 'NEOUSDT',
    'OGNUSDT', 'OPUSDT', 'PAXGUSDT', 'POLUSDT', 'QNTUSDT', 'RUNEUSDT',
    'SOLUSDT', 'SSVUSDT', 'STXUSDT', 'SUIUSDT', 'SUSHIUSDT', 'TAOUSDT',
    'THETAUSDT', 'TRBUSDT', 'UNIUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT',
    'ZECUSDT', 'ZRXUSDT'
]

SYMBOL_LEVERAGE = {s: 20 for s in SYMBOLS}

def sign_params(params):
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

class SymbolInfo:
    """Sembollerin filtre ve hassasiyet bilgilerini tutar."""
    def __init__(self, symbol):
        self.symbol = symbol
        self.price_precision = 0 
        self.min_price = 0.0
        self.max_price = 0.0
        self.tick_size = 0.0     
        self.quantity_precision = 0 
        self.min_quantity = 0.0
        self.max_quantity = 0.0
        self.step_size = 0.0     
        self.min_notional = 0.0

def supertrend(df, period=7, multiplier=2):
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
    
    if atr.isnull().all():
        logging.warning("ATR tamamen NaN, SuperTrend hesaplanamadı.")
        df['SuperTrend'] = np.nan
        df['SuperTrend_Up'] = np.nan
        return df

    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    
    supertrend_val = np.zeros(len(df))
    direction = np.ones(len(df), dtype=int)
    
    for i in range(1, len(df)):
        if pd.isna(supertrend_val[i-1]) or supertrend_val[i-1] == 0:
            supertrend_val[i] = final_lowerband.iloc[i] if not pd.isna(final_lowerband.iloc[i]) else 0
            direction[i] = 1
        elif df['close'].iloc[i - 1] > supertrend_val[i - 1]:
            supertrend_val[i] = max(final_lowerband.iloc[i], supertrend_val[i - 1])
            direction[i] = 1
        else:
            supertrend_val[i] = min(final_upperband.iloc[i], supertrend_val[i - 1])
            direction[i] = 0

    df['SuperTrend'] = supertrend_val
    df['SuperTrend_Up'] = direction
    return df

def tema(series, n=10):
    if len(series) < n:
        return pd.Series([np.nan] * len(series), index=series.index)
    ema1 = series.ewm(span=n, adjust=False).mean()
    ema2 = ema1.ewm(span=n, adjust=False).mean()
    ema3 = ema2.ewm(span=n, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

def teknik_analiz(df):
    df_copy = df.copy()
    df_copy['TEMA_10'] = tema(df_copy['close'], n=10)
    df_copy = supertrend(df_copy, period=7, multiplier=2)
    
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=14)
    df_copy['MACD'] = ta.trend.macd_diff(df_copy['close'])
    if len(df_copy) >= 14:
        df_copy['VWAP'] = ta.volume.volume_weighted_average_price(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'])
    else:
        df_copy['VWAP'] = np.nan

    trend_up = False
    trend_down = False
    
    if len(df_copy) >= 3 and \
       not pd.isna(df_copy['TEMA_10'].iloc[-1]) and \
       not pd.isna(df_copy['TEMA_10'].iloc[-3]) and \
       not pd.isna(df_copy['VWAP'].iloc[-1]) and \
       not pd.isna(df_copy['SuperTrend_Up'].iloc[-1]) and \
       not pd.isna(df_copy['RSI'].iloc[-1]) and \
       not pd.isna(df_copy['MACD'].iloc[-1]):
        
        trend_up = (df_copy['SuperTrend_Up'].iloc[-1] == 1 and 
                    df_copy['TEMA_10'].iloc[-1] > df_copy['TEMA_10'].iloc[-3] and
                    df_copy['close'].iloc[-1] > df_copy['VWAP'].iloc[-1] and
                    df_copy['RSI'].iloc[-1] > 50 and # RSI 50'nin üzerinde olmalı
                    df_copy['MACD'].iloc[-1] > 0)    # MACD pozitif olmalı
        
        trend_down = (df_copy['SuperTrend_Up'].iloc[-1] == 0 and
                      df_copy['TEMA_10'].iloc[-1] < df_copy['TEMA_10'].iloc[-3] and
                      df_copy['close'].iloc[-1] < df_copy['VWAP'].iloc[-1] and
                      df_copy['RSI'].iloc[-1] < 50 and # RSI 50'nin altında olmalı
                      df_copy['MACD'].iloc[-1] < 0)     # MACD negatif olmalı
    
    return trend_up, trend_down, df_copy

# ========== İSTATİSTİK MODÜLÜ ==========
class TradeStatistics:
    def __init__(self):
        self.trades = []
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'peak_equity': 0.0,
            'max_drawdown': 0.0,
            'daily_volume': 0.0
        }

    def add_trade(self, symbol, direction, entry, exit_price, quantity, pnl_reason='Unknown'):
        pnl = (exit_price - entry) * quantity if direction == 'LONG' else (entry - exit_price) * quantity
        
        if abs(pnl) < 0.01 and pnl != 0:
            if pnl < 0:
                pnl = -0.01
            else:
                pnl = 0.01
        else:
            pnl = round(pnl, 2)
            
        is_profitable = pnl > 0

        trade_record = {
            'timestamp': datetime.now(IST_TIMEZONE),
            'symbol': symbol,
            'direction': direction,
            'entry': entry,
            'exit': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'success': is_profitable,
            'reason': pnl_reason
        }

        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['profitable_trades'] += int(is_profitable)
        self.performance_metrics['total_pnl'] += pnl
        self.performance_metrics['daily_volume'] += abs(quantity * entry)

        if self.performance_metrics['total_pnl'] > self.performance_metrics['peak_equity']:
            self.performance_metrics['peak_equity'] = self.performance_metrics['total_pnl']
        current_drawdown = self.performance_metrics['peak_equity'] - self.performance_metrics['total_pnl']
        if current_drawdown > self.performance_metrics['max_drawdown']:
            self.performance_metrics['max_drawdown'] = current_drawdown

        self.trades.append(trade_record)
        self._clean_old_trades()

    def _clean_old_trades(self):
        cutoff = datetime.now(IST_TIMEZONE) - timedelta(days=30)
        self.trades = [t for t in self.trades if t['timestamp'] > cutoff]

    def get_stats(self, period_hours=24):
        cutoff = datetime.now(IST_TIMEZONE) - timedelta(hours=period_hours)
        recent_trades = [t for t in self.trades if t['timestamp'] > cutoff]

        if not recent_trades:
            return {
                'win_rate': 0.0,
                'total_trades': 0,
                'total_pnl_period': 0.0,
                'avg_pnl': 0.0,
                'max_drawdown_period': 0.0,
                'volume': 0.0,
                'overall_max_drawdown': self.performance_metrics['max_drawdown']
            }

        profitable = sum(1 for t in recent_trades if t['success'])
        total = len(recent_trades)
        total_pnl_period = sum(t['pnl'] for t in recent_trades)
        avg_pnl = total_pnl_period / total
        volume = sum(abs(t['quantity'] * t['entry']) for t in recent_trades)

        period_equity_curve = [0] + [t['pnl'] for t in recent_trades]
        cumulative_pnl_period = np.cumsum(period_equity_curve)
        peak_equity_period = 0.0
        max_drawdown_period = 0.0
        for pnl_val in cumulative_pnl_period:
            if pnl_val > peak_equity_period:
                peak_equity_period = pnl_val
            current_drawdown_period = peak_equity_period - pnl_val
            if current_drawdown_period > max_drawdown_period:
                max_drawdown_period = current_drawdown_period

        return {
            'win_rate': profitable / total * 100,
            'total_trades': total,
            'total_pnl_period': total_pnl_period,
            'avg_pnl': avg_pnl,
            'max_drawdown_period': max_drawdown_period,
            'volume': volume,
            'overall_max_drawdown': self.performance_metrics['max_drawdown']
        }

class QuantumTrader:
    
    async def get_futures_price(self, symbol):
        """Binance Futures fiyatını çeker."""
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        async with self.session.get(url) as resp:
            data = await resp.json()
            return float(data['price'])

    def format_quantity(self, symbol, quantity):
        info = self.symbol_info.get(symbol)
        if not info:
            logging.error(f"[{symbol}] Sembol bilgisi bulunamadı, varsayılan miktar hassasiyeti kullanılıyor.")
            return round(quantity, 3)
        
        qty = (int(quantity / info.step_size)) * info.step_size
        qty = round(qty, info.quantity_precision)
        return max(qty, info.min_quantity)

    def format_price(self, symbol, price):
        info = self.symbol_info.get(symbol)
        if not info:
            logging.error(f"[{symbol}] Sembol bilgisi bulunamadı, varsayılan fiyat hassasiyeti kullanılıyor.")
            return round(price, 2)
        
        return round(price, info.price_precision)

    def __init__(self, symbols_to_trade, telegram_token, chat_id):
        self.session = None
        self.headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        self.symbols_to_trade = symbols_to_trade
        self.models = {'xgb': {}, 'lgbm': {}, 'gbt': {}, 'lstm': {}}
        self.scalers = {}
        self.required_features = ['TEMA_10', 'SuperTrend', 'SuperTrend_Up', 'RSI', 'MACD', 'VWAP']
        self.market_phase = "INITIALIZING"
        self.cooldowns = {}
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.lock = asyncio.Lock()
        
        self.active_positions = {} 
        self.last_error_notified = {} 
        self.error_notification_cooldown = timedelta(hours=1)
        
        self.stats = TradeStatistics()
        self.report_task = None
        self.last_model_update = datetime.now(IST_TIMEZONE) - MODEL_UPDATE_INTERVAL
        
        self.symbol_info = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        
        await self.fetch_exchange_info()

        for symbol in self.symbols_to_trade:
            if symbol not in self.symbol_info:
                logging.warning(f"[{symbol}] Exchange bilgisi yüklü değil, kaldıraç ayarı atlanıyor.")
                continue
            await self.set_leverage(symbol, SYMBOL_LEVERAGE.get(symbol, 20))
        
        logging.info("Binance'deki mevcut açık pozisyonlar yükleniyor...")
        current_open_positions_binance = await self.get_all_open_positions_with_details()
        for pos_details in current_open_positions_binance:
            symbol = pos_details['symbol']
            if symbol in self.symbols_to_trade and symbol in self.symbol_info:
                amt = float(pos_details['positionAmt'])
                direction = 'LONG' if amt > 0 else 'SHORT'
                self.active_positions[symbol] = {
                    'entry': float(pos_details['entryPrice']),
                    'quantity': abs(amt),
                    'original_total_quantity': abs(amt), # <<--- BURAYA EKLENMELİ
                    'direction': direction,
                    'stop_loss_price': None,
                    'tp1_price': None,
                    'tp2_price': None,
                    'tp3_price': None,
                    'sl_moved_to_entry': False,
                    # Aşağıdaki anahtarları da eklemek iyi olur, böylece SL/TP yeniden gönderme mantığı daha tutarlı çalışır
                    'main_sl_order_id': None, # Bilinmiyor, belki daha sonra Binance'den çekilebilir
                    'tp1_order_id': None,
                    'tp2_order_id': None,
                    'tp3_order_id': None,
                    'tp1_status': 'PENDING', # Ya da 'UNKNOWN' ve sonra emirleri kontrol et
                    'tp2_status': 'PENDING',
                    'tp3_status': 'PENDING'
                }
                logging.info(f"[{symbol}] Mevcut pozisyon yüklendi: Yön={direction}, Giriş={float(pos_details['entryPrice']):.{self.symbol_info[symbol].price_precision}f}, Miktar={abs(amt):.{self.symbol_info[symbol].quantity_precision}f}")
            else:
                logging.warning(f"[{symbol}] Mevcut pozisyon botun takip listesinde değil veya exchange bilgisi yok. Bu pozisyon yönetilmeyecek.")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def fetch_exchange_info(self):
        """Binance Futures exchangeInfo'dan sembol filtrelerini çeker."""
        url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                if not data or 'symbols' not in data:
                    logging.error("Binance exchangeInfo çekilemedi veya boş geldi.")
                    return

                for s_info in data['symbols']:
                    symbol = s_info['symbol']
                    if symbol not in self.symbols_to_trade:
                        continue

                    info = SymbolInfo(symbol)
                    info.price_precision = s_info['pricePrecision']
                    info.quantity_precision = s_info['quantityPrecision']
                    
                    for f in s_info['filters']:
                        if f['filterType'] == 'PRICE_FILTER':
                            info.tick_size = float(f['tickSize'])
                            info.min_price = float(f['minPrice'])
                            info.max_price = float(f['maxPrice'])
                        elif f['filterType'] == 'LOT_SIZE':
                            info.step_size = float(f['stepSize'])
                            info.min_quantity = float(f['minQty'])
                            info.max_quantity = float(f['maxQty'])
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            info.min_notional = float(f['notional'])
                    
                    self.symbol_info[symbol] = info
                logging.info("Binance exchangeInfo başarıyla yüklendi.")
        except aiohttp.ClientError as e:
            logging.error(f"Binance exchangeInfo çekilirken ağ hatası: {e}")
        except Exception as e:
            logging.error(f"Binance exchangeInfo çekilirken beklenmeyen hata: {e}", exc_info=True)

    async def send_telegram_message(self, message: str):
        """Doğrudan Telegram API'sine mesaj gönderir."""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            params = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            async with self.session.post(url, json=params) as response:
                if response.status != 200:
                    error_data = await response.text()
                    logging.error(f"Telegram API hatası: {response.status} - {error_data}")
                    return False
            return True
        except Exception as e:
            logging.error(f"Telegram mesaj gönderme hatası: {str(e)}")
            return False

    async def send_smart_telegram_message(self, message: str, msg_type: str = 'INFO', symbol: str = None):
        """
        Cooldown uygulayarak Telegram mesajı gönderir.
        msg_type: 'INFO', 'WARNING', 'ERROR'. Cooldown sadece WARNING ve ERROR için geçerlidir.
        """
        if msg_type == 'ERROR' or msg_type == 'WARNING':
            if symbol:
                key = f"{msg_type}_{symbol}"
            else:
                key = msg_type
            
            now = datetime.now(IST_TIMEZONE)
            if key in self.last_error_notified and (now - self.last_error_notified[key]) < self.error_notification_cooldown:
                logging.info(f"Telegram {msg_type} mesajı cooldown'da, gönderilmiyor. Key: {key}")
                return False
            self.last_error_notified[key] = now

        await self.send_telegram_message(message)
        return True

    async def handle_position_close(self, symbol):
        pos_info = self.active_positions.get(symbol)
        if not pos_info:
            logging.warning(f"[{symbol}] Kapatılan pozisyon için aktif kayıt bulunamadı (handle_position_close). Atlanıyor.")
            return

        entry = pos_info['entry']
        original_quantity = pos_info['original_total_quantity']
        direction = pos_info['direction']
        open_timestamp = pos_info.get('open_timestamp', 0)

        pnl = 0.0
        exit_price_for_stats = entry
        pnl_reason = "Unknown"

        try:
            pnl_data = await self.get_realized_pnl_from_trades(symbol, open_timestamp)
            pnl = pnl_data['realized_pnl']
            exit_price_for_stats = pnl_data['avg_exit_price'] if pnl_data['avg_exit_price'] > 0 else entry

            if pnl > 0:
                if pos_info.get('tp3_status') == 'EXECUTED':
                    pnl_reason = "Take Profit (TP3)"
                elif pos_info.get('tp2_status') == 'EXECUTED':
                    pnl_reason = "Take Profit (TP2)"
                elif pos_info.get('tp1_status') == 'EXECUTED':
                    pnl_reason = "Take Profit (TP1)"
                else:
                    pnl_reason = "Take Profit (Unknown Reason)"
            elif pnl < 0:
                pnl_reason = "Stop Loss"
            else:
                pnl_reason = "Breakeven"

        except Exception as e:
            logging.error(f"[{symbol}] PnL çekilirken/hesaplanırken hata (handle_position_close): {e}. PnL 0 olarak kabul edildi.", exc_info=True)
            pnl = 0.0
            pnl_reason = "PnL Hatası"

        # ... (Telegram mesajı gönderme kısmı) ...

        if pnl_reason.startswith("Stop Loss"):
            self.cooldowns[symbol] = datetime.now(IST_TIMEZONE) + timedelta(minutes=COOLDOWN_MINUTES)
            logging.info(f"[{symbol}] {pnl_reason} nedeniyle {COOLDOWN_MINUTES} dakika cooldown'a alındı.")

        try:
            # Marjin temizleme işlemi burada yapılır
            await self.clean_margin(symbol) # Yeni metod
        except Exception as e_clean:
             logging.error(f"[{symbol}] Marjin temizleme işlemi sırasında hata: {e_clean}", exc_info=True)
             

        # Pozisyonu en sonda sil (hata olsa bile)
        del self.active_positions[symbol]
        logging.info(f"[{symbol}] Aktif pozisyon kaydı (handle_position_close) bot içinden silindi.")

    async def get_realized_pnl_from_trades(self, symbol, open_timestamp):
        """
        Bir sembol için belirli bir zamandan sonraki tüm trade'lerin gerçekleşen PnL'sini toplar.
        Bu, kısmi TP'lerle kapanan pozisyonlar için toplam PnL'yi doğru bulur.
        Dönüş değeri: {'realized_pnl': float, 'avg_exit_price': float, 'total_closed_quantity': float}
        """
        url = BASE_URL + "/fapi/v1/userTrades"
        realized_pnl = 0.0
        total_closed_quantity = 0.0
        total_exit_value = 0.0 # Exit price * quantity for weighted average
        
        # Binance /fapi/v1/userTrades en fazla 7 günlük veri çeker.
        # Eğer open_timestamp 7 günden eskiyse, sorguyu son 7 günle sınırla.
        # Ancak, pozisyonun kendi açılış zamanından daha eski trade'leri almamalıyız.
        start_time_api_limit = int((datetime.now(IST_TIMEZONE) - timedelta(days=6, hours=23)).timestamp() * 1000) # Son 7 güne yakın
        
        # open_timestamp'ın milisaniye cinsinden olduğundan emin olmalıyız. 
        # Eğer __aenter__ veya execute_trade'de saniye cinsinden kaydediliyorsa, burada *1000 yapılmalı.
        # Mevcut kodunuzda execute_trade'de 'updateTime' (milisaniye) kullanılıyor.
        # __aenter__'da 'time' (milisaniye) kullanılıyor olmalı.
        
        actual_start_time_for_query = max(open_timestamp, start_time_api_limit)
        
        params = {
            "symbol": symbol,
            "startTime": actual_start_time_for_query,
            # "endTime": int(time.time() * 1000), # endTime belirtmek bazen son trade'leri kaçırabilir, belirtmeyebiliriz.
            "limit": 1000, # Max limit
            "timestamp": int(time.time() * 1000)
        }
        signed_params = sign_params(params)
        
        try:
            async with self.session.get(url, headers=self.headers, params=signed_params) as resp:
                resp.raise_for_status()  # HTTP durum kodlarını kontrol et (4xx, 5xx)
                data = await resp.json()

                if not isinstance(data, list):
                    logging.error(f"[{symbol}] Trade geçmişi alınamadı veya format yanlış: {data}")
                    return {'realized_pnl': 0.0, 'avg_exit_price': 0.0, 'total_closed_quantity': 0.0}

                # Trade'leri zamana göre sırala (Binance genellikle sıralı verir ama garanti değil)
                # data.sort(key=lambda x: int(x.get('time', 0))) # Opsiyonel, eğer sıralı gelmiyorsa

                for trade in data:
                    trade_time = int(trade.get('time', 0))
                    # Sadece pozisyonun açılış zamanından sonraki veya eşit olan trade'leri dikkate al
                    if trade_time >= open_timestamp:
                        pnl_from_trade = float(trade.get('realizedPnl', 0))
                        
                        # Sadece PnL'si olan (yani bir kapanışa işaret eden) trade'leri topla
                        # VEYA daha güvenli bir yol: 'side' ve 'positionSide' ile kapanış trade'lerini belirle.
                        # Şimdilik realizedPnl != 0 yeterli olabilir.
                        if pnl_from_trade != 0:
                            realized_pnl += pnl_from_trade
                            closed_qty_trade = float(trade.get('qty', 0))
                            closed_price_trade = float(trade.get('price', 0))
                            
                            total_closed_quantity += closed_qty_trade
                            total_exit_value += closed_qty_trade * closed_price_trade
            
            avg_exit_price = total_exit_value / total_closed_quantity if total_closed_quantity > 0 else 0.0
            
            logging.info(f"[{symbol}] Gerçekleşen PnL (userTrades): {realized_pnl:.2f}, Ort. Çıkış Fiyatı: {avg_exit_price:.{self.symbol_info.get(symbol, SymbolInfo(symbol)).price_precision}f}, Kapanan Miktar: {total_closed_quantity}")
            return {'realized_pnl': realized_pnl, 'avg_exit_price': avg_exit_price, 'total_closed_quantity': total_closed_quantity}

        except aiohttp.ClientResponseError as e:
            logging.error(f"[{symbol}] Trade geçmişi alınırken API hatası ({url}): {e.status} {e.message}")
            return {'realized_pnl': 0.0, 'avg_exit_price': 0.0, 'total_closed_quantity': 0.0}
        except aiohttp.ClientError as e: # Ağ hatalarını yakala
            logging.error(f"[{symbol}] Trade geçmişi alınırken ağ hatası: {e}")
            return {'realized_pnl': 0.0, 'avg_exit_price': 0.0, 'total_closed_quantity': 0.0}
        except Exception as e_fetch:
            logging.error(f"[{symbol}] Trade geçmişi çekilirken beklenmedik hata: {e_fetch}", exc_info=True)
            return {'realized_pnl': 0.0, 'avg_exit_price': 0.0, 'total_closed_quantity': 0.0}

    async def send_performance_report(self):
        report = self.stats.get_stats(24)
        
        trades = self.stats.trades
        cutoff = datetime.now(IST_TIMEZONE) - timedelta(hours=24)
        recent_trades = [t for t in trades if t['timestamp'] > cutoff]
        
        long_count = sum(1 for t in recent_trades if t['direction'] == 'LONG')
        short_count = sum(1 for t in recent_trades if t['direction'] == 'SHORT')
        long_win = sum(1 for t in recent_trades if t['direction'] == 'LONG' and t['success'])
        short_win = sum(1 for t in recent_trades if t['direction'] == 'SHORT' and t['success'])
        
        long_loss = long_count - long_win
        short_loss = short_count - short_win
        
        long_winrate = (long_win / long_count * 100) if long_count else 0
        short_winrate = (short_win / short_count * 100) if short_count else 0
        
        best_trade = max(recent_trades, key=lambda t: t['pnl'], default=None)
        worst_trade = min(recent_trades, key=lambda t: t['pnl'], default=None)
        
        from collections import Counter
        symbol_counter = Counter(t['symbol'] for t in recent_trades)
        most_traded = symbol_counter.most_common(1)[0][0] if symbol_counter else '-'
        
        volume_explanation = 'Hacim: Son 24 saatte açılan tüm işlemlerin toplam büyüklüğü (USDT cinsinden, giriş fiyatı x miktar).'
        maxdd_explanation = 'Max Çekilme: Son 24 saatteki işlemlere göre oluşan en büyük dönemsel kayıp (tepeden düşüş).'
        overall_maxdd_explanation = 'Genel Max Çekilme: Botun başlangıcından itibaren toplam kar/zararın gördüğü en büyük dönemsel kayıp.'

        best_trade_price_precision = self.symbol_info.get(best_trade['symbol']).price_precision if best_trade and best_trade['symbol'] in self.symbol_info else 4
        worst_trade_price_precision = self.symbol_info.get(worst_trade['symbol']).price_precision if worst_trade and worst_trade['symbol'] in self.symbol_info else 4

        message = (
            "📊 24 Saatlık Performans Raporu\n\n"
            f"• Win Rate: {report['win_rate']:.1f}%\n"
            f"• Toplam İşlem: {report['total_trades']}\n"
            f"• Long İşlem: {long_count} (Başarı: {long_win}, Başarısız: {long_loss}, Win Rate: {long_winrate:.1f}%)\n"
            f"• Short İşlem: {short_count} (Başarı: {short_win}, Başarısız: {short_loss}, Win Rate: {short_winrate:.1f}%)\n"
            f"• Ort. Kazanç: ${report['avg_pnl']:.2f}\n"
            f"• Toplam Kar/Zarar: ${report['total_pnl_period']:.2f}\n"
            f"• Hacim: ${report['volume']:,.2f}\n  ({volume_explanation})\n"
            f"• Max Çekilme (24s): ${report['max_drawdown_period']:,.2f}\n  ({maxdd_explanation})\n"
            f"• Genel Max Çekilme: ${report['overall_max_drawdown']:,.2f}\n  ({overall_maxdd_explanation})\n"
            f"• En iyi işlem: {best_trade['symbol']} {best_trade['direction']} ${best_trade['pnl']:.2f} (Entry: {best_trade['entry']}, Exit: {best_trade['exit']:.{best_trade_price_precision}f})\n" if best_trade else "• En iyi işlem: -\n"
            f"• En kötü işlem: {worst_trade['symbol']} {worst_trade['direction']} ${worst_trade['pnl']:.2f} (Entry: {worst_trade['entry']}, Exit: {worst_trade['exit']:.{worst_trade_price_precision}f})\n" if worst_trade else "• En kötü işlem: -\n"
            f"• En çok işlem yapılan coin: {most_traded}\n"
            f"\n• ⏰ Zaman: {datetime.now(IST_TIMEZONE).strftime('%Y-%m-%d %H:%M')}"
        )
        await self.send_telegram_message(message)

    # Yeni Model Kaydetme Fonksiyonu
    async def save_models(self, symbol):
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)

        try:
            if symbol in self.models['xgb']:
                joblib.dump(self.models['xgb'][symbol], os.path.join(model_dir, f"{symbol}_xgb.pkl"))
            if symbol in self.models['lgbm']:
                joblib.dump(self.models['lgbm'][symbol], os.path.join(model_dir, f"{symbol}_lgbm.pkl"))
            if symbol in self.models['gbt']:
                joblib.dump(self.models['gbt'][symbol], os.path.join(model_dir, f"{symbol}_gbt.pkl"))
            if symbol in self.models['lstm']:
                self.models['lstm'][symbol].save(os.path.join(model_dir, f"{symbol}_lstm.h5"))
            if symbol in self.scalers:
                joblib.dump(self.scalers[symbol], os.path.join(model_dir, f"{symbol}_scaler.pkl"))
            
            logging.info(f"[{symbol}] Modeller ve Scaler başarıyla kaydedildi.")
        except Exception as e:
            logging.error(f"[{symbol}] Modeller ve Scaler kaydedilirken hata: {e}", exc_info=True)

    # Yeni Model Yükleme Fonksiyonu
    async def load_models(self, symbol):
        model_dir = "saved_models"
        try:
            xgb_path = os.path.join(model_dir, f"{symbol}_xgb.pkl")
            if os.path.exists(xgb_path):
                self.models['xgb'][symbol] = joblib.load(xgb_path)
                logging.info(f"[{symbol}] XGBoost modeli yüklendi.")
            else:
                return False

            lgbm_path = os.path.join(model_dir, f"{symbol}_lgbm.pkl")
            if os.path.exists(lgbm_path):
                self.models['lgbm'][symbol] = joblib.load(lgbm_path)
                logging.info(f"[{symbol}] LightGBM modeli yüklendi.")
            else:
                return False

            gbt_path = os.path.join(model_dir, f"{symbol}_gbt.pkl")
            if os.path.exists(gbt_path):
                self.models['gbt'][symbol] = joblib.load(gbt_path)
                logging.info(f"[{symbol}] GradientBoosting modeli yüklendi.")
            else:
                return False

            lstm_path = os.path.join(model_dir, f"{symbol}_lstm.h5")
            if os.path.exists(lstm_path):
                # Keras modelini yüklerken custom_objects geçirmek gerekebilir
                self.models['lstm'][symbol] = tf.keras.models.load_model(lstm_path)
                logging.info(f"[{symbol}] LSTM modeli yüklendi.")
            else:
                return False

            scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scalers[symbol] = joblib.load(scaler_path)
                logging.info(f"[{symbol}] Scaler yüklendi.")
            else:
                return False
            
            return True
        except Exception as e:
            logging.error(f"[{symbol}] Modeller ve Scaler yüklenirken hata: {e}", exc_info=True)
            return False

    async def find_most_volatile_symbols(self, interval='3m', lookback=100, top_n=10):
        volatilities = []
        logging.info(f"Volatil sembol taraması başlatıldı. Toplam sembol: {len(self.symbols_to_trade)}")
        for symbol in self.symbols_to_trade:
            if symbol not in self.symbol_info:
                logging.warning(f"[{symbol}] Sembol bilgisi bulunamadı, volatilite analizi atlanıyor.")
                continue

            df = await self.fetch_data_multi(symbol, interval=interval, total_limit=lookback)
            if df is None or df.empty or len(df) < 20:
                logging.info(f"[{symbol}] Yeterli geçmiş veri yok ({len(df)} mum), volatilite analizi atlanıyor.")
                continue
            
            atr_series = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            if atr_series.empty or pd.isna(atr_series.iloc[-1]) or atr_series.iloc[-1] <= 0:
                logging.info(f"[{symbol}] Geçersiz ATR değeri: {atr_series.iloc[-1] if not atr_series.empty else 'N/A'}. Volatilite analizi atlanıyor.")
                continue
            
            atr = atr_series.iloc[-1]
            rel_vol = atr / df['close'].iloc[-1] if df['close'].iloc[-1] != 0 else 0
            volatilities.append((symbol, rel_vol))
        volatilities.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, v in volatilities[:top_n]]

        logging.info(f"Volatilite taraması tamamlandı. Uygun {len(volatilities)} sembol bulundu. En volatil {len(top_symbols)} sembol seçildi.")
        return top_symbols, volatilities

     # QuantumTrader sınıfı içinde bir yere ekleyin (örn. find_most_volatile_symbols'ın altına)
    
    async def get_market_direction(self):
        """
        Belirlenen ana sembollerin daha uzun zaman dilimindeki trendine göre
        genel piyasa yönünü belirler.
        Dönüş değeri: 'LONG', 'SHORT', 'NEUTRAL'
        """
        long_count = 0
        short_count = 0
        
        for symbol in MARKET_DIRECTION_SYMBOLS:
            # Sembol bilgisi yoksa veya bot takip etmiyorsa atla
            if symbol not in self.symbol_info or symbol not in SYMBOLS:
                logging.warning(f"[{symbol}] Piyasa yönü belirleme için sembol bilgisi eksik veya listede yok. Atlanıyor.")
                continue

            df = await self.fetch_data_multi(symbol, interval=MARKET_DIRECTION_INTERVAL, total_limit=MARKET_DIRECTION_LOOKBACK)
            if df.empty or len(df) < 20: # Yeterli veri yoksa
                logging.warning(f"[{symbol}] Piyasa yönü için yeterli veri yok, bu sembol atlanıyor.")
                continue
            
            # Daha uzun vadeli SuperTrend kullanabiliriz
            df = supertrend(df.copy(), period=10, multiplier=3) # Daha uzun periyotlu SuperTrend
            
            # Trend belirleme
            # SuperTrend'in son değeri yükseliş trendiyse LONG, düşüş trendiyse SHORT
            if not df.empty and not pd.isna(df['SuperTrend_Up'].iloc[-1]):
                if df['SuperTrend_Up'].iloc[-1] == 1: # SuperTrend yukarıysa
                    long_count += 1
                else: # SuperTrend aşağıysa
                    short_count += 1

        total_symbols_checked = len(MARKET_DIRECTION_SYMBOLS)
        if total_symbols_checked == 0:
            return 'NEUTRAL' # Hiç sembol kontrol edilemediyse nötr

        # Çoğunluk hangi yöndeyse piyasa o yöndedir.
        if long_count > total_symbols_checked / 2: # Yarısından fazlası LONG ise
            logging.info(f"Genel Piyasa Yönü: LONG ({long_count}/{total_symbols_checked} sembol)")
            return 'LONG'
        elif short_count > total_symbols_checked / 2: # Yarısından fazlası SHORT ise
            logging.info(f"Genel Piyasa Yönü: SHORT ({short_count}/{total_symbols_checked} sembol)")
            return 'SHORT'
        else:
            logging.info(f"Genel Piyasa Yönü: NEUTRAL (Belirgin bir yön yok)")
            return 'NEUTRAL'   

    async def clean_margin(self, symbol):
        try:
            pos_data = await self.get_open_position(symbol)
            if not pos_data or 'amt' not in pos_data:
                logging.warning(f"[{symbol}] Pozisyon bilgisi alınamadı veya 'amt' anahtarı eksik: {pos_data}. Marjin temizlemeye gerek yok.")
                return

            quantity_to_close = abs(pos_data['amt'])
            if quantity_to_close <= 0:  # quantity_to_close 0 veya negatif ise işlem yapma
                logging.info(f"[{symbol}] Kapatılacak pozisyon miktarı sıfır veya negatif. Marjin temizlemeye gerek yok.")
                return

            close_order_result = await self.send_binance_market_order(symbol, 'SELL' if pos_data['amt'] > 0 else 'BUY', quantity_to_close, SYMBOL_LEVERAGE.get(symbol, 20))
            if close_order_result and 'orderId' in close_order_result:
                logging.info(f"[{symbol}] Marjin temizleme emri gönderildi. OrderID: {close_order_result['orderId']}")
            else:
                logging.error(f"[{symbol}] Marjin temizleme emri gönderilemedi. Detay: {close_order_result}")

        except Exception as e:
            logging.error(f"[{symbol}] Marjin temizleme işlemi sırasında hata oluştu: {e}", exc_info=True)

    async def get_all_open_positions(self):
        url = BASE_URL + "/fapi/v2/positionRisk"
        params = {"timestamp": int(time.time() * 1000)}
        signed = sign_params(params)
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        async with self.session.get(url, headers=headers, params=signed) as resp:
            data = await resp.json()
            open_positions = []
            if isinstance(data, list):
                for pos in data:
                    amt = float(pos['positionAmt'])
                    if abs(amt) > 0:
                        open_positions.append(pos['symbol'])
            else:
                logging.error(f"Binance pozisyon bilgisi alınamadı: {data}")
            return open_positions

    async def get_all_open_positions_with_details(self):
        url = BASE_URL + "/fapi/v2/positionRisk"
        params = {"timestamp": int(time.time() * 1000)}
        signed = sign_params(params)
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        async with self.session.get(url, headers=headers, params=signed) as resp:
            data = await resp.json()
            detailed_positions = []
            if isinstance(data, list):
                for pos in data:
                    amt = float(pos['positionAmt'])
                    if abs(amt) > 0:
                        detailed_positions.append(pos)
            else:
                logging.error(f"Binance pozisyon bilgisi alınamadı: {data}")
            return detailed_positions

    async def fetch_data_multi(self, symbol, interval='5m', total_limit=1500):
        limit_per_call = 1500
        all_data = []
        end_time = int(time.time() * 1000)
        while total_limit > 0:
            limit = min(limit_per_call, total_limit)
            url = f"{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={end_time}"
            async with self.session.get(url) as response:
                data = await response.json()
                if not data or not isinstance(data, list) or len(data) == 0:
                    logging.warning(f"[{symbol}] İçin veri çekilemedi veya boş geldi: {data}")
                    break
                
                all_data = data + all_data 
                end_time = data[0][0] - 1 
                
                if len(data) < limit:
                    break
                total_limit -= limit
        
        if not all_data:
            return pd.DataFrame()
        
        all_data = all_data[-total_limit:]

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(IST_TIMEZONE)
        return df.dropna().reset_index(drop=True)

    async def get_binance_balance(self):
        url = BASE_URL + "/fapi/v2/balance"
        params = {"timestamp": int(time.time() * 1000)}
        signed_params = sign_params(params) 
        async with self.session.get(url, headers=self.headers, params=signed_params) as resp:
            data = await resp.json()
            if not isinstance(data, list):
                logging.error(f"Binance bakiye bilgisi alınamadı: {data}")
                return 0.0
            for asset in data:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0.0

    async def get_open_position(self, symbol):
        url = BASE_URL + "/fapi/v2/positionRisk"
        params = {"symbol": symbol, "timestamp": int(time.time() * 1000)}
        signed = sign_params(params)
        async with self.session.get(url, headers=self.headers, params=signed) as resp:
            data = await resp.json()
            if not isinstance(data, list):
                logging.error(f"Binance pozisyon riski bilgisi alınamadı: {data}")
                return None
            for pos in data:
                if pos['symbol'] == symbol:
                    amt = float(pos['positionAmt'])
                    entry = float(pos['entryPrice'])
                    unRealized = float(pos['unRealizedProfit'])
                    return {"amt": amt, "entry": entry, "unRealizedProfit": unRealized}
            return None

    async def cancel_order(self, symbol, order_id):
        url = BASE_URL + "/fapi/v1/order"
        params = {"symbol": symbol, "orderId": order_id, "timestamp": int(time.time() * 1000)}
        signed = sign_params(params)
        async with self.session.delete(url, headers=self.headers, params=signed) as resp:
            data = await resp.json()
            if resp.status == 200:
                logging.info(f"[{symbol}] {order_id} no'lu emir iptal edildi: {data.get('orderId')}")
            else:
                logging.error(f"[{symbol}] {order_id} no'lu emir iptal edilemedi: {data}")

    async def cancel_open_orders(self, symbol):
        url = BASE_URL + "/fapi/v1/openOrders"
        params = {"symbol": symbol, "timestamp": int(time.time() * 1000)}
        signed = sign_params(params)

        try:
            async with self.session.get(url, headers=self.headers, params=signed) as resp:
                resp.raise_for_status()
                orders = await resp.json()
                if not isinstance(orders, list):
                    logging.error(f"[{symbol}] Açık emirler alınamadı: {orders}")
                    return

                for order in orders:
                    try:
                        await self.cancel_order(symbol, order['orderId'])
                    except Exception as e:
                        logging.error(f"[{symbol}] {order['orderId']} no'lu emrin iptalinde hata: {e}")

        except aiohttp.ClientResponseError as e:
            logging.error(f"[{symbol}] Açık emirler çekilirken API hatası: {e.status} {e.message}")
        except aiohttp.ClientError as e: # Ağ hataları
            logging.error(f"[{symbol}] Açık emirler çekilirken ağ hatası: {e}")
        except Exception as e:
            logging.error(f"[{symbol}] Açık emirler çekilirken beklenmedik hata: {e}", exc_info=True)

    async def get_open_sl_tp_orders_for_symbol(self, symbol):
        url = BASE_URL + "/fapi/v1/openOrders"
        params = {"symbol": symbol, "timestamp": int(time.time() * 1000)}
        signed = sign_params(params)

        try:
            async with self.session.get(url, headers=self.headers, params=signed) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if not isinstance(data, list):
                    logging.error(f"[{symbol}] Açık emirler alınamadı: {data}")
                    return []

                return [order for order in data if order.get('reduceOnly', False) or order.get('closePosition', False)]

        except aiohttp.ClientResponseError as e:
            logging.error(f"[{symbol}] Açık emirler çekilirken API hatası: {e.status} {e.message}")
            return []
        except aiohttp.ClientError as e:
            logging.error(f"[{symbol}] Açık emirler çekilirken ağ hatası: {e}")
            return []
        except Exception as e:
            logging.error(f"[{symbol}] Açık emirler çekilirken beklenmedik hata: {e}", exc_info=True)
            return []

    async def set_leverage(self, symbol, leverage):
        try:
            lev_params = {
                "symbol": symbol,
                "leverage": leverage,
                "timestamp": int(time.time() * 1000)
            }
            signed = sign_params(lev_params)
            url = BASE_URL + "/fapi/v1/leverage"
            async with self.session.post(url, headers=self.headers, data=signed) as resp:
                data = await resp.json()
                if resp.status == 200:
                    logging.info(f"[{symbol}] Kaldıraç ayarlandı: {leverage}x")
                else:
                    logging.error(f"[{symbol}] Kaldıraç ayarlanamadı: {data}")
        except Exception as e:
            logging.error(f"[{symbol}] Kaldıraç ayarı hatası: {str(e)}")

    async def send_binance_market_order(self, symbol, side, quantity, leverage):
        try:
            await self.set_leverage(symbol, leverage)
            
            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": self.format_quantity(symbol, quantity),
                "timestamp": int(time.time() * 1000)
            }
            signed = sign_params(params)
            url = BASE_URL + "/fapi/v1/order"
            async with self.session.post(url, headers=self.headers, data=signed) as resp:
                data = await resp.json()
                if resp.status == 200 and data.get('orderId'):
                    logging.info(f"[BINANCE] Market emir gönderildi: {side} {symbol} {params['quantity']} - OrderId: {data.get('orderId')}")
                else:
                    logging.error(f"[BINANCE] Market emir gönderilemedi: {data} (Status: {resp.status})")
                return data
        except Exception as e:
            logging.error(f"Market emir gönderme hatası: {str(e)}", exc_info=True)
            return None

    async def send_initial_sl_and_tps(self, symbol, entry_price, total_quantity, stop_loss, tp1, tp2, tp3):
        close_side = 'SELL' if self.active_positions[symbol]['direction'] == 'LONG' else 'BUY'

        main_sl_order_data = await self.place_only_sl_order(symbol, close_side, total_quantity, stop_loss, "MAIN_SL")
        if main_sl_order_data and 'orderId' in main_sl_order_data:
            self.active_positions[symbol]['main_sl_order_id'] = str(main_sl_order_data['orderId'])

        qty_tp1 = total_quantity * 0.3
        qty_tp2 = total_quantity * 0.3
        qty_tp3 = total_quantity - qty_tp1 - qty_tp2

        tp1_order_data = await self.place_tp_order(symbol, close_side, qty_tp1, tp1, "TP1") 
        if tp1_order_data and 'orderId' in tp1_order_data:
            self.active_positions[symbol]['tp1_order_id'] = str(tp1_order_data['orderId'])

        tp2_order_data = await self.place_tp_order(symbol, close_side, qty_tp2, tp2, "TP2")
        if tp2_order_data and 'orderId' in tp2_order_data:
            self.active_positions[symbol]['tp2_order_id'] = str(tp2_order_data['orderId'])
            
        tp3_order_data = await self.place_tp_order(symbol, close_side, qty_tp3, tp3, "TP3")
        if tp3_order_data and 'orderId' in tp3_order_data:
            self.active_positions[symbol]['tp3_order_id'] = str(tp3_order_data['orderId'])

    async def place_only_sl_order(self, symbol, side, quantity, stop_loss, label="SL"):
        qty = self.format_quantity(symbol, quantity)
        if qty <= 0:
            logging.warning(f"[{symbol}][{label}] SL emri için miktar sıfır veya negatif ({qty}). Emir gönderilmedi.")
            return None

        url = BASE_URL + "/fapi/v1/order"
        max_retries = 3
        s_info = self.symbol_info.get(symbol)
        min_tick_size = s_info.tick_size if s_info else 0.001  # Varsayılan tick size

        for attempt in range(max_retries):
            stop_price = self.format_price(symbol, stop_loss + min_tick_size * (attempt + 1)) # formatlama burada
            sl_params = {
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "quantity": qty,
                "stopPrice": stop_price,
                "closePosition": "false",
                "reduceOnly": "true",
                "timestamp": int(time.time() * 1000)
            }
            sl_signed = sign_params(sl_params)

            try:
                async with self.session.post(url, headers=self.headers, data=sl_signed) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if resp.status == 200:
                        logging.info(f"[{symbol}][{label}] STOP-LOSS emri gönderildi. OrderId: {data.get('orderId')}")
                        return data
                    elif resp.status == 400 and data.get('code') == -2021:
                        logging.warning(f"[{symbol}][{label}] STOP-LOSS emri gönderilemedi (-2021). stopPrice {stop_price}'a yükseltildi ({min_tick_size:.8f} artırıldı). Tekrar denenecek...")
                        await asyncio.sleep(1)
                    else:
                        error_msg = f"[{symbol}][{label}] STOP-LOSS emri gönderilemedi! Hata Kodu: {data.get('code')}, Hata Mesajı: {data.get('msg')}, Durum Kodu: {resp.status}"
                        logging.error(error_msg)
                        await self.send_smart_telegram_message(f"❌ <b>{symbol} STOP-LOSS</b> ({label}) emri <b>gönderilemedi</b>! Hata: {error_msg}", msg_type='ERROR', symbol=symbol)
                        raise Exception(error_msg)

            except Exception as e:
                logging.error(f"[{symbol}][{label}] STOP-LOSS emri gönderme hatası: {e}", exc_info=True)
                await self.send_smart_telegram_message(f"❌ <b>{symbol} STOP-LOSS</b> ({label}) emri <b>gönderilemedi</b>! Hata: {e}", msg_type='ERROR', symbol=symbol)
                return None
        return None

    async def place_tp_order(self, symbol, side, quantity, take_profit, label="TP", close_entire_position=False): # Yeni parametre eklendi
        qty = self.format_quantity(symbol, quantity)
        tp_price = self.format_price(symbol, take_profit)

        if qty <= 0 and not close_entire_position: # Eğer pozisyonu kapatmayacaksa ve miktar sıfırsa gönderme
            logging.warning(f"[{symbol}][{label}] TP emri için miktar sıfır veya negatif ({qty}). Emir gönderilmedi.")
            return None

        logging.info(f"[EMIR] {symbol} TP: miktar={qty}, TP={tp_price}, label={label}{', POZİSYONU KAPAT' if close_entire_position else ''}")
        url = BASE_URL + "/fapi/v1/order"
        tp_params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": qty, # closePosition=true olduğunda Binance bu miktarı dikkate almayabilir ama göndermek iyi bir pratik.
            "stopPrice": tp_price,
            # "reduceOnly": "true", # Binance dokümanlarına göre: closePosition=true ise reduceOnly göz ardı edilir.
            "timestamp": int(time.time() * 1000)
        }

        if close_entire_position:
            tp_params["closePosition"] = "true"
            # closePosition true olduğunda reduceOnly Binance tarafından göz ardı edilir.
            # Bu yüzden reduceOnly'yi burada açıkça false yapmaya veya kaldırmaya gerek yok,
            # ancak API'nin nasıl davrandığına bağlı olarak "reduceOnly": "false" eklemek de düşünülebilir.
            # Şimdilik varsayılan (veya mevcut) reduceOnly davranışını koruyalım, closePosition öncelikli olmalı.
            logging.info(f"[{symbol}][{label}] TP emri TÜM POZİSYONU KAPATACAK şekilde ayarlandı.")
        else:
            tp_params["reduceOnly"] = "true" # Sadece pozisyonu kapatmıyorsa reduceOnly ekle

        tp_signed = sign_params(tp_params)
        async with self.session.post(url, headers=self.headers, data=tp_signed) as resp:
            data = await resp.json()
            if resp.status == 200:
                logging.info(f"[{symbol}][{label}] TAKE PROFIT emri gönderildi. OrderId: {data.get('orderId')}")
                return data
            else:
                logging.error(f"[{symbol}][{label}] TAKE PROFIT emri gönderilemedi! Hata: {data} (Status: {resp.status})")
                await self.send_smart_telegram_message(f"❌ <b>{symbol} {label}</b> TP emri <b>gönderilemedi</b>! Hata: {data}", msg_type='ERROR', symbol=symbol)
                return None
    
    async def move_stop_to_entry(self, symbol, quantity, entry_price):
        main_sl_order_id = self.active_positions[symbol].get('main_sl_order_id')
        
        if main_sl_order_id:
            logging.info(f"[{symbol}] SL girişe çekilirken eski ana SL emri iptal ediliyor: {main_sl_order_id}")
            await self.cancel_order(symbol, main_sl_order_id)
            self.active_positions[symbol]['main_sl_order_id'] = None 
            await asyncio.sleep(0.5)
        else:
            logging.warning(f"[{symbol}] SL girişe çekilirken iptal edilecek ana SL emri bulunamadı. Tüm açık reduceOnly emirleri kontrol ediliyor.")
            await self.cancel_open_orders(symbol) 
            await asyncio.sleep(0.5)

        close_side = 'SELL' if self.active_positions[symbol]['direction'] == 'LONG' else 'BUY'
        
        url = BASE_URL + "/fapi/v1/order"
        sl_params = {
            "symbol": symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "quantity": self.format_quantity(symbol, quantity),
            "stopPrice": self.format_price(symbol, entry_price),
            "closePosition": "false",
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }
        sl_signed = sign_params(sl_params)
        async with self.session.post(url, headers=self.headers, data=sl_signed) as resp:
            data = await resp.json()
            if resp.status == 200:
                logging.info(f"[{symbol}] STOP-LOSS entryye çekildi: OrderId: {data.get('orderId')}")
                self.active_positions[symbol]['main_sl_order_id'] = str(data.get('orderId')) if data.get('orderId') else None
            else:
                logging.error(f"[{symbol}] STOP-LOSS entryye çekilemedi! Hata: {data} (Status: {resp.status})")
                await self.send_smart_telegram_message(f"❌ <b>{symbol} STOP-LOSS</b> entryye <b>çekilemedi</b>! Hata: {data}", msg_type='ERROR', symbol=symbol)

    async def execute_trade(self, symbol: str, direction: str, current_price: float, confidence: float, atr: float):
        try:
            if symbol not in self.symbol_info:
                logging.error(f"[{symbol}] Exchange bilgisi yüklenmedi. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için borsa bilgisi alınamadı. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return
            
            s_info = self.symbol_info[symbol]

            logging.info(f"[{symbol}] Model Confidence: {confidence}")
            logging.info(f"[{symbol}] Trade açma işlemi başlıyor. Yön: {direction}, Fiyat: {current_price}, ATR: {atr}")
            
            if current_price < s_info.min_price or atr < s_info.tick_size: 
                logging.error(f"[{symbol}] ATR veya fiyat geçersiz veya çok küçük: ATR={atr:.{s_info.price_precision}f}, Price={current_price:.{s_info.price_precision}f}. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için fiyat ({current_price:.{s_info.price_precision}f}) veya ATR ({atr:.{s_info.price_precision}f}) geçersiz veya çok düşük. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            min_tick = s_info.tick_size
            max_atr_multiplier = 0.03 
            min_atr_multiplier = 0.0005 
            atr_calculated = max(min(current_price * max_atr_multiplier, atr), current_price * min_atr_multiplier)
            
            min_sl_percentage = 0.003
            sl_distance_base = max(atr_calculated * 2.5, current_price * min_sl_percentage)
            sl_distance = max(sl_distance_base, min_tick * 3)
            sl_distance = (round(sl_distance / min_tick)) * min_tick
            
            entry = current_price
            close_side = 'SELL' if direction == 'LONG' else 'BUY'

            if direction == 'LONG':
                stop_loss = entry - sl_distance
                if stop_loss <= s_info.min_price or (entry - stop_loss < min_tick * 2):
                    logging.error(f"[{symbol}] stop_loss geçersiz veya girişe çok yakın: {stop_loss:.{s_info.price_precision}f}. Trade açılamadı. (Entry: {entry:.{s_info.price_precision}f}, SL_dist: {sl_distance:.{s_info.price_precision}f})")
                    return
                tp1 = entry + (sl_distance * 1.0)
                tp2 = entry + (sl_distance * 1.5)
                tp3 = entry + (sl_distance * 2.5)
                order_side = "BUY"
            else: # SHORT
                stop_loss = entry + sl_distance
                if stop_loss >= s_info.max_price or (stop_loss - entry < min_tick * 2):
                    logging.error(f"[{symbol}] stop_loss geçersiz veya girişe çok yakın: {stop_loss:.{s_info.price_precision}f}. Trade açılamadı. (Entry: {entry:.{s_info.price_precision}f}, SL_dist: {sl_distance:.{s_info.price_precision}f})")
                    return
                tp1 = entry - (sl_distance * 1.0)
                tp2 = entry - (sl_distance * 1.5)
                tp3 = entry - (sl_distance * 2.5)
                order_side = "SELL"
            
            entry_fmt_val = self.format_price(symbol, entry)
            stop_loss_fmt_val = self.format_price(symbol, stop_loss)
            tp1_fmt_val = self.format_price(symbol, tp1)
            tp2_fmt_val = self.format_price(symbol, tp2)
            tp3_fmt_val = self.format_price(symbol, tp3)

            prices_to_check = [entry_fmt_val, stop_loss_fmt_val, tp1_fmt_val, tp2_fmt_val, tp3_fmt_val]
            unique_prices = len(set(prices_to_check))
            if unique_prices < 5:
                logging.info(f"{symbol} için Entry/SL/TP fiyatları birbirine çok yakın veya eşit, trade açılamadı.")
                await self.send_smart_telegram_message(f"⚠️ {symbol} için Entry/SL/TP fiyatları birbirine çok yakın veya eşit, trade açılamadı.", msg_type='WARNING', symbol=symbol)
                return

            rr = abs(tp3 - entry) / abs(entry - stop_loss) if abs(entry - stop_loss) > 0 else 0
            if rr < 1.3:
                logging.info(f"[{symbol}] için RR çok düşük ({rr:.2f}), trade açılamadı.")
                await self.send_smart_telegram_message(
                    f"⚠️ {symbol} için RR çok düşük ({rr:.2f}) veya TP3 aralığı çok kısa, trade açılamadı.", msg_type='WARNING', symbol=symbol
                )
                return

            balance = await self.get_binance_balance()
            if balance <= 0:
                logging.error(f"[{symbol}] Bakiye sıfır veya negatif. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için bakiye yetersiz. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            risk_percent = 0.01
            risk_amount = balance * risk_percent
            
            stop_distance_actual = abs(entry - stop_loss)
            if stop_distance_actual == 0:
                logging.error(f"[{symbol}] Stop mesafesi sıfır. Trade açılamadı.")
                return

            position_size = risk_amount / stop_distance_actual
            
            calculated_notional = entry * position_size
            if calculated_notional < s_info.min_notional:
                position_size = s_info.min_notional / entry
                logging.warning(f"[{symbol}] Hesaplanan notional ({calculated_notional:.2f}) min notional'dan ({s_info.min_notional}) küçük. Miktar {position_size:.{s_info.quantity_precision}f} olarak ayarlandı.")

            final_quantity_to_send = self.format_quantity(symbol, position_size)
            
            if final_quantity_to_send < s_info.min_quantity or final_quantity_to_send > s_info.max_quantity:
                logging.error(f"[{symbol}] Nihai miktar ({final_quantity_to_send:.{s_info.quantity_precision}f}) Binance min/max miktar limitleri dışında. Trade açılamadı. (Min: {s_info.min_quantity}, Max: {s_info.max_quantity})")
                await self.send_smart_telegram_message(f"⛔ {symbol} için miktar limitleri karşılanamadı. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            actual_notional_after_format = entry * final_quantity_to_send
            if actual_notional_after_format < s_info.min_notional:
                logging.error(f"[{symbol}] Formatlandıktan sonra bile notional ({actual_notional_after_format:.2f}) hala min notional ({s_info.min_notional}) gereksinimini karşılamıyor. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için notional gereksinimleri karşılanamadı ({actual_notional_after_format:.2f}). Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            leverage = SYMBOL_LEVERAGE.get(symbol, 20)
            required_margin = (entry * final_quantity_to_send) / leverage
            
            if required_margin > balance * 0.98:
                logging.error(f"[{symbol}] Yetersiz marjin: Gereken marjin ({required_margin:.2f} USDT) bakiyeden ({balance:.2f} USDT) fazla. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için yetersiz marjin. Trade açılamadı. Gereken: {required_margin:.2f}, Mevcut: {balance:.2f} (Kullanılabilir: {balance * 0.98:.2f})", msg_type='ERROR', symbol=symbol)
                return

            if final_quantity_to_send <= 0:
                logging.error(f"[{symbol}] Hesaplanan pozisyon miktarı sıfır veya negatif: {final_quantity_to_send}. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için pozisyon miktarı sıfır. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            order_result = await self.send_binance_market_order(
                symbol=symbol,
                side=order_side,
                quantity=final_quantity_to_send,
                leverage=leverage
            )

            order_id = order_result.get('orderId') if order_result else None
            
            if not order_id:
                logging.error(f"[{symbol}] Binance MARKET emir gönderilemedi veya OrderId dönmedi. Detay: {order_result}")
                await self.send_smart_telegram_message(f"⚠️ [BINANCE] {symbol} MARKET emri gönderilemedi. Lütfen manuel kontrol edin.", msg_type='ERROR', symbol=symbol)
                return

            poll_attempts = 0
            max_poll_attempts = 20
            order_filled = False
            final_order_status = None

            while poll_attempts < max_poll_attempts:
                await asyncio.sleep(0.5)
                status_check_params = {
                    "symbol": symbol,
                    "orderId": order_id,
                    "timestamp": int(time.time() * 1000)
                }
                signed_status_check = sign_params(status_check_params)
                status_url = BASE_URL + "/fapi/v1/order"
                
                async with self.session.get(status_url, headers=self.headers, params=signed_status_check) as resp:
                    check_data = await resp.json()
                    
                    if resp.status == 200 and check_data and check_data.get('status'):
                        current_status = check_data['status']
                        executed_qty = float(check_data.get('executedQty', '0.0'))
                        
                        if current_status == 'FILLED' or (current_status == 'PARTIALLY_FILLED' and executed_qty > 0):
                            final_order_status = check_data
                            order_filled = True
                            logging.info(f"[{symbol}] MARKET emir başarıyla dolduruldu! Statu: {current_status}, Executed Qty: {executed_qty}")
                            break
                        elif current_status in ['CANCELED', 'EXPIRED', 'REJECTED']:
                            logging.error(f"[{symbol}] MARKET emir {current_status} oldu, doldurulamadı. Detay: {check_data}")
                            break
                        else:
                            logging.info(f"[{symbol}] MARKET emir hala bekliyor... Mevcut durum: {current_status}, Executed Qty: {executed_qty}")
                    else:
                        logging.warning(f"[{symbol}] Emir durumu kontrol edilemedi: {check_data}")
                
                poll_attempts += 1

            if not order_filled:
                logging.error(f"[{symbol}] MARKET emir zaman aşımına uğradı veya TAMAMEN DOLMADI. Son durum: {final_order_status}")
                await self.send_smart_telegram_message(f"⚠️ [BINANCE] {symbol} MARKET emri zaman aşımına uğradı veya dolmadı. Lütfen manuel kontrol edin. Emir ID: {order_id}", msg_type='ERROR', symbol=symbol)
                return
            
            actual_entry_price = float(final_order_status.get('avgPrice', current_price))
            actual_filled_quantity = float(final_order_status.get('executedQty', final_quantity_to_send))

            logging.info(f"[{symbol}] Pozisyon başarıyla açıldı ve kaydedildi: Entry={actual_entry_price:.{s_info.price_precision}f}, Quantity={actual_filled_quantity:.{s_info.quantity_precision}f}")
            
            self.active_positions[symbol] = {
                'entry': actual_entry_price,
                'quantity': actual_filled_quantity, # Bu, mevcut (kalan) miktar olacak
                'original_total_quantity': actual_filled_quantity, # <<--- BU SATIRI EKLEYİN
                'direction': direction,
                'stop_loss_price': stop_loss,
                'tp1_price': tp1,
                'tp2_price': tp2,
                'tp3_price': tp3,
                'sl_moved_to_entry': False,
                'main_sl_order_id': None,
                'tp1_order_id': None,
                'tp2_order_id': None,
                'tp3_order_id': None,
                'tp1_status': 'PENDING',
                'tp2_status': 'PENDING',
                'tp3_status': 'PENDING'
            }

            # SL/TP emirlerini yerleştir
            main_sl_order_data = await self.place_only_sl_order(symbol, close_side, actual_filled_quantity, stop_loss, "MAIN_SL")
            if main_sl_order_data and 'orderId' in main_sl_order_data:
                self.active_positions[symbol]['main_sl_order_id'] = str(main_sl_order_data['orderId'])

            tp1_order_data = await self.place_tp_order(symbol, close_side, actual_filled_quantity * 0.3, tp1, "TP1")
            if tp1_order_data and 'orderId' in tp1_order_data:
                self.active_positions[symbol]['tp1_order_id'] = str(tp1_order_data['orderId'])

            tp2_order_data = await self.place_tp_order(symbol, close_side, actual_filled_quantity * 0.3, tp2, "TP2")
            if tp2_order_data and 'orderId' in tp2_order_data:
                self.active_positions[symbol]['tp2_order_id'] = str(tp2_order_data['orderId'])

            # TP3 için kalan miktar (orijinalin %40'ı) veya pozisyonu kapatma emri
            # Miktarı yine de gönderiyoruz, Binance closePosition=true olduğunda bunu nasıl ele alacağına karar verir.
            # Genellikle kalan tüm pozisyonu kapatır.
            tp3_quantity_nominal = actual_filled_quantity * 0.4 # Bu sadece bir referans, asıl işi closePosition yapacak
            tp3_order_data = await self.place_tp_order(symbol, close_side, tp3_quantity_nominal, tp3, "TP3", close_entire_position=True) # Değişiklik burada
            if tp3_order_data and 'orderId' in tp3_order_data:
                self.active_positions[symbol]['tp3_order_id'] = str(tp3_order_data['orderId'])
            
            message = (
                f"🔻 Quantum AI Trader 🔻\n"
                f"• 🔔 Sembol: {symbol}\n"
                f"• 📉 Trend: {direction}\n"
                f"• 💲 Marjin: {required_margin:.2f} USDT\n"
                f"• 📈 Kaldırac: {leverage}x\n"
                "------------------\n"
                f"• 🟢 ENTRY: {self.format_price(symbol, actual_entry_price)}\n"
                f"• 🚫 Stop Loss: {stop_loss_fmt_val}\n"
                f"• 💸 TP1: {tp1_fmt_val}\n"
                f"• 👑 TP2: {tp2_fmt_val}\n"
                f"• 💎 TP3: {tp3_fmt_val}\n"
                f"• 💰 Bakiye: {balance:.2f}\n"
                f"\n• ⏰ Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n"
            )
            await self.send_telegram_message(message)

        except Exception as e:
            logging.error(f"Execute Trade Hatası: {e}", exc_info=True)

    async def process_symbol(self, symbol):
        logging.info(f"[{symbol}] --- Yeni analiz başlıyor ---")
        try:
            if symbol not in self.symbol_info:
                logging.warning(f"[{symbol}] Sembol bilgisi bulunamadı, analiz atlanıyor.")
                return

            df = await self.fetch_data_multi(symbol, interval='3m', total_limit=1500)
            min_raw_candles_needed = max(14, 30) + N_FUTURE_CANDLES
            if df.empty or len(df) < min_raw_candles_needed:
                logging.warning(f"[{symbol}] Yeterli ham veri çekilemedi veya DataFrame boş! (Mevcut: {len(df)} mum, Minimum: {min_raw_candles_needed} mum gerekli)")
                return
            
            trend_up, trend_down, df_processed_ta = teknik_analiz(df)
            df = df_processed_ta

            target_profit_threshold = 0.007

            df['long_target'] = (df['close'].shift(-N_FUTURE_CANDLES) > df['close'] * (1 + target_profit_threshold)).astype(int)
            df['short_target'] = (df['close'].shift(-N_FUTURE_CANDLES) < df['close'] * (1 - target_profit_threshold)).astype(int)
            df['target'] = np.where(df['long_target'] == 1, 1, np.where(df['short_target'] == 1, 0, -1))

            df = df.dropna(subset=self.required_features + ['target'])
            df_train_predict = df[df['target'] != -1].copy()

            if df_train_predict.empty or len(df_train_predict) < 20: 
                logging.warning(f"[{symbol}] Eğitim veya tahmin için yeterli feature/target verisi yok! (Mevcut: {len(df_train_predict)})")
                return
            
            features = df_train_predict[self.required_features]
            targets = df_train_predict['target']

            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()
                self.scalers[symbol].fit(features.values)
            
            X_last = pd.DataFrame([features.iloc[-1]], columns=features.columns)
            X_scaled_last = self.scalers[symbol].transform(X_last.values)

            if symbol not in self.models['xgb']:
                logging.info(f"[{symbol}] Modeller kontrol ediliyor/yükleniyor...")
                models_loaded = await self.load_models(symbol)
                
                if not models_loaded:
                    logging.info(f"[{symbol}] Kaydedilmiş modeller bulunamadı veya yüklenemedi. Modeller ilk defa eğitiliyor...")
                    self.models['xgb'][symbol] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    self.models['lgbm'][symbol] = LGBMClassifier()
                    self.models['gbt'][symbol] = GradientBoostingClassifier()
                    
                    self.models['lstm'][symbol] = Sequential([
                        LSTM(16, input_shape=(1, X_scaled_last.shape[1]), activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    self.models['lstm'][symbol].compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
                    
                    try:
                        self.models['xgb'][symbol].fit(self.scalers[symbol].transform(features.values), targets.values)
                        self.models['lgbm'][symbol].fit(self.scalers[symbol].transform(features.values), targets.values)
                        self.models['gbt'][symbol].fit(self.scalers[symbol].transform(features.values), targets.values)
                        
                        X_lstm_train = self.scalers[symbol].transform(features.values).reshape(-1, 1, features.shape[1])
                        self.models['lstm'][symbol].fit(X_lstm_train, targets, epochs=20, batch_size=32, verbose=0)
                        logging.info(f"[{symbol}] Modeller başarıyla eğitildi.")
                        await self.save_models(symbol)
                    except Exception as e:
                        logging.error(f"[{symbol}] Modellerin ilk eğitimi sırasında hata: {e}")
                        return
                else:
                    logging.info(f"[{symbol}] Modeller başarıyla yüklendi.")

            X_lstm_predict_tensor = tf.constant(X_scaled_last.reshape(1, 1, -1), dtype=tf.float32)

            predictions = {}
            try:
                predictions['xgb'] = self.models['xgb'][symbol].predict_proba(X_scaled_last)[0][1]
                predictions['lgbm'] = self.models['lgbm'][symbol].predict_proba(X_scaled_last)[0][1]
                predictions['gbt'] = self.models['gbt'][symbol].predict_proba(X_scaled_last)[0][1]
                predictions['lstm'] = float(self.models['lstm'][symbol].predict(X_lstm_predict_tensor)[0][0])
            except Exception as e:
                logging.error(f"[{symbol}] Model tahminleri sırasında hata: {e}", exc_info=True)
                return

            avg_prediction = sum(predictions.values()) / len(predictions)
            current_atr_series = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            current_atr = current_atr_series.iloc[-1] if not current_atr_series.empty and not pd.isna(current_atr_series.iloc[-1]) else 0
            current_price = df['close'].iloc[-1]

            now = datetime.now(IST_TIMEZONE)
            logging.info(f"[{symbol}] Model prediction: {avg_prediction:.3f}, trend_up: {trend_up}, trend_down: {trend_down}, ATR: {current_atr:.4f}, Price: {current_price:.2f}")
            
            if symbol in self.cooldowns and now < self.cooldowns[symbol]:
                logging.info(f"[{symbol}] Cooldown sürüyor ({self.cooldowns[symbol].strftime('%H:%M')}'e kadar), yeni trade açılmayacak.")
                return

            pos = await self.get_open_position(symbol)
            pozisyon_acik_binance = pos and abs(pos['amt']) > 0
            pozisyon_acik_bot = symbol in self.active_positions

            logging.info(f"[{symbol}] Binance'de pozisyon açık mı? {pozisyon_acik_binance} (Miktar: {pos['amt'] if pos else 'N/A'})")
            logging.info(f"[{symbol}] Bot tarafında pozisyon açık mı? {pozisyon_acik_bot}")

            if pozisyon_acik_bot and not pozisyon_acik_binance:
                logging.warning(f"[{symbol}] Botta açık pozisyon görünüyor ancak Binance'de kapalı. Pozisyon kapatma işlemini tetikliyorum.")
                await self.cancel_open_orders(symbol)
                await self.handle_position_close(symbol)
                pozisyon_acik_bot = False

            elif not pozisyon_acik_bot and pozisyon_acik_binance:
                logging.warning(f"[{symbol}] Binance'de açık pozisyon var ancak bot kaydında yok. Bu pozisyonu bot takip etmiyor. Manuel kapatma veya inceleme gerekebilir.")
                return

            if pozisyon_acik_bot:
                logging.info(f"[{symbol}] Zaten açık pozisyon var (Bot tarafından yönetiliyor), yeni trade açılmayacak.")
                return

            market_direction = await self.get_market_direction() # Genel piyasa yönünü çek
            logging.info(f"[{symbol}] Piyasa Yönü Filtresi: {market_direction}")
            # DÜZELTME BİTİŞİ

            if avg_prediction > 0.52 and trend_up:
                # DÜZELTME: LONG sinyali için piyasa yönü 'LONG' veya 'NEUTRAL' olmalı
                if market_direction == 'LONG' or market_direction == 'NEUTRAL':
                    logging.info(f"[{symbol}] LONG sinyali: {avg_prediction:.3f} - Piyasa Yönü Uygun: {market_direction}")
                    await self.execute_trade(symbol, 'LONG', current_price, avg_prediction, current_atr)
                else:
                    logging.debug(f"[{symbol}] LONG sinyali var ancak piyasa yönü uygun değil ({market_direction}). İşlem açılmayacak.")
            elif avg_prediction < 0.48 and trend_down:
                # DÜZELTME: SHORT sinyali için piyasa yönü 'SHORT' veya 'NEUTRAL' olmalı
                if market_direction == 'SHORT' or market_direction == 'NEUTRAL':
                    logging.info(f"[{symbol}] SHORT sinyali: {avg_prediction:.3f} - Piyasa Yönü Uygun: {market_direction}")
                    await self.execute_trade(symbol, 'SHORT', current_price, avg_prediction, current_atr)
                else:
                    logging.debug(f"[{symbol}] SHORT sinyali var ancak piyasa yönü uygun değil ({market_direction}). İşlem açılmayacak.")
            else:
                logging.debug(f"[{symbol}] için sinyal yok. Tahmin: {avg_prediction:.3f}")

        except Exception as e:
            logging.error(f"[{symbol}] analiz hatası: {str(e)}", exc_info=True)

    async def update_models(self):
        logging.info("Tüm ML modelleri yeniden eğitiliyor...")
        for symbol in self.symbols_to_trade:
            try:
                if symbol not in self.symbol_info:
                    logging.warning(f"[{symbol}] Sembol bilgisi bulunamadı, model eğitimi atlanıyor.")
                    continue

                df = await self.fetch_data_multi(symbol, interval='3m', total_limit=1500)
                min_raw_candles_needed = max(14, 30) + N_FUTURE_CANDLES
                if df.empty or len(df) < min_raw_candles_needed:
                    logging.warning(f"[{symbol}] Model eğitimi için yeterli veri yok, atlanıyor. (Mevcut: {len(df)} mum)")
                    continue
                
                trend_up, trend_down, df_processed_ta = teknik_analiz(df)
                df_processed = df_processed_ta

                target_profit_threshold = 0.007

                df_processed['long_target'] = (df_processed['close'].shift(-N_FUTURE_CANDLES) > df_processed['close'] * (1 + target_profit_threshold)).astype(int)
                df_processed['short_target'] = (df_processed['close'].shift(-N_FUTURE_CANDLES) < df_processed['close'] * (1 - target_profit_threshold)).astype(int)
                df_processed['target'] = np.where(df_processed['long_target'] == 1, 1, np.where(df_processed['short_target'] == 1, 0, -1))

                df_train = df_processed.dropna(subset=self.required_features + ['target'])
                df_train = df_train[df_train['target'] != -1].copy()

                if df_train.empty or len(df_train) < 50:
                    logging.warning(f"[{symbol}] Eğitim için feature/target verisi çok az. Atlanıyor. (Mevcut: {len(df_train)})")
                    continue

                features_train = df_train[self.required_features]
                targets_train = df_train['target']

                scaler = StandardScaler()
                scaler.fit(features_train.values)
                self.scalers[symbol] = scaler
                X_scaled_train = scaler.transform(features_train.values)

                logging.info(f"[{symbol}] Modeller yeniden eğitiliyor...")
                self.models['xgb'][symbol] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                self.models['lgbm'][symbol] = LGBMClassifier()
                self.models['gbt'][symbol] = GradientBoostingClassifier()
                
                self.models['xgb'][symbol].fit(X_scaled_train, targets_train.values)
                self.models['lgbm'][symbol].fit(X_scaled_train, targets_train.values)
                self.models['gbt'][symbol].fit(X_scaled_train, targets_train.values)

                self.models['lstm'][symbol] = Sequential([
                    LSTM(16, input_shape=(1, X_scaled_train.shape[1]), activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                self.models['lstm'][symbol].compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
                X_lstm_train = X_scaled_train.reshape(-1, 1, X_scaled_train.shape[1])
                self.models['lstm'][symbol].fit(X_lstm_train, targets_train, epochs=20, batch_size=32, verbose=0)
                
                logging.info(f"[{symbol}] Modeller başarıyla güncellendi.")
                await self.save_models(symbol) # Modelleri kaydet

            except Exception as e:
                logging.error(f"[{symbol}] Model güncelleme hatası: {str(e)}", exc_info=True)
        self.last_model_update = datetime.now(IST_TIMEZONE)
        logging.info("Tüm ML modelleri güncelleme tamamlandı.")

    async def run_analysis_loop(self):
        logging.info("Analiz döngüsü başlatılıyor...")
        MAX_ACTIVE_TRADES = 8
        TOP_VOLATILE_COUNT = 20
        MIN_MARGIN_THRESHOLD = 1.0 

        await self.update_models() 
        self.last_model_update = datetime.now(IST_TIMEZONE)

        self.report_task = asyncio.create_task(self._report_scheduler())

        while True:
            top_symbols = [] 
            all_vols = []
            try:
                # 1. Periyodik Bakiye Kontrolü (Acil Durum Kapanışı)
                balance = await self.get_binance_balance()
                
                if balance < MIN_MARGIN_THRESHOLD:
                    logging.critical(f"!!! KRİTİK UYARI: Bakiye eşiğin altına düştü! Mevcut Bakiye: {balance:.2f} USDT. Tüm açık pozisyonlar kapatılıyor !!!")
                    await self.send_telegram_message(f"🚨 <b>KRİTİK UYARI:</b> Bakiye eşiğin altına düştü ({balance:.2f} USDT). Tüm açık pozisyonlar kapatılıyor.")
                    
                    open_positions_to_close = await self.get_all_open_positions()
                    if open_positions_to_close:
                        logging.info(f"Açık pozisyonlar kapatılıyor: {open_positions_to_close}")
                        for symbol_to_close in open_positions_to_close:
                            try:
                                if symbol_to_close in self.active_positions:
                                    pos_qty = abs(self.active_positions[symbol_to_close]['quantity'])
                                    pos_direction = self.active_positions[symbol_to_close]['direction']
                                    close_side = 'SELL' if pos_direction == 'LONG' else 'BUY'
                                else:
                                    pos_info_binance = await self.get_open_position(symbol_to_close)
                                    if pos_info_binance and abs(pos_info_binance['amt']) > 0:
                                        pos_qty = abs(pos_info_binance['amt'])
                                        close_side = 'SELL' if pos_info_binance['amt'] > 0 else 'BUY'
                                    else:
                                        pos_qty = 0
                                        logging.warning(f"[{symbol_to_close}] Manuel kapanış için pozisyon miktarı bulunamadı, atlanıyor.")
                                        continue
                                if pos_qty > 0:
                                    await self.send_binance_market_order(
                                        symbol=symbol_to_close,
                                        side=close_side,
                                        quantity=pos_qty,
                                        leverage=SYMBOL_LEVERAGE.get(symbol_to_close, 20)
                                    )
                                    logging.info(f"[{symbol_to_close}] Marjin eşiği nedeniyle pozisyon kapatma emri gönderildi.")
                                    
                                    await asyncio.sleep(2) 
                                    
                                    if symbol_to_close in self.active_positions:
                                         await self.handle_position_close(symbol_to_close)
                                    else:
                                        logging.info(f"[{symbol_to_close}] Botun takip etmediği pozisyon marjin eşiği nedeniyle kapatıldı.")
                                else:
                                    logging.info(f"[{symbol_to_close}] Kapatılacak aktif pozisyon bulunamadı veya miktarı sıfır.")

                            except Exception as close_e:
                                logging.error(f"[{symbol_to_close}] Marjin eşiği nedeniyle pozisyon kapatılırken hata oluştu: {close_e}", exc_info=True)
                                await self.send_telegram_message(f"❌ <b>{symbol_to_close}</b> pozisyon kapatılırken hata: {close_e}")
                    logging.info("Bakiye eşiği nedeniyle tüm pozisyonlar kapatıldı. Yeni analiz için bekleniyor...")
                    await asyncio.sleep(60)
                    continue

                # 2. Model Güncelleme Kontrolü
                if (datetime.now(IST_TIMEZONE) - self.last_model_update) > MODEL_UPDATE_INTERVAL:
                    logging.info("Model güncelleme zamanı geldi.")
                    await self.update_models()
                    self.last_model_update = datetime.now(IST_TIMEZONE)

                # 3. VOLATİL COİNLERİ BUL (Buraya taşındı!)
                top_symbols, all_vols = await self.find_most_volatile_symbols(interval='3m', lookback=120, top_n=TOP_VOLATILE_COUNT)
                
                if not top_symbols:
                    logging.info("find_most_volatile_symbols boş liste döndürdü. Uygun volatil coin bulunamadı.")
                    await asyncio.sleep(60) # Eğer uygun volatil coin yoksa bekle
                    continue

                msg = "En volatil coinler:\n" + "\n".join([f"{s}: {v*100:.2f}%" for s,v in all_vols[:TOP_VOLATILE_COUNT]])
                logging.info(msg)


                # 4. Pozisyon Senkronizasyonu ve Yönetimi (Kapanan Pozisyonları Temizle)
                open_positions_binance = await self.get_all_open_positions()
                open_positions_bot_keys = list(self.active_positions.keys())

                for symbol_in_bot in open_positions_bot_keys:
                    if symbol_in_bot not in open_positions_binance:
                        logging.warning(f"[{symbol_in_bot}] Botta açık pozisyon görünüyor ancak Binance'de kapalı. Pozisyon kapatma işlemini tetikliyorum.")
                        await self.cancel_open_orders(symbol_in_bot)
                        await self.handle_position_close(symbol_in_bot)
                
                for symbol_in_binance in open_positions_binance:
                    if symbol_in_binance not in open_positions_bot_keys:
                        logging.warning(f"[{symbol_in_binance}] Binance'de açık pozisyon var ancak bot kaydında yok. Bu pozisyonu bot takip etmiyor. Manuel kapatma veya inceleme gerekebilir.")
                
                current_active_symbols = list(self.active_positions.keys())
                logging.info(f"Bot tarafından yönetilen açık pozisyonlar: {current_active_symbols} (max {MAX_ACTIVE_TRADES})")

                # 5. Mevcut Açık Pozisyonların Durumunu Kontrol Etme (SL/TP Yönetimi)
                analyzed_open_positions_count = 0 # Yeni değişken: Sadece açık pozisyonları saymak için
                for symbol in current_active_symbols: # Bu TEK ve DOĞRU döngüdür.
                    logging.info(f"[{symbol}] Zaten açık pozisyonda (bot yönetiyor). SL/TP durumları kontrol ediliyor.")
                    pos_details = self.active_positions[symbol]
                    
                    close_side = 'SELL' if pos_details['direction'] == 'LONG' else 'BUY'
                    
                    binance_pos_info = await self.get_open_position(symbol)
                    if binance_pos_info and abs(binance_pos_info['amt']) > 0:
                        current_binance_quantity = abs(binance_pos_info['amt'])
                        
                        # Eğer botun kaydındaki miktar farklıysa güncelle ve TP durumlarını tahmin et
                        if current_binance_quantity != pos_details['quantity']:
                            logging.info(f"[{symbol}] Pozisyon miktarı Binance'de değişmiş: {pos_details['quantity']:.{self.symbol_info[symbol].quantity_precision}f} -> {current_binance_quantity:.{self.symbol_info[symbol].quantity_precision}f}. Bot kaydı ve TP durumları güncelleniyor.")
                            
                            original_qty_at_entry = pos_details['original_total_quantity']
                            
                            # TP yüzdeleri
                            tp1_qty_share = 0.3 
                            tp2_qty_share = 0.3 
                            # TP3 için kalan %40
                            
                            # DÜZELTME: TP Durumlarını Güncelle (miktar değişimine göre)
                            # TP1 gerçekleşti mi? (Orijinal miktarın %30'u veya daha azı eksikse)
                            if pos_details['tp1_status'] == 'PENDING' and original_qty_at_entry > 0 and \
                               abs(current_binance_quantity - original_qty_at_entry * (1 - tp1_qty_share)) <= original_qty_at_entry * 0.01: # %1 tolerans
                                logging.info(f"[{symbol}] Pozisyon miktarı değişiminden TP1'in vurulduğu tahmin edildi. Miktar: {current_binance_quantity:.{self.symbol_info[symbol].quantity_precision}f} / {original_qty_at_entry * (1 - tp1_qty_share):.{self.symbol_info[symbol].quantity_precision}f}")
                                pos_details['tp1_status'] = 'EXECUTED'
                                await self.send_telegram_message(f"✅ <b>{symbol}</b> pozisyonunda TP1 <b>gerçekleşti!</b>")

                            # TP2 gerçekleşti mi? (Eğer TP1 gerçekleştiyse ve miktar orijinalin %60'ı veya daha azı eksikse)
                            if pos_details['tp2_status'] == 'PENDING' and original_qty_at_entry > 0 and \
                               abs(current_binance_quantity - original_qty_at_entry * (1 - tp1_qty_share - tp2_qty_share)) <= original_qty_at_entry * 0.01:
                                logging.info(f"[{symbol}] Pozisyon miktarı değişiminden TP2'nin vurulduğu tahmin edildi.")
                                pos_details['tp2_status'] = 'EXECUTED'
                                await self.send_telegram_message(f"✅ <b>{symbol}</b> pozisyonunda TP2 <b>gerçekleşti!</b>")

                            # TP3 gerçekleşti mi? (Eğer TP1 ve TP2 gerçekleştiyse ve miktar neredeyse sıfırsa)
                            # Bu kontrol, botun pozisyonun büyük çoğunluğunun kapandığını anlamasını sağlar.
                            if pos_details['tp3_status'] == 'PENDING' and original_qty_at_entry > 0 and current_binance_quantity <= original_qty_at_entry * 0.01:
                                logging.info(f"[{symbol}] Pozisyon miktarı değişiminden TP3'ün vurulduğu tahmin edildi.")
                                pos_details['tp3_status'] = 'EXECUTED'
                                await self.send_telegram_message(f"✅ <b>{symbol}</b> pozisyonunda TP3 <b>gerçekleşti!</b>")
                            
                            pos_details['quantity'] = current_binance_quantity # Botun aktif miktarını güncelledik (KRİTİK)

                    else: # Binance'de pozisyon yoksa ama bot hala takip ediyorsa, kapatma mantığı zaten yukarıda var
                        logging.warning(f"[{symbol}] Bot takip ediyor ama Binance'de pozisyon yok gibi. Bu pozisyon handle_position_close tarafından kapatılacaktır.")
                        continue # Bu durumda SL/TP kontrol etme, bir sonraki sembole geç
                    
                    current_market_price = await self.get_futures_price(symbol)
                    s_info = self.symbol_info[symbol]
                    
                    # Stop-Even (Breakeven) Mantığı - TP1'e ulaşıldığında SL'i girişe çek
                    if pos_details.get('tp1_price') is not None and not pos_details.get('sl_moved_to_entry', False) and pos_details['tp1_status'] == 'PENDING':
                        tp1_price_val = pos_details['tp1_price']
                        entry_price_val = pos_details['entry']
                        should_move_sl = False
                        tolerance = s_info.tick_size * 0.5 

                        if pos_details['direction'] == 'LONG':
                            if current_market_price >= (tp1_price_val - tolerance): 
                                should_move_sl = True
                        elif pos_details['direction'] == 'SHORT':
                            if current_market_price <= (tp1_price_val + tolerance):
                                should_move_sl = True
                                
                        if should_move_sl:
                            logging.info(f"[{symbol}] TP1 seviyesine ulaşıldı ({tp1_price_val:.{s_info.price_precision}f}). Stop-loss girişe çekiliyor ({entry_price_val:.{s_info.price_precision}f}).")
                            await self.move_stop_to_entry(symbol, current_binance_quantity, entry_price_val)
                            pos_details['sl_moved_to_entry'] = True
                            pos_details['tp1_status'] = 'EXECUTED'
                            pos_details['tp1_order_id'] = None
                            await self.send_telegram_message(f"✅ <b>{symbol}</b> pozisyonunda TP1'e ulaşıldı. STOP-LOSS giriş fiyatına <b>çekildi ve TP1 EXECUTED olarak işaretlendi!</b>")
                            await asyncio.sleep(1)  # kısa bekleme ekledik
                            logging.info(f"DEBUG: Stop-Even sonrası {symbol} tp1_status: {self.active_positions[symbol].get('tp1_status')}")
                        # DÜZELTME: TP1 vurulduysa, durumunu 'EXECUTED' olarak işaretle (Bu zaten Stop-Even içinde vardı, teyit edildi)
                        # pos_details['tp1_status'] = 'EXECUTED' # Bu satır aslında yukarıdaki miktara göre güncelleme bloğunda daha iyi yönetilir.
                                                                    # Ama buradaki satırı da bırakabiliriz.
                    
                    # SL/TP Emirlerinin Kontrolü ve Yeniden Yerleştirme
                    # 1. Beklenen açık emir sayısını hesaplama (Bu kısım bilgilendirme amaçlı olabilir, asıl karar 'missing_x' flag'leriyle veriliyor)
                    expected_sl_tp_orders_count_for_logging = 0 # Loglama için farklı bir isim kullanalım
                    if pos_details.get('main_sl_order_id'): # Eğer bot bir SL emri bekliyorsa
                        expected_sl_tp_orders_count_for_logging += 1
                    if pos_details.get('tp1_status') == 'PENDING': expected_sl_tp_orders_count_for_logging += 1
                    if pos_details.get('tp2_status') == 'PENDING': expected_sl_tp_orders_count_for_logging += 1
                    if pos_details.get('tp3_status') == 'PENDING': expected_sl_tp_orders_count_for_logging += 1

                    # Sadece pozisyon için SL ve tüm TP fiyatları tanımlıysa devam et
                    if pos_details.get('stop_loss_price') is not None and \
                       pos_details.get('tp1_price') is not None and \
                       pos_details.get('tp2_price') is not None and \
                       pos_details.get('tp3_price') is not None:

                        current_open_sl_tp_orders = await self.get_open_sl_tp_orders_for_symbol(symbol)
                        current_tp_order_ids_on_binance = [str(order['orderId']) for order in current_open_sl_tp_orders if order.get('type') == 'TAKE_PROFIT_MARKET']
                        main_sl_is_open_on_binance = any(
                            pos_details.get('main_sl_order_id') and # Botun bir SL ID'si olmalı
                            str(order['orderId']) == str(pos_details.get('main_sl_order_id')) # ID'ler eşleşmeli
                            for order in current_open_sl_tp_orders if order.get('type') == 'STOP_MARKET'
                        )

                        missing_sl = False
                        # Eğer bot bir ana SL bekliyorsa (ID'si kayıtlıysa) VE Binance'de bu ID ile açık bir SL yoksa
                        if pos_details.get('main_sl_order_id') and not main_sl_is_open_on_binance:
                            missing_sl = True

                        missing_tp1 = False
                        if pos_details.get('tp1_status') == 'PENDING' and self.active_positions[symbol].get('tp1_status') == 'PENDING' and \
                           (not pos_details.get('tp1_order_id') or str(pos_details.get('tp1_order_id')) not in current_tp_order_ids_on_binance):
                            missing_tp1 = True

                        missing_tp2 = False
                        if pos_details.get('tp2_status') == 'PENDING' and self.active_positions[symbol].get('tp2_status') == 'PENDING' and \
                           (not pos_details.get('tp2_order_id') or str(pos_details.get('tp2_order_id')) not in current_tp_order_ids_on_binance):
                            missing_tp2 = True

                        missing_tp3 = False
                        if pos_details.get('tp3_status') == 'PENDING' and self.active_positions[symbol].get('tp3_status') == 'PENDING' and \
                           (not pos_details.get('tp3_order_id') or str(pos_details.get('tp3_order_id')) not in current_tp_order_ids_on_binance):
                            missing_tp3 = True

                        if missing_sl or missing_tp1 or missing_tp2 or missing_tp3:
                            logging.warning(f"[{symbol}] Pozisyonda eksik SL/TP emirleri bulundu. Yeniden göndermeyi deniyor. (SL: {missing_sl}, TP1: {missing_tp1}, TP2: {missing_tp2}, TP3: {missing_tp3}). "
                                            f"Binance'de açık SL/TP sayısı: {len(current_open_sl_tp_orders)}, Beklenen (log için): {expected_sl_tp_orders_count_for_logging}")

                        # original_qty, execute_trade'de 'original_total_quantity' olarak kaydedilmiş olmalı
                        original_qty = pos_details.get('original_total_quantity')
                        if original_qty is None: # Eğer bir şekilde kaydedilmemişse
                            logging.error(f"[{symbol}] 'original_total_quantity' bulunamadı! TP miktarları yanlış olabilir. Fallback: current_binance_quantity.")
                            original_qty = current_binance_quantity # Bu current_binance_quantity'nin güncel olması lazım

                        tps_re_sent_labels_log = []

                        # ÖNEMLİ: Emir göndermeden önce açık emirleri iptal et
                        await self.cancel_open_orders(symbol)
                        await asyncio.sleep(0.5) # Binance'in API kısıtlamaları nedeniyle biraz beklemek iyi olabilir

                        # 1. Ana SL'i Yeniden Gönder
                        if missing_sl:
                            sl_price_for_re_placement = pos_details['entry'] if pos_details.get('sl_moved_to_entry', False) else pos_details['stop_loss_price']
                            main_sl_order_data = await self.place_only_sl_order(symbol, close_side, current_binance_quantity, sl_price_for_re_placement, "MAIN_SL_RE_SEND")
                            if main_sl_order_data and 'orderId' in main_sl_order_data:
                                pos_details['main_sl_order_id'] = str(main_sl_order_data['orderId'])
                            logging.info(f"[{symbol}] Ana SL emri yeniden gönderildi.")
                        # Eğer SL gönderilemezse (örn: bakiye yetersiz, fiyat çok yakın vb.), main_sl_order_id güncellenmez ve bir sonraki döngüde tekrar denenir.

                        # 2. Eksik TP Emirlerini Yeniden Gönder
                        # TP1
                        if missing_tp1:
                            qty_tp1 = original_qty * 0.3 # R-çarpanlarınıza göre (1.5R, 1.0R, 1.5R) miktarlar: %30, %30, %40
                            tp_order_data = await self.place_tp_order(symbol, close_side, qty_tp1, pos_details['tp1_price'], "TP1")
                            if tp_order_data and 'orderId' in tp_order_data:
                                pos_details['tp1_order_id'] = str(tp_order_data['orderId'])
                                tps_re_sent_labels_log.append("TP1")
                            elif tp_order_data and tp_order_data.get('code') == -2021:
                                logging.warning(f"[{symbol}] TP1 yeniden gönderilirken 'Order would immediately trigger' hatası. Muhtemelen tetiklenmiş.")
                                pos_details['tp1_status'] = 'EXECUTED' # BU ÇALIŞMALI
                                pos_details['tp1_order_id'] = None   # BU DA ÇALIŞMALI

                        # TP2
                        if missing_tp2:
                            qty_tp2 = original_qty * 0.3
                            tp_order_data = await self.place_tp_order(symbol, close_side, qty_tp2, pos_details['tp2_price'], "TP2")
                            if tp_order_data and 'orderId' in tp_order_data:
                                pos_details['tp2_order_id'] = str(tp_order_data['orderId'])
                                tps_re_sent_labels_log.append("TP2")
                            elif tp_order_data and tp_order_data.get('code') == -2021:
                                logging.warning(f"[{symbol}] TP2 yeniden gönderilirken 'Order would immediately trigger' hatası. Muhtemelen tetiklenmiş.")
                                pos_details['tp2_status'] = 'EXECUTED' # DİKKAT: Bu satırın girintisi ÖNCEKİ KODDA YANLIŞTI, şimdi doğru olmalı.
                                pos_details['tp2_order_id'] = None   # DİKKAT: Bu satırın girintisi ÖNCEKİ KODDA YANLIŞTI, şimdi doğru olmalı.

                        # TP3
                        if missing_tp3:
                            qty_tp3_nominal = original_qty * 0.4
                            tp_order_data = await self.place_tp_order(symbol, close_side, qty_tp3_nominal, pos_details['tp3_price'], "TP3", close_entire_position=True)
                            if tp_order_data and 'orderId' in tp_order_data:
                                pos_details['tp3_order_id'] = str(tp_order_data['orderId'])
                                tps_re_sent_labels_log.append("TP3 (Poz. Kapat)")
                            elif tp_order_data and tp_order_data.get('code') == -2021:
                                logging.warning(f"[{symbol}] TP3 yeniden gönderilirken 'Order would immediately trigger' hatası. Muhtemelen tetiklenmiş.")
                                pos_details['tp3_status'] = 'EXECUTED'
                                pos_details['tp3_order_id'] = None

                        if tps_re_sent_labels_log or missing_sl:
                            await self.send_telegram_message(f"🔄 <b>{symbol}</b> pozisyonunda eksik SL/TP emirleri <b>yeniden yerleştirildi.</b> (Yeniden gönderilen TP'ler: {', '.join(tps_re_sent_labels_log) if tps_re_sent_labels_log else 'Yok'}, SL Yeniden Gönderildi: {'Evet' if missing_sl else 'Hayır'})")

                    # === BU KISIM TAMAMEN KALDIRILMALI ===
                    # if tps_to_resend_details:
                    #    logging.info(f"[{symbol}] Yeniden gönderilecek TP'ler: {[tp['label'] for tp in tps_to_resend_details]}")
                    #    # ... (önceki koddan kalan ve artık gereksiz olan uzun bir blok) ...
                    # await self.send_telegram_message(f"🔄 <b>{symbol}</b> pozisyonunda eksik SL/TP emirleri <b>yeniden yerleştirildi.</b>")
                    # === BU KISIM TAMAMEN KALDIRILMALI ===

                # DÜZELTME BİTİŞİ: Mevcut Açık Pozisyonların Durumunu Kontrol Etme (YENİDEN DÜZENLENDİ)

                # 6. Yeni pozisyon açma döngüsü
                analyzed_count = 0 
                for symbol in top_symbols:
                    
                    now = datetime.now(IST_TIMEZONE)
                    if symbol in self.cooldowns and now < self.cooldowns[symbol]:
                        logging.info(f"[{symbol}] Cooldown sürüyor ({self.cooldowns[symbol].strftime('%H:%M')}'e kadar), yeni trade açılmayacak.")
                        continue

                    if symbol not in self.symbol_info:
                        logging.warning(f"[{symbol}] Sembol bilgisi bulunamadı, analiz atlanıyor.")
                        continue

                    if symbol in current_active_symbols:
                        logging.info(f"[{symbol}] Zaten açık pozisyonda (bot yönetiyor), yeni işlem açılmayacak.")
                        continue

                    if len(current_active_symbols) + analyzed_count >= MAX_ACTIVE_TRADES:
                        logging.info(f"Maksimum aktif işlem limiti ({MAX_ACTIVE_TRADES}) doldu, yeni işlem açılmayacak.")
                        break
                    
                    logging.info(f"[DEBUG] {symbol} analiz başlıyor (volatiliteye göre seçildi)...")
                    await self.process_symbol(symbol) 
                    logging.info(f"[DEBUG] {symbol} analiz tamamlandı.")
                    
                    if symbol in self.active_positions: 
                        analyzed_count += 1
                        
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                logging.info("Analiz döngüsü durduruldu.")
                if self.report_task:
                    self.report_task.cancel()
                break
            except Exception as e:
                if isinstance(e, KeyError):
                    logging.error(f"Analiz döngüsünde KeyError oluştu: {e}. Sembol bilgisi hatalı olabilir veya beklenmedik bir durum. Sembol: '{e.args[0]}'")
                else:
                    logging.error(f"Analiz döngüsünde beklenmeyen hata: {str(e)}", exc_info=True)
                await asyncio.sleep(5)

    async def _report_scheduler(self):
        """Performans raporunu düzenli aralıklarla gönderir."""
        while True:
            await asyncio.sleep(6 * 3600)
            try:
                logging.info("Performans raporu gönderiliyor...")
                await self.send_performance_report()
            except Exception as e:
                logging.error(f"Rapor gönderme hatası: {str(e)}")

async def main():
    load_dotenv()
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    CHAT_ID = os.getenv('CHAT_ID')
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logging.error("Binance API anahtarları eksik. Lütfen .env dosyasını kontrol edin.")
        return
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("Telegram API bilgileri eksik. Lütfen .env dosyasını kontrol edin.")
        return

    symbols = SYMBOLS
    async with QuantumTrader(symbols, TELEGRAM_TOKEN, CHAT_ID) as trader:
        await trader.run_analysis_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot güvenli şekilde durduruldu.")
    except Exception as e:
        logging.critical(f"Bot çalışırken beklenmeyen bir hata oluştu: {e}", exc_info=True)