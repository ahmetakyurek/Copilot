# -*- coding: utf-8 -*-
# AI-Powered QuantumTrade Bot v4.2 (Düzeltilmiş ve Test Edilmiş Sürüm)

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import xgboost as xgb  # Changed this line
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import ta
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ===== KONFİGÜRASYON =====
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'BNBUSDT', 'XRPUSDT']
INITIAL_PORTFOLIO = 1000  # Başlangıç bakiyesi
MAX_RISK_PER_TRADE = 0.05  # Her trade için maksimum risk %5'e çıkarıldı
MIN_CONFIDENCE = 0.65
SYMBOL_LEVERAGE = {
    'BTCUSDT': 20, 
    'ETHUSDT': 20, 
    'SOLUSDT': 20, 
    'SUIUSDT': 20, 
    'BNBUSDT': 20, 
    'XRPUSDT': 20
}
COOLDOWN_MINUTES = 10
DEBUG_MODE = True
MMODEL_UPDATE_INTERVAL = timedelta(hours=6)
LSTM_WINDOW_SIZE = 30 # LSTM sequence pencere boyutu
IST_TIMEZONE = ZoneInfo("Europe/Istanbul")
N_FUTURE_CANDLES = 5 # Modelin kaç mum sonrasını tahmin edeceği (initialize_models'da target için)

# Binance endpoint ayarı
def get_binance_base_url():
    return "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"

BASE_URL = get_binance_base_url()

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_trade.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class VirtualPortfolio:
    def __init__(self, initial_balance: float = 1000.0, max_risk_per_trade: float = 0.05):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.trades = []  # Tüm işlemlerin geçmişi
        self.open_positions = {}  # Aktif pozisyonlar
        self.total_profit = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.start_time = datetime.now(IST_TIMEZONE)
        self.cooldowns = {}  # Cooldown süreleri
        self.used_margin = 0.0  # Kullanılan toplam marjin
        self.symbol_margins = {}  # Sembol başına kullanılan marjin
        
    def add_position(self, symbol: str, direction: str, entry_price: float, 
                    size: float, leverage: int, stop_loss: float, take_profits: list):
        """Yeni bir pozisyon ekler"""
        # Marjin hesaplama
        margin_used = (entry_price * size) / leverage
        
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'size': size,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'entry_time': datetime.now(IST_TIMEZONE),
            'initial_risk': abs(entry_price - stop_loss) * size,
            'status': 'OPEN',
            'margin': margin_used  # Marjin bilgisini pozisyona ekle
        }
        self.open_positions[symbol] = position
        self.trades.append(position)
        
        # Kullanılan marjini güncelle
        self.used_margin += margin_used
        self.symbol_margins[symbol] = margin_used
        
        # Bakiyeden marjini düşme - bu satırı kaldırın
        # self.current_balance -= margin_used
        
        return position
        
    def close_position(self, symbol: str, exit_price: float, reason: str):
        """Pozisyonu kapatır ve P/L hesaplar"""
        if symbol not in self.open_positions:
            return None
            
        position = self.open_positions[symbol]
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now(IST_TIMEZONE)
        position['duration'] = position['exit_time'] - position['entry_time']
        position['status'] = 'CLOSED'
        position['close_reason'] = reason
        
        # P/L hesapla
        if position['direction'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) * position['size']
            
        position['pnl'] = pnl
        self.total_profit += pnl
        
        # İstatistikleri güncelle
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        # Marjini serbest bırak
        margin_used = position.get('margin', (position['entry_price'] * position['size']) / position['leverage'])
        self.used_margin -= margin_used
        if symbol in self.symbol_margins:
            del self.symbol_margins[symbol]
        
        # Bakiyeyi sadece PnL ile güncelle
        self.current_balance += pnl
        
        # Pozisyonu kapat
        del self.open_positions[symbol]
        
        return position
        
    def get_portfolio_stats(self):
        """Portfolio istatistiklerini döndürür"""
        total_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        roi = ((self.current_balance - self.initial_balance) / self.initial_balance * 100) 
        available_balance = self.current_balance - self.used_margin
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'available_balance': available_balance,  # Kullanılabilir bakiye
            'used_margin': self.used_margin,  # Kullanılan toplam marjin
            'total_profit': self.total_profit,
            'roi_percent': roi,
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'open_positions': len(self.open_positions),
            'running_time': datetime.now(IST_TIMEZONE) - self.start_time
        }
        
    def can_open_position(self, required_margin: float):
        """Yeni pozisyon açılıp açılamayacağını kontrol eder"""
        # Mevcut bakiyeden kullanılan marjini çıkararak kullanılabilir bakiyeyi bul
        available_balance = self.current_balance - self.used_margin
        
        # Toplam bakiyenin belirli bir yüzdesini aşmamalı
        max_risk = self.initial_balance * self.max_risk_per_trade
        
        # Gerekli marjin, maksimum riski aşmamalı ve kullanılabilir bakiye yeterli olmalı
        return required_margin <= max_risk and required_margin <= available_balance

    def is_in_cooldown(self, symbol: str):
        """Belirli bir sembol için cooldown süresinde olup olmadığını kontrol eder"""
        if symbol in self.cooldowns:
            cooldown_end = self.cooldowns[symbol]
            if cooldown_end > datetime.now(IST_TIMEZONE):
                return True
        return False

    def set_cooldown(self, symbol: str, duration: timedelta = timedelta(minutes=COOLDOWN_MINUTES)):
        """Belirli bir sembol için cooldown süresini başlatır"""
        self.cooldowns[symbol] = datetime.now(IST_TIMEZONE) + duration

class QuantumTrader:
    def __init__(self, symbols_to_trade, telegram_token, chat_id, portfolio=10000):
        self.symbols_to_trade = symbols_to_trade
        self.models = {'xgb': {}, 'lgbm': {}, 'gbt': {}, 'lstm': {}}
        self.scalers = {}
        self.active_positions = {}
        # self.required_features listesinin tamamı burada olmalı
        self.required_features = ['SMA_10', 'EMA_20', 'RSI_14', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR_14', 'OBV', 'Stoch_k']
        self.market_phase = "INITIALIZING"
        self.portfolio = VirtualPortfolio(initial_balance=portfolio, max_risk_per_trade=MAX_RISK_PER_TRADE)
        self.telegram_token = telegram_token # YENİ: Init'te ata
        self.chat_id = chat_id           # YENİ: Init'te ata
        self.session = None              # YENİ: aiohttp session için
        self.lock = asyncio.Lock()       # Async lock (eğer daha önce yoksa)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession() # YENİ: Session'ı burada başlat
        logging.info("Aiohttp session başlatıldı.") # YENİ LOG
        await self.initialize_models()
        # initialize_models çağrısından sonra modellerin durumunu logla
        logging.debug(f"initialize_models sonrası TÜM self.models['xgb'] anahtarları: {list(self.models.get('xgb', {}).keys())}") # YENİ LOG
        logging.debug(f"initialize_models sonrası TÜM self.models['lgbm'] anahtarları: {list(self.models.get('lgbm', {}).keys())}") # YENİ LOG
        logging.debug(f"initialize_models sonrası TÜM self.models['gbt'] anahtarları: {list(self.models.get('gbt', {}).keys())}") # YENİ LOG
        logging.debug(f"initialize_models sonrası TÜM self.models['lstm'] anahtarları: {list(self.models.get('lstm', {}).keys())}") # YENİ LOG
        logging.debug(f"initialize_models sonrası TÜM self.scalers anahtarları: {list(self.scalers.keys())}") # YENİ LOG
        logging.info("Tüm semboller için model başlatma işlemi tamamlandı.") # Bu log zaten vardı ve yeri doğru.
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session: # YENİ: Session varsa kapat
            await self.session.close()
            logging.info("Aiohttp session kapatıldı.") # YENİ LOG
        logging.info("Bot oturumu kapatılıyor.")

    async def send_telegram_message(self, message: str):
        """Telegram'a mesaj gönderir."""
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
                    logging.error(f"Telegram API hatası: {error_data}")
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Telegram mesaj gönderme hatası: {str(e)}")
            return False 

    async def fetch_data(self, symbol: str, interval: str = '1h', limit: int = 1000) -> pd.DataFrame:
        try:
            # Pandas interval mapping (güncel formatlar)
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            interval_pd = interval_map.get(interval, '1h')  # Varsayılan 1h
        
            url = f'{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
            async with self.session.get(url) as response:
                data = await response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            # Numerik dönüşümler
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Zaman damgası işleme
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(IST_TIMEZONE)

            # Zaman damgası kontrolü (sadece veri varsa)
            if not df.empty:
                try:
                    # Eksik mum kontrolü
                    expected_freq = pd.Timedelta(interval_pd)
                    time_diff = df['timestamp'].diff().dropna()

                    if not time_diff.empty:
                        missing_periods = time_diff[time_diff > expected_freq * 1.1]  # %10 tolerans
                        if not missing_periods.empty:
                            logging.warning(
                                f"{symbol} {interval}'de {len(missing_periods)} eksik mum. "
                                f"En uzun boşluk: {missing_periods.max()}"
                            )

                except Exception as e:
                    logging.error(f"Zaman kontrol hatası: {str(e)}")

            return df.dropna().reset_index(drop=True) # dropna() sonrası index'i sıfırla
        
        except aiohttp.ClientError as ce:
            logging.error(f"Bağlantı hatası ({symbol}): {str(ce)}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Genel veri çekme hatası ({symbol}): {str(e)}")
            return pd.DataFrame()
                
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Tüm özellik hesaplamalarını tek seferde yap
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            logging.error("ATR için gerekli sütunlar eksik!")
            return df

        try:
            # Trend göstergeleri
            df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
                                    
            # Momentum göstergesi
            df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)

            # MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD_signal'] = macd.macd_signal() # MACD sinyal hattı
 
            # Volatilite (ATR için özel kontrol)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                atr = ta.volatility.AverageTrueRange(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=14,
                    fillna=False
                )
                df['ATR_14'] = atr.average_true_range()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()

            # Volume
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14, # %K için pencere
                smooth_window=3, # %D için pencere (genellikle %K'nın düzeltilmiş hali)
                fillna=False
            )
            df['Stoch_k'] = stoch.stoch() # %K değeri
            # df['Stoch_d'] = stoch.stoch_signal() # %D değeri (eğer istenirse)

        except Exception as e:
            logging.error(f"Özellik hesaplama hatası: {str(e)}")
            # Hata durumunda bile bazı temel sütunları koruyarak boş olmayan DF döndürmeye çalış
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"Hesaplama sonrası eksik temel sütunlar: {missing_cols}")
                return pd.DataFrame() # Temel sütunlar yoksa boş döndür

            # Hata oluşursa, hesaplanan özellikleri NaN ile doldurmak yerine mevcut haliyle bırakabiliriz
            # ya da sadece hatalı olanları NaN yapabiliriz. Şimdilik mevcut haliyle bırakıyoruz.
            # return pd.DataFrame() # Eski: Hata durumunda boş DataFrame döndür
        
        # Önemli: Tüm gerekli özellikler hesaplandıktan sonra NaN içeren satırları kaldır
        # Bu, özellikle indikatörlerin ısınma periyotları nedeniyle önemlidir.
        return df.dropna(subset=self.required_features).reset_index(drop=True)

    async def initialize_models(self):
        logging.info("Modeller başlatılıyor...")
        for symbol in self.symbols_to_trade:
            try:
                logging.debug(f"[{symbol}] için initialize_models içinde veri çekiliyor...")
                df = await self.fetch_data(symbol)
                if df.empty:
                    logging.error(f"{symbol} için veri çekilemedi!")
                    continue

                df = self.calculate_features(df)
                
                # Prepare data for training
                X = df[self.required_features].values
                y = (df['close'].shift(-N_FUTURE_CANDLES) > df['close']).astype(int).values[:-N_FUTURE_CANDLES]
                
                # Remove last N_FUTURE_CANDLES rows from X to match y length
                X = X[:-N_FUTURE_CANDLES]
                
                # Train/test split (use most recent data for testing)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[symbol] = scaler
                
                # Train XGBoost
                logging.info(f"{symbol} için XGBoost modeli eğitiliyor...")
                self.models['xgb'][symbol] = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                ).fit(X_train_scaled, y_train)
                
                # Train LightGBM
                logging.info(f"{symbol} için LightGBM modeli eğitiliyor...")
                self.models['lgbm'][symbol] = LGBMClassifier(random_state=42).fit(X_train_scaled, y_train)
                
                # Train Gradient Boosting
                logging.info(f"{symbol} için Gradient Boosting modeli eğitiliyor...")
                self.models['gbt'][symbol] = GradientBoostingClassifier(random_state=42).fit(X_train_scaled, y_train)
                
                # Train LSTM
                logging.info(f"{symbol} için LSTM modeli eğitiliyor...")
                X_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                
                lstm_model = Sequential([
                    LSTM(50, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
                    LSTM(50),
                    Dense(1, activation='sigmoid')
                ])
                
                lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])
                                 
                lstm_model.fit(X_lstm, y_train, epochs=10, batch_size=32, verbose=0)
                self.models['lstm'][symbol] = lstm_model
                
                logging.info(f"{symbol} için tüm modeller başarıyla eğitildi.")
                
            except Exception as e:
                logging.error(f"Model eğitim hatası ({symbol}): {str(e)}")
                continue
            continue # Bir sonraki sembole geç
         # initialize_models metodunun sonundaki bu log __aenter__ içinde zaten var.
         # logging.info("Tüm semboller için model başlatma işlemi tamamlandı.")
    def build_lstm_model(self):
        model = Sequential([
            LSTM(64, input_shape=(LSTM_WINDOW_SIZE, len(self.required_features))), # Sabit kullan
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']) # Metrik ekle
        return model

    def create_sequences(self, data, window_size):
        sequences = []
        # Veri en az pencere boyutu kadar uzun olmalı
        if len(data) >= window_size:
            for i in range(len(data) - window_size + 1):
                sequences.append(data[i:i+window_size])
        return np.array(sequences) # sequences boş olsa bile numpy array döndürür
        
    async def analyze_market(self):
        last_update = datetime.now(IST_TIMEZONE) # Zaman dilimi ekle
        logging.info("Piyasa analizi başlatıldı")
        while True:
            now = datetime.now(IST_TIMEZONE) # Zaman dilimi ekle
            if (now - last_update) > MODEL_UPDATE_INTERVAL:
               logging.info("Periyodik model güncelleme zamanı.")
               await self.initialize_models() # Periyodik model yenileme
               last_update = now # Zamanı güncelle
            try:
                await self.update_market_phase()
                
                for symbol in self.symbols_to_trade: # self.symbols_to_trade kullanıldı
                    await self.process_symbol(symbol)
                await asyncio.sleep(300)
            except Exception as e:
                logging.error(f"Ana döngü hatası: {str(e)}")

    async def update_market_phase(self):
        try:
            btc_df = await self.fetch_data('BTCUSDT', '1d', 100)
            if not btc_df.empty:
                rsi = ta.momentum.rsi(btc_df['close'], 14).iloc[-1]
                self.market_phase = 'bullish' if rsi > 60 else 'bearish' if rsi < 40 else 'neutral'
        except Exception as e:
            logging.error(f"Piyasa fazı güncelleme hatası: {str(e)}")

    async def process_symbol(self, symbol):
        """Belirli bir sembol için analiz ve işlem gerçekleştirir."""
        try:
            # Cooldown kontrolü
            if self.portfolio.is_in_cooldown(symbol):
                logging.debug(f"[{symbol}] Cooldown süresinde, işlem yapılmayacak")
                return
                
            # Aktif pozisyon kontrolü
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                # Son 1 dakikalık veriyi al
                last_candle = await self.fetch_data(symbol, interval='1m', limit=1)
                if last_candle.empty:
                    logging.warning(f"[{symbol}] için fiyat verisi alınamadı")
                    return
                    
                current_price = float(last_candle['close'].iloc[-1])
                period_high = float(last_candle['high'].iloc[-1])
                period_low = float(last_candle['low'].iloc[-1])
                
                # Stop loss veya TP3 kontrolü
                if position['direction'] == 'LONG':
                    # Stop loss kontrolü - periyottaki en düşük fiyat
                    if period_low <= position['sl']:
                        logging.info(f"[{symbol}] Stop Loss tetiklendi - En Düşük: {period_low:.2f}")
                        # Virtual portfolio'yu güncelle
                        closed_position = self.portfolio.close_position(symbol, period_low, "Stop Loss")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))  # 10dk cooldown
                        
                        # Telegram bildirimi
                        stats = self.portfolio.get_portfolio_stats()
                        await self.send_telegram_message(
                            f"🔴 {symbol} LONG Pozisyon Kapatıldı\n"
                            f"• Sebep: Stop Loss\n"
                            f"• Giriş: {position['entry']:.2f}\n"
                            f"• Çıkış: {period_low:.2f}\n"
                            f"• Kar/Zarar: {closed_position['pnl']:.4f} USDT\n"  # 2 yerine 4 ondalık basamak {closed_position['pnl']:.2f} USDT\n"
                            f"• Süre: {closed_position['duration']}\n"
                            f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                            f"📊 Portfolio Durumu:\n"
                            f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                            f"• Toplam Kar/Zarar: {stats['total_profit']:.2f} USDT\n"
                            f"• ROI: {stats['roi_percent']:.2f}%\n"
                            f"• Başarı Oranı: {stats['win_rate']:.1f}%"
                        )
                        del self.active_positions[symbol]
                    # TP3 kontrolü - periyottaki en yüksek fiyat
                    elif period_high >= position['tp3']:
                        logging.info(f"[{symbol}] TP3 hedefine ulaşıldı - En Yüksek: {period_high:.2f}")
                        # Virtual portfolio'yu güncelle
                        closed_position = self.portfolio.close_position(symbol, period_high, "TP3")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))  # 10dk cooldown
                        
                        # Telegram bildirimi
                        stats = self.portfolio.get_portfolio_stats()
                        await self.send_telegram_message(
                            f"🟢 {symbol} LONG Pozisyon Kapatıldı\n"
                            f"• Sebep: TP3 Hedefi\n"
                            f"• Giriş: {position['entry']:.2f}\n"
                            f"• Çıkış: {period_high:.2f}\n"
                            f"• Kar/Zarar: {closed_position['pnl']:.4f} USDT\n"
                            f"• Süre: {closed_position['duration']}\n"
                            f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                            f"📊 Portfolio Durumu:\n"
                            f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                            f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"
                            f"• Toplam Kar/Zarar: {stats['total_profit']:.2f} USDT\n"
                            f"• ROI: {stats['roi_percent']:.2f}%\n"
                            f"• Başarı Oranı: {stats['win_rate']:.1f}%"
                        )
                        del self.active_positions[symbol]
                    else:
                        # TP1 ve TP2 bilgilendirmeleri (LONG pozisyonlar için)
                        if period_high >= position['tp1'] and not position.get('tp1_hit'):
                            position['tp1_hit'] = True
                            # Stop loss'u giriş fiyatına çek (break-even)
                            position['sl'] = position['entry']
                            stats = self.portfolio.get_portfolio_stats()
                            await self.send_telegram_message(
                                f"📈 {symbol} TP1 Hedefine Ulaşıldı\n"
                                f"• Fiyat: {period_high:.2f}\n"
                                f"• Anlık Kar: {(period_high - position['entry']) * position['size']:.4f} USDT\n"
                                f"• Marjin: {position.get('margin', 0):.2f} USDT\n"  # Marjin bilgisini ekle
                                f"• Stop Loss Break-Even'a çekildi\n"
                                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                                f"📊 Portfolio Durumu:\n"
                                f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                                f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"  # Kullanılabilir bakiye ekle
                                f"• Kullanılan Marjin: {stats['used_margin']:.2f} USDT"  # Kullanılan marjin ekle
                            )
                        if period_high >= position['tp2'] and not position.get('tp2_hit'):
                            position['tp2_hit'] = True
                            stats = self.portfolio.get_portfolio_stats()
                            await self.send_telegram_message(
                                f"📈 {symbol} TP2 Hedefine Ulaşıldı\n"
                                f"• Fiyat: {period_high:.2f}\n"
                                f"• Anlık Kar: {(period_high - position['entry']) * position['size']:.2f} USDT\n"
                                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                                f"📊 Portfolio Durumu:\n"
                                f"• Bakiye: {stats['current_balance']:.2f} USDT"
                            )
                        logging.debug(f"[{symbol}] Aktif LONG pozisyon devam ediyor")
                        return
                else:  # SHORT
                    # Stop loss kontrolü - periyottaki en yüksek fiyat
                    if period_high >= position['sl']:
                        logging.info(f"[{symbol}] Stop Loss tetiklendi - En Yüksek: {period_high:.2f}")
                        # Virtual portfolio'yu güncelle
                        closed_position = self.portfolio.close_position(symbol, period_high, "Stop Loss")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))  # 10dk cooldown
                        
                        # Telegram bildirimi
                        stats = self.portfolio.get_portfolio_stats()
                        await self.send_telegram_message(
                            f"🔴 {symbol} SHORT Pozisyon Kapatıldı\n"
                            f"• Sebep: Stop Loss\n"
                            f"• Giriş: {position['entry']:.2f}\n"
                            f"• Çıkış: {period_high:.2f}\n"
                            f"• Kar/Zarar: {closed_position['pnl']:.2f} USDT\n"
                            f"• Süre: {closed_position['duration']}\n"
                            f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                            f"📊 Portfolio Durumu:\n"
                            f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                            f"• Toplam Kar/Zarar: {stats['total_profit']:.2f} USDT\n"
                            f"• ROI: {stats['roi_percent']:.2f}%\n"
                            f"• Başarı Oranı: {stats['win_rate']:.1f}%"
                        )
                        del self.active_positions[symbol]
                    # TP3 kontrolü - periyottaki en düşük fiyat
                    elif period_low <= position['tp3']:
                        logging.info(f"[{symbol}] TP3 hedefine ulaşıldı - En Düşük: {period_low:.2f}")
                        # Virtual portfolio'yu güncelle
                        closed_position = self.portfolio.close_position(symbol, period_low, "TP3")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))  # 10dk cooldown
                        
                        # Telegram bildirimi
                        stats = self.portfolio.get_portfolio_stats()
                        await self.send_telegram_message(
                            f"🟢 {symbol} SHORT Pozisyon Kapatıldı\n"
                            f"• Sebep: TP3 Hedefi\n"
                            f"• Giriş: {position['entry']:.2f}\n"
                            f"• Çıkış: {period_low:.2f}\n"
                            f"• Kar/Zarar: {closed_position['pnl']:.4f} USDT\n"
                            f"• Süre: {closed_position['duration']}\n"
                            f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                            f"📊 Portfolio Durumu:\n"
                            f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                            f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"
                            f"• Toplam Kar/Zarar: {stats['total_profit']:.2f} USDT\n"
                            f"• ROI: {stats['roi_percent']:.2f}%\n"
                            f"• Başarı Oranı: {stats['win_rate']:.1f}%"
                        )
                        del self.active_positions[symbol]
                    else:
                        # TP1 ve TP2 bilgilendirmeleri (SHORT pozisyonlar için)
                        if period_low <= position['tp1'] and not position.get('tp1_hit'):
                            position['tp1_hit'] = True
                            # Stop loss'u giriş fiyatına çek (break-even)
                            position['sl'] = position['entry']
                            stats = self.portfolio.get_portfolio_stats()
                            await self.send_telegram_message(
                                f"📉 {symbol} TP1 Hedefine Ulaşıldı\n"
                                f"• Fiyat: {period_low:.2f}\n"
                                f"• Anlık Kar: {(position['entry'] - period_low) * position['size']:.4f} USDT\n"
                                f"• Marjin: {position.get('margin', 0):.2f} USDT\n"  # Marjin bilgisini ekle
                                f"• Stop Loss Break-Even'a çekildi\n"
                                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                                f"📊 Portfolio Durumu:\n"
                                f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                                f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"  # Kullanılabilir bakiye ekle
                                f"• Kullanılan Marjin: {stats['used_margin']:.2f} USDT"  # Kullanılan marjin ekle
                            )
                        if period_low <= position['tp2'] and not position.get('tp2_hit'):
                            position['tp2_hit'] = True
                            stats = self.portfolio.get_portfolio_stats()
                            await self.send_telegram_message(
                                f"📉 {symbol} TP2 Hedefine Ulaşıldı\n"
                                f"• Fiyat: {period_low:.2f}\n"
                                f"• Anlık Kar: {(position['entry'] - period_low) * position['size']:.2f} USDT\n"
                                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                                f"📊 Portfolio Durumu:\n"
                                f"• Bakiye: {stats['current_balance']:.2f} USDT"
                            )
                        logging.debug(f"[{symbol}] Aktif SHORT pozisyon devam ediyor")
                        return

            logging.debug(f"[{symbol}] analiz ediliyor...")
            
            # Son verileri çek
            df = await self.fetch_data(symbol, interval='15m', limit=100)
            if df.empty:
                logging.warning(f"[{symbol}] için veri alınamadı.")
                return
                
            # Özellikleri hesapla
            df = self.calculate_features(df)
            if df.empty:
                logging.warning(f"[{symbol}] için özellikler hesaplanamadı.")
                return
                
            # Son veriyi al ve ölçekle
            latest_features = df[self.required_features].iloc[-1:].values
            X_scaled = self.scalers[symbol].transform(latest_features)
            
            # Model tahminlerini al
            predictions = {
                'xgb': self.models['xgb'][symbol].predict_proba(X_scaled)[0][1],
                'lgbm': self.models['lgbm'][symbol].predict_proba(X_scaled)[0][1],
                'gbt': self.models['gbt'][symbol].predict_proba(X_scaled)[0][1],
                'lstm': float(self.models['lstm'][symbol].predict(X_scaled.reshape(1, 1, -1))[0][0])
            }
            
            # Ortalama tahmin
            avg_prediction = sum(predictions.values()) / len(predictions)
            
            # Son ATR değerini al
            current_atr = df['ATR_14'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # İşlem sinyali oluştur
            if avg_prediction > 0.55:  # Alış sinyali
                logging.info(f"[{symbol}] LONG sinyali: {avg_prediction:.3f}")
                await self.execute_trade(symbol, 'LONG', current_price, avg_prediction, current_atr)
            elif avg_prediction < 0.45:  # Satış sinyali
                logging.info(f"[{symbol}] SHORT sinyali: {avg_prediction:.3f}")
                await self.execute_trade(symbol, 'SHORT', current_price, avg_prediction, current_atr)
            else:
                logging.debug(f"[{symbol}] için sinyal yok. Tahmin: {avg_prediction:.3f}")
                
        except Exception as e:
            logging.error(f"[{symbol}] analiz hatası: {str(e)}")

    async def execute_trade(self, symbol: str, direction: str, current_price: float, confidence: float, atr: float):
        """İşlem sinyalini işler ve Telegram'a bildirim gönderir."""
        try:
            # Risk ve ATR kontrolleri
            if atr <= 0:
                logging.error(f"[{symbol}] ATR değeri geçersiz: {atr}")
                return
                
            # Stop loss ve take profit hesaplamaları (ATR tabanlı)
            sl_distance = 1.5 * atr  # Stop loss için 1.5 ATR
            sl_percent = sl_distance / current_price  # Stop loss yüzdesi
            
            if direction == 'LONG':
                stop_loss = current_price - sl_distance
                tp1 = current_price + (sl_distance * 1.0)  # Risk:Reward = 1:1
                tp2 = current_price + (sl_distance * 1.5)  # Risk:Reward = 1:1.5
                tp3 = current_price + (sl_distance * 2.0)  # Risk:Reward = 1:2
            else:  # SHORT
                stop_loss = current_price + sl_distance
                tp1 = current_price - (sl_distance * 1.0)
                tp2 = current_price - (sl_distance * 1.5)
                tp3 = current_price - (sl_distance * 2.0)
                
            # Pozisyon büyüklüğü hesaplama
            leverage = SYMBOL_LEVERAGE.get(symbol, 1)
            
            # Toplam bakiyenin belirli bir yüzdesini kullan
            max_risk = self.portfolio.initial_balance * self.portfolio.max_risk_per_trade
            
            # Hedef marjin değeri (her sembol için yaklaşık aynı marjin kullanmak için)
            target_margin = max_risk / len(SYMBOLS)  # Sembol başına eşit marjin dağıtımı
            
            # Hedef marjin değerine göre pozisyon büyüklüğünü hesapla
            position_size = (target_margin * leverage) / current_price
            
            # Minimum ve maksimum pozisyon kontrolleri - fiyata göre ayarlanmış
            if current_price < 1:  # Düşük fiyatlı kriptolar için daha yüksek limit
                min_position = 10.0
                max_position = 100.0
            elif current_price < 10:  # Orta-düşük fiyatlı kriptolar
                min_position = 1.0
                max_position = 50.0
            elif current_price < 100:  # Orta fiyatlı kriptolar
                min_position = 0.1
                max_position = 10.0
            elif current_price < 1000:  # Yüksek fiyatlı kriptolar
                min_position = 0.01
                max_position = 1.0
            else:  # Çok yüksek fiyatlı kriptolar (BTC gibi)
                min_position = 0.001
                max_position = 0.1
            
            position_size = max(min_position, min(position_size, max_position))
            position_size = round(position_size, 3)  # 3 ondalık basamak
            
            # Marjin hesaplama
            required_margin = (current_price * position_size) / leverage
            
            # Marjin kontrolü
            if not self.portfolio.can_open_position(required_margin):
                logging.warning(f"[{symbol}] Yetersiz marjin veya risk limiti aşıldı - Gerekli: {required_margin:.2f} USDT")
                return
                
            # Pozisyonu portfolio'ya ekle
            position = self.portfolio.add_position(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                size=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profits=[tp1, tp2, tp3]
            )
            
            # Aktif pozisyonlar listesine ekle
            self.active_positions[symbol] = {
                'direction': direction,
                'entry': current_price,
                'sl': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'size': position_size,
                'leverage': leverage,
                'margin': required_margin,  # Marjin bilgisini ekle
                'timestamp': datetime.now(IST_TIMEZONE)
            }
            
            # Portfolio durumunu al
            stats = self.portfolio.get_portfolio_stats()
            
            # Telegram mesajı
            message = (
                f"🔻 QuantumTrade Sinyali 🔻\n"
                f"• Sembol: {symbol}\n"
                f"• Yön: {direction}\n"
                f"• Giriş: {current_price:.2f}\n"
                f"• Boyut: {position_size}\n"
                f"• Marjin: {required_margin:.2f} USDT\n"  # Marjin bilgisini ekle
                f"• Kaldıraç: {leverage}x\n"
                f"• Stop Loss: {stop_loss:.2f}\n"
                f"• TP1: {tp1:.2f}\n"
                f"• TP2: {tp2:.2f}\n"
                f"• TP3: {tp3:.2f}\n"
                f"• Piyasa Fazı: {self.market_phase}\n"
                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                f"📊 Portfolio Durumu:\n"
                f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"  # Kullanılabilir bakiye ekle
                f"• Kullanılan Marjin: {stats['used_margin']:.2f} USDT\n"  # Kullanılan marjin ekle
                f"• Toplam Kar/Zarar: {stats['total_profit']:.2f} USDT\n"
                f"• ROI: {stats['roi_percent']:.2f}%\n"
                f"• Başarı Oranı: {stats['win_rate']:.1f}%\n"
                f"• Toplam İşlem: {stats['total_trades']}"
            )
            
            # Mesajı gönder
            success = await self.send_telegram_message(message)
            if success:
                logging.info(f"[{symbol}] işlem sinyali Telegram'a gönderildi")
            else:
                logging.error(f"[{symbol}] işlem sinyali Telegram'a gönderilemedi")
            
        except Exception as e:
            logging.error(f"execute_trade içinde hata: {str(e)}")
            logging.exception("execute_trade içinde detaylı hata:")
    
    async def run_analysis_loop(self):
        """Ana analiz döngüsü. Sürekli olarak sembolleri izler ve işlem sinyallerini değerlendirir."""
        logging.info("Analiz döngüsü başlatılıyor...")
        while True:
            try:
                for symbol in self.symbols_to_trade:
                    await self.process_symbol(symbol)
                    
                # Her sembol için analiz tamamlandıktan sonra kısa bir bekleme
                await asyncio.sleep(60)  # 1 dakika bekle
                
            except asyncio.CancelledError:
                logging.info("Analiz döngüsü durduruldu.")
                break
            except Exception as e:
                logging.error(f"Analiz döngüsünde hata: {str(e)}")
                await asyncio.sleep(5)  # Hata durumunda 5 saniye bekle

async def main():
    # .env dosyasını yükle (main fonksiyonunun başında veya global scope'ta bir kere)
    load_dotenv()
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    CHAT_ID = os.getenv('CHAT_ID')

    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("Telegram token veya chat ID ortam değişkenlerinde bulunamadı. Lütfen .env dosyanızı kontrol edin.")
        return # Token veya ID yoksa botu başlatma

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'BNBUSDT', 'XRPUSDT']
    
    # QuantumTrader'ı başlatırken token ve chat_id'yi ver
    async with QuantumTrader(symbols, TELEGRAM_TOKEN, CHAT_ID, portfolio=1000) as trader:
        await trader.run_analysis_loop()
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot güvenli şekilde durduruldu.")