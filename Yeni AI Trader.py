# -*- coding: utf-8 -*-
# AI-Powered QuantumTrade Bot v4.2 (Binance Testnet Emir Gönderen Sürüm)

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
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
import hmac
import hashlib
import urllib.parse

# ===== KONFİGÜRASYON =====
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'BNBUSDT', 'XRPUSDT']
INITIAL_PORTFOLIO = 1000
MAX_RISK_PER_TRADE = 0.05
MIN_CONFIDENCE = 0.65
SYMBOL_LEVERAGE = {s: 20 for s in SYMBOLS}
COOLDOWN_MINUTES = 10
DEBUG_MODE = True
MODEL_UPDATE_INTERVAL = timedelta(hours=6)
LSTM_WINDOW_SIZE = 30
IST_TIMEZONE = ZoneInfo("Europe/Istanbul")
N_FUTURE_CANDLES = 5

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

# ========== YARDIMCI FONKSİYON ==========
def format_quantity(symbol, quantity):
    precision_map = {
        'BTCUSDT': 3,
        'ETHUSDT': 3,
        'BNBUSDT': 2,
        'XRPUSDT': 0,
        'SOLUSDT': 2,
        'SUIUSDT': 0
    }
    step_map = {
        'BTCUSDT': 0.001,
        'ETHUSDT': 0.001,
        'BNBUSDT': 0.01,
        'XRPUSDT': 1,
        'SOLUSDT': 0.01,
        'SUIUSDT': 1
    }
    prec = precision_map.get(symbol, 3)
    step = step_map.get(symbol, 0.001)
    quantity = (int(quantity / step)) * step
    fmt_str = "{:." + str(prec) + "f}"
    return fmt_str.format(quantity)

def sign_params(params):
    # Binance parametrelerini imzalar
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

def parse_signal(signal_text):
    # Telegram sinyali gibi gelen metni parse eder
    lines = signal_text.splitlines()
    symbol = lines[1].split(":")[1].strip()
    direction = lines[2].split(":")[1].strip()
    entry = float(lines[3].split(":")[1].strip())
    quantity = float(lines[4].split(":")[1].strip())
    leverage = int(lines[5].split(":")[1].replace("x", "").strip())
    stop_loss = float(lines[6].split(":")[1].strip())
    tp1 = float(lines[7].split(":")[1].strip())
    tp2 = float(lines[8].split(":")[1].strip())
    tp3 = float(lines[9].split(":")[1].strip())
    return {
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "quantity": quantity,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profits": [tp1, tp2, tp3]
    }

class VirtualPortfolio:
    def __init__(self, initial_balance: float = 1000.0, max_risk_per_trade: float = 0.05):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.trades = []
        self.open_positions = {}
        self.total_profit = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.start_time = datetime.now(IST_TIMEZONE)
        self.cooldowns = {}
        self.used_margin = 0.0
        self.symbol_margins = {}

    def add_position(self, symbol: str, direction: str, entry_price: float,
                    size: float, leverage: int, stop_loss: float, take_profits: list):
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
            'margin': margin_used
        }
        self.open_positions[symbol] = position
        self.trades.append(position)
        self.used_margin += margin_used
        self.symbol_margins[symbol] = margin_used
        return position

    def close_position(self, symbol: str, exit_price: float, reason: str):
        if symbol not in self.open_positions:
            return None
        position = self.open_positions[symbol]
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now(IST_TIMEZONE)
        position['duration'] = position['exit_time'] - position['entry_time']
        position['status'] = 'CLOSED'
        position['close_reason'] = reason
        if position['direction'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']
        position['pnl'] = pnl
        self.total_profit += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        margin_used = position.get('margin', (position['entry_price'] * position['size']) / position['leverage'])
        self.used_margin -= margin_used
        if symbol in self.symbol_margins:
            del self.symbol_margins[symbol]
        self.current_balance += pnl
        del self.open_positions[symbol]
        return position

    def get_portfolio_stats(self):
        total_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        roi = ((self.current_balance - self.initial_balance) / self.initial_balance * 100)
        available_balance = self.current_balance - self.used_margin
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'available_balance': available_balance,
            'used_margin': self.used_margin,
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
        available_balance = self.current_balance - self.used_margin
        max_risk = self.initial_balance * self.max_risk_per_trade
        return required_margin <= max_risk and required_margin <= available_balance

    def is_in_cooldown(self, symbol: str):
        if symbol in self.cooldowns:
            cooldown_end = self.cooldowns[symbol]
            if cooldown_end > datetime.now(IST_TIMEZONE):
                return True
        return False

    def set_cooldown(self, symbol: str, duration: timedelta = timedelta(minutes=COOLDOWN_MINUTES)):
        self.cooldowns[symbol] = datetime.now(IST_TIMEZONE) + duration

class QuantumTrader:
    def __init__(self, symbols_to_trade, telegram_token, chat_id, portfolio=10000):
        self.symbols_to_trade = symbols_to_trade
        self.models = {'xgb': {}, 'lgbm': {}, 'gbt': {}, 'lstm': {}}
        self.scalers = {}
        self.active_positions = {}
        self.required_features = ['SMA_10', 'EMA_20', 'RSI_14', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR_14', 'OBV', 'Stoch_k']
        self.market_phase = "INITIALIZING"
        self.portfolio = VirtualPortfolio(initial_balance=portfolio, max_risk_per_trade=MAX_RISK_PER_TRADE)
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.session = None
        self.headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        logging.info("Aiohttp session başlatıldı.")
        await self.initialize_models()
        logging.info("Tüm semboller için model başlatma işlemi tamamlandı.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            logging.info("Aiohttp session kapatıldı.")
        logging.info("Bot oturumu kapatılıyor.")
    
    


    async def send_telegram_message(self, message: str):
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
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            interval_pd = interval_map.get(interval, '1h')
            url = f'{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
            async with self.session.get(url) as response:
                data = await response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(IST_TIMEZONE)
            if not df.empty:
                try:
                    expected_freq = pd.Timedelta(interval_pd)
                    time_diff = df['timestamp'].diff().dropna()
                    if not time_diff.empty:
                        missing_periods = time_diff[time_diff > expected_freq * 1.1]
                        if not missing_periods.empty:
                            logging.warning(
                                f"{symbol} {interval}'de {len(missing_periods)} eksik mum. "
                                f"En uzun boşluk: {missing_periods.max()}"
                            )
                except Exception as e:
                    logging.error(f"Zaman kontrol hatası: {str(e)}")
            return df.dropna().reset_index(drop=True)
        except aiohttp.ClientError as ce:
            logging.error(f"Bağlantı hatası ({symbol}): {str(ce)}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Genel veri çekme hatası ({symbol}): {str(e)}")
            return pd.DataFrame()

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            logging.error("ATR için gerekli sütunlar eksik!")
            return df
        try:
            df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)
            macd = ta.trend.MACD(df['close'])
            df['MACD_signal'] = macd.macd_signal()
            if all(col in df.columns for col in ['high', 'low', 'close']):
                atr = ta.volatility.AverageTrueRange(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=14,
                    fillna=False
                )
                df['ATR_14'] = atr.average_true_range()
            bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                smooth_window=3,
                fillna=False
            )
            df['Stoch_k'] = stoch.stoch()
        except Exception as e:
            logging.error(f"Özellik hesaplama hatası: {str(e)}")
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"Hesaplama sonrası eksik temel sütunlar: {missing_cols}")
                return pd.DataFrame()
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
                X = df[self.required_features].values
                y = (df['close'].shift(-N_FUTURE_CANDLES) > df['close']).astype(int).values[:-N_FUTURE_CANDLES]
                X = X[:-N_FUTURE_CANDLES]
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[symbol] = scaler
                logging.info(f"{symbol} için XGBoost modeli eğitiliyor...")
                self.models['xgb'][symbol] = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                ).fit(X_train_scaled, y_train)
                logging.info(f"{symbol} için LightGBM modeli eğitiliyor...")
                self.models['lgbm'][symbol] = LGBMClassifier(random_state=42).fit(X_train_scaled, y_train)
                logging.info(f"{symbol} için Gradient Boosting modeli eğitiliyor...")
                self.models['gbt'][symbol] = GradientBoostingClassifier(random_state=42).fit(X_train_scaled, y_train)
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

    # --- BURADAN SONRA BINANCE ORDER FONKSİYONU EKLENDİ ---
    async def set_leverage(self, symbol, leverage):
        lev_params = {
            "symbol": symbol,
            "leverage": leverage,
            "timestamp": int(time.time() * 1000)
        }
        signed = sign_params(lev_params)
        url = BASE_URL + "/fapi/v1/leverage"
        async with self.session.post(url, headers=self.headers, data=signed) as resp:
            data = await resp.text()
            logging.info(f"Kaldıraç ayarlandı: {symbol} - {leverage}x (Yanıt: {data})")

    sync def send_binance_order(self, symbol, side, quantity, leverage):
        await self.set_leverage(symbol, leverage)
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": format_quantity(symbol, quantity),
            "timestamp": int(time.time() * 1000)
        }
        signed = sign_params(params)
        url = BASE_URL + "/fapi/v1/order"
        async with self.session.post(url, headers=self.headers, data=signed) as resp:
            data = await resp.json()
            if resp.status == 200:
                logging.info(f"[BINANCE] Emir gönderildi: {side} {symbol} {quantity} - {data.get('orderId')}")
            else:
                logging.error(f"[BINANCE] Emir gönderilemedi: {data}")
            return data

    async def send_stop_tp_orders(self, symbol, side, quantity, stop_loss, take_profits):
        # Pozisyonu kapatacak yön
        close_side = 'SELL' if side == 'BUY' else 'BUY'
        # STOP LOSS
        sl_params = {
            "symbol": symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "quantity": format_quantity(symbol, quantity),
            "stopPrice": str(stop_loss),
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }
        sl_signed = sign_params(sl_params)
        url = BASE_URL + "/fapi/v1/order"
        async with self.session.post(url, headers=self.headers, data=sl_signed) as resp:
            data = await resp.json()
            if resp.status == 200:
                logging.info(f"STOP-LOSS emri gönderildi: {symbol} {close_side} {stop_loss}")
            else:
                logging.error(f"STOP-LOSS gönderilemedi: {data}")

        # Take Profits (Her biri ayrı gönderiliyor)
        for i, tp in enumerate(take_profits, 1):
            tp_params = {
                "symbol": symbol,
                "side": close_side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": format_quantity(symbol, quantity),  # Hepsi aynı miktar
                "stopPrice": str(tp),
                "reduceOnly": "true",
                "timestamp": int(time.time() * 1000)
            }
            tp_signed = sign_params(tp_params)
            async with self.session.post(url, headers=self.headers, data=tp_signed) as resp:
                data = await resp.json()
                if resp.status == 200:
                    logging.info(f"TP{i} emri gönderildi: {symbol} {close_side} {tp}")
                else:
                    logging.error(f"TP{i} gönderilemedi: {data}")

    async def trade_with_signal(self, signal_text):
        # Sinyali parse et
        data = parse_signal(signal_text)
        symbol = data["symbol"]
        direction = data["direction"]
        quantity = data["quantity"]
        leverage = data["leverage"]
        stop_loss = data["stop_loss"]
        take_profits = data["take_profits"]
        # LONG/SHORT'u BUY/SELL'e çevir
        side = "BUY" if direction == "LONG" else "SELL"

        # 1. Market emir gönder
        await self.send_binance_order(symbol, side, quantity, leverage)
        # 2. Stop-Loss ve Take-Profits gönder
        await self.send_stop_tp_orders(symbol, side, quantity, stop_loss, take_profits)

    # ------------------------------------------------------

    async def execute_trade(self, symbol: str, direction: str, current_price: float, confidence: float, atr: float):
        try:
            if atr <= 0 or current_price <= 0:
                logging.error(f"[{symbol}] ATR veya fiyat geçersiz: ATR={atr}, Price={current_price}")
                return

            max_atr = current_price * 0.3
            atr = min(atr, max_atr)
            sl_distance = min(1.5 * atr, current_price * 0.2)

            if direction == 'LONG':
                stop_loss = current_price - sl_distance
                if stop_loss <= 0:
                    logging.error(f"[{symbol}] stop_loss geçersiz: {stop_loss}. Trade açılmadı.")
                    return
                tp1 = current_price + (sl_distance * 1.0)
                tp2 = current_price + (sl_distance * 1.5)
                tp3 = current_price + (sl_distance * 2.0)
                order_side = "BUY"
            else:
                stop_loss = current_price + sl_distance
                tp1 = current_price - (sl_distance * 1.0)
                tp2 = current_price - (sl_distance * 1.5)
                tp3 = current_price - (sl_distance * 2.0)
                order_side = "SELL"

            leverage = SYMBOL_LEVERAGE.get(symbol, 1)
            max_risk = self.portfolio.initial_balance * self.portfolio.max_risk_per_trade
            target_margin = max_risk / len(SYMBOLS)
            position_size = (target_margin * leverage) / current_price

            if current_price < 1:
                min_position = 10.0
                max_position = 100.0
            elif current_price < 10:
                min_position = 1.0
                max_position = 50.0
            elif current_price < 100:
                min_position = 0.1
                max_position = 10.0
            elif current_price < 1000:
                min_position = 0.01
                max_position = 1.0
            else:
                min_position = 0.001
                max_position = 0.1

            position_size = max(min_position, min(position_size, max_position))
            position_size = round(position_size, 3)
            required_margin = (current_price * position_size) / leverage

            if direction == 'LONG' and stop_loss <= 0:
                logging.error(f"[{symbol}] stop_loss geçersiz: {stop_loss}. Trade açılmadı.")
                return

            if not self.portfolio.can_open_position(required_margin):
                logging.warning(f"[{symbol}] Yetersiz marjin veya risk limiti aşıldı - Gerekli: {required_margin:.2f} USDT")
                return

            # --- Binance'e gerçek emir gönder ---
            order_result = await self.send_binance_order(
                symbol=symbol,
                side=order_side,
                quantity=position_size,
                leverage=leverage,
                reduce_only=False
            )

            if order_result is None or "orderId" not in order_result:
                logging.error(f"[{symbol}] Binance emir başarısız, sanal ekleniyor.")
                await self.send_telegram_message(f"⚠️ [BINANCE] {symbol} için emir gönderilemedi, simülasyon olarak sanal portföye eklendi.")
            else:
                logging.info(f"[{symbol}] Binance emir başarıyla gönderildi: {order_result.get('orderId')}")

            # Sanal portföyde de tut
            position = self.portfolio.add_position(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                size=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profits=[tp1, tp2, tp3]
            )

            self.active_positions[symbol] = {
                'direction': direction,
                'entry': current_price,
                'sl': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'size': position_size,
                'leverage': leverage,
                'margin': required_margin,
                'timestamp': datetime.now(IST_TIMEZONE)
            }

            stats = self.portfolio.get_portfolio_stats()
            message = (
                f"🔻 QuantumTrade Sinyali 🔻\n"
                f"• Sembol: {symbol}\n"
                f"• Yön: {direction}\n"
                f"• Giriş: {current_price:.2f}\n"
                f"• Boyut: {position_size}\n"
                f"• Marjin: {required_margin:.2f} USDT\n"
                f"• Kaldıraç: {leverage}x\n"
                f"• Stop Loss: {stop_loss:.2f}\n"
                f"• TP1: {tp1:.2f}\n"
                f"• TP2: {tp2:.2f}\n"
                f"• TP3: {tp3:.2f}\n"
                f"• Piyasa Fazı: {self.market_phase}\n"
                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                f"📊 Portfolio Durumu:\n"
                f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"
                f"• Kullanılan Marjin: {stats['used_margin']:.2f} USDT\n"
                f"• Toplam Kar/Zarar: {stats['total_profit']:.2f} USDT\n"
                f"• ROI: {stats['roi_percent']:.2f}%\n"
                f"• Başarı Oranı: {stats['win_rate']:.1f}%\n"
                f"• Toplam İşlem: {stats['total_trades']}"
            )

            success = await self.send_telegram_message(message)
            if success:
                logging.info(f"[{symbol}] işlem sinyali Telegram'a gönderildi")
            else:
                logging.error(f"[{symbol}] işlem sinyali Telegram'a gönderilemedi")

        except Exception as e:
            logging.error(f"execute_trade içinde hata: {str(e)}")
            logging.exception("execute_trade içinde detaylı hata:")

    # Diğer metodlar (analyze_market, process_symbol, vb.) değişmedi, yukarıdaki kodla aynı şekilde devam edecek

    async def process_symbol(self, symbol):
        try:
            if self.portfolio.is_in_cooldown(symbol):
                logging.debug(f"[{symbol}] Cooldown süresinde, işlem yapılmayacak")
                return
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                last_candle = await self.fetch_data(symbol, interval='1m', limit=1)
                if last_candle.empty:
                    logging.warning(f"[{symbol}] için fiyat verisi alınamadı")
                    return
                current_price = float(last_candle['close'].iloc[-1])
                period_high = float(last_candle['high'].iloc[-1])
                period_low = float(last_candle['low'].iloc[-1])
                if position['direction'] == 'LONG':
                    if period_low <= position['sl']:
                        logging.info(f"[{symbol}] Stop Loss tetiklendi - En Düşük: {period_low:.2f}")
                        closed_position = self.portfolio.close_position(symbol, period_low, "Stop Loss")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))
                        stats = self.portfolio.get_portfolio_stats()
                        await self.send_telegram_message(
                            f"🔴 {symbol} LONG Pozisyon Kapatıldı\n"
                            f"• Sebep: Stop Loss\n"
                            f"• Giriş: {position['entry']:.2f}\n"
                            f"• Çıkış: {period_low:.2f}\n"
                            f"• Kar/Zarar: {closed_position['pnl']:.4f} USDT\n"
                            f"• Süre: {closed_position['duration']}\n"
                            f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                            f"📊 Portfolio Durumu:\n"
                            f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                            f"• Toplam Kar/Zarar: {stats['total_profit']:.2f} USDT\n"
                            f"• ROI: {stats['roi_percent']:.2f}%\n"
                            f"• Başarı Oranı: {stats['win_rate']:.1f}%"
                        )
                        del self.active_positions[symbol]
                    elif period_high >= position['tp3']:
                        logging.info(f"[{symbol}] TP3 hedefine ulaşıldı - En Yüksek: {period_high:.2f}")
                        closed_position = self.portfolio.close_position(symbol, period_high, "TP3")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))
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
                        if period_high >= position['tp1'] and not position.get('tp1_hit'):
                            position['tp1_hit'] = True
                            position['sl'] = position['entry']
                            stats = self.portfolio.get_portfolio_stats()
                            await self.send_telegram_message(
                                f"📈 {symbol} TP1 Hedefine Ulaşıldı\n"
                                f"• Fiyat: {period_high:.2f}\n"
                                f"• Anlık Kar: {(period_high - position['entry']) * position['size']:.4f} USDT\n"
                                f"• Marjin: {position.get('margin', 0):.2f} USDT\n"
                                f"• Stop Loss Break-Even'a çekildi\n"
                                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                                f"📊 Portfolio Durumu:\n"
                                f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                                f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"
                                f"• Kullanılan Marjin: {stats['used_margin']:.2f} USDT"
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
                else:
                    if period_high >= position['sl']:
                        logging.info(f"[{symbol}] Stop Loss tetiklendi - En Yüksek: {period_high:.2f}")
                        closed_position = self.portfolio.close_position(symbol, period_high, "Stop Loss")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))
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
                    elif period_low <= position['tp3']:
                        logging.info(f"[{symbol}] TP3 hedefine ulaşıldı - En Düşük: {period_low:.2f}")
                        closed_position = self.portfolio.close_position(symbol, period_low, "TP3")
                        self.portfolio.set_cooldown(symbol, duration=timedelta(minutes=COOLDOWN_MINUTES))
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
                        if period_low <= position['tp1'] and not position.get('tp1_hit'):
                            position['tp1_hit'] = True
                            position['sl'] = position['entry']
                            stats = self.portfolio.get_portfolio_stats()
                            await self.send_telegram_message(
                                f"📉 {symbol} TP1 Hedefine Ulaşıldı\n"
                                f"• Fiyat: {period_low:.2f}\n"
                                f"• Anlık Kar: {(position['entry'] - period_low) * position['size']:.4f} USDT\n"
                                f"• Marjin: {position.get('margin', 0):.2f} USDT\n"
                                f"• Stop Loss Break-Even'a çekildi\n"
                                f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n\n"
                                f"📊 Portfolio Durumu:\n"
                                f"• Bakiye: {stats['current_balance']:.2f} USDT\n"
                                f"• Kullanılabilir: {stats['available_balance']:.2f} USDT\n"
                                f"• Kullanılan Marjin: {stats['used_margin']:.2f} USDT"
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
            df = await self.fetch_data(symbol, interval='15m', limit=100)
            if df.empty:
                logging.warning(f"[{symbol}] için veri alınamadı.")
                return
            df = self.calculate_features(df)
            if df.empty:
                logging.warning(f"[{symbol}] için özellikler hesaplanamadı.")
                return
            latest_features = df[self.required_features].iloc[-1:].values
            X_scaled = self.scalers[symbol].transform(latest_features)
            predictions = {
                'xgb': self.models['xgb'][symbol].predict_proba(X_scaled)[0][1],
                'lgbm': self.models['lgbm'][symbol].predict_proba(X_scaled)[0][1],
                'gbt': self.models['gbt'][symbol].predict_proba(X_scaled)[0][1],
                'lstm': float(self.models['lstm'][symbol].predict(X_scaled.reshape(1, 1, -1))[0][0])
            }
            avg_prediction = sum(predictions.values()) / len(predictions)
            current_atr = df['ATR_14'].iloc[-1]
            current_price = df['close'].iloc[-1]
            if avg_prediction > 0.55:
                logging.info(f"[{symbol}] LONG sinyali: {avg_prediction:.3f}")
                await self.execute_trade(symbol, 'LONG', current_price, avg_prediction, current_atr)
            elif avg_prediction < 0.45:
                logging.info(f"[{symbol}] SHORT sinyali: {avg_prediction:.3f}")
                await self.execute_trade(symbol, 'SHORT', current_price, avg_prediction, current_atr)
            else:
                logging.debug(f"[{symbol}] için sinyal yok. Tahmin: {avg_prediction:.3f}")
        except Exception as e:
            logging.error(f"[{symbol}] analiz hatası: {str(e)}")

    async def run_analysis_loop(self):
        logging.info("Analiz döngüsü başlatılıyor...")
        while True:
            try:
                for symbol in self.symbols_to_trade:
                    await self.process_symbol(symbol)
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logging.info("Analiz döngüsü durduruldu.")
                break
            except Exception as e:
                logging.error(f"Analiz döngüsünde hata: {str(e)}")
                await asyncio.sleep(5)

async def main():
    load_dotenv()
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    CHAT_ID = os.getenv('CHAT_ID')
    symbols = SYMBOLS
    async with QuantumTrader(symbols, TELEGRAM_TOKEN, CHAT_ID, portfolio=1000) as trader:
        await trader.run_analysis_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot güvenli şekilde durduruldu.")
