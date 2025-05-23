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
import pandas_ta as pta
import pytz
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
import hmac
import hashlib
import urllib.parse
import requests
from datetime import datetime

# ===== KONFİGÜRASYON =====
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
IST_TIMEZONE = pytz.timezone("Europe/Istanbul")
COOLDOWN_MINUTES = 10
MODEL_UPDATE_INTERVAL = timedelta(hours=6)
LSTM_WINDOW_SIZE = 30
N_FUTURE_CANDLES = 3
BASE_URL = "https://fapi.binance.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== YARDIMCI FONKSİYON ==========
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'UNIUSDT', 'BCHUSDT', 'DOTUSDT', 'POLUSDT',
    'LINKUSDT', 'ETCUSDT', 'FILUSDT', 'SUIUSDT', 'LDOUSDT', 'ARBUSDT',
    'APTUSDT', 'OPUSDT', '1000PEPEUSDT', 'RNDRUSDT', 'STXUSDT', 'BLURUSDT', 'INJUSDT',
    'ENAUSDT', 'ATAUSDT', 'ARKUSDT', '1000SHIBUSDT', '1000BONKUSDT',
    '1000FLOKIUSDT', '1000SATSUSDT', '10000RATSUSDT', 'APEUSDT', 'CYBERUSDT', 'JTOUSDT'
]
SYMBOL_LEVERAGE = {s: 20 for s in SYMBOLS}
precision_map = {
    'BTCUSDT': 3, 'ETHUSDT': 3, 'BNBUSDT': 2, 'SOLUSDT': 2, 'XRPUSDT': 0, 'SUIUSDT': 0,
    'DOGEUSDT': 0, 'ADAUSDT': 0, 'AVAXUSDT': 2, 'UNIUSDT': 2, 'BCHUSDT': 3, 'DOTUSDT': 1,
    'POLUSDT': 1, 'LINKUSDT': 2, 'ETCUSDT': 3, 'FILUSDT': 2, 'LDOUSDT': 2, 'ARBUSDT': 2,
    'APTUSDT': 2, 'OPUSDT': 2, '1000PEPEUSDT': 0, 'RNDRUSDT': 2, 'STXUSDT': 2, 'BLURUSDT': 0, 'INJUSDT': 2,
    'ENAUSDT': 2, 'ATAUSDT': 1, 'ARKUSDT': 1, '1000SHIBUSDT': 0, '1000BONKUSDT': 0, '1000FLOKIUSDT': 0,
    '1000SATSUSDT': 0, '10000RATSUSDT': 0, 'APEUSDT': 1, 'CYBERUSDT': 2, 'JTOUSDT': 2
}
step_map = {
    'BTCUSDT': 0.001, 'ETHUSDT': 0.001, 'BNBUSDT': 0.01, 'SOLUSDT': 0.01, 'XRPUSDT': 1, 'SUIUSDT': 1,
    'DOGEUSDT': 1, 'ADAUSDT': 1, 'AVAXUSDT': 0.01, 'UNIUSDT': 0.01, 'BCHUSDT': 0.001, 'DOTUSDT': 0.1,
    'POLUSDT': 0.1, 'LINKUSDT': 0.01, 'ETCUSDT': 0.001, 'FILUSDT': 0.01, 'LDOUSDT': 0.01, 'ARBUSDT': 0.01,
    'APTUSDT': 0.01, 'OPUSDT': 0.01, '1000PEPEUSDT': 1, 'RNDRUSDT': 0.01, 'STXUSDT': 0.01, 'BLURUSDT': 1, 'INJUSDT': 0.01,
    'ENAUSDT': 0.01, 'ATAUSDT': 0.1, 'ARKUSDT': 0.1, '1000SHIBUSDT': 1, '1000BONKUSDT': 1, '1000FLOKIUSDT': 1,
    '1000SATSUSDT': 1, '10000RATSUSDT': 1, 'APEUSDT': 0.1, 'CYBERUSDT': 0.01, 'JTOUSDT': 0.01
}
price_precisions = {
    'BTCUSDT': 1, 'ETHUSDT': 2, 'BNBUSDT': 2, 'SOLUSDT': 2, 'XRPUSDT': 4, 'SUIUSDT': 4,
    'DOGEUSDT': 4, 'ADAUSDT': 4, 'AVAXUSDT': 3, 'UNIUSDT': 3, 'BCHUSDT': 2, 'DOTUSDT': 3,
    'POLUSDT': 4, 'LINKUSDT': 3, 'ETCUSDT': 2, 'FILUSDT': 3, 'LDOUSDT': 4, 'ARBUSDT': 4,
    'APTUSDT': 4, 'OPUSDT': 4, '1000PEPEUSDT': 5, 'RNDRUSDT': 4, 'STXUSDT': 4, 'BLURUSDT': 5, 'INJUSDT': 3,
    'ENAUSDT': 4, 'ATAUSDT': 4, 'ARKUSDT': 4, '1000SHIBUSDT': 1, '1000BONKUSDT': 4, '1000FLOKIUSDT': 4,
    '1000SATSUSDT': 4, '10000RATSUSDT': 5, 'APEUSDT': 4, 'CYBERUSDT': 4, 'JTOUSDT': 4
}

def format_quantity(symbol, quantity):
    precision = precision_map.get(symbol, 3)
    step = step_map.get(symbol, 0.01)
    qty = round(quantity // step * step, precision)
    return max(qty, step)

def format_price(symbol, price):
    precision = price_precisions.get(symbol, 2)
    return round(price, precision)

# ========== İSTATİSTİK MODÜLÜ ==========
class TradeStatistics:
    def __init__(self):
        self.trades = []
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'daily_volume': 0.0
        }

    def add_trade(self, symbol, direction, entry, exit_price, quantity):
        pnl = (exit_price - entry) * quantity if direction == 'LONG' else (entry - exit_price) * quantity
        is_profitable = pnl > 0

        trade_record = {
            'timestamp': datetime.now(IST_TIMEZONE),
            'symbol': symbol,
            'direction': direction,
            'entry': entry,
            'exit': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'success': is_profitable
        }

        # Metrikleri güncelle
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['profitable_trades'] += int(is_profitable)
        self.performance_metrics['total_pnl'] += pnl
        self.performance_metrics['daily_volume'] += abs(quantity * entry)

        # Max drawdown güncelleme
        current_equity = self.performance_metrics['total_pnl']
        if current_equity < self.performance_metrics['max_drawdown']:
            self.performance_metrics['max_drawdown'] = current_equity

        self.trades.append(trade_record)
        self._clean_old_trades()

    def _clean_old_trades(self):
        # 30 günden eskinleri temizle
        cutoff = datetime.now(IST_TIMEZONE) - timedelta(days=30)
        self.trades = [t for t in self.trades if t['timestamp'] > cutoff]

    def get_stats(self, period_hours=24):
        cutoff = datetime.now(IST_TIMEZONE) - timedelta(hours=period_hours)
        recent_trades = [t for t in self.trades if t['timestamp'] > cutoff]

        if not recent_trades:
            return {
                'win_rate': 0.0,
                'total_trades': 0,
                'avg_pnl': 0.0,
                'max_drawdown': 0.0,
                'volume': 0.0
            }

        profitable = sum(1 for t in recent_trades if t['success'])
        total = len(recent_trades)
        avg_pnl = sum(t['pnl'] for t in recent_trades) / total
        volume = sum(abs(t['quantity'] * t['entry']) for t in recent_trades)

        return {
            'win_rate': profitable / total * 100,
            'total_trades': total,
            'avg_pnl': avg_pnl,
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'volume': volume
        }

def sign_params(params):
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

def supertrend(self, df, period=7, multiplier=2):
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
        final_upperband = hl2 + (multiplier * atr)
        final_lowerband = hl2 - (multiplier * atr)
        supertrend = np.zeros(len(df))
        direction = np.ones(len(df), dtype=bool)
        for i in range(1, len(df)):
            if supertrend[i - 1] == 0:
                supertrend[i] = final_lowerband.iloc[i]
                direction[i] = True
            elif df['close'].iloc[i - 1] > supertrend[i - 1]:
                supertrend[i] = final_lowerband.iloc[i]
                direction[i] = True
            else:
                supertrend[i] = final_upperband.iloc[i]
                direction[i] = False
        df['SuperTrend'] = supertrend
        df['SuperTrend_Up'] = direction.astype(int)
        return df

def tema(series, n=10):
    ema1 = series.ewm(span=n, adjust=False).mean()
    ema2 = ema1.ewm(span=n, adjust=False).mean()
    ema3 = ema2.ewm(span=n, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

def teknik_analiz(self, df):
        df['TEMA_10'] = self.tema(df['close'], n=10)
        df = self.supertrend(df, period=7, multiplier=2)
        trend_up = df['SuperTrend_Up'].iloc[-1] and df['TEMA_10'].iloc[-1] > df['close'].iloc[-2]
        trend_down = (not df['SuperTrend_Up'].iloc[-1]) and df['TEMA_10'].iloc[-1] < df['close'].iloc[-2]
        return trend_up, trend_down, df

class QuantumTrader:
    def __init__(self, symbols_to_trade, telegram_token, chat_id):
        self.session = None
        self.headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        self.symbols_to_trade = symbols_to_trade
        self.models = {'xgb': {}, 'lgbm': {}, 'gbt': {}, 'lstm': {}}
        self.scalers = {}
        self.required_features = ['TEMA_10', 'SuperTrend', 'SuperTrend_Up']
        self.market_phase = "INITIALIZING"
        self.cooldowns = {}
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.lock = asyncio.Lock()
        self.last_position_status = {}
        self.stats = TradeStatistics()      # İstatistik modülünü de ekle!
        self.report_task = None             # Rapor görevini de ekle!

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def send_telegram_message(self, message: str):
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            params = {
                "chat_id": self.chat_id,           # veya self.telegram_chat_id, senin sistemine göre
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

    async def send_performance_report(self):
        report = self.stats.get_stats(24)
        message = (
            "📊 24 Saatlik Performans Raporu\n"
            f"• Win Rate: {report['win_rate']:.1f}%\n"
            f"• Toplam İşlem: {report['total_trades']}\n"
            f"• Ort. Kazanç: ${report['avg_pnl']:.2f}\n"
            f"• Hacim: ${report['volume']:,.2f}\n"
            f"• Max Çekilme: ${report['max_drawdown']:,.2f}"
        )
        await self.send_telegram_message(message)

    # --- VOLATILITE ANALIZI ---
    async def find_most_volatile_symbols(self, interval='3m', lookback=100, top_n=5):
        volatilities = []
        for symbol in self.symbols_to_trade:
            df = await self.fetch_data_multi(symbol, interval=interval, total_limit=lookback)
            if df is None or df.empty or len(df) < 20:
                continue
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
            rel_vol = atr / df['close'].iloc[-1]
            volatilities.append((symbol, rel_vol))
        volatilities.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, v in volatilities[:top_n]]
        return top_symbols, volatilities

    # --- AÇIK POZISYONLARI BUL ---
    async def get_all_open_positions(self):
        url = "https://fapi.binance.com/fapi/v2/positionRisk"
        # Binance API için auth ve sign_params fonksiyonu gereklidir!
        params = {"timestamp": int(time.time() * 1000)}
        # signed = sign_params(params)  # Burada auth fonksiyonunu eklemelisin!
        headers = {"X-MBX-APIKEY": self.api_key}
        async with self.session.get(url, headers=headers, params=params) as resp:
            data = await resp.json()
            open_positions = []
            for pos in data:
                amt = float(pos['positionAmt'])
                if abs(amt) > 0:
                    open_positions.append(pos['symbol'])
            return open_positions

    async def fetch_data_multi(self, symbol, interval='5m', total_limit=3000):
        limit_per_call = 1500
        all_data = []
        end_time = int(time.time() * 1000)
        while total_limit > 0:
            limit = min(limit_per_call, total_limit)
            url = f"{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={end_time}"
            async with self.session.get(url) as response:
                data = await response.json()
                if not data or isinstance(data, dict):
                    logging.warning(f"{symbol} için veri çekilemedi veya boş geldi: {data}")
                    break
                all_data = data + all_data
                end_time = data[0][0] - 1
                if len(data) < limit:
                    break
                total_limit -= limit
        if not all_data:
            return pd.DataFrame()
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
            logging.info(f"{symbol} {order_id} no'lu emir iptal edildi: {data}")

    async def cancel_open_orders(self, symbol):
        url = BASE_URL + "/fapi/v1/openOrders"
        params = {"symbol": symbol, "timestamp": int(time.time() * 1000)}
        signed = sign_params(params)
        async with self.session.get(url, headers=self.headers, params=signed) as resp:
            orders = await resp.json()
            for order in orders:
                if (order.get('reduceOnly', False) or order.get('closePosition', False)) and order['type'] in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                    await self.cancel_order(symbol, order['orderId'])

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

    async def send_binance_market_order(self, symbol, side, quantity, leverage):
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
                logging.info(f"[BINANCE] Market emir gönderildi: {side} {symbol} {quantity} - {data.get('orderId')}")
            else:
                logging.error(f"[BINANCE] Market emir gönderilemedi: {data}")
            return data

    async def send_partial_tp_and_move_sl(self, symbol, entry_price, total_quantity, stop_loss, tp1, tp2, tp3):
        close_side = 'SELL' if self.last_direction == 'BUY' else 'BUY'

        # Pozisyonun bölümleri
        qty_tp1 = round(total_quantity * 0.3, 6)
        qty_tp2 = round(total_quantity * 0.3, 6)
        qty_tp3 = total_quantity - qty_tp1 - qty_tp2  # kalan %40

        # 1. Stop Loss ve 3 TP için emir aç
        await self.place_sl_tp_order(symbol, close_side, qty_tp1, stop_loss, tp1, "TP1")
        await self.place_tp_order(symbol, close_side, qty_tp2, tp2, "TP2")
        await self.place_tp_order(symbol, close_side, qty_tp3, tp3, "TP3")

    async def place_sl_tp_order(self, symbol, side, quantity, stop_loss, take_profit, label="TP1"):
        # Hem stop-loss hem TP için emir açılır
        url = BASE_URL + "/fapi/v1/order"
        # Stop-Loss
        sl_params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": format_quantity(symbol, quantity),
            "stopPrice": format_price(symbol, stop_loss),
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }
        sl_signed = sign_params(sl_params)
        async with self.session.post(url, headers=self.headers, data=sl_signed) as resp:
            data = await resp.json()
            logging.info(f"[{symbol}][{label}] STOP-LOSS gönderildi. Yanıt: {data}, Status: {resp.status}")

        # TP
        tp_params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": format_quantity(symbol, quantity),
            "stopPrice": format_price(symbol, take_profit),
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }
        tp_signed = sign_params(tp_params)
        async with self.session.post(url, headers=self.headers, data=tp_signed) as resp:
            data = await resp.json()
            logging.info(f"[{symbol}][{label}] TAKE PROFIT gönderildi. Yanıt: {data}, Status: {resp.status}")

    async def place_tp_order(self, symbol, side, quantity, take_profit, label="TP"):
        # Sadece TP için (stop yok)
        url = BASE_URL + "/fapi/v1/order"
        tp_params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": format_quantity(symbol, quantity),
            "stopPrice": format_price(symbol, take_profit),
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }
        tp_signed = sign_params(tp_params)
        async with self.session.post(url, headers=self.headers, data=tp_signed) as resp:
            data = await resp.json()
            logging.info(f"[{symbol}][{label}] TAKE PROFIT gönderildi. Yanıt: {data}, Status: {resp.status}")
    
    async def move_stop_to_entry(self, symbol, quantity, entry_price):
        # Eski stop emirlerini iptal et
        await self.cancel_open_orders(symbol)
        # Yeni stop-loss'u entry fiyatına koy
        close_side = 'SELL' if self.last_direction == 'BUY' else 'BUY'
        url = BASE_URL + "/fapi/v1/order"
        sl_params = {
            "symbol": symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "quantity": format_quantity(symbol, quantity),
            "stopPrice": format_price(symbol, entry_price),
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }
        sl_signed = sign_params(sl_params)
        async with self.session.post(url, headers=self.headers, data=sl_signed) as resp:
            data = await resp.json()
            logging.info(f"[{symbol}] STOP-LOSS entryye çekildi: {data}, Status: {resp.status}")

    async def execute_trade(self, symbol: str, direction: str, current_price: float, confidence: float, atr: float):
        try:
            logging.info(f"[{symbol}] Trade açma işlemi başlıyor. Yön: {direction}, Fiyat: {current_price}, ATR: {atr}")
            if atr <= 0 or current_price <= 0:
                logging.error(f"[{symbol}] ATR veya fiyat geçersiz: ATR={atr}, Price={current_price}")
                return

            max_atr = current_price * 0.3
            atr = min(atr, max_atr)
            sl_distance = min(2.0 * atr, current_price * 0.3)

            if direction == 'LONG':
                entry = current_price
                stop_loss = current_price - sl_distance
                if stop_loss <= 0:
                    logging.error(f"[{symbol}] stop_loss geçersiz: {stop_loss}. Trade açılmadı.")
                    return
                tp1 = entry + (sl_distance * 1.0)
                tp2 = entry + (sl_distance * 1.5)
                tp3 = entry + (sl_distance * 2.5)
                order_side = "BUY"
                self.last_direction = "BUY"
            else:
                entry = current_price
                stop_loss = current_price + sl_distance
                tp1 = entry - (sl_distance * 1.0)
                tp2 = entry - (sl_distance * 1.5)
                tp3 = entry - (sl_distance * 2.5)
                order_side = "SELL"
                self.last_direction = "SELL"

            min_tp_gap = entry * 0.003
            tp3 = entry + (sl_distance * 2.5) if direction == 'LONG' else entry - (sl_distance * 2.5)
            rr = abs(tp3 - entry) / abs(entry - stop_loss)

            if rr < 1.3 or abs(tp3 - entry) < min_tp_gap:
                await self.send_telegram_message(
                    f"{symbol} için RR çok düşük ({rr:.2f}) veya TP3 aralığı çok kısa (min %0.5), trade açılmadı."
                )
                logging.info(f"{symbol} için RR ({rr:.2f}) veya TP3-giriş ({abs(tp3-entry):.4f}) yetersiz, trade açılmadı.")
                return

            balance = await self.get_binance_balance()
            risk_percent = 0.01
            risk_amount = balance * risk_percent
            stop_distance = abs(entry - stop_loss)
            position_size = risk_amount / stop_distance

            leverage = SYMBOL_LEVERAGE.get(symbol, 20)
            required_margin = (entry * position_size) / leverage

            order_result = await self.send_binance_market_order(
                symbol=symbol,
                side=order_side,
                quantity=position_size,
                leverage=leverage
            )

            if order_result is None or "orderId" not in order_result:
                logging.error(f"[{symbol}] Binance emir başarısız: {order_result}")
                return

            for _ in range(10):
                await asyncio.sleep(0.5)
                pos = await self.get_open_position(symbol)
                if pos and abs(pos['amt']) > 0:
                    break
            else:
                logging.error("Pozisyon açılmadı, SL/TP emirleri gönderilmeyecek!")
                await self.send_telegram_message(f"⚠️ [BINANCE] {symbol} pozisyonu açılmadı, SL/TP emirleri gönderilmedi.")
                return

            await self.send_partial_tp_and_move_sl(
                symbol=symbol,
                entry_price=entry,
                total_quantity=position_size,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3
            )

            message = (
                f"🔻 Quantum AI Trader 🔻\n"
                f"• 🔔 Sembol: {symbol}\n"
                f"• 📉 Trend: {direction}\n"
                f"• 💲 Marjin: {required_margin:.2f} USDT\n"
                f"• 📈 Kaldırac: {leverage}x\n"
                "------------------\n"
                f"• 🟢 ENTRY: {current_price:.2f}\n"
                f"• 🚫 Stop Loss: {stop_loss:.2f}\n"
                f"• 💸 TP1: {tp1:.2f}\n"
                f"• 👑 TP2: {tp2:.2f}\n"
                f"• 💎 TP3: {tp3:.2f}\n"
                f"• 💰 Bakiye: {balance:.2f}\n"
                f"\n• ⏰ Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n"
            )
            await self.send_telegram_message(message)
        except Exception as e:
            logging.error(f"execute_trade içinde hata: {str(e)}")
            logging.exception("execute_trade içinde detaylı hata:")

    async def process_symbol(self, symbol):
        df = await self.fetch_data_multi(symbol, interval='3m', total_limit=200)
        if df is None or df.empty:
            return
        trend_up, trend_down, df = self.teknik_analiz(df)
        entry = df['close'].iloc[-1]
        position_size = 1  # Örnek, gerçek pozisyon büyüklüğü hesaplaması eklemelisin

        if trend_up:
            direction = 'LONG'
            await self.open_position(symbol, direction, entry, position_size)
        elif trend_down:
            direction = 'SHORT'
            await self.open_position(symbol, direction, entry, position_size)

    async def open_position(self, symbol, direction, entry, quantity):
        # Burada Binance API ile emir açma kodu olmalı
        # Emir açıldıktan sonra pozisyonu izle ve kapanınca istatistiğe kaydet
        exit_price = await self.wait_for_exit(symbol)
        if exit_price:
            await self.after_position_closed(symbol, direction, entry, exit_price, quantity)

    async def wait_for_exit(self, symbol, timeout=86400):
        start_time = time.time()
        while time.time() - start_time < timeout:
            position = await self.get_position(symbol)
            if not position or position['amount'] == 0:
                ticker = await self.get_ticker(symbol)
                return ticker['last_price']
            await asyncio.sleep(60)
        return None

    async def get_position(self, symbol):
        # Pozisyon bilgisini çeken kod burada olmalı (ör: Binance API)
        # {"amount": poz_büyüklüğü} veya benzeri bir sözlük dönmeli
        return {"amount": 0}  # Örnek, gerçeğini eklemelisin

    async def get_ticker(self, symbol):
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
        async with self.session.get(url) as res:
            data = await res.json()
            return {
                'last_price': float(data['lastPrice']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice'])
            }

    async def after_position_closed(self, symbol, direction, entry, exit_price, quantity):
        self.stats.add_trade(
            symbol=symbol,
            direction=direction,
            entry=entry,
            exit_price=exit_price,
            quantity=quantity
        )
       
        features = df[self.required_features]  # DataFrame olarak

        # SCALER
        scaler = self.scalers.get(symbol)
        if scaler is None:
            scaler = StandardScaler().fit(features)
            self.scalers[symbol] = scaler

        # Son veri DataFrame olarak (tek satırlık)
        X = pd.DataFrame([features.iloc[-1]], columns=features.columns)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features.columns)

        # MODEL OLUŞTURMA (ilk defa ise)
        if symbol not in self.models['xgb']:
                # Her zaman DataFrame ile fit!
                self.models['xgb'][symbol] = xgb.XGBClassifier().fit(features, np.random.randint(0,2,size=len(df)))
                self.models['lgbm'][symbol] = LGBMClassifier().fit(features, np.random.randint(0,2,size=len(df)))
                self.models['gbt'][symbol] = GradientBoostingClassifier().fit(features, np.random.randint(0,2,size=len(df)))
                self.models['lstm'][symbol] = Sequential([
                    LSTM(16, input_shape=(1, X_scaled.shape[1])),
                    Dense(1, activation='sigmoid')
                ])
                self.models['lstm'][symbol].compile(optimizer=Adam(), loss='binary_crossentropy')
                # Dummy fit veya yükleme burada yapılabilir

        # TAHMİNLER (predict kısmı)
        predictions = {
            'xgb': self.models['xgb'][symbol].predict_proba(X_scaled_df)[0][1],
            'lgbm': self.models['lgbm'][symbol].predict_proba(X_scaled_df)[0][1],
            'gbt': self.models['gbt'][symbol].predict_proba(X_scaled_df)[0][1],
            'lstm': float(self.models['lstm'][symbol].predict(X_scaled.reshape(1, 1, -1))[0][0])
            }

        avg_prediction = sum(predictions.values()) / len(predictions)
        current_atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
        current_price = df['close'].iloc[-1]

        now = datetime.now(IST_TIMEZONE)
        logging.info(f"[{symbol}] Model prediction: {avg_prediction:.3f}, trend_up: {trend_up}, trend_down: {trend_down}, ATR: {current_atr:.4f}, Price: {current_price:.2f}")
        if symbol in self.cooldowns and now < self.cooldowns[symbol]:
            logging.info(f"[{symbol}] cooldown sürüyor, yeni trade açılmayacak.")
            return

        pos = await self.get_open_position(symbol)
        last_pos_status = self.last_position_status.get(symbol, None)
        pozisyon_acik = pos and abs(pos['amt']) > 0
        logging.info(f"[{symbol}] Pozisyon açık mı? {pozisyon_acik} (Açık pozisyon miktarı: {pos['amt'] if pos else None})")

        if last_pos_status is True and not pozisyon_acik:
            logging.info(f"[{symbol}] Pozisyon kapanmış, açık TP/SL emirleri temizleniyor.")
            await self.cancel_open_orders(symbol)
            self.last_position_status[symbol] = pozisyon_acik
            if pozisyon_acik:
                logging.info(f"[{symbol}] zaten açık pozisyon var, yeni trade açılmayacak.")
                return

            if avg_prediction > 0.52 and trend_up:
                logging.info(f"[{symbol}] LONG sinyali: {avg_prediction:.3f}")
                await self.execute_trade(symbol, 'LONG', current_price, avg_prediction, current_atr)
                self.cooldowns[symbol] = now + timedelta(minutes=COOLDOWN_MINUTES)
            elif avg_prediction < 0.48 and trend_down:
                logging.info(f"[{symbol}] SHORT sinyali: {avg_prediction:.3f}")
                await self.execute_trade(symbol, 'SHORT', current_price, avg_prediction, current_atr)
                self.cooldowns[symbol] = now + timedelta(minutes=COOLDOWN_MINUTES)
            else:
                logging.debug(f"[{symbol}] için sinyal yok. Tahmin: {avg_prediction:.3f}")

        except Exception as e:
            logging.error(f"[{symbol}] analiz hatası: {str(e)}")

    async def run_analysis_loop(self):
        logging.info("Analiz döngüsü başlatılıyor...")
        MAX_ACTIVE_TRADES = 3
        TOP_VOLATILE_COUNT = 5

        while True:
            try:
                top_symbols, all_vols = await self.find_most_volatile_symbols(interval='3m', lookback=120, top_n=TOP_VOLATILE_COUNT)
                msg = "En volatil coinler:\n" + "\n".join([f"{s}: {v*100:.2f}%" for s,v in all_vols[:TOP_VOLATILE_COUNT]])
                logging.info(msg)

                open_positions = await self.get_all_open_positions()
                logging.info(f"Açık pozisyonlar: {open_positions} (max {MAX_ACTIVE_TRADES})")

                analyzed_count = 0
                for symbol in top_symbols:
                    if symbol in open_positions:
                        logging.info(f"{symbol} zaten açık pozisyonda, atlanıyor.")
                        continue
                    if len(open_positions) + analyzed_count >= MAX_ACTIVE_TRADES:
                        logging.info(f"Maksimum aktif işlem limiti ({MAX_ACTIVE_TRADES}) doldu, yeni işlem açılmayacak.")
                        break
                    logging.info(f"[DEBUG] {symbol} analiz başlıyor (volatiliteye göre seçildi)...")
                    await self.process_symbol(symbol)
                    logging.info(f"[DEBUG] {symbol} analiz tamamlandı.")
                    analyzed_count += 1
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logging.info("Analiz döngüsü durduruldu.")
                break
            except Exception as e:
                logging.error(f"Analiz döngüsünde hata: {str(e)}")
                await asyncio.sleep(5)

    async def find_most_volatile_symbols(self, interval='3m', lookback=100, top_n=5):
        volatilities = []
        for symbol in self.symbols_to_trade:
            df = await self.fetch_data_multi(symbol, interval=interval, total_limit=lookback)
            if df.empty or len(df) < 20:
                continue
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
            rel_vol = atr / df['close'].iloc[-1]
            volatilities.append((symbol, rel_vol))
        volatilities.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, v in volatilities[:top_n]]
        return top_symbols, volatilities

    async def get_all_open_positions(self):
        # Tüm semboller için açık pozisyonları döndür
        url = BASE_URL + "/fapi/v2/positionRisk"
        params = {"timestamp": int(time.time() * 1000)}
        signed = sign_params(params)
        async with self.session.get(url, headers=self.headers, params=signed) as resp:
            data = await resp.json()
            open_positions = []
            for pos in data:
                amt = float(pos['positionAmt'])
                if abs(amt) > 0:
                    open_positions.append(pos['symbol'])
            return open_positions

    async def run_analysis_loop(self):
        logging.info("Analiz döngüsü başlatılıyor...")
        MAX_ACTIVE_TRADES = 3   # aynı anda en fazla 3 açık işlem
        TOP_VOLATILE_COUNT = 5  # analiz edilecek en volatil ilk 5 coin

        while True:
            try:
                # 1. En volatil coinleri bul
                top_symbols, all_vols = await self.find_most_volatile_symbols(interval='3m', lookback=120, top_n=TOP_VOLATILE_COUNT)
                msg = "En volatil coinler:\n" + "\n".join([f"{s}: {v*100:.2f}%" for s,v in all_vols[:TOP_VOLATILE_COUNT]])
                logging.info(msg)

                # 2. O anda açık pozisyonları bul
                open_positions = await self.get_all_open_positions()
                logging.info(f"Açık pozisyonlar: {open_positions} (max {MAX_ACTIVE_TRADES})")

                # 3. Eğer açık pozisyon sayısı < MAX_ACTIVE_TRADES ise ve volatil coinlerde yeni pozisyon yoksa, analiz ve işlem aç
                analyzed_count = 0
                for symbol in top_symbols:
                    if symbol in open_positions:
                        logging.info(f"{symbol} zaten açık pozisyonda, atlanıyor.")
                        continue
                    if len(open_positions) + analyzed_count >= MAX_ACTIVE_TRADES:
                        logging.info(f"Maksimum aktif işlem limiti ({MAX_ACTIVE_TRADES}) doldu, yeni işlem açılmayacak.")
                        break
                    print(f"[DEBUG] {symbol} analiz başlıyor (volatiliteye göre seçildi)...")
                    await self.process_symbol(symbol)
                    print(f"[DEBUG] {symbol} analiz tamamlandı.")
                    analyzed_count += 1
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logging.info("Analiz döngüsü durduruldu.")
                break
            except Exception as e:
                logging.error(f"Analiz döngüsünde hata: {str(e)}")
                await asyncio.sleep(5)
async def main():
    # API KEY/SECRET ve Telegram bilgilerinizi buraya girin
    API_KEY = "YOUR_BINANCE_API_KEY"
    API_SECRET = "YOUR_BINANCE_API_SECRET"
    TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

    trader = QuantumTrader(API_KEY, API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    await trader.__aenter__()
    trader.report_task = asyncio.create_task(trader.start_periodic_reporting())
    try:
        await trader.run_analysis_loop()
    finally:
        trader.report_task.cancel()
        await trader.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())