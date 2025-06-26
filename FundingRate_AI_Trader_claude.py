# -*- coding: utf-8 -*-
# FR_AI_HUNTER_IMPROVED.py
# Geliştirilmiş, Güvenli ve Daha Akıllı Funding Rate AI Trader

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dotenv import load_dotenv
import ta
import pytz
import logging
from datetime import datetime, timedelta
import time
import hmac
import hashlib
import urllib.parse
import warnings
import joblib
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# ===== ENHANCED CONFIGURATION =====
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

if not all([BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_TOKEN, CHAT_ID]):
    logging.critical("Gerekli .env değişkenleri bulunamadı! Program durduruluyor.")
    exit()

# === ENHANCED STRATEGY PARAMETERS ===
class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"

@dataclass
class TradingConfig:
    # Funding Rate Thresholds
    FR_LONG_THRESHOLD: float = -0.08
    FR_EXTREME_THRESHOLD: float = -0.15
    
    # AI & Risk Parameters
    AI_CONFIDENCE_THRESHOLD: float = 0.70
    MIN_CONFIDENCE_FOR_EXTREME: float = 0.80
    BASE_RISK_PERCENT: float = 0.015  # 1.5% base risk
    MAX_RISK_PERCENT: float = 0.04    # 4% max risk per trade
    
    # Position Management
    MAX_ACTIVE_TRADES: int = 3
    MAX_CORRELATION_THRESHOLD: float = 0.7
    
    # Timing
    MARKET_SCAN_INTERVAL: int = 600   # 10 minutes
    MIN_TIME_BEFORE_FUNDING: int = 45  # minutes
    MAX_TIME_BEFORE_FUNDING: int = 390 # minutes
    SNIPER_MODE_WINDOW: int = 25      # minutes
    
    # Technical
    BATCH_SIZE: int = 30
    BATCH_DELAY: float = 0.8
    MAX_SLIPPAGE_TOLERANCE: float = 0.005  # 0.5%
    
    # Model Parameters
    N_FUTURE_CANDLES: int = 6
    MODEL_UPDATE_HOURS: int = 12
    MIN_TRAINING_SAMPLES: int = 200
    
    # Trading Mode
    TRADING_MODE: TradingMode = TradingMode.PAPER
    PAPER_BALANCE: float = 10000.0

config = TradingConfig()

# === CONSTANTS ===
IST_TIMEZONE = pytz.timezone("Europe/Istanbul")
BASE_URL = "https://fapi.binance.com"
BLACKLISTED_SYMBOLS = [
    'USDCUSDT', 'FDUSDUSDT', 'BTCDOMUSDT', 'AEURUSDT', 
    'TUSDUSDT', 'USDPUSDT', 'EURUSDT', 'COCOSUSDT'
]
MODEL_DIR = "saved_models_enhanced"
PERFORMANCE_DIR = "performance_data"

# Create directories
for directory in [MODEL_DIR, PERFORMANCE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler('fr_ai_hunter_enhanced.log', 'w', 'utf-8'),
        logging.StreamHandler()
    ]
)

@dataclass
# 1. ENHANCED MODEL VALIDATION
class ModelValidator:
    @staticmethod
    def comprehensive_validation(model, X_test, y_test, returns_test):
        """Detaylı model performans validasyonu"""
        predictions = model.predict_proba(X_test)[:, 1]
        
        # Trading metrics
        threshold = 0.7
        trades = predictions > threshold
        
        if trades.sum() == 0:
            return {'valid': False, 'reason': 'no_trades'}
        
        trade_returns = returns_test[trades]
        
        # Performance metrics
        win_rate = (trade_returns > 0).mean()
        avg_return = trade_returns.mean()
        sharpe_ratio = trade_returns.mean() / trade_returns.std() if trade_returns.std() > 0 else 0
        max_drawdown = (trade_returns.cumsum().expanding().max() - trade_returns.cumsum()).max()
        
        # Validation criteria
        criteria = {
            'min_win_rate': 0.45,
            'min_avg_return': -0.01,
            'min_sharpe': 0.5,
            'max_drawdown': 0.15,
            'min_trades': 10
        }
        
        valid = (
            win_rate >= criteria['min_win_rate'] and
            avg_return >= criteria['min_avg_return'] and
            sharpe_ratio >= criteria['min_sharpe'] and
            max_drawdown <= criteria['max_drawdown'] and
            trades.sum() >= criteria['min_trades']
        )
        
        return {
            'valid': valid,
            'metrics': {
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': trades.sum()
            }
        }

# 2. ENHANCED RISK MANAGEMENT
class PortfolioRiskManager:
    def __init__(self, max_portfolio_risk=0.1, max_correlation=0.7):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.correlation_cache = {}
        self.last_correlation_update = {}
    
    async def check_portfolio_risk(self, active_positions, new_symbol, new_risk):
        """Portfolio seviyesinde risk kontrolü"""
        # Mevcut portfolio riski
        current_risk = sum(pos.quantity * pos.entry_price * 0.02 for pos in active_positions.values())
        
        # Yeni risk eklendiğinde limit kontrolü
        if (current_risk + new_risk) > self.max_portfolio_risk:
            return False, "portfolio_risk_exceeded"
        
        # Correlation kontrolü
        if await self.check_correlation_risk(active_positions, new_symbol):
            return False, "high_correlation"
        
        return True, "approved"
    
    async def update_correlations(self, symbols, session):
        """Correlation matrix güncelleme"""
        from itertools import combinations
        
        now = datetime.now()
        
        for symbol1, symbol2 in combinations(symbols, 2):
            cache_key = tuple(sorted([symbol1, symbol2]))
            
            # Cache kontrolü (1 saatte bir güncelle)
            if (cache_key in self.last_correlation_update and 
                (now - self.last_correlation_update[cache_key]).seconds < 3600):
                continue
            
            try:
                # Her iki symbol için de fiyat verisi al
                corr = await self.calculate_correlation(symbol1, symbol2, session)
                self.correlation_cache[cache_key] = corr
                self.last_correlation_update[cache_key] = now
                
            except Exception as e:
                logging.error(f"Correlation calculation error for {symbol1}-{symbol2}: {e}")
    
    async def calculate_correlation(self, symbol1, symbol2, session):
        """İki symbol arasındaki korelasyon hesaplama"""
        # Implementation for correlation calculation
        # Bu kısım API call'ları gerektirir
        pass

# 3. MEMORY MANAGEMENT
class MemoryManager:
    @staticmethod
    def cleanup_trade_history(trade_history, max_records=1000):
        """Trade history temizliği"""
        if len(trade_history) > max_records:
            # Sadece son N kayıtları tut
            return trade_history[-max_records:]
        return trade_history
    
    @staticmethod
    def cleanup_model_cache(models_dict, max_age_hours=24):
        """Model cache temizliği"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for symbol, (model, timestamp) in models_dict.items():
            if timestamp < cutoff_time:
                to_remove.append(symbol)
        
        for symbol in to_remove:
            del models_dict[symbol]
            logging.info(f"Removed cached model for {symbol}")

# 4. ENHANCED ERROR HANDLING
def safe_dataframe_operations(func):
    """DataFrame operasyonları için decorator"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Sonuç kontrolü
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    logging.warning(f"{func.__name__}: Empty DataFrame returned")
                    return pd.DataFrame()
                
                # NaN kontrolü
                if result.isnull().all().all():
                    logging.warning(f"{func.__name__}: All NaN DataFrame")
                    return pd.DataFrame()
                
                # Infinite values kontrolü
                if np.isinf(result.select_dtypes(include=[np.number])).any().any():
                    logging.warning(f"{func.__name__}: Infinite values detected")
                    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return pd.DataFrame() if 'DataFrame' in str(type(e)) else None
    
    return wrapper

# 5. API RATE LIMITER
class APIRateLimiter:
    def __init__(self, max_requests_per_minute=1000):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Rate limit kontrolü"""
        async with self.lock:
            now = time.time()
            
            # Son 1 dakikadaki istekleri filtrele
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # Limit kontrolü
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    logging.warning(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self.requests.append(now)

# 6. ENHANCED CONFIGURATION VALIDATION
def validate_configuration():
    """Konfigürasyon doğrulama"""
    errors = []
    
    if config.FR_LONG_THRESHOLD >= 0:
        errors.append("FR_LONG_THRESHOLD should be negative for long positions")
    
    if config.AI_CONFIDENCE_THRESHOLD < 0.5 or config.AI_CONFIDENCE_THRESHOLD > 1.0:
        errors.append("AI_CONFIDENCE_THRESHOLD should be between 0.5 and 1.0")
    
    if config.MAX_RISK_PERCENT > 0.1:
        errors.append("MAX_RISK_PERCENT should not exceed 10%")
    
    if config.BATCH_SIZE > 50:
        errors.append("BATCH_SIZE too large, may cause rate limiting")
    
    if errors:
        for error in errors:
            logging.error(f"Configuration error: {error}")
        raise ValueError("Invalid configuration")

# 7. ROBUST TIMEZONE HANDLING
def ensure_timezone(dt):
    """Timezone'ın her zaman belirtildiğinden emin ol"""
    if dt.tzinfo is None:
        return IST_TIMEZONE.localize(dt)
    return dt.astimezone(IST_TIMEZONE)

# 8. DATABASE INTEGRATION (Optional)
class DatabaseManager:
    def __init__(self, db_path="trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """SQLite database initialization"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                confidence REAL,
                funding_rate REAL,
                exit_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                daily_pnl REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                balance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

# 9. SIGNAL QUALITY METRICS
class SignalQualityAnalyzer:
    @staticmethod
    def analyze_signal_quality(df_featured, funding_rate, confidence):
        """Signal kalitesi analizi"""
        score = 0
        max_score = 100
        
        # Funding rate score (40 points)
        if funding_rate < -0.15:
            score += 40
        elif funding_rate < -0.08:
            score += 25
        elif funding_rate < -0.05:
            score += 15
        
        # AI confidence score (30 points)
        if confidence > 0.9:
            score += 30
        elif confidence > 0.8:
            score += 25
        elif confidence > 0.7:
            score += 20
        
        # Technical indicators score (30 points)
        if not df_featured.empty:
            latest = df_featured.iloc[-1]
            
            # RSI oversold
            if latest.get('rsi', 50) < 30:
                score += 10
            
            # Volume spike
            if latest.get('volume_ratio', 1) > 2:
                score += 10
            
            # Low volatility (better for funding plays)
            if latest.get('volatility_regime', 1) == 0:
                score += 10
        
        return min(score, max_score)


class TradeMetrics:
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    confidence: float
    funding_rate: float
    stop_loss: float
    take_profit: float
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None

class SymbolInfo:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_precision = 4
        self.quantity_precision = 1
        self.tick_size = 0.0001
        self.step_size = 0.001
        self.min_quantity = 0.001
        self.min_notional = 5.0

class EnsemblePredictor:
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble of models with different strengths"""
        self.models = {
            'xgb': xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss',
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lgb': LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        }
        
        # Train models and calculate weights based on validation performance
        tscv = TimeSeriesSplit(n_splits=3)
        model_scores = {}
        
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                scores.append(precision_score(y_val, pred, zero_division=0))
            
            model_scores[name] = np.mean(scores)
            # Final fit on all data
            model.fit(X, y)
        
        # Calculate weights based on performance
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.model_weights = {name: score/total_score for name, score in model_scores.items()}
        else:
            self.model_weights = {name: 1/len(self.models) for name in self.models.keys()}
        
        self.is_trained = True
        logging.info(f"Ensemble trained with weights: {self.model_weights}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions = []
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)
            if pred_proba.shape[1] > 1:
                predictions.append(pred_proba[:, 1] * self.model_weights[name])
            else:
                predictions.append(np.zeros(X.shape[0]))
        
        if not predictions:
            return np.zeros((X.shape[0], 2))
        
        ensemble_pred = np.sum(predictions, axis=0)
        # Return as probability matrix
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

class MarketRegimeDetector:
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> str:
        """Detect current market regime"""
        recent_data = df.tail(50)
        
        # Volatility regime
        current_vol = recent_data['atr_percent'].iloc[-1]
        avg_vol = recent_data['atr_percent'].mean()
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        # Trend regime
        sma_short = recent_data['close'].rolling(10).mean().iloc[-1]
        sma_long = recent_data['close'].rolling(30).mean().iloc[-1]
        trend_strength = abs(sma_short - sma_long) / sma_long if sma_long > 0 else 0
        
        if vol_ratio > 1.5:
            return "HIGH_VOLATILITY"
        elif vol_ratio < 0.7:
            return "LOW_VOLATILITY"
        elif trend_strength > 0.05:
            return "TRENDING"
        else:
            return "RANGING"

class EnhancedAITrader:
    def __init__(self, telegram_token: str, chat_id: str):
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        
        # Enhanced model storage
        self.ensemble_models: Dict[str, EnsemblePredictor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.symbol_info: Dict[str, SymbolInfo] = {}
        self.last_model_update: Dict[str, datetime] = {}
        
        # Trading state
        self.active_positions: Dict[str, TradeMetrics] = {}
        self.paper_positions: Dict[str, TradeMetrics] = {}
        self.all_symbols: List[str] = []
        self.symbol_correlations: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.trade_history: List[TradeMetrics] = []
        self.daily_pnl: Dict[str, float] = {}
        self.paper_balance = config.PAPER_BALANCE
        
        # Communication
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        
        # Enhanced features list
        self.required_features = [
            'fundingRate', 'funding_rate_momentum', 'fr_trend', 'fr_volatility',
            'rsi', 'rsi_divergence', 'macd_diff', 'macd_signal_cross',
            'bb_width', 'bb_position', 'atr_percent', 'atr_trend',
            'volume_ratio', 'volume_trend', 'price_momentum',
            'volatility_regime', 'market_hour'
        ]
        
        self.model_update_interval = timedelta(hours=config.MODEL_UPDATE_HOURS)
        
        # Risk management
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.consecutive_losses = 0
        self.last_loss_reset = datetime.now(IST_TIMEZONE)

        # Yeni componentler
        self.risk_manager = PortfolioRiskManager()
        self.rate_limiter = APIRateLimiter()
        self.memory_manager = MemoryManager()
        self.db_manager = DatabaseManager()  # Optional
    
        # Configuration validation
        validate_configuration()

    async def initialize(self):
        """Initialize the trading system"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30, connect=10)
        )
        await self.update_and_filter_symbols()
        await self.load_initial_positions()
        await self.load_trade_history()
        
        mode_msg = "📝 PAPER TRADING" if config.TRADING_MODE == TradingMode.PAPER else "🔴 LIVE TRADING"
        await self.send_telegram(f"🤖 <b>Enhanced FR AI Hunter Başlatıldı!</b>\n\n{mode_msg} Modunda")

    async def close(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()
        await self.save_performance_data()

    async def _api_request_with_retry(self, method: str, path: str, params: Optional[Dict] = None, 
                                    signed: bool = False, max_retries: int = 3) -> Optional[Dict]:
        """Enhanced API request with retry logic"""
        params = params or {}
        
        for attempt in range(max_retries):
            try:
                if signed:
                    params['timestamp'] = int(time.time() * 1000)
                    query_string = urllib.parse.urlencode(params, True)
                    params['signature'] = hmac.new(
                        BINANCE_API_SECRET.encode('utf-8'),
                        msg=query_string.encode('utf-8'),
                        digestmod=hashlib.sha256
                    ).hexdigest()
                
                url = BASE_URL + path
                
                async with self.session.request(
                    method.upper(), url, params=params, headers=self.headers
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:  # Rate limit
                        retry_after = int(resp.headers.get('Retry-After', 60))
                        logging.warning(f"Rate limit hit, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    elif resp.status in [500, 502, 503, 504] and attempt < max_retries - 1:
                        # Server errors, retry with exponential backoff
                        wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logging.warning(f"API Error: {method} {path} | Status: {resp.status}")
                        if attempt == max_retries - 1:
                            return None
                        
            except aiohttp.ClientTimeout:
                logging.warning(f"Timeout: {method} {path} | Attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as e:
                logging.error(f"Network error: {method} {path} | {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)
        
        return None

    async def send_telegram(self, message: str):
        """Send telegram message with retry"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            params = {
                "chat_id": self.chat_id,
                "text": message[:4000],  # Telegram limit
                "parse_mode": "HTML"
            }
            async with self.session.post(url, json=params) as resp:
                if resp.status != 200:
                    logging.error(f"Telegram error: {resp.status}")
        except Exception as e:
            logging.error(f"Telegram send error: {e}")

    def add_advanced_features(self, df: pd.DataFrame, current_fr: float = 0, momentum: float = 0) -> pd.DataFrame:
        """Add comprehensive technical indicators and features"""
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        
        df_copy = df.copy().dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        if len(df_copy) < 20:
            return pd.DataFrame()
            
        close = df_copy['close']
        high, low = df_copy['high'], df_copy['low']
        volume = df_copy['volume']
        
        try:
            # Funding Rate Features
            df_copy['fundingRate'] = current_fr
            df_copy['funding_rate_momentum'] = momentum
            df_copy['fr_trend'] = df_copy['fundingRate'].rolling(3).mean() - df_copy['fundingRate'].rolling(10).mean()
            df_copy['fr_volatility'] = df_copy['fundingRate'].rolling(10).std()
            
            # Enhanced RSI
            df_copy['rsi'] = ta.momentum.rsi(close, window=14)
            df_copy['rsi_divergence'] = (df_copy['rsi'].diff() * close.pct_change()).rolling(5).mean()
            
            # Enhanced MACD
            macd_line = ta.trend.macd(close)
            macd_signal = ta.trend.macd_signal(close)
            df_copy['macd_diff'] = macd_line - macd_signal
            df_copy['macd_signal_cross'] = ((macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))).astype(int)
            
            # Enhanced Bollinger Bands
            bb = ta.volatility.BollingerBands(close)
            df_copy['bb_width'] = bb.bollinger_wband()
            df_copy['bb_position'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            
            # Enhanced ATR
            atr = ta.volatility.average_true_range(high, low, close)
            df_copy['atr_percent'] = (atr / close) * 100
            df_copy['atr_trend'] = df_copy['atr_percent'].rolling(5).mean() - df_copy['atr_percent'].rolling(20).mean()
            
            # Volume Analysis
            volume_sma = volume.rolling(20).mean()
            df_copy['volume_ratio'] = volume / volume_sma
            df_copy['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
            
            # Price Momentum
            df_copy['price_momentum'] = close.pct_change(5) * 100
            
            # Market Regime
            current_vol = df_copy['atr_percent'].iloc[-1] if not df_copy['atr_percent'].empty else 0
            avg_vol = df_copy['atr_percent'].rolling(20).mean().iloc[-1] if not df_copy['atr_percent'].empty else 1
            df_copy['volatility_regime'] = (current_vol > avg_vol * 1.2).astype(int)
            
            # Time Features
            df_copy['market_hour'] = (datetime.now(IST_TIMEZONE).hour % 24) / 24
            
            # Fill any remaining NaN values
            df_copy = df_copy.fillna(method='ffill').fillna(0)
            
            return df_copy.dropna()
            
        except Exception as e:
            logging.error(f"Feature engineering error: {e}")
            return pd.DataFrame()

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        ticker = await self._api_request_with_retry('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
        return float(ticker['price']) if ticker else None

    def calculate_position_size(self, balance: float, confidence: float, volatility: float, current_price: float) -> float:
        """Dynamic position sizing based on confidence and market conditions"""
        # Base risk adjustment
        base_risk = config.BASE_RISK_PERCENT
        
        # Confidence multiplier (higher confidence = larger position)
        confidence_multiplier = min(confidence * 1.5, 2.0)
        
        # Volatility adjustment (higher volatility = smaller position)
        volatility_adjustment = max(0.5, 2.0 - (volatility / 2.0))
        
        # Calculate final risk percentage
        risk_percent = min(
            base_risk * confidence_multiplier * volatility_adjustment,
            config.MAX_RISK_PERCENT
        )
        
        # Convert to position size
        risk_amount = balance * risk_percent
        return risk_amount

    def is_optimal_funding_window(self, next_funding_time: int) -> Tuple[bool, str]:
        """Enhanced timing analysis for funding rate trades"""
        current_time = int(time.time() * 1000)
        time_to_funding = (next_funding_time - current_time) / 60000  # minutes
        
        # Avoid funding payment periods (30 min before and 10 min after)
        if time_to_funding < config.MIN_TIME_BEFORE_FUNDING:
            return False, "too_close_to_funding"
        
        if time_to_funding > config.MAX_TIME_BEFORE_FUNDING:
            return False, "too_far_from_funding"
        
        # Sniper mode - very close to funding (higher urgency)
        funding_cycle_minutes = 8 * 60  # 8 hours
        time_since_last_funding = funding_cycle_minutes - time_to_funding
        
        if time_since_last_funding <= config.SNIPER_MODE_WINDOW:
            return True, "sniper_mode"
        
        return True, "standard_mode"

    async def check_correlation_risk(self, symbol: str) -> bool:
        """Check if adding this position would create too much correlation risk"""
        if len(self.active_positions) == 0:
            return True
        
        # Simple correlation check based on symbol patterns
        active_symbols = list(self.active_positions.keys())
        
        # Group symbols by base asset
        symbol_base = symbol.replace('USDT', '')
        for active_symbol in active_symbols:
            active_base = active_symbol.replace('USDT', '')
            
            # Check for high correlation (same base asset or known correlations)
            if symbol_base == active_base:
                return False
            
            # Additional correlation rules
            btc_correlated = ['ETH', 'BNB', 'ADA', 'DOT', 'LINK']
            if symbol_base in btc_correlated and active_base in btc_correlated:
                return False
        
        return True

    async def validate_model_performance(self, symbol: str) -> bool:
        """Validate model performance before using for live trading"""
        if symbol not in self.ensemble_models:
            return False
        
        # Check if we have performance data
        perf_file = os.path.join(PERFORMANCE_DIR, f"{symbol}_performance.json")
        if not os.path.exists(perf_file):
            return True  # Allow first-time usage
        
        try:
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
            
            # Check minimum requirements
            if perf_data.get('total_trades', 0) < 5:
                return True  # Not enough data yet
            
            win_rate = perf_data.get('win_rate', 0)
            avg_return = perf_data.get('avg_return', 0)
            
            # Minimum performance thresholds
            if win_rate < 0.4 or avg_return < -0.02:  # Less than 40% win rate or -2% avg return
                logging.warning(f"[{symbol}] Model performance below threshold: WR={win_rate:.2%}, AR={avg_return:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating model performance for {symbol}: {e}")
            return True  # Default to allowing trade if validation fails

    async def get_or_train_ensemble_model(self, symbol: str) -> bool:
        """Get or train ensemble model for symbol"""
        now = datetime.now(IST_TIMEZONE)
        
        # Check if model exists and is recent
        if (symbol in self.ensemble_models and 
            symbol in self.last_model_update and 
            (now - self.last_model_update[symbol]) < self.model_update_interval):
            return True

        # Try to load existing model
        model_path = os.path.join(MODEL_DIR, f"{symbol}_ensemble.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.ensemble_models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                self.last_model_update[symbol] = datetime.fromtimestamp(
                    os.path.getmtime(model_path), tz=IST_TIMEZONE
                )
                
                if (now - self.last_model_update[symbol]) < self.model_update_interval:
                    logging.info(f"[{symbol}] Loaded existing ensemble model")
                    return True
            except Exception as e:
                logging.warning(f"[{symbol}] Error loading saved model: {e}")

        # Train new model
        logging.info(f"[{symbol}] Training new ensemble model...")
        
        # Get historical data
        klines = await self._api_request_with_retry(
            'GET', '/fapi/v1/klines',
            {'symbol': symbol, 'interval': '5m', 'limit': 1000}
        )
        
        if not klines or len(klines) < config.MIN_TRAINING_SAMPLES:
            logging.warning(f"[{symbol}] Insufficient data for training")
            return False

        # Prepare data
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df[['open', 'high', 'low', 'close', 'volume']] = df[
            ['open', 'high', 'low', 'close', 'volume']
        ].apply(pd.to_numeric, errors='coerce')
        
        # Add features
        df_featured = self.add_advanced_features(df)
        if df_featured.empty or len(df_featured) < 100:
            logging.warning(f"[{symbol}] Feature engineering failed")
            return False

        # Create target variable (future price increase)
        df_featured['future_high'] = df_featured['high'].rolling(
            window=config.N_FUTURE_CANDLES
        ).max().shift(-config.N_FUTURE_CANDLES)
        
        df_featured['target'] = (
            df_featured['future_high'] > df_featured['close'] * 1.008
        ).astype(int)
        
        # Prepare training data
        df_train = df_featured.dropna(subset=self.required_features + ['target']).copy()
        
        if len(df_train) < config.MIN_TRAINING_SAMPLES or df_train['target'].sum() < 10:
            logging.warning(f"[{symbol}] Insufficient positive samples for training")
            return False

        # Train ensemble
        try:
            features = df_train[self.required_features]
            targets = df_train['target']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features.values)
            
            # Train ensemble
            ensemble = EnsemblePredictor()
            ensemble.fit(X_scaled, targets.values)
            
            # Save models
            self.ensemble_models[symbol] = ensemble
            self.scalers[symbol] = scaler
            self.last_model_update[symbol] = now
            
            joblib.dump(ensemble, model_path)
            joblib.dump(scaler, scaler_path)
            
            logging.info(f"[{symbol}] Ensemble model trained and saved successfully")
            return True
            
        except Exception as e:
            logging.error(f"[{symbol}] Model training failed: {e}")
            return False

    async def execute_market_order_with_protection(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """Execute market order with slippage protection"""
        try:
            # Get pre-trade price
            pre_price = await self.get_current_price(symbol)
            if not pre_price:
                logging.error(f"[{symbol}] Could not get pre-trade price")
                return None
            
            # Execute order
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity
            }
            
            order = await self._api_request_with_retry('POST', '/fapi/v1/order', order_params, signed=True)
            
            if not order or 'orderId' not in order:
                logging.error(f"[{symbol}] Market order failed")
                return None
            
            # Wait for fill
            await asyncio.sleep(1)
            
            # Check fill
            filled_order = await self._api_request_with_retry(
                'GET', '/fapi/v1/order',
                {'symbol': symbol, 'orderId': order['orderId']},
                signed=True
            )
            
            if not filled_order or filled_order.get('status') != 'FILLED':
                logging.error(f"[{symbol}] Order not filled: {filled_order}")
                return None
            
            # Check slippage
            avg_price = float(filled_order['avgPrice'])
            slippage = abs(avg_price - pre_price) / pre_price
            
            if slippage > config.MAX_SLIPPAGE_TOLERANCE:
                logging.warning(f"[{symbol}] High slippage detected: {slippage:.3%}")
                # In real scenario, might want to close position immediately
            
            return filled_order
            
        except Exception as e:
            logging.error(f"[{symbol}] Order execution error: {e}")
            return None

    async def update_and_filter_symbols(self):
        """Update symbol list with enhanced filtering"""
        logging.info("Updating symbol list...")
        
        data = await self._api_request_with_retry('GET', '/fapi/v1/exchangeInfo')
        if not data:
            self.all_symbols = []
            return
        
       # Filter symbols
        self.all_symbols = []
        valid_symbols = data.get('symbols', [])
        
        for symbol_data in valid_symbols:
            symbol = symbol_data['symbol']
            
            # Basic filters
            if (symbol.endswith('USDT') and 
                symbol_data['status'] == 'TRADING' and
                symbol not in BLACKLISTED_SYMBOLS):
                
                # Additional filters
                filters = {f['filterType']: f for f in symbol_data['filters']}
                
                # Check minimum notional
                min_notional_filter = filters.get('MIN_NOTIONAL')
                if min_notional_filter and float(min_notional_filter['minNotional']) > 100:
                    continue
                
                # Store symbol info
                symbol_info = SymbolInfo(symbol)
                
                # Get precision info
                symbol_info.price_precision = symbol_data['pricePrecision']
                symbol_info.quantity_precision = symbol_data['quantityPrecision']
                
                # Get tick size and step size
                price_filter = filters.get('PRICE_FILTER')
                if price_filter:
                    symbol_info.tick_size = float(price_filter['tickSize'])
                
                lot_size_filter = filters.get('LOT_SIZE')
                if lot_size_filter:
                    symbol_info.step_size = float(lot_size_filter['stepSize'])
                    symbol_info.min_quantity = float(lot_size_filter['minQty'])
                
                if min_notional_filter:
                    symbol_info.min_notional = float(min_notional_filter['minNotional'])
                
                self.symbol_info[symbol] = symbol_info
                self.all_symbols.append(symbol)
        
        logging.info(f"Filtered symbols: {len(self.all_symbols)} valid futures symbols")

    async def load_initial_positions(self):
        """Load existing positions"""
        if config.TRADING_MODE == TradingMode.LIVE:
            positions = await self._api_request_with_retry('GET', '/fapi/v2/positionRisk', signed=True)
            if positions:
                for pos in positions:
                    if float(pos['positionAmt']) != 0:
                        symbol = pos['symbol']
                        # Create position record if not exists
                        if symbol not in self.active_positions:
                            self.active_positions[symbol] = TradeMetrics(
                                symbol=symbol,
                                entry_time=datetime.now(IST_TIMEZONE),
                                entry_price=float(pos['entryPrice']),
                                quantity=abs(float(pos['positionAmt'])),
                                confidence=0.5,  # Unknown confidence for existing positions
                                funding_rate=0.0,
                                stop_loss=0.0,
                                take_profit=0.0
                            )
                logging.info(f"Loaded {len(self.active_positions)} existing positions")

    async def load_trade_history(self):
        """Load trade history from file"""
        history_file = os.path.join(PERFORMANCE_DIR, "trade_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for trade_data in history_data:
                    # Convert to TradeMetrics object
                    trade = TradeMetrics(
                        symbol=trade_data['symbol'],
                        entry_time=datetime.fromisoformat(trade_data['entry_time']),
                        entry_price=trade_data['entry_price'],
                        quantity=trade_data['quantity'],
                        confidence=trade_data['confidence'],
                        funding_rate=trade_data['funding_rate'],
                        stop_loss=trade_data['stop_loss'],
                        take_profit=trade_data['take_profit']
                    )
                    
                    if trade_data.get('exit_time'):
                        trade.exit_time = datetime.fromisoformat(trade_data['exit_time'])
                        trade.exit_price = trade_data.get('exit_price')
                        trade.pnl = trade_data.get('pnl')
                        trade.exit_reason = trade_data.get('exit_reason')
                    
                    self.trade_history.append(trade)
                
                logging.info(f"Loaded {len(self.trade_history)} historical trades")
            except Exception as e:
                logging.error(f"Error loading trade history: {e}")

    async def save_performance_data(self):
        """Save performance data to files"""
        try:
            # Save trade history
            history_file = os.path.join(PERFORMANCE_DIR, "trade_history.json")
            history_data = []
            
            for trade in self.trade_history:
                trade_dict = {
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time.isoformat(),
                    'entry_price': trade.entry_price,
                    'quantity': trade.quantity,
                    'confidence': trade.confidence,
                    'funding_rate': trade.funding_rate,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit
                }
                
                if trade.exit_time:
                    trade_dict.update({
                        'exit_time': trade.exit_time.isoformat(),
                        'exit_price': trade.exit_price,
                        'pnl': trade.pnl,
                        'exit_reason': trade.exit_reason
                    })
                
                history_data.append(trade_dict)
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Save daily PnL
            pnl_file = os.path.join(PERFORMANCE_DIR, "daily_pnl.json")
            with open(pnl_file, 'w') as f:
                json.dump(self.daily_pnl, f, indent=2)
            
            logging.info("Performance data saved")
        except Exception as e:
            logging.error(f"Error saving performance data: {e}")

    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Comprehensive symbol analysis"""
        try:
            # Get historical data
            klines = await self._api_request_with_retry(
                'GET', '/fapi/v1/klines',
                {'symbol': symbol, 'interval': '5m', 'limit': 200}
            )
            
            if not klines or len(klines) < 50:
                return None
            
            # Get funding rate info
            funding_info = await self._api_request_with_retry(
                'GET', '/fapi/v1/premiumIndex',
                {'symbol': symbol}
            )
            
            if not funding_info:
                return None
            
            current_fr = float(funding_info['lastFundingRate'])
            next_funding_time = int(funding_info['nextFundingTime'])
            
            # Timing check
            timing_ok, timing_reason = self.is_optimal_funding_window(next_funding_time)
            if not timing_ok:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df[['open', 'high', 'low', 'close', 'volume']] = df[
                ['open', 'high', 'low', 'close', 'volume']
            ].apply(pd.to_numeric, errors='coerce')
            
            # Calculate funding rate momentum
            fr_history = await self._api_request_with_retry(
                'GET', '/fapi/v1/fundingRate',
                {'symbol': symbol, 'limit': 10}
            )
            
            fr_momentum = 0
            if fr_history and len(fr_history) >= 2:
                recent_rates = [float(r['fundingRate']) for r in fr_history[-5:]]
                fr_momentum = np.mean(np.diff(recent_rates))
            
            # Add features
            df_featured = self.add_advanced_features(df, current_fr, fr_momentum)
            if df_featured.empty:
                return None
            
            # Market regime detection
            market_regime = MarketRegimeDetector.detect_regime(df_featured)
            
            # Get current market data
            current_price = float(df_featured['close'].iloc[-1])
            volatility = df_featured['atr_percent'].iloc[-1]
            
            return {
                'symbol': symbol,
                'funding_rate': current_fr,
                'fr_momentum': fr_momentum,
                'current_price': current_price,
                'volatility': volatility,
                'market_regime': market_regime,
                'timing_reason': timing_reason,
                'next_funding_time': next_funding_time,
                'df_featured': df_featured
            }
            
        except Exception as e:
            logging.error(f"[{symbol}] Analysis error: {e}")
            return None

    async def generate_trading_signal(self, analysis: Dict) -> Optional[Dict]:
        """Generate trading signal using ensemble model"""
        symbol = analysis['symbol']
        funding_rate = analysis['funding_rate']
        df_featured = analysis['df_featured']
        
        # Funding rate filter
        if funding_rate >= config.FR_LONG_THRESHOLD:
            return None
        
        # Get or train model
        if not await self.get_or_train_ensemble_model(symbol):
            return None
        
        # Validate model performance
        if not await self.validate_model_performance(symbol):
            return None
        
        # Prepare features
        try:
            latest_features = df_featured[self.required_features].iloc[-1:].values
            
            if symbol not in self.scalers:
                logging.error(f"[{symbol}] Scaler not found")
                return None
            
            features_scaled = self.scalers[symbol].transform(latest_features)
            
            # Get prediction
            pred_proba = self.ensemble_models[symbol].predict_proba(features_scaled)
            confidence = pred_proba[0][1] if len(pred_proba[0]) > 1 else 0
            
            # Confidence thresholds
            min_confidence = (config.MIN_CONFIDENCE_FOR_EXTREME 
                            if funding_rate <= config.FR_EXTREME_THRESHOLD 
                            else config.AI_CONFIDENCE_THRESHOLD)
            
            if confidence < min_confidence:
                return None
            
            # Risk management checks
            if not await self.check_correlation_risk(symbol):
                return None
            
            if len(self.active_positions) >= config.MAX_ACTIVE_TRADES:
                return None
            
            return {
                'symbol': symbol,
                'side': 'BUY',  # Long position for negative funding
                'confidence': confidence,
                'funding_rate': funding_rate,
                'current_price': analysis['current_price'],
                'volatility': analysis['volatility'],
                'market_regime': analysis['market_regime'],
                'timing_reason': analysis['timing_reason']
            }
            
        except Exception as e:
            logging.error(f"[{symbol}] Signal generation error: {e}")
            return None

    async def execute_trade(self, signal: Dict) -> bool:
        """Execute trade based on signal"""
        symbol = signal['symbol']
        confidence = signal['confidence']
        current_price = signal['current_price']
        volatility = signal['volatility']
        
        try:
            # Calculate position size
            balance = (self.paper_balance if config.TRADING_MODE == TradingMode.PAPER 
                      else await self.get_account_balance())
            
            if not balance or balance < 100:
                logging.warning(f"Insufficient balance: {balance}")
                return False
            
            position_value = self.calculate_position_size(balance, confidence, volatility, current_price)
            quantity = position_value / current_price
            
            # Round quantity according to symbol precision
            if symbol in self.symbol_info:
                precision = self.symbol_info[symbol].quantity_precision
                quantity = round(quantity, precision)
                
                # Check minimum quantity
                if quantity < self.symbol_info[symbol].min_quantity:
                    logging.warning(f"[{symbol}] Quantity too small: {quantity}")
                    return False
            
            # Calculate stop loss and take profit
            stop_loss_pct = max(0.015, volatility * 0.5)  # Dynamic stop loss
            take_profit_pct = stop_loss_pct * 2.5  # 2.5:1 risk-reward
            
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
            
            # Execute trade
            if config.TRADING_MODE == TradingMode.PAPER:
                # Paper trading
                self.paper_balance -= position_value
                executed = True
                order_id = f"PAPER_{int(time.time())}"
            else:
                # Live trading
                order = await self.execute_market_order_with_protection(symbol, 'BUY', quantity)
                executed = order is not None
                order_id = order.get('orderId') if order else None
                current_price = float(order['avgPrice']) if order else current_price
            
            if executed:
                # Create trade record
                trade = TradeMetrics(
                    symbol=symbol,
                    entry_time=datetime.now(IST_TIMEZONE),
                    entry_price=current_price,
                    quantity=quantity,
                    confidence=confidence,
                    funding_rate=signal['funding_rate'],
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                self.active_positions[symbol] = trade
                
                # Send notification
                await self.send_telegram(
                    f"🟢 <b>LONG POZİSYON AÇILDI</b>\n\n"
                    f"📊 <b>{symbol}</b>\n"
                    f"💰 Giriş: ${current_price:.4f}\n"
                    f"📦 Miktar: {quantity:.3f}\n"
                    f"🎯 Güven: {confidence:.1%}\n"
                    f"📈 FR: {signal['funding_rate']:.4%}\n"
                    f"🛑 Stop: ${stop_loss:.4f}\n"
                    f"🎯 Hedef: ${take_profit:.4f}\n"
                    f"⚡ Mod: {config.TRADING_MODE.value.upper()}"
                )
                
                logging.info(f"[{symbol}] Trade executed: ${current_price:.4f} | Qty: {quantity:.3f}")
                return True
            
        except Exception as e:
            logging.error(f"[{symbol}] Trade execution error: {e}")
        
        return False

    async def enhanced_execute_trade(self, signal: Dict) -> bool:
        # Rate limiting
        await self.rate_limiter.acquire()
    
        # Risk kontrolü
        risk_ok, reason = await self.risk_manager.check_portfolio_risk(
            self.active_positions, signal['symbol'], signal.get('risk_amount', 0)
        )
    
        if not risk_ok:
            logging.info(f"Trade rejected: {reason}")
            return False
    
        # Signal quality check
        quality_score = SignalQualityAnalyzer.analyze_signal_quality(
        signal.get('df_featured'), signal['funding_rate'], signal['confidence']
        )
    
        if quality_score < 60:  # Minimum quality threshold
            logging.info(f"Signal quality too low: {quality_score}")
            return False
    
        # Proceed with trade execution
        return await self.execute_trade(signal) 

    async def get_account_balance(self) -> float:
        """Get account balance"""
        try:
            account = await self._api_request_with_retry('GET', '/fapi/v2/account', signed=True)
            if account:
                return float(account['totalWalletBalance'])
        except Exception as e:
            logging.error(f"Error getting balance: {e}")
        return 0.0

    async def manage_active_positions(self):
        """Manage active positions with trailing stops and exit conditions"""
        if not self.active_positions:
            return
        
        positions_to_close = []
        
        for symbol, trade in list(self.active_positions.items()):
            try:
                current_price = await self.get_current_price(symbol)
                if not current_price:
                    continue
                
                # Calculate current P&L
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                pnl_dollar = pnl_pct * trade.quantity * trade.entry_price
                
                # Update max profit and drawdown
                trade.max_profit = max(trade.max_profit, pnl_pct)
                trade.max_drawdown = min(trade.max_drawdown, pnl_pct)
                
                # Exit conditions
                exit_reason = None
                should_exit = False
                
                # Stop loss hit
                if current_price <= trade.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # Take profit hit
                elif current_price >= trade.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                
                # Time-based exit (24 hours max)
                elif (datetime.now(IST_TIMEZONE) - trade.entry_time).total_seconds() > 86400:
                    should_exit = True
                    exit_reason = "time_exit"
                
                # Trailing stop (if profit > 2%)
                elif trade.max_profit > 0.02:
                    trailing_stop = current_price * 0.985  # 1.5% trailing
                    if trailing_stop > trade.stop_loss:
                        trade.stop_loss = trailing_stop
                        logging.info(f"[{symbol}] Trailing stop updated: ${trailing_stop:.4f}")
                
                # Emergency exit on large drawdown
                elif pnl_pct < -0.05:  # -5% emergency exit
                    should_exit = True
                    exit_reason = "emergency_exit"
                
                if should_exit:
                    positions_to_close.append((trade, exit_reason, current_price, pnl_dollar))
                
            except Exception as e:
                logging.error(f"[{symbol}] Position management error: {e}")
        
        # Close positions
        for trade, exit_reason, exit_price, pnl_dollar in positions_to_close:
            await self.close_position(trade, exit_reason, exit_price, pnl_dollar)

    async def close_position(self, trade: TradeMetrics, exit_reason: str, exit_price: float, pnl_dollar: float):
        """Close position and update records"""
        symbol = trade.symbol
        
        try:
            # Execute close order
            if config.TRADING_MODE == TradingMode.PAPER:
                # Paper trading
                self.paper_balance += (trade.quantity * exit_price)
                executed = True
            else:
                # Live trading
                order = await self.execute_market_order_with_protection(symbol, 'SELL', trade.quantity)
                executed = order is not None
                if order:
                    exit_price = float(order['avgPrice'])
            
            if executed:
                # Update trade record
                trade.exit_time = datetime.now(IST_TIMEZONE)
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl = pnl_dollar
                
                # Move to history
                self.trade_history.append(trade)
                del self.active_positions[symbol]
                
                # Update daily PnL
                today = datetime.now(IST_TIMEZONE).strftime('%Y-%m-%d')
                self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl_dollar
                
                # Send notification
                pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                emoji = "🟢" if pnl_dollar > 0 else "🔴"
                
                await self.send_telegram(
                    f"{emoji} <b>POZİSYON KAPATILDI</b>\n\n"
                    f"📊 <b>{symbol}</b>\n"
                    f"💰 Giriş: ${trade.entry_price:.4f}\n"
                    f"💰 Çıkış: ${exit_price:.4f}\n"
                    f"📈 P&L: ${pnl_dollar:.2f} ({pnl_pct:.2%})\n"
                    f"⏰ Süre: {trade.exit_time - trade.entry_time}\n"
                    f"🏷️ Sebep: {exit_reason}\n"
                    f"💼 Günlük: ${self.daily_pnl.get(today, 0):.2f}"
                )
                
                logging.info(f"[{symbol}] Position closed: {exit_reason} | P&L: ${pnl_dollar:.2f}")
                
        except Exception as e:
            logging.error(f"[{symbol}] Position close error: {e}")

    async def scan_markets(self):
        """Scan markets for trading opportunities"""
        logging.info("🔍 Scanning markets...")
        
        opportunities = []
        
        # Process symbols in batches
        for i in range(0, len(self.all_symbols), config.BATCH_SIZE):
            batch = self.all_symbols[i:i + config.BATCH_SIZE]
            
            # Analyze batch
            tasks = [self.analyze_symbol(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    # Generate signal
                    signal = await self.generate_trading_signal(result)
                    if signal:
                        opportunities.append(signal)
            
            # Rate limiting
            await asyncio.sleep(config.BATCH_DELAY)
        
        # Sort by confidence and funding rate
        opportunities.sort(key=lambda x: (x['confidence'], abs(x['funding_rate'])), reverse=True)
        
        # Execute top opportunities
        executed_trades = 0
        for signal in opportunities:
            if executed_trades >= config.MAX_ACTIVE_TRADES - len(self.active_positions):
                break
                
            if await self.execute_trade(signal):
                executed_trades += 1
                await asyncio.sleep(2)  # Delay between trades
        
        if opportunities:
            logging.info(f"Found {len(opportunities)} opportunities, executed {executed_trades} trades")

    async def daily_report(self):
        """Generate daily performance report"""
        today = datetime.now(IST_TIMEZONE).strftime('%Y-%m-%d')
        daily_pnl = self.daily_pnl.get(today, 0)
        
        # Calculate statistics
        active_count = len(self.active_positions)
        total_trades = len(self.trade_history)
        
        if total_trades > 0:
            winning_trades = sum(1 for t in self.trade_history if t.pnl and t.pnl > 0)
            win_rate = winning_trades / total_trades
            avg_return = np.mean([t.pnl for t in self.trade_history if t.pnl])
        else:
            win_rate = 0
            avg_return = 0
        
        current_balance = (self.paper_balance if config.TRADING_MODE == TradingMode.PAPER 
                          else await self.get_account_balance())
        
        report = (
            f"📊 <b>GÜNLÜK RAPOR</b> - {today}\n\n"
            f"💰 Günlük P&L: ${daily_pnl:.2f}\n"
            f"💼 Mevcut Bakiye: ${current_balance:.2f}\n"
            f"📈 Aktif Pozisyonlar: {active_count}\n"
            f"📊 Toplam İşlem: {total_trades}\n"
            f"🎯 Kazanma Oranı: {win_rate:.1%}\n"
            f"📈 Ortalama Getiri: ${avg_return:.2f}\n"
            f"⚡ Mod: {config.TRADING_MODE.value.upper()}"
        )
        
        await self.send_telegram(report)

    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logging.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Close all positions
            for symbol in list(self.active_positions.keys()):
                trade = self.active_positions[symbol]
                current_price = await self.get_current_price(symbol)
                if current_price:
                    pnl = (current_price - trade.entry_price) / trade.entry_price * trade.quantity * trade.entry_price
                    await self.close_position(trade, "emergency_shutdown", current_price, pnl)
            
            # Save data
            await self.save_performance_data()
            
            await self.send_telegram("🚨 <b>ACİL DURUM KAPATMA</b>\n\nTüm pozisyonlar kapatıldı ve sistem durduruldu.")
            
        except Exception as e:
            logging.error(f"Emergency shutdown error: {e}")

    async def run(self):
        """Main trading loop"""
        await self.initialize()
        
        last_daily_report = datetime.now(IST_TIMEZONE).date()
        scan_counter = 0
        
        try:
            while True:
                current_time = datetime.now(IST_TIMEZONE)
                
                # Daily report
                if current_time.date() > last_daily_report:
                    await self.daily_report()
                    last_daily_report = current_time.date()
                
                # Position management
                await self.manage_active_positions()
                
                # Market scanning
                scan_counter += 1
                if scan_counter % 2 == 0:  # Every other cycle
                    await self.scan_markets()
                
                # Check for emergency conditions
                if self.consecutive_losses >= 5:
                    await self.emergency_shutdown()
                    break
                
                # Daily loss check
                today = current_time.strftime('%Y-%m-%d')
                if self.daily_pnl.get(today, 0) < -self.max_daily_loss * 1000:
                    logging.warning("Daily loss limit reached")
                    await self.send_telegram("⚠️ Günlük zarar limiti aşıldı. İşlemler durduruldu.")
                    await asyncio.sleep(3600)  # Wait 1 hour
                
                # Save performance data periodically
                if scan_counter % 10 == 0:
                    await self.save_performance_data()
                
                # Wait for next cycle
                await asyncio.sleep(config.MARKET_SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            logging.info("Shutdown signal received")
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            await self.emergency_shutdown()
        finally:
            await self.close()

async def main():
    """Main function"""
    if not all([TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
        logging.critical("Missing required environment variables!")
        return
    
    trader = EnhancedAITrader(TELEGRAM_TOKEN, CHAT_ID)
    
    try:
        await trader.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await trader.close()

if __name__ == "__main__":
    asyncio.run(main())