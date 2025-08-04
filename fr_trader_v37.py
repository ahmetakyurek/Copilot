# -*- coding: utf-8 -*-
"""
FR Hunter V15.3 - Advanced AI Trading Bot
=========================================

Features:
- Neural Networks with TensorFlow integration
- Advanced memory management with proactive cleanup
- Smart model caching with LRU eviction
- Dynamic SL/TP management
- Race condition safe operations
- Enhanced feature engineering
- Real-time system monitoring
"""

# ===== CORE IMPORTS =====
import os
import gc
import sys
import time
import hmac
import json
import hashlib
import warnings
import traceback
import urllib.parse
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, Counter

# ===== ASYNC & NETWORKING =====
import asyncio
import aiohttp

# ===== DATA PROCESSING =====
import pandas as pd
import numpy as np

# ===== MACHINE LEARNING =====
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

# ===== SYSTEM MONITORING =====
import psutil
import sqlite3

# ===== TECHNICAL ANALYSIS =====
import ta

# ===== CONFIGURATION & LOGGING =====
import pytz
import logging
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv
import joblib

# ===== OPTIONAL IMPORTS =====
try:
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    SKLEARN_FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    SKLEARN_FEATURE_SELECTION_AVAILABLE = False
    logging.warning("sklearn.feature_selection not available, using manual feature selection")

# ===== ENVIRONMENT SETUP =====
warnings.filterwarnings("ignore")
load_dotenv()

# ===== GLOBAL CONSTANTS =====
# Environment Variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Validation
if not all([BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_TOKEN, CHAT_ID]):
    print("❌ Critical environment variables are missing! Check your .env file.")
    sys.exit(1)

# Trading Configuration
IST_TIMEZONE = pytz.timezone("Europe/Istanbul")
BASE_URL = "https://fapi.binance.com"
BLACKLISTED_SYMBOLS = ['USDCUSDT', 'FDUSDUSDT', 'BTCDOMUSDT']

# Directory Structure
MODEL_DIR = "models_v15"
PERFORMANCE_DIR = "performance_v15"
DATABASE_PATH = "trades_v15.db"

# Create directories
for directory in [MODEL_DIR, PERFORMANCE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== LOGGING CONFIGURATION =====
def setup_logging():
    """Configure advanced logging with rotation"""
    logger = logging.getLogger()
    logger.setLevel(os.getenv('LOG_LEVEL', 'INFO').upper())
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler
    file_handler = TimedRotatingFileHandler(
        filename='fr_hunter_v15.log',
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized with rotation")

# Initialize logging
setup_logging()

# ===== HELPER FUNCTIONS =====
def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safe division with zero protection"""
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return fallback
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return fallback
        return result
    except:
        return fallback

def safe_percentage(part: float, whole: float, fallback: float = 0.0) -> float:
    """Safe percentage calculation"""
    return safe_divide(part, whole, fallback) * 100

# ===== DECORATORS =====
def safe_async_operation(func):
    """Decorator for safe async operations"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper

def safe_dataframe_operations(func):
    """Decorator for safe DataFrame operations"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return pd.DataFrame()
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and np.isinf(result[numeric_cols]).any().any():
                    result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return pd.DataFrame()
    return wrapper

# ===== CORE ENUMS =====
class TradingMode(Enum):
    """Trading mode enumeration"""
    LIVE = "live"
    PAPER = "paper"

class MarketRegime(Enum):
    """Market regime classification"""
    UPTREND = "Uptrend"
    DOWNTREND = "Downtrend"
    SIDEWAYS_RANGING = "Ranging"
    HIGH_VOLATILITY = "High Volatility"
    UNKNOWN = "Unknown"

 # ===== DATA CLASSES =====
@dataclass
class TradeMetrics:
    """Comprehensive trade metrics storage"""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    confidence: float
    funding_rate: float
    stop_loss: float
    take_profit: float
    volatility: float = 0.0
    market_regime: Optional[str] = None
    smart_sl_activated: bool = False
    bot_mode_on_entry: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    time_to_funding_on_entry: Optional[float] = None
    trailing_stop_activated: bool = False
    highest_price_seen: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None

@dataclass
class SymbolInfo:
    """Symbol trading information"""
    symbol: str
    price_precision: int
    quantity_precision: int
    min_notional: float

@dataclass
class SignalData:
    """Trading signal data"""
    symbol: str
    side: str
    confidence: float
    funding_rate: float
    current_price: float
    volatility: float
    market_regime: MarketRegime
    next_funding_time: Optional[int] = None

# ==================================================================
# ADIM 1: YENİ VERİ SINIFINI BURAYA EKLEYİN
# ==================================================================
@dataclass
class SignalFunnelStats:
    """Sinyal hunisindeki adayların akışını takip eder."""
    total_symbols: int = 0
    initial_fr_candidates: List[str] = field(default_factory=list)
    after_timing_cooldown: List[str] = field(default_factory=list)
    after_correlation: List[str] = field(default_factory=list)
    after_deep_analysis: List[SignalData] = field(default_factory=list)
    reasons_for_rejection: Counter = field(default_factory=Counter)

# ===== TRADING CONFIGURATION =====
class TradingConfig:
    """Centralized configuration management"""
    
    def __init__(self):
        self.load_settings_from_env()
        logging.info("Trading configuration initialized")

    def load_settings_from_env(self):
        """Load all settings from environment variables"""
        
        # ===== TRADING MODE =====
        self.TRADING_MODE = TradingMode(os.getenv('TRADING_MODE', 'live').lower())
        self.PAPER_BALANCE = float(os.getenv('PAPER_BALANCE', '100.0'))
        
        # ===== STRATEGY THRESHOLDS =====
        self.FR_LONG_THRESHOLD = float(os.getenv('FR_LONG_THRESHOLD', '-0.30'))
        self.AI_CONFIDENCE_THRESHOLD = float(os.getenv('AI_CONFIDENCE_THRESHOLD', '0.65'))
        
        # ===== RISK MANAGEMENT =====
        self.MAX_ACTIVE_TRADES = int(os.getenv('MAX_ACTIVE_TRADES', '5'))
        self.BASE_RISK_PERCENT = float(os.getenv('BASE_RISK_PERCENT', '0.01'))
        self.MAX_RISK_PERCENT = float(os.getenv('MAX_RISK_PERCENT', '0.025'))
        self.MAX_LOSS_PER_TRADE_USD = float(os.getenv('MAX_LOSS_PER_TRADE_USD', '15.0'))
        
        # ===== TIMING FILTERS =====
        self.MARKET_SCAN_INTERVAL = int(os.getenv('MARKET_SCAN_INTERVAL', '600'))
        self.MIN_TIME_BEFORE_FUNDING = int(os.getenv('MIN_TIME_BEFORE_FUNDING', '30'))
        self.MAX_TIME_BEFORE_FUNDING = int(os.getenv('MAX_TIME_BEFORE_FUNDING', '220'))
        self.WAIT_TIME_AFTER_FUNDING = int(os.getenv('WAIT_TIME_AFTER_FUNDING', '25'))
        
        # ===== DYNAMIC SETTINGS =====
        self.ENABLE_DYNAMIC_SETTINGS = os.getenv('ENABLE_DYNAMIC_SETTINGS', 'True').lower() == 'true'
        self.RELAX_MODE_START_HOUR = int(os.getenv('RELAX_MODE_START_HOUR', '4'))
        self.DEFENSIVE_MODE_START_HOUR = int(os.getenv('DEFENSIVE_MODE_START_HOUR', '12'))
        self.RELAX_FR_THRESHOLD = float(os.getenv('RELAX_FR_THRESHOLD', '-0.12'))
        self.RELAX_AI_CONFIDENCE = float(os.getenv('RELAX_AI_CONFIDENCE', '0.65'))
        self.DEFENSIVE_FR_THRESHOLD = float(os.getenv('DEFENSIVE_FR_THRESHOLD', '-0.35'))
        self.DEFENSIVE_AI_CONFIDENCE = float(os.getenv('DEFENSIVE_AI_CONFIDENCE', '0.80'))
        
        # ===== ADVANCED FEATURES =====
        self.ENABLE_COOLDOWN = os.getenv('ENABLE_COOLDOWN', 'True').lower() == 'true'
        self.COOLDOWN_PERIOD_MINUTES = int(os.getenv('COOLDOWN_PERIOD_MINUTES', '10'))
        self.ANALYSIS_REPORT_DAYS = int(os.getenv('ANALYSIS_REPORT_DAYS', '7'))
        
        # ===== TRAILING STOP =====
        self.ENABLE_TRAILING_STOP = os.getenv('ENABLE_TRAILING_STOP', 'True').lower() == 'true'
        self.TRAILING_STOP_ACTIVATION_PERCENT = float(os.getenv('TRAILING_STOP_ACTIVATION_PERCENT', '0.02'))
        self.TRAILING_STOP_DISTANCE_ATR_MULTIPLIER = float(os.getenv('TRAILING_STOP_DISTANCE_ATR_MULTIPLIER', '1.5'))
        
        # ===== HIGH CONFIDENCE SETTINGS =====
        self.HIGH_CONFIDENCE_SL_MULTIPLIER = float(os.getenv('HIGH_CONFIDENCE_SL_MULTIPLIER', '1.10'))
        self.HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.85'))
        self.DANGEROUS_COUNTDOWN_MINUTES = int(os.getenv('DANGEROUS_COUNTDOWN_MINUTES', '45'))
        
        # ===== PUMP FILTER =====
        self.ENABLE_PUMP_FILTER = os.getenv('ENABLE_PUMP_FILTER', 'True').lower() == 'true'
        self.PUMP_SENSITIVITY_UPTREND = float(os.getenv('PUMP_SENSITIVITY_UPTREND', '4.0'))
        self.PUMP_SENSITIVITY_RANGING = float(os.getenv('PUMP_SENSITIVITY_RANGING', '2.5'))
        self.PUMP_FILTER_RECENT_CANDLES = int(os.getenv('PUMP_FILTER_RECENT_CANDLES', '3'))
        self.PUMP_FILTER_LOOKBACK_CANDLES = int(os.getenv('PUMP_FILTER_LOOKBACK_CANDLES', '20'))
        
        # ===== CORRELATION FILTER =====
        self.ENABLE_CORRELATION_FILTER = os.getenv('ENABLE_CORRELATION_FILTER', 'True').lower() == 'true'
        self.MAX_CORRELATION_THRESHOLD = float(os.getenv('MAX_CORRELATION_THRESHOLD', '0.80'))
        
        # ===== AUTO TUNING =====
        self.ENABLE_AUTO_PARAM_TUNING = os.getenv('ENABLE_AUTO_PARAM_TUNING', 'False').lower() == 'true'
        self.AUTO_TUNE_INTERVAL_HOURS = int(os.getenv('AUTO_TUNE_INTERVAL_HOURS', '12'))
        self.AUTO_TUNE_MIN_TRADES = int(os.getenv('AUTO_TUNE_MIN_TRADES', '20'))
        self.AUTO_TUNE_STEP = float(os.getenv('AUTO_TUNE_STEP', '0.01'))

        # ===== TECHNICAL SETTINGS =====
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', '40'))
        self.BATCH_DELAY = float(os.getenv('BATCH_DELAY', '0.5'))
        self.MODEL_UPDATE_HOURS = int(os.getenv('MODEL_UPDATE_HOURS', '12'))
        self.MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '200'))
        self.N_FUTURE_CANDLES = int(os.getenv('N_FUTURE_CANDLES', '6'))
        
        # ===== MEMORY SETTINGS =====
        self.MAX_MODEL_CACHE_SIZE = int(os.getenv('MAX_MODEL_CACHE_SIZE', '25'))
        self.MAX_MODEL_MEMORY_MB = int(os.getenv('MAX_MODEL_MEMORY_MB', '200'))

        # ===== CACHE & HISTORY SETTINGS =====
        self.MAX_TRADE_HISTORY_SIZE = int(os.getenv('MAX_TRADE_HISTORY_SIZE', '1000'))
        self.MODEL_CACHE_MAX_AGE_HOURS = int(os.getenv('MODEL_CACHE_MAX_AGE_HOURS', '2'))
    
    def set_value(self, key: str, value_str: str) -> bool:
        """Set configuration value dynamically"""
        key = key.upper()
        if hasattr(self, key):
            try:
                attr_type = type(getattr(self, key))
                if attr_type == bool:
                    new_value = value_str.lower() in ['true', '1', 't', 'y', 'yes']
                else:
                    new_value = attr_type(value_str)
                setattr(self, key, new_value)
                logging.info(f"CONFIG UPDATED: {key} set to {new_value}")
                return True
            except Exception as e:
                logging.error(f"Failed to set {key}: {e}")
                return False
        return False

    def adjust_confidence_threshold(self, adjustment: float):
        """Adjust AI confidence threshold for auto-tuning"""
        current = config.AI_CONFIDENCE_THRESHOLD
        new = np.clip(current + adjustment, 0.50, 0.90)
        config.AI_CONFIDENCE_THRESHOLD = new
        logging.info(f"AUTO-TUNE: AI Confidence Threshold adjusted: {current:.3f} -> {new:.3f}")

# Initialize global configuration
config = TradingConfig()

# ===== EXCEPTION CLASSES =====
class TradingError(Exception):
    """Base trading exception"""
    pass

class APIError(TradingError):
    """API related errors"""
    pass

class MemoryError(TradingError):
    """Memory management errors"""
    pass

# ===== UTILITY CLASSES =====
class APIRateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 1000):
        self.max_requests = max_requests
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit permission"""
        async with self.lock:
            now = time.time()
            self.requests = [r for r in self.requests if now - r < 60]
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0])
                await asyncio.sleep(sleep_time)
            self.requests.append(time.time())

class EnhancedMemoryManager:
    """
    V15.3 - Proactive Memory Management with Smart Coordination
    
    Features:
    - Proactive cleanup before memory pressure
    - Dynamic cache sizing based on memory availability
    - Coordinated cleanup across all cache systems
    - Memory health scoring and alerts
    - Adaptive thresholds based on market activity
    - Emergency memory recovery protocols
    """
    
    def __init__(self, max_memory_mb: int = 512, bot_instance=None):
        self.max_memory_mb = max_memory_mb
        self.bot = bot_instance
        
        # Adaptive thresholds (percentage of max memory)
        self.thresholds = {
            'green': 0.60,      # < 60%: All systems go
            'yellow': 0.75,     # 60-75%: Start proactive cleanup
            'orange': 0.85,     # 75-85%: Aggressive cleanup
            'red': 0.95         # > 95%: Emergency protocols
        }
        
        # Memory zones and actions
        self.memory_zones = {
            'green': {'model_cache_size': 25, 'feature_cache_size': 100},
            'yellow': {'model_cache_size': 20, 'feature_cache_size': 75},
            'orange': {'model_cache_size': 15, 'feature_cache_size': 50},
            'red': {'model_cache_size': 10, 'feature_cache_size': 25}
        }
        
        # Monitoring state
        self.last_cleanup_time = datetime.now(IST_TIMEZONE)
        self.cleanup_interval = timedelta(minutes=5)  # Check every 5 minutes
        self.memory_history = deque(maxlen=20)  # Track last 20 readings
        self.cleanup_effectiveness = deque(maxlen=10)  # Track cleanup results
        
        # Process monitoring
        try:
            self.process = psutil.Process(os.getpid())
        except:
            self.process = None
            logging.warning("psutil not available, limited memory monitoring")
        
        # Performance metrics
        self.metrics = {
            'total_cleanups': 0,
            'proactive_cleanups': 0,
            'emergency_cleanups': 0,
            'memory_freed_mb': 0.0,
            'avg_cleanup_effectiveness': 0.0
        }
        
        logging.info(f"EnhancedMemoryManager initialized: max={max_memory_mb}MB, adaptive thresholds enabled")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Basic memory usage (compatibility with old interface)
        
        Returns:
            Dict with basic memory information for backward compatibility
        """
        try:
            detailed_stats = self.get_detailed_memory_stats()
            if 'error' in detailed_stats:
                return {'error': detailed_stats['error']}
            
            # Return in old format for compatibility
            return {
                'rss_mb': detailed_stats['rss_mb'],
                'percentage': detailed_stats['system_memory_percent'],
                'available_mb': detailed_stats['available_system_mb']
            }
            
        except Exception as e:
            logging.error(f"Error getting basic memory usage: {e}")
            return {'error': str(e)}

    def get_detailed_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        
        Returns:
            Dict with detailed memory information
        """
        try:
            if not self.process:
                return {'error': 'Process monitoring not available'}
            
            # System memory info
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            # Calculate usage metrics
            rss_mb = memory_info.rss / 1024 / 1024
            memory_percent = (rss_mb / self.max_memory_mb) * 100
            system_percent = self.process.memory_percent()
            
            # Determine current zone
            current_zone = self._get_memory_zone(memory_percent / 100)
            
            # Cache-specific memory estimates (safe access)
            model_memory = 0
            feature_memory = 0
            
            try:
                if self.bot and hasattr(self.bot, 'model_cache'):
                    model_memory = getattr(self.bot.model_cache, 'estimated_memory_mb', 0)
                if self.bot and hasattr(self.bot, 'feature_engineer'):
                    feature_engineer = self.bot.feature_engineer
                    if hasattr(feature_engineer, 'feature_cache'):
                        feature_memory = getattr(feature_engineer.feature_cache, 'estimated_memory_mb', 0)
            except Exception as cache_error:
                logging.debug(f"Cache memory estimation error: {cache_error}")
            
            stats = {
                'rss_mb': rss_mb,
                'max_memory_mb': self.max_memory_mb,
                'memory_percent': memory_percent,
                'system_memory_percent': system_percent,
                'available_system_mb': system_memory.available / 1024 / 1024,
                'current_zone': current_zone,
                'zone_thresholds': {
                    zone: self.max_memory_mb * threshold 
                    for zone, threshold in self.thresholds.items()
                },
                'cache_memory': {
                    'model_cache_mb': model_memory,
                    'feature_cache_mb': feature_memory,
                    'total_cache_mb': model_memory + feature_memory
                },
                'recommendations': self.memory_zones[current_zone],
                'metrics': self.metrics.copy()
            }
            
            # Add to history
            self.memory_history.append({
                'timestamp': datetime.now(IST_TIMEZONE),
                'rss_mb': rss_mb,
                'zone': current_zone
            })
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting detailed memory stats: {e}")
            return {'error': str(e)}
    
    def _get_memory_zone(self, usage_ratio: float) -> str:
        """Determine current memory zone based on usage ratio"""
        if usage_ratio >= self.thresholds['red']:
            return 'red'
        elif usage_ratio >= self.thresholds['orange']:
            return 'orange'
        elif usage_ratio >= self.thresholds['yellow']:
            return 'yellow'
        else:
            return 'green'
    
    async def proactive_memory_check(self) -> Dict[str, Any]:
        """
        Proactive memory monitoring and cleanup
        
        Returns:
            Dict with actions taken and results
        """
        stats = self.get_detailed_memory_stats()
        if 'error' in stats:
            return stats
        
        current_zone = stats['current_zone']
        actions_taken = []
        memory_freed = 0.0
        
        # Zone-based actions
        if current_zone == 'green':
            actions_taken.append("System healthy - no action needed")
            
        elif current_zone in ['yellow', 'orange', 'red']:
            # Basic cleanup for all non-green zones
            logging.info(f"Memory in {current_zone.upper()} zone - performing cleanup")
            
            # Force garbage collection
            gc_freed = self._force_garbage_collection()
            memory_freed += gc_freed
            actions_taken.append(f"Garbage collection: {gc_freed:.1f}MB")
            
            # Update metrics
            if current_zone in ['orange', 'red']:
                self.metrics['emergency_cleanups'] += 1
            else:
                self.metrics['proactive_cleanups'] += 1
        
        # Update metrics
        self.metrics['total_cleanups'] += 1
        self.metrics['memory_freed_mb'] += memory_freed
        self.cleanup_effectiveness.append(memory_freed)
        
        if self.cleanup_effectiveness:
            self.metrics['avg_cleanup_effectiveness'] = np.mean(self.cleanup_effectiveness)
        
        self.last_cleanup_time = datetime.now(IST_TIMEZONE)
        
        result = {
            'zone': current_zone,
            'memory_before_mb': stats['rss_mb'],
            'memory_freed_mb': memory_freed,
            'actions_taken': actions_taken,
            'effectiveness_score': min(memory_freed / 10.0, 1.0)  # 0-1 score
        }
        
        if memory_freed > 0:
            logging.info(f"Proactive memory check completed: {current_zone} zone, {memory_freed:.1f}MB freed")
        
        return result
    
    def _force_garbage_collection(self) -> float:
        """Force garbage collection and estimate memory freed"""
        memory_before = 0
        memory_after = 0
        
        try:
            if self.process:
                memory_before = self.process.memory_info().rss / 1024 / 1024
            
            # Force GC on all generations
            collected = 0
            for generation in range(3):
                collected += gc.collect(generation)
            
            if self.process:
                memory_after = self.process.memory_info().rss / 1024 / 1024
                freed = max(0, memory_before - memory_after)
            else:
                freed = max(0, collected * 0.001)  # Rough estimate
            
            logging.debug(f"Garbage collection freed ~{freed:.1f}MB, collected {collected} objects")
            return freed
            
        except Exception as e:
            logging.error(f"Error in garbage collection: {e}")
            return 0.0
    
    def should_run_check(self) -> bool:
        """Determine if memory check should run"""
        return datetime.now(IST_TIMEZONE) - self.last_cleanup_time > self.cleanup_interval
    
    def force_cleanup(self) -> float:
        """Force cleanup (compatibility method)"""
        return self._force_garbage_collection()
    
    def check_memory_health(self) -> bool:
        """Check memory health (compatibility method)"""
        try:
            stats = self.get_detailed_memory_stats()
            if 'error' in stats:
                return True  # Assume healthy if can't check
            
            return stats['current_zone'] in ['green', 'yellow']
        except:
            return True
    
    def get_memory_health_score(self) -> float:
        """
        Calculate memory health score (0-100)
        
        Returns:
            float: Health score where 100 = excellent, 0 = critical
        """
        try:
            stats = self.get_detailed_memory_stats()
            if 'error' in stats:
                return 50.0  # Unknown, assume average
            
            memory_percent = stats['memory_percent']
            
            # Base score from memory usage
            if memory_percent < 60:
                base_score = 100
            elif memory_percent < 75:
                base_score = 85
            elif memory_percent < 85:
                base_score = 70
            elif memory_percent < 95:
                base_score = 40
            else:
                base_score = 10
            
            # Adjust based on cleanup effectiveness
            if self.cleanup_effectiveness:
                avg_effectiveness = np.mean(self.cleanup_effectiveness)
                effectiveness_bonus = min(avg_effectiveness * 2, 10)  # Max 10 point bonus
                base_score += effectiveness_bonus
            
            return max(0, min(100, base_score))
            
        except Exception as e:
            logging.error(f"Error calculating memory health score: {e}")
            return 50.0
    
    async def generate_memory_report(self) -> str:
        """Generate comprehensive memory report"""
        try:
            stats = self.get_detailed_memory_stats()
            health_score = self.get_memory_health_score()
            
            if 'error' in stats:
                return f"Memory Report Error: {stats['error']}"
            
            # Zone emoji
            zone_emoji = {
                'green': '🟢',
                'yellow': '🟡', 
                'orange': '🟠',
                'red': '🔴'
            }
            
            report = f"""📊 <b>Memory Health Report</b>

            {zone_emoji.get(stats['current_zone'], '⚪')} <b>Status:</b> {stats['current_zone'].upper()} Zone
            💯 <b>Health Score:</b> {health_score:.0f}/100

            📈 <b>Memory Usage:</b>
            - Current: {stats['rss_mb']:.1f}MB ({stats['memory_percent']:.1f}%)
            - Limit: {stats['max_memory_mb']}MB
            - Available: {stats['available_system_mb']:.0f}MB system

            🗂️ <b>Cache Memory:</b>
            - Model Cache: {stats['cache_memory']['model_cache_mb']:.1f}MB
            - Feature Cache: {stats['cache_memory']['feature_cache_mb']:.1f}MB  
            - Total Cache: {stats['cache_memory']['total_cache_mb']:.1f}MB

            📊 <b>Cleanup Stats:</b>
            - Total Cleanups: {stats['metrics']['total_cleanups']}
            - Proactive: {stats['metrics']['proactive_cleanups']}
            - Emergency: {stats['metrics']['emergency_cleanups']}
            - Total Freed: {stats['metrics']['memory_freed_mb']:.1f}MB"""

            # Add trend information if available
            if len(self.memory_history) >= 3:
                recent = list(self.memory_history)[-3:]
                trend_mb = recent[-1]['rss_mb'] - recent[0]['rss_mb']
                if abs(trend_mb) > 1:
                    trend_arrow = "📈" if trend_mb > 0 else "📉"
                    report += f"\n\n{trend_arrow} <b>Trend:</b> {trend_mb:+.1f}MB (last 3 checks)"
            
            return report
            
        except Exception as e:
            return f"Error generating memory report: {e}"

# ===== TENSORFLOW INITIALIZATION =====
def initialize_tensorflow():
    """Initialize TensorFlow with proper configuration"""
    print("🔍 Starting TensorFlow import...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully: {tf.__version__}")
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.regularizers import l2
        
        # Configure TensorFlow
        tf.get_logger().setLevel('ERROR')
        
        print("🧠 Neural Networks will be ENABLED")
        logging.info(f"TensorFlow {tf.__version__} loaded successfully for Neural Networks")
        
        return True, tf
        
    except ImportError as e:
        print(f"❌ TensorFlow ImportError: {e}")
        logging.warning(f"TensorFlow not available. ImportError: {e}")
        return False, None
    except Exception as e:
        print(f"❌ TensorFlow Exception: {e}")
        logging.warning(f"TensorFlow error: {e}")
        return False, None

# Initialize TensorFlow (SADECE BİR KERE)
TENSORFLOW_AVAILABLE, tf_module = initialize_tensorflow()
print(f"🎯 TENSORFLOW_AVAILABLE = {TENSORFLOW_AVAILABLE}")
logging.info(f"TensorFlow availability: {TENSORFLOW_AVAILABLE}")

# ===== LOGGING BANNER =====
logging.info("=" * 60)
logging.info("FR HUNTER V15.3 - ADVANCED AI TRADING BOT")
logging.info("=" * 60)
logging.info(f"Trading Mode: {config.TRADING_MODE.value.upper()}")
logging.info(f"TensorFlow: {'ENABLED' if TENSORFLOW_AVAILABLE else 'DISABLED'}")
logging.info(f"Memory Management: ENHANCED")
logging.info(f"Thread Safety: RACE CONDITION PROTECTED")
logging.info("=" * 60)


class SystemHealthMonitor:
    """Bot'un sistem sağlığını ve performansını izler"""
    
    def __init__(self):
        self.start_time = datetime.now(IST_TIMEZONE)
        self.error_count = 0
        self.api_errors = 0
        self.telegram_errors = 0
        self.last_errors = deque(maxlen=50)  # Son 50 hata
        self.performance_metrics = {
            'trades_opened': 0,
            'trades_closed': 0,
            'api_calls_total': 0,
            'api_calls_failed': 0,
            'memory_warnings': 0,
            'database_operations': 0,
            'telegram_messages_sent': 0
        }
        self.daily_metrics = {}  # Günlük metrikler
        self.last_health_check = datetime.now(IST_TIMEZONE)
        
    def get_uptime(self) -> timedelta:
        """Bot'un ne kadar süredir çalıştığını döndür"""
        return datetime.now(IST_TIMEZONE) - self.start_time
    
    def record_error(self, error_type: str, error_message: str, source: str = "unknown"):
        """Hata kaydet"""
        self.error_count += 1
        
        # Hata tipine göre sayaçları artır
        if 'api' in error_type.lower():
            self.api_errors += 1
        elif 'telegram' in error_type.lower():
            self.telegram_errors += 1
        
        # Hata detayını kaydet
        error_record = {
            'timestamp': datetime.now(IST_TIMEZONE),
            'type': error_type,
            'message': error_message[:200],  # İlk 200 karakter
            'source': source
        }
        self.last_errors.append(error_record)
        
        logging.debug(f"Health Monitor: Recorded error - {error_type} from {source}")
    
    def record_metric(self, metric_name: str, increment: int = 1):
        """Metrik kaydet"""
        if metric_name in self.performance_metrics:
            self.performance_metrics[metric_name] += increment
        else:
            self.performance_metrics[metric_name] = increment
            
        # Günlük metrikler
        today = datetime.now(IST_TIMEZONE).date()
        if today not in self.daily_metrics:
            self.daily_metrics[today] = {}
        
        if metric_name not in self.daily_metrics[today]:
            self.daily_metrics[today][metric_name] = 0
        self.daily_metrics[today][metric_name] += increment
    
    def get_error_rate(self, hours: int = 1) -> float:
        """Son X saatteki hata oranını hesapla"""
        if not self.last_errors:
            return 0.0
        
        cutoff_time = datetime.now(IST_TIMEZONE) - timedelta(hours=hours)
        recent_errors = [e for e in self.last_errors if e['timestamp'] > cutoff_time]
        
        # Basit hata oranı: errors per hour
        return len(recent_errors) / hours
    
    def get_api_success_rate(self) -> float:
        """API başarı oranını hesapla"""
        total_calls = self.performance_metrics['api_calls_total']
        failed_calls = self.performance_metrics['api_calls_failed']
        
        if total_calls == 0:
            return 100.0
        
        success_rate = ((total_calls - failed_calls) / total_calls) * 100
        return round(success_rate, 2)
    
    def get_daily_performance(self, days: int = 7) -> Dict:
        """Son X günün performans özetini döndür"""
        today = datetime.now(IST_TIMEZONE).date()
        performance_data = {}
        
        for i in range(days):
            date = today - timedelta(days=i)
            if date in self.daily_metrics:
                performance_data[date.strftime('%Y-%m-%d')] = self.daily_metrics[date]
            else:
                performance_data[date.strftime('%Y-%m-%d')] = {}
        
        return performance_data
    
    def check_system_health(self) -> Dict[str, any]:
        """Sistem sağlığını kontrol et ve durum döndür"""
        uptime = self.get_uptime()
        error_rate = self.get_error_rate(hours=1)
        api_success_rate = self.get_api_success_rate()
        
        # Sağlık skoru hesapla (0-100)
        health_score = 100
        
        # Hata oranına göre skor düşür
        if error_rate > 10:  # Saatte 10'dan fazla hata
            health_score -= 30
        elif error_rate > 5:
            health_score -= 15
        elif error_rate > 2:
            health_score -= 5
        
        # API başarı oranına göre skor düşür
        if api_success_rate < 80:
            health_score -= 25
        elif api_success_rate < 90:
            health_score -= 10
        elif api_success_rate < 95:
            health_score -= 5
        
        # Uptime kontrolü (çok kısa süreli çalışma suspicious)
        if uptime.total_seconds() < 300:  # 5 dakikadan az
            health_score -= 10
        
        health_score = max(0, health_score)  # Negatif olmasın
        
        # Durum belirleme
        if health_score >= 90:
            status = "Excellent"
            status_emoji = "🟢"
        elif health_score >= 70:
            status = "Good"
            status_emoji = "🟡"
        elif health_score >= 50:
            status = "Warning"
            status_emoji = "🟠"
        else:
            status = "Critical"
            status_emoji = "🔴"
        
        return {
            'health_score': health_score,
            'status': status,
            'status_emoji': status_emoji,
            'uptime_hours': uptime.total_seconds() / 3600,
            'error_rate_per_hour': error_rate,
            'api_success_rate': api_success_rate,
            'total_errors': self.error_count,
            'recent_errors': len([e for e in self.last_errors if e['timestamp'] > datetime.now(IST_TIMEZONE) - timedelta(hours=1)]),
            'last_check': datetime.now(IST_TIMEZONE)
        }
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Performans özetini döndür"""
        uptime = self.get_uptime()
        today_metrics = self.daily_metrics.get(datetime.now(IST_TIMEZONE).date(), {})
        
        return {
            'uptime_days': uptime.days,
            'uptime_hours': int(uptime.total_seconds() // 3600) % 24,
            'uptime_minutes': int((uptime.total_seconds() % 3600) // 60),
            'total_metrics': self.performance_metrics.copy(),
            'today_metrics': today_metrics.copy(),
            'error_breakdown': {
                'total_errors': self.error_count,
                'api_errors': self.api_errors,
                'telegram_errors': self.telegram_errors,
                'other_errors': self.error_count - self.api_errors - self.telegram_errors
            }
        }
    
    def cleanup_old_data(self):
        """Eski metrikleri temizle (30 günden eski)"""
        cutoff_date = datetime.now(IST_TIMEZONE).date() - timedelta(days=30)
        old_dates = [date for date in self.daily_metrics.keys() if date < cutoff_date]
        
        for old_date in old_dates:
            del self.daily_metrics[old_date]
        
        if old_dates:
            logging.info(f"Health Monitor: Cleaned {len(old_dates)} old daily metrics")

class ErrorRecoveryManager:
    """Kritik hatalardan otomatik kurtarma ve sistem stabilitesi yönetimi"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.circuit_breakers = {}  # Service başına circuit breaker
        self.recovery_attempts = {}  # Recovery deneme sayıları
        self.last_recovery_time = {}  # Son recovery zamanları
        self.emergency_mode = False
        self.emergency_start_time = None
        self.max_recovery_attempts = 3
        self.recovery_cooldown = timedelta(minutes=5)
        
        # Circuit breaker ayarları
        self.circuit_breaker_config = {
            'api': {'failure_threshold': 5, 'timeout': 300},  # 5 hata, 5 dk timeout
            'telegram': {'failure_threshold': 3, 'timeout': 180},  # 3 hata, 3 dk timeout
            'database': {'failure_threshold': 3, 'timeout': 240},  # 3 hata, 4 dk timeout
        }
        
        # Emergency mode tetikleyicileri
        self.emergency_triggers = {
            'consecutive_api_failures': 10,
            'memory_usage_critical': 90,  # %90 üzeri
            'error_rate_critical': 20,  # Saatte 20+ hata
            'health_score_critical': 30  # Health score 30 altı
        }

    def should_trigger_emergency(self) -> Dict[str, bool]:
        """Emergency mode tetiklenmeli mi kontrol et"""
        triggers = {}
        
        try:
            # API failure kontrolü
            api_failures = self.bot.health_monitor.performance_metrics.get('api_calls_failed', 0)
            api_total = self.bot.health_monitor.performance_metrics.get('api_calls_total', 1)
            api_failure_rate = (api_failures / api_total) * 100 if api_total > 0 else 0
            triggers['consecutive_api_failures'] = api_failure_rate > 50  # %50+ failure rate
            
            # Memory kontrolü
            memory_stats = self.bot.memory_manager.get_memory_usage()
            memory_percentage = memory_stats.get('percentage', 0)
            triggers['memory_usage_critical'] = memory_percentage > self.emergency_triggers['memory_usage_critical']
            
            # Error rate kontrolü
            error_rate = self.bot.health_monitor.get_error_rate(hours=1)
            triggers['error_rate_critical'] = error_rate > self.emergency_triggers['error_rate_critical']
            
            # Health score kontrolü
            health_data = self.bot.health_monitor.check_system_health()
            triggers['health_score_critical'] = health_data['health_score'] < self.emergency_triggers['health_score_critical']
            
        except Exception as e:
            logging.error(f"Error checking emergency triggers: {e}")
            triggers['system_error'] = True
        
        return triggers

    async def handle_critical_error(self, error_type: str, error_details: str, source: str = "unknown"):
        """Kritik hataları handle et ve recovery dene"""
        logging.critical(f"CRITICAL ERROR: {error_type} from {source}: {error_details}")
        
        # Health monitor'a kaydet
        self.bot.health_monitor.record_error(f"CRITICAL_{error_type}", error_details, source)
        
        # Emergency mode kontrolü
        emergency_triggers = self.should_trigger_emergency()
        if any(emergency_triggers.values()) and not self.emergency_mode:
            await self.activate_emergency_mode(emergency_triggers)
            return
        
        # Circuit breaker kontrolü
        service = self._determine_service_from_source(source)
        if await self._check_circuit_breaker(service, error_type):
            logging.warning(f"Circuit breaker activated for {service}")
            return
        
        # Recovery deneme
        recovery_success = await self._attempt_recovery(error_type, source)
        if not recovery_success:
            await self._escalate_error(error_type, error_details, source)

    def _determine_service_from_source(self, source: str) -> str:
        """Source'a göre hangi service olduğunu belirle"""
        source_lower = source.lower()
        if 'api' in source_lower or 'binance' in source_lower:
            return 'api'
        elif 'telegram' in source_lower:
            return 'telegram'
        elif 'database' in source_lower or 'db' in source_lower:
            return 'database'
        else:
            return 'unknown'

    async def _check_circuit_breaker(self, service: str, error_type: str) -> bool:
        """Circuit breaker logic"""
        if service not in self.circuit_breaker_config:
            return False
        
        now = datetime.now(IST_TIMEZONE)
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = {
                'failures': 0,
                'last_failure': now,
                'circuit_open': False,
                'circuit_open_time': None
            }
        
        breaker = self.circuit_breakers[service]
        config = self.circuit_breaker_config[service]
        
        # Circuit açıksa ve timeout geçmediyse, istekleri blokla
        if breaker['circuit_open']:
            if now - breaker['circuit_open_time'] < timedelta(seconds=config['timeout']):
                return True  # Circuit hala açık
            else:
                # Timeout geçti, circuit'i kapat
                breaker['circuit_open'] = False
                breaker['failures'] = 0
                logging.info(f"Circuit breaker for {service} closed after timeout")
                return False
        
        # Failure sayısını artır
        breaker['failures'] += 1
        breaker['last_failure'] = now
        
        # Threshold aşıldıysa circuit'i aç
        if breaker['failures'] >= config['failure_threshold']:
            breaker['circuit_open'] = True
            breaker['circuit_open_time'] = now
            logging.warning(f"Circuit breaker opened for {service} after {breaker['failures']} failures")
            
            try:
                await self.bot.send_telegram(
                    f"🔌 <b>Circuit Breaker Activated</b>\n"
                    f"Service: {service.upper()}\n"
                    f"Failures: {breaker['failures']}\n"
                    f"Timeout: {config['timeout']}s"
                )
            except:
                pass
            
            return True
        
        return False

    async def _attempt_recovery(self, error_type: str, source: str) -> bool:
        """Recovery deneme"""
        recovery_key = f"{error_type}_{source}"
        now = datetime.now(IST_TIMEZONE)
        
        # Recovery cooldown kontrolü
        if recovery_key in self.last_recovery_time:
            if now - self.last_recovery_time[recovery_key] < self.recovery_cooldown:
                logging.info(f"Recovery cooldown active for {recovery_key}")
                return False
        
        # Recovery deneme sayısını kontrol et
        if recovery_key not in self.recovery_attempts:
            self.recovery_attempts[recovery_key] = 0
        
        if self.recovery_attempts[recovery_key] >= self.max_recovery_attempts:
            logging.error(f"Max recovery attempts reached for {recovery_key}")
            return False
        
        # Recovery deneme
        self.recovery_attempts[recovery_key] += 1
        self.last_recovery_time[recovery_key] = now
        
        recovery_success = False
        try:
            if 'api' in error_type.lower() or 'api' in source.lower():
                recovery_success = await self._recover_api_service()
            elif 'telegram' in error_type.lower() or 'telegram' in source.lower():
                recovery_success = await self._recover_telegram_service()
            elif 'memory' in error_type.lower():
                recovery_success = await self._recover_memory_issues()
            elif 'database' in error_type.lower():
                recovery_success = await self._recover_database_service()
            else:
                recovery_success = await self._general_recovery()
                
        except Exception as e:
            logging.error(f"Recovery attempt failed: {e}")
            recovery_success = False
        
        if recovery_success:
            logging.info(f"Recovery successful for {recovery_key}")
            self.recovery_attempts[recovery_key] = 0  # Reset counter
            try:
                await self.bot.send_telegram(f"✅ <b>Recovery Successful</b>\n{error_type} issue resolved")
            except:
                pass
        else:
            logging.warning(f"Recovery failed for {recovery_key}")
        
        return recovery_success

    async def _recover_api_service(self) -> bool:
        """API service recovery"""
        try:
            # API session'ı yeniden oluştur
            if self.bot.session:
                await self.bot.session.close()
                await asyncio.sleep(2)
            
            self.bot.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            
            # Test API call
            test_result = await self.bot._api_request_with_retry('GET', '/fapi/v1/ping')
            return test_result is not None
            
        except Exception as e:
            logging.error(f"API recovery failed: {e}")
            return False

    async def _recover_telegram_service(self) -> bool:
        """Telegram service recovery"""
        try:
            # Test message gönder
            url = f"https://api.telegram.org/bot{self.bot.telegram_token}/getMe"
            async with self.bot.session.get(url) as resp:
                return resp.status == 200
                
        except Exception as e:
            logging.error(f"Telegram recovery failed: {e}")
            return False

    async def _recover_memory_issues(self) -> bool:
        """Memory issues recovery"""
        try:
            # Agresif cleanup
            cleaned_items = self.bot._cleanup_old_data()
            freed_mb = self.bot.memory_manager.force_cleanup()
            
            # Memory durumunu kontrol et
            memory_stats = self.bot.memory_manager.get_memory_usage()
            return memory_stats['rss_mb'] < 400  # 400MB altında kabul edilebilir
            
        except Exception as e:
            logging.error(f"Memory recovery failed: {e}")
            return False

    async def _recover_database_service(self) -> bool:
        """Database service recovery"""
        try:
            # Database connection test
            db_info = await self.bot.db_manager.get_database_info()
            return 'error' not in db_info
            
        except Exception as e:
            logging.error(f"Database recovery failed: {e}")
            return False

    async def _general_recovery(self) -> bool:
        """Genel recovery"""
        try:
            # Memory cleanup
            self.bot._cleanup_old_data()
            gc.collect()
            
            # Wait and retry
            await asyncio.sleep(5)
            return True
            
        except Exception as e:
            logging.error(f"General recovery failed: {e}")
            return False

    async def _escalate_error(self, error_type: str, error_details: str, source: str):
        """Hata escalation"""
        try:
            escalation_message = (
                f"🚨 <b>CRITICAL ERROR ESCALATION</b>\n\n"
                f"<b>Type:</b> {error_type}\n"
                f"<b>Source:</b> {source}\n"
                f"<b>Details:</b> {error_details[:200]}...\n\n"
                f"<b>Recovery failed after multiple attempts.</b>\n"
                f"Manual intervention may be required."
            )
            await self.bot.send_telegram(escalation_message)
        except:
            logging.critical(f"Failed to send escalation message for {error_type}")

    async def activate_emergency_mode(self, triggers: Dict[str, bool]):
        """Emergency mode'u aktifleştir"""
        self.emergency_mode = True
        self.emergency_start_time = datetime.now(IST_TIMEZONE)
        
        active_triggers = [trigger for trigger, active in triggers.items() if active]
        
        logging.critical(f"EMERGENCY MODE ACTIVATED! Triggers: {active_triggers}")
        
        try:
            emergency_message = (
                f"🆘 <b>EMERGENCY MODE ACTIVATED</b>\n\n"
                f"<b>Triggers:</b>\n"
            )
            
            for trigger in active_triggers:
                emergency_message += f"• {trigger.replace('_', ' ').title()}\n"
            
            emergency_message += (
                f"\n<b>Actions taken:</b>\n"
                f"• All non-essential operations suspended\n"
                f"• Positions monitoring only\n"
                f"• Aggressive cleanup initiated\n"
                f"• New trades blocked"
            )
            
            await self.bot.send_telegram(emergency_message)
        except:
            logging.critical("Failed to send emergency mode notification")
        
        # Emergency actions
        await self._execute_emergency_actions()

    async def _execute_emergency_actions(self):
        """Emergency mode actions"""
        try:
            # 1. Agresif memory cleanup
            self.bot._cleanup_old_data()
            self.bot.memory_manager.force_cleanup()
            
            # 2. Model cache'i temizle
            if len(self.bot.ensemble_models) > 10:
                # Sadece son 10 modeli bırak
                symbols_to_remove = list(self.bot.ensemble_models.keys())[:-10]
                for symbol in symbols_to_remove:
                    if symbol in self.bot.ensemble_models:
                        del self.bot.ensemble_models[symbol]
                    if symbol in self.bot.scalers:
                        del self.bot.scalers[symbol]
            
            # 3. Trade history'yi kısalt
            if len(self.bot.trade_history) > 100:
                # Sadece son 100 trade'i bırak
                recent_trades = list(self.bot.trade_history)[-100:]
                self.bot.trade_history.clear()
                self.bot.trade_history.extend(recent_trades)
            
            # 4. Database cleanup
            await self.bot.db_manager.cleanup_old_trades(days_to_keep=30)  # Daha agresif
            
            logging.info("Emergency actions completed")
            
        except Exception as e:
            logging.critical(f"Emergency actions failed: {e}")

    def should_exit_emergency_mode(self) -> bool:
        """Emergency mode'dan çıkılmalı mı kontrol et"""
        if not self.emergency_mode:
            return False
        
        # En az 10 dakika emergency mode'da kal
        if datetime.now(IST_TIMEZONE) - self.emergency_start_time < timedelta(minutes=10):
            return False
        
        try:
            # Sistemin durumunu kontrol et
            health_data = self.bot.health_monitor.check_system_health()
            memory_stats = self.bot.memory_manager.get_memory_usage()
            
            # Çıkış kriterleri
            conditions_met = [
                health_data['health_score'] > 70,
                memory_stats['rss_mb'] < 300,
                health_data['error_rate_per_hour'] < 5,
                health_data['api_success_rate'] > 90
            ]
            
            return all(conditions_met)
            
        except Exception as e:
            logging.error(f"Error checking emergency exit conditions: {e}")
            return False

    async def exit_emergency_mode(self):
        """Emergency mode'dan çık"""
        self.emergency_mode = False
        self.emergency_start_time = None
        
        logging.info("Emergency mode deactivated - system stabilized")
        
        try:
            await self.bot.send_telegram(
                f"✅ <b>Emergency Mode Deactivated</b>\n\n"
                f"System has stabilized and returned to normal operations."
            )
        except:
            logging.error("Failed to send emergency mode exit notification")

    def is_service_available(self, service: str) -> bool:
        """Service'in kullanılabilir olup olmadığını kontrol et"""
        if service not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[service]
        if not breaker['circuit_open']:
            return True
        
        # Circuit açıksa ve timeout geçmişse, kullanılabilir
        config = self.circuit_breaker_config.get(service, {'timeout': 300})
        if datetime.now(IST_TIMEZONE) - breaker['circuit_open_time'] > timedelta(seconds=config['timeout']):
            return True
        
        return False

class SmartModelCache:
    """
    V15.3 - Intelligent Model Cache with Memory Management
    
    Features:
    - LRU eviction policy
    - Memory-aware cache sizing
    - Model quality-based retention
    - Thread-safe operations
    - Performance tracking
    """
    
    def __init__(self, max_size: int = 30, max_memory_mb: int = 200):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        
        # Cache storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # LRU tracking
        self.access_order = deque()  # Most recently used at end
        self.access_count: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}
        
        # Memory tracking
        self.estimated_memory_mb = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self.cache_lock = asyncio.Lock()
        
        logging.info(f"SmartModelCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    async def get_model(self, symbol: str) -> Optional[Tuple[Any, StandardScaler]]:
        """
        Thread-safe model retrieval with LRU tracking
        
        Returns:
            Tuple of (model, scaler) or None if not found
        """
        async with self.cache_lock:
            if symbol in self.models:
                # ✅ Cache HIT
                self._update_access_tracking(symbol)
                self.cache_hits += 1
                
                model = self.models[symbol]
                scaler = self.scalers.get(symbol)
                
                logging.debug(f"[{symbol}] Model cache HIT (total hits: {self.cache_hits})")
                return model, scaler
            else:
                # ❌ Cache MISS
                self.cache_misses += 1
                logging.debug(f"[{symbol}] Model cache MISS (total misses: {self.cache_misses})")
                return None
    
    async def put_model(self, symbol: str, model: Any, scaler: StandardScaler, 
                       confidence: float, training_time: float) -> bool:
        """
        Thread-safe model storage with smart eviction
        
        Args:
            symbol: Trading symbol
            model: Trained model object
            scaler: Associated scaler
            confidence: Model performance score
            training_time: Time taken to train
            
        Returns:
            bool: True if successfully cached
        """
        async with self.cache_lock:
            # Estimate memory usage for this model
            estimated_size_mb = self._estimate_model_size(model, scaler)
            
            # Check if we need to make space
            while (len(self.models) >= self.max_size or 
                   self.estimated_memory_mb + estimated_size_mb > self.max_memory_mb):
                
                if not await self._evict_least_valuable():
                    logging.warning("Could not evict models to make space")
                    return False
            
            # Store model and metadata
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.model_metadata[symbol] = {
                'confidence': confidence,
                'training_time': training_time,
                'created_at': datetime.now(IST_TIMEZONE),
                'estimated_size_mb': estimated_size_mb,
                'model_type': type(model).__name__
            }
            
            # Update tracking
            self._update_access_tracking(symbol)
            self.estimated_memory_mb += estimated_size_mb
            
            logging.info(f"[{symbol}] Model cached: {estimated_size_mb:.1f}MB, "
                        f"total cache: {self.estimated_memory_mb:.1f}MB ({len(self.models)} models)")
            
            return True
    
    async def _evict_least_valuable(self) -> bool:
        """
        Smart eviction: Remove least valuable model based on multiple factors
        
        Factors considered:
        1. Last access time (LRU)
        2. Access frequency
        3. Model performance (confidence)
        4. Memory usage
        
        Returns:
            bool: True if eviction successful
        """
        if not self.models:
            return False
        
        # Calculate value score for each model
        model_scores = {}
        now = datetime.now(IST_TIMEZONE)
        
        for symbol in self.models.keys():
            metadata = self.model_metadata.get(symbol, {})
            last_access = self.last_access.get(symbol, now)
            access_count = self.access_count.get(symbol, 0)
            confidence = metadata.get('confidence', 0.5)
            size_mb = metadata.get('estimated_size_mb', 10.0)
            
            # Calculate composite score (higher = more valuable)
            time_score = 1.0 / (1 + (now - last_access).total_seconds() / 3600)  # Decay over hours
            frequency_score = min(access_count / 10.0, 1.0)  # Normalize to 0-1
            performance_score = confidence
            size_penalty = 1.0 / (1 + size_mb / 50.0)  # Penalty for large models
            
            composite_score = (time_score * 0.3 + 
                             frequency_score * 0.2 + 
                             performance_score * 0.4 + 
                             size_penalty * 0.1)
            
            model_scores[symbol] = composite_score
        
        # Find least valuable model
        victim_symbol = min(model_scores.keys(), key=lambda s: model_scores[s])
        victim_metadata = self.model_metadata.get(victim_symbol, {})
        victim_size = victim_metadata.get('estimated_size_mb', 0)
        
        # Remove victim
        del self.models[victim_symbol]
        if victim_symbol in self.scalers:
            del self.scalers[victim_symbol]
        if victim_symbol in self.model_metadata:
            del self.model_metadata[victim_symbol]
        
        # Update tracking
        if victim_symbol in self.access_order:
            self.access_order.remove(victim_symbol)
        if victim_symbol in self.access_count:
            del self.access_count[victim_symbol]
        if victim_symbol in self.last_access:
            del self.last_access[victim_symbol]
        
        self.estimated_memory_mb -= victim_size
        
        logging.info(f"[{victim_symbol}] Model evicted (score: {model_scores[victim_symbol]:.3f}, "
                    f"freed: {victim_size:.1f}MB)")
        
        return True
    
    def _update_access_tracking(self, symbol: str):
        """Update LRU and access tracking"""
        now = datetime.now(IST_TIMEZONE)
        
        # Update access order (LRU)
        if symbol in self.access_order:
            self.access_order.remove(symbol)
        self.access_order.append(symbol)
        
        # Update access count and time
        self.access_count[symbol] = self.access_count.get(symbol, 0) + 1
        self.last_access[symbol] = now
    
    def _estimate_model_size(self, model: Any, scaler: StandardScaler) -> float:
        """
        Estimate memory usage of model + scaler in MB
        
        This is a rough estimation based on model type and parameters
        """
        try:
            # Base size for different model types
            if hasattr(model, 'get_model_info'):
                # Advanced ensemble
                info = model.get_model_info()
                model_count = info.get('total_models', 2)
                nn_trained = info.get('neural_network_trained', False)
                
                base_size = model_count * 5.0  # ~5MB per traditional model
                if nn_trained:
                    base_size += 20.0  # Additional 20MB for neural networks
            else:
                # Legacy ensemble or single model
                base_size = 10.0  # Default 10MB
            
            # Add scaler size (usually small)
            scaler_size = 0.5  # ~0.5MB for scaler
            
            total_size = base_size + scaler_size
            return max(total_size, 1.0)  # Minimum 1MB
            
        except Exception as e:
            logging.error(f"Error estimating model size: {e}")
            return 15.0  # Conservative default
    
    async def cleanup_stale_models(self, max_age_hours: int = 24):
        """
        Remove models older than specified age
        
        Args:
            max_age_hours: Maximum age in hours before forced removal
        """
        async with self.cache_lock:
            now = datetime.now(IST_TIMEZONE)
            stale_symbols = []
            
            for symbol, metadata in self.model_metadata.items():
                created_at = metadata.get('created_at', now)
                age_hours = (now - created_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    stale_symbols.append(symbol)
            
            # Remove stale models
            removed_size = 0.0
            for symbol in stale_symbols:
                size = self.model_metadata[symbol].get('estimated_size_mb', 0)
                removed_size += size
                
                del self.models[symbol]
                if symbol in self.scalers:
                    del self.scalers[symbol]
                del self.model_metadata[symbol]
                
                # Cleanup tracking
                if symbol in self.access_order:
                    self.access_order.remove(symbol)
                if symbol in self.access_count:
                    del self.access_count[symbol]
                if symbol in self.last_access:
                    del self.last_access[symbol]
            
            self.estimated_memory_mb -= removed_size
            
            if stale_symbols:
                logging.info(f"Cleaned {len(stale_symbols)} stale models, freed {removed_size:.1f}MB")
    
    async def force_cleanup(self, target_memory_mb: Optional[float] = None) -> float:
        """
        Aggressive cleanup to reach target memory usage
        
        Args:
            target_memory_mb: Target memory usage, defaults to max_memory_mb * 0.7
            
        Returns:
            float: Amount of memory freed in MB
        """
        if target_memory_mb is None:
            target_memory_mb = self.max_memory_mb * 0.7
        
        freed_mb = 0.0
        initial_memory = self.estimated_memory_mb
        
        async with self.cache_lock:
            while (self.estimated_memory_mb > target_memory_mb and 
                   len(self.models) > 1):  # Keep at least 1 model
                
                if not await self._evict_least_valuable():
                    break
            
            freed_mb = initial_memory - self.estimated_memory_mb
        
        if freed_mb > 0:
            logging.info(f"Force cleanup freed {freed_mb:.1f}MB "
                        f"(before: {initial_memory:.1f}MB, after: {self.estimated_memory_mb:.1f}MB)")
        
        return freed_mb
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'cache_size': len(self.models),
            'max_size': self.max_size,
            'estimated_memory_mb': self.estimated_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'utilization_pct': (len(self.models) / self.max_size) * 100,
            'memory_utilization_pct': (self.estimated_memory_mb / self.max_memory_mb) * 100
        }

class FeatureCache:
    """
    V15.3 - Smart Feature Engineering Cache
    
    Features:
    - Hash-based price data caching
    - TTL (Time To Live) expiration
    - Memory-aware eviction
    - Thread-safe operations
    - Hit/miss rate tracking
    - Automatic invalidation
    """
    
    def __init__(self, max_entries: int = 100, ttl_minutes: int = 15, max_memory_mb: int = 50):
        self.max_entries = max_entries
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_memory_mb = max_memory_mb
        
        # Cache storage
        self.cache: Dict[str, Dict] = {}  # cache_key -> {features, metadata}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.estimated_memory_mb = 0.0
        
        # Thread safety
        self.cache_lock = asyncio.Lock()
        
        logging.info(f"FeatureCache initialized: max_entries={max_entries}, ttl={ttl_minutes}min, max_memory={max_memory_mb}MB")
    
    def _generate_cache_key(self, df: pd.DataFrame, funding_rate: float) -> str:
        """
        Generate unique cache key from price data + funding rate
        
        Uses:
        - Last 10 rows of price data (recent candles)
        - Funding rate
        - Data shape info
        
        Returns:
            str: Unique cache key
        """
        try:
            # Use last 10 rows for key generation (most recent data)
            if len(df) >= 10:
                recent_data = df[['open', 'high', 'low', 'close', 'volume']].tail(10)
            else:
                recent_data = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Create hashable string
            data_string = (
                f"{recent_data.values.tobytes().hex()}"
                f"{funding_rate:.6f}"
                f"{len(df)}"
            )
            
            # Generate MD5 hash
            cache_key = hashlib.md5(data_string.encode()).hexdigest()[:16]  # First 16 chars
            return cache_key
            
        except Exception as e:
            logging.error(f"Error generating cache key: {e}")
            # Fallback: timestamp-based key (no caching benefit but won't crash)
            return f"fallback_{int(datetime.now(IST_TIMEZONE).timestamp())}"
    
    async def get_features(self, df: pd.DataFrame, funding_rate: float) -> Optional[pd.DataFrame]:
        """
        Get cached features for given price data
        
        Args:
            df: Price data DataFrame
            funding_rate: Current funding rate
            
        Returns:
            Optional[pd.DataFrame]: Cached features or None if not found/expired
        """
        cache_key = self._generate_cache_key(df, funding_rate)
        
        async with self.cache_lock:
            if cache_key not in self.cache:
                self.miss_count += 1
                logging.debug(f"Feature cache MISS: {cache_key[:8]}... (total misses: {self.miss_count})")
                return None
            
            cached_entry = self.cache[cache_key]
            cached_time = cached_entry['timestamp']
            
            # Check TTL expiration
            if datetime.now(IST_TIMEZONE) - cached_time > self.ttl:
                # Expired, remove from cache
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
                
                self.miss_count += 1
                logging.debug(f"Feature cache EXPIRED: {cache_key[:8]}... (total misses: {self.miss_count})")
                return None
            
            # Cache HIT
            self.hit_count += 1
            self.access_times[cache_key] = datetime.now(IST_TIMEZONE)
            
            cached_features = cached_entry['features']
            logging.debug(f"Feature cache HIT: {cache_key[:8]}... (total hits: {self.hit_count})")
            
            return cached_features.copy()  # Return copy to prevent modification
    
    async def put_features(self, df: pd.DataFrame, funding_rate: float, features: pd.DataFrame) -> bool:
        """
        Cache computed features
        
        Args:
            df: Original price data
            funding_rate: Funding rate used
            features: Computed features to cache
            
        Returns:
            bool: True if successfully cached
        """
        cache_key = self._generate_cache_key(df, funding_rate)
        
        async with self.cache_lock:
            # Estimate memory usage
            estimated_size_mb = self._estimate_features_size(features)
            
            # Check if we need to make space
            while (len(self.cache) >= self.max_entries or 
                   self.estimated_memory_mb + estimated_size_mb > self.max_memory_mb):
                
                if not self._evict_lru():
                    logging.warning("Could not evict from feature cache")
                    return False
            
            # Store features
            cache_entry = {
                'features': features.copy(),
                'timestamp': datetime.now(IST_TIMEZONE),
                'funding_rate': funding_rate,
                'data_length': len(df),
                'estimated_size_mb': estimated_size_mb
            }
            
            self.cache[cache_key] = cache_entry
            self.access_times[cache_key] = datetime.now(IST_TIMEZONE)
            self.estimated_memory_mb += estimated_size_mb
            
            logging.debug(f"Features cached: {cache_key[:8]}... "
                         f"({estimated_size_mb:.1f}MB, total: {self.estimated_memory_mb:.1f}MB)")
            
            return True
    
    def _evict_lru(self) -> bool:
        """
        Evict least recently used cache entry
        
        Returns:
            bool: True if eviction successful
        """
        if not self.cache:
            return False
        
        # Find LRU entry
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove entry
        if lru_key in self.cache:
            removed_size = self.cache[lru_key].get('estimated_size_mb', 0)
            del self.cache[lru_key]
            self.estimated_memory_mb -= removed_size
            
            logging.debug(f"Feature cache LRU evicted: {lru_key[:8]}... (freed: {removed_size:.1f}MB)")
        
        if lru_key in self.access_times:
            del self.access_times[lru_key]
        
        return True
    
    def _estimate_features_size(self, features: pd.DataFrame) -> float:
        """
        Estimate memory usage of features DataFrame in MB
        
        Args:
            features: Features DataFrame
            
        Returns:
            float: Estimated size in MB
        """
        try:
            # Rough estimation: rows * columns * 8 bytes (float64) + overhead
            num_elements = features.shape[0] * features.shape[1]
            size_bytes = num_elements * 8  # 8 bytes per float64
            overhead = size_bytes * 0.3    # 30% overhead for pandas structures
            
            total_size_mb = (size_bytes + overhead) / (1024 * 1024)
            return max(total_size_mb, 0.1)  # Minimum 0.1MB
            
        except Exception as e:
            logging.error(f"Error estimating features size: {e}")
            return 1.0  # Conservative default
    
    async def cleanup_expired(self):
        """
        Remove all expired entries from cache
        """
        async with self.cache_lock:
            now = datetime.now(IST_TIMEZONE)
            expired_keys = []
            
            for cache_key, entry in self.cache.items():
                if now - entry['timestamp'] > self.ttl:
                    expired_keys.append(cache_key)
            
            # Remove expired entries
            freed_memory = 0.0
            for key in expired_keys:
                if key in self.cache:
                    freed_memory += self.cache[key].get('estimated_size_mb', 0)
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            self.estimated_memory_mb -= freed_memory
            
            if expired_keys:
                logging.info(f"Feature cache cleanup: {len(expired_keys)} expired entries removed, "
                           f"{freed_memory:.1f}MB freed")
    
    async def force_cleanup(self, target_size: int = None) -> int:
        """
        Aggressive cleanup to reduce cache size
        
        Args:
            target_size: Target number of entries, defaults to max_entries * 0.7
            
        Returns:
            int: Number of entries removed
        """
        if target_size is None:
            target_size = int(self.max_entries * 0.7)
        
        async with self.cache_lock:
            initial_count = len(self.cache)
            
            while len(self.cache) > target_size:
                if not self._evict_lru():
                    break
            
            removed_count = initial_count - len(self.cache)
            
            if removed_count > 0:
                logging.info(f"Feature cache force cleanup: {removed_count} entries removed "
                           f"(before: {initial_count}, after: {len(self.cache)})")
            
            return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_entries': self.max_entries,
            'estimated_memory_mb': self.estimated_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'ttl_minutes': self.ttl.total_seconds() / 60,
            'utilization_pct': (len(self.cache) / self.max_entries) * 100,
            'memory_utilization_pct': (self.estimated_memory_mb / self.max_memory_mb) * 100
        }

# ===== DATABASE MANAGER (Orijinal Koddan Alındı) =====
class DatabaseManager:
    """V15.1 - Optimized database operations with connection pooling and async support"""
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._connection_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent connections
        self._last_cleanup = datetime.now(IST_TIMEZONE)
        self._cleanup_interval = timedelta(days=7)  # Haftalık cleanup
        self.init_database()
    
    def init_database(self):
        """Database ve tabloları başlat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Ana trades tablosu
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        quantity REAL NOT NULL,
                        pnl REAL,
                        confidence REAL NOT NULL,
                        funding_rate REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        max_profit REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        exit_reason TEXT,
                        market_regime TEXT,
                        volatility REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Performance için indexler
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at)')
                
                # Statistics tablosu (günlük performans özeti için)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE UNIQUE NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        best_trade REAL DEFAULT 0,
                        worst_trade REAL DEFAULT 0,
                        avg_confidence REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logging.info("Database initialized successfully")
                
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            raise

    async def save_trade(self, trade: TradeMetrics):
        """Trade'i async olarak ve güvenli formatta kaydet"""
        async with self._connection_semaphore:
            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    
                    # Trade verilerini veritabanına kaydet
                    cursor.execute('''
                        INSERT INTO trades (
                            symbol, entry_time, exit_time, entry_price, exit_price,
                            quantity, pnl, confidence, funding_rate, stop_loss,
                            take_profit, max_profit, max_drawdown, exit_reason,
                            market_regime, volatility
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade.symbol,
                        trade.entry_time.isoformat() if trade.entry_time else None,
                        trade.exit_time.isoformat() if trade.exit_time else None,
                        trade.entry_price,
                        trade.exit_price,
                        trade.quantity,
                        trade.pnl,
                        trade.confidence,
                        trade.funding_rate,
                        trade.stop_loss,
                        trade.take_profit,
                        trade.max_profit or 0.0,
                        trade.max_drawdown or 0.0,
                        trade.exit_reason,
                        trade.market_regime,
                        trade.volatility or 0.0
                    ))
                    
                    conn.commit()
                    
                    # Günlük istatistikleri güncelle (sadece kapalı trade'ler için)
                    if trade.exit_time and trade.pnl is not None:
                        await self._update_daily_stats(trade, conn)
                    
                    logging.info(f"✅ Trade saved to database: {trade.symbol}")
                    
            except Exception as e:
                logging.error(f"❌ Error saving trade to database: {e}")
                # Database hatası trade'i engellememelidir, sadece logla

    async def _update_daily_stats(self, trade: TradeMetrics, conn: sqlite3.Connection):
        """Günlük istatistikleri güncelle"""
        try:
            trade_date = trade.entry_time.date() if trade.entry_time else datetime.now(IST_TIMEZONE).date()
            cursor = conn.cursor()
            
            # Mevcut stats'ı al
            cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (trade_date,))
            existing = cursor.fetchone()
            
            if existing:
                # Güncelle
                cursor.execute('''
                    UPDATE daily_stats SET 
                        total_trades = total_trades + 1,
                        winning_trades = winning_trades + ?,
                        total_pnl = total_pnl + ?,
                        max_drawdown = MIN(max_drawdown, ?),
                        best_trade = MAX(best_trade, ?),
                        worst_trade = MIN(worst_trade, ?)
                    WHERE date = ?
                ''', (
                    1 if trade.pnl > 0 else 0,
                    trade.pnl,
                    trade.max_drawdown,
                    trade.pnl,
                    trade.pnl,
                    trade_date
                ))
            else:
                # Yeni kayıt
                cursor.execute('''
                    INSERT INTO daily_stats (
                        date, total_trades, winning_trades, total_pnl, 
                        max_drawdown, best_trade, worst_trade, avg_confidence
                    ) VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_date,
                    1 if trade.pnl > 0 else 0,
                    trade.pnl,
                    trade.max_drawdown,
                    trade.pnl,
                    trade.pnl,
                    trade.confidence
                ))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating daily stats: {e}")

    async def load_recent_trades(self, days: int) -> List[Dict]:
        """Son X gündeki trades'leri async olarak yükle (Geliştirilmiş Sorgu)"""
        async with self._connection_semaphore:
            trades = []
            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # ----- SORGUDOĞRU DÜZELTME -----
                    # Sorguyu, kapanış zamanı (exit_time) yerine giriş zamanına (entry_time) göre yap.
                    # Bu, son 7 gün içinde açılmış tüm işlemleri getirir, kapanmamış olsalar bile.
                    # Bu, history'nin boş gelme olasılığını büyük ölçüde azaltır.
                    # Ayrıca, start_date'i de ISO formatında bir string'e çevirerek
                    # veritabanındaki metinle güvenli bir şekilde karşılaştırılmasını sağlıyoruz.
                    
                    start_date_iso = (datetime.now(IST_TIMEZONE) - timedelta(days=days)).isoformat()
                    
                    cursor.execute('''
                        SELECT * FROM trades 
                        WHERE entry_time >= ? 
                        ORDER BY entry_time DESC 
                        LIMIT 1000
                    ''', (start_date_iso,))
                    # ----- DÜZELTME SONU -----
                    
                    rows = cursor.fetchall()
                    trades = [dict(row) for row in rows]
                    
                # Bu log mesajı artık bize doğru bilgiyi verecek
                logging.info(f"Loaded {len(trades)} recent trades from database (based on entry_time).")
                
            except Exception as e:
                logging.error(f"Error loading recent trades from DB: {e}", exc_info=True)
                
            return trades

    async def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """Son X günün günlük istatistiklerini al"""
        async with self._connection_semaphore:
            stats = []
            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    start_date = datetime.now(IST_TIMEZONE).date() - timedelta(days=days)
                    cursor.execute('''
                        SELECT * FROM daily_stats 
                        WHERE date >= ? 
                        ORDER BY date DESC
                    ''', (start_date,))
                    
                    rows = cursor.fetchall()
                    stats = [dict(row) for row in rows]
                    
            except Exception as e:
                logging.error(f"Error loading daily stats: {e}")
                
            return stats

    async def get_stats_for_date(self, target_date: date) -> Optional[Dict]:
        """Belirli bir tarihin günlük istatistiklerini veritabanından alır."""
        async with self._connection_semaphore:
            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (target_date,))
                    
                    row = cursor.fetchone()
                    if row:
                        return dict(row)
                    else:
                        # O gün için hiç veri yoksa, boş bir sözlük döndür
                        return None
                        
            except Exception as e:
                logging.error(f"Error loading stats for date {target_date}: {e}")
                return None

    async def cleanup_old_trades(self, days_to_keep: int = 90):
        """Eski trade kayıtlarını temizle (90 günden eski)"""
        async with self._connection_semaphore:
            try:
                cutoff_date = datetime.now(IST_TIMEZONE) - timedelta(days=days_to_keep)
                
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    cursor = conn.cursor()
                    
                    # Eski trades'leri sil
                    cursor.execute('DELETE FROM trades WHERE entry_time < ?', (cutoff_date,))
                    deleted_trades = cursor.rowcount
                    
                    # Eski daily stats'leri sil (6 aydan eski)
                    stats_cutoff = datetime.now(IST_TIMEZONE).date() - timedelta(days=180)
                    cursor.execute('DELETE FROM daily_stats WHERE date < ?', (stats_cutoff,))
                    deleted_stats = cursor.rowcount
                    
                    conn.commit()
                    
                    if deleted_trades > 0 or deleted_stats > 0:
                        logging.info(f"Database cleanup: Deleted {deleted_trades} trades and {deleted_stats} daily stats")
                        
                        # VACUUM for space reclaim
                        cursor.execute("VACUUM")
                        logging.info("Database vacuumed")
                        
                    self._last_cleanup = datetime.now(IST_TIMEZONE)
                    
            except Exception as e:
                logging.error(f"Error cleaning old trades: {e}")

    async def get_database_info(self) -> Dict[str, any]:
        """Database bilgilerini al"""
        async with self._connection_semaphore:
            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    
                    # Toplam trade sayısı
                    cursor.execute('SELECT COUNT(*) FROM trades')
                    total_trades = cursor.fetchone()[0]
                    
                    # Son 30 günün trade sayısı
                    cutoff = datetime.now(IST_TIMEZONE) - timedelta(days=30)
                    cursor.execute('SELECT COUNT(*) FROM trades WHERE entry_time >= ?', (cutoff,))
                    recent_trades = cursor.fetchone()[0]
                    
                    # Database dosya boyutu
                    try:
                        file_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
                    except:
                        file_size_mb = 0
                    
                    return {
                        'total_trades': total_trades,
                        'recent_trades_30d': recent_trades,
                        'file_size_mb': file_size_mb,
                        'last_cleanup': self._last_cleanup,
                        'next_cleanup': self._last_cleanup + self._cleanup_interval
                    }
                    
            except Exception as e:
                logging.error(f"Error getting database info: {e}")
                return {'error': str(e)}

    async def should_cleanup(self) -> bool:
        """Cleanup gerekli mi kontrol et"""
        return datetime.now(IST_TIMEZONE) > (self._last_cleanup + self._cleanup_interval)

class AdvancedFeatureEngineer:
    """
    V15.3 Refactored - Centralized and robust feature engineering powerhouse.
    - Handles advanced, simple, and fallback feature creation internally.
    - Manages its own feature cache.
    - Provides a single, reliable entry point: create_features.
    """
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50]
        self.volatility_windows = [10, 20, 30]
        self.momentum_periods = [3, 5, 10, 15]
        
        self.feature_cache = FeatureCache(
            max_entries=100,
            ttl_minutes=15,
            max_memory_mb=50
        )
        logging.info("AdvancedFeatureEngineer initialized with caching.")

    @safe_dataframe_operations
    async def create_features(self, df: pd.DataFrame, funding_rate: float = 0.0) -> pd.DataFrame:
        cached_features = await self.feature_cache.get_features(df, funding_rate)
        if cached_features is not None:
            return cached_features

        try:
            loop = asyncio.get_running_loop()
            features = await loop.run_in_executor(
                None, self._create_advanced_features_sync, df.copy(), funding_rate
            )
            if features is not None and not features.empty:
                await self.feature_cache.put_features(df, funding_rate, features)
                return features
            else:
                logging.warning("Advanced feature creation returned empty. Falling back to simple features.")
        except Exception as e:
            logging.error(f"Advanced feature creation failed in executor: {e}", exc_info=True)

        simple_features = self._create_simple_features(df, funding_rate)
        return simple_features
        
    def _create_advanced_features_sync(self, df: pd.DataFrame, funding_rate: float = 0.0) -> pd.DataFrame:
        """
        FINAL & CORRECTED VERSION of advanced feature creation.
        It now preserves ALL original columns to prevent feature mismatch errors.
        """
        if df.empty or len(df) < 50:
            return pd.DataFrame()

        try:
            # ÖNEMLİ: Orijinal sütunları kaybetmemek için bir kopya ile başla
            df_features = df.copy()
            
            # Veriyi temizle
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            
            df_features.dropna(subset=required_cols, inplace=True)
            df_features.replace([np.inf, -np.inf], 0, inplace=True)
            df_features['close'] = df_features['close'].replace(0, 1e-9)
            
            if len(df_features) < 50: return pd.DataFrame()

            # Hesaplamalar için serileri al
            close = df_features['close']
            high = df_features['high']
            low = df_features['low']
            volume = df_features['volume']
            
            # --- Yeni Özellikleri Ekle ---
            df_features['fundingRate'] = funding_rate
            df_features['rsi_14'] = ta.momentum.rsi(close, window=14).fillna(50)
            df_features['macd_diff'] = ta.trend.macd_diff(close).fillna(0)
            
            atr = ta.volatility.average_true_range(high, low, close, window=14)
            df_features['atr_percent'] = (atr / close * 100).fillna(1.5).replace(0, 1.5)
            
            bbands = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            df_features['bb_width'] = bbands.bollinger_wband().fillna(0)

            for p in [5, 10, 20]:
                df_features[f'momentum_{p}'] = close.pct_change(p).fillna(0) * 100

            df_features['ema_5_20_diff'] = ((ta.trend.ema_indicator(close, 5) - ta.trend.ema_indicator(close, 20)) / close * 100).fillna(0)

            volume_sma_20 = volume.rolling(20).mean().replace(0, 1e-9)
            df_features['volume_ratio'] = (volume / volume_sma_20).fillna(1)
            
            # Son temizlik (Sütun SİLME YOK!)
            df_features.fillna(0, inplace=True)

            logging.info(f"Advanced features created. Total columns now: {len(df_features.columns)}")
            return df_features

        except Exception as e:
            logging.error(f"CRITICAL ERROR during advanced feature creation: {e}", exc_info=True)
            return pd.DataFrame()

    def _create_simple_features(self, df: pd.DataFrame, funding_rate: float) -> pd.DataFrame:
        """Fallback: Creates a basic, reliable set of features while preserving original OHLCV columns."""
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if df.empty or not all(col in df.columns for col in required_cols):
                return pd.DataFrame()
            
            df_simple = df[required_cols].copy()
            high, low, close = df_simple['high'], df_simple['low'], df_simple['close']
            
            # Yeni basit özellikleri ekle
            df_simple['fundingRate'] = funding_rate
            df_simple['rsi'] = ta.momentum.rsi(close, window=14).fillna(50)
            df_simple['macd_diff'] = ta.trend.macd_diff(close).fillna(0)
            df_simple['price_momentum_5'] = close.pct_change(5).fillna(0) * 100
            
            # Bollinger Bands
            indicator_bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            df_simple['bb_width'] = indicator_bb.bollinger_wband().fillna(0)
            
            # --- ATR HESAPLAMASINI GÜVENLİ HALE GETİRME ---
            atr_series = ta.volatility.average_true_range(high, low, close, window=14)
            # ATR'yi fiyata oranla yüzdeye çevir
            atr_percent_series = (atr_series / close * 100)
            
            # Eğer ATR hesaplanamazsa veya sıfırsa, %1.5'luk bir varsayılan volatilite ata
            # Bu, sıfır volatilite sorununu KESİNLİKLE çözer.
            df_simple['atr_percent'] = atr_percent_series.fillna(1.5).replace(0, 1.5)

            # Final cleanup
            df_simple = df_simple.replace([np.inf, -np.inf], 0).fillna(0)
            logging.info("Created simple fallback features, preserving OHLCV. Ensured non-zero ATR.")
            return df_simple
            
        except Exception as e:
            logging.critical(f"Even simple feature creation failed: {e}")
            return pd.DataFrame()

    def get_feature_names(self) -> List[str]:
        """Tüm feature isimlerini döndür"""
        basic_features = ['fundingRate', 'rsi', 'macd_diff', 'bb_width', 'atr_percent', 'volume_ratio', 'price_momentum']
        
        advanced_features = []
        
        # Momentum features
        for period in self.momentum_periods:
            advanced_features.extend([f'momentum_{period}', f'rsi_{period}', f'volume_momentum_{period}'])
        
        # Volatility features
        for window in self.volatility_windows:
            advanced_features.extend([f'realized_vol_{window}', f'hl_volatility_{window}'])
            if window > 10:
                advanced_features.append(f'vol_ratio_{window}')
        
        # Trend features
        for fast, slow in [(5, 10), (10, 20), (20, 50)]:
            advanced_features.extend([
                f'ema_cross_{fast}_{slow}', f'ema_spread_{fast}_{slow}', f'trend_strength_{fast}'
            ])
        
        # Other advanced features
        other_features = [
            'dist_to_high', 'dist_to_low', 'vpt', 'vpt_momentum', 'obv', 'obv_momentum',
            'vwap_distance', 'spread_proxy', 'spread_avg', 'trade_intensity', 'trade_intensity_norm',
            'regime_short', 'regime_medium', 'regime_long', 'z_score_20', 'z_score_50',
            'returns_skew_20', 'returns_kurt_20', 'fr_momentum', 'fr_signal_strength', 'fr_regime'
        ]
        
        return basic_features + advanced_features + other_features
    
    def select_best_features(self, df: pd.DataFrame, target: pd.Series, max_features: int = 30) -> List[str]:
        """Feature selection - en iyi features'ları seç"""
        try:
            # Sklearn feature selection varsa kullan
            if 'SKLEARN_FEATURE_SELECTION_AVAILABLE' in globals() and SKLEARN_FEATURE_SELECTION_AVAILABLE:
                from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
                
                # Sadece numeric features
                numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_features) <= max_features:
                    return numeric_features
                
                # Missing values'ları handle et
                df_clean = df[numeric_features].fillna(0)
                target_clean = target.fillna(0)
                
                # Mutual information ile feature selection
                selector = SelectKBest(score_func=mutual_info_classif, k=min(max_features, len(numeric_features)))
                selector.fit(df_clean, target_clean)
                
                # Seçilen features
                selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
                
                logging.info(f"Feature selection: {len(selected_features)}/{len(numeric_features)} features selected")
                
                return selected_features
            else:
                # Sklearn yoksa manuel seçim
                logging.info("Using manual feature selection (sklearn not available)")
                return self._manual_feature_selection(df, max_features)
                
        except Exception as e:
            logging.error(f"Feature selection failed: {e}")
            # Fallback: manuel seçim
            return self._manual_feature_selection(df, max_features)
    
    def _manual_feature_selection(self, df: pd.DataFrame, max_features: int) -> List[str]:
        """Manuel feature selection (fallback)"""
        # En önemli features (domain knowledge'a göre)
        priority_features = [
            'fundingRate', 'rsi', 'macd_diff', 'bb_width', 'atr_percent', 'volume_ratio', 'price_momentum',
            'momentum_5', 'momentum_10', 'rsi_10', 'realized_vol_20', 'ema_spread_10_20', 
            'trend_strength_10', 'vwap_distance', 'regime_short', 'z_score_20',
            'obv_momentum', 'fr_signal_strength', 'vol_ratio_20'
        ]
        
        # Mevcut features'lar içinden priority olanları seç
        available_features = df.columns.tolist()
        selected = [f for f in priority_features if f in available_features]
        
        # Yeterli değilse diğerlerinden ekle
        if len(selected) < max_features:
            remaining = [f for f in available_features if f not in selected and df[f].dtype in ['float64', 'int64']]
            selected.extend(remaining[:max_features - len(selected)])
        
        return selected[:max_features]

# === NEURAL NETWORK PREDICTOR SINIFINI AdvancedFeatureEngineer'DAN SONRA EKLEYİN ===

try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully: {tf.__version__}")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    
    TENSORFLOW_AVAILABLE = True
    
    # TensorFlow loglarını azalt
    tf.get_logger().setLevel('ERROR')
    
    print("🧠 Neural Networks will be ENABLED")
    logging.info(f"TensorFlow {tf.__version__} loaded successfully for Neural Networks")
    
except ImportError as e:
    print(f"❌ TensorFlow ImportError: {e}")
    TENSORFLOW_AVAILABLE = False
    logging.warning(f"TensorFlow not available. ImportError: {e}")
    
except Exception as e:
    print(f"❌ TensorFlow Exception: {e}")
    TENSORFLOW_AVAILABLE = False
    logging.warning(f"TensorFlow error: {e}")

class NeuralNetworkPredictor:
    """TensorFlow ile Neural Network modelleri - daha akıllı predictions"""
    
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.input_shape = None
        self.sequence_length = 10  # LSTM için sequence length
        
    def is_available(self) -> bool:
        """TensorFlow kullanılabilir mi?"""
        return TENSORFLOW_AVAILABLE
    
    def _create_dense_model(self, input_dim: int):  # Type hint kaldırıldı
        """Dense Neural Network modeli oluştur"""
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not available")
            
        model = Sequential([
            # Input layer
            Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_lstm_model(self, input_dim: int, sequence_length: int):  # Type hint kaldırıldı
        """LSTM modeli oluştur (sequential data için)"""
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not available")
            
        model = Sequential([
            # LSTM layers
            LSTM(32, return_sequences=True, input_shape=(sequence_length, input_dim)),
            Dropout(0.2),
            
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_lstm_data(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        """LSTM için sequential data hazırla"""
        if len(X) < sequence_length:
            return None, None
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Neural Network modellerini train et"""
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not available for Neural Network training")
        
        if len(X) < 100:  # NN için daha fazla data gerekir
            raise ValueError("Insufficient data for Neural Network training (minimum: 100)")
        
        if y.sum() < 10:  # Yeterli positive sample
            raise ValueError("Insufficient positive samples for Neural Network training")
        
        self.input_shape = X.shape[1]
        
        try:
            # Train/validation split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=0
            )
            
            callbacks = [early_stopping, reduce_lr]
            
            # ===== 1. DENSE MODEL =====
            try:
                logging.debug("Training Dense Neural Network...")
                dense_model = self._create_dense_model(self.input_shape)
                
                dense_history = dense_model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Model performance
                val_loss = min(dense_history.history['val_loss'])
                val_accuracy = max(dense_history.history['val_accuracy'])
                
                self.models['dense'] = {
                    'model': dense_model,
                    'score': val_accuracy,
                    'loss': val_loss,
                    'type': 'dense'
                }
                
                logging.debug(f"Dense NN trained - Accuracy: {val_accuracy:.3f}, Loss: {val_loss:.3f}")
                
            except Exception as e:
                logging.warning(f"Dense NN training failed: {e}")
            
            # ===== 2. LSTM MODEL =====
            try:
                if len(X) >= self.sequence_length * 2:  # LSTM için yeterli data var mı?
                    logging.debug("Training LSTM Neural Network...")
                    
                    # LSTM data preparation
                    X_lstm, y_lstm = self._prepare_lstm_data(X, y, self.sequence_length)
                    
                    if X_lstm is not None and len(X_lstm) > 50:
                        # LSTM train/val split
                        lstm_split = int(len(X_lstm) * 0.8)
                        X_lstm_train, X_lstm_val = X_lstm[:lstm_split], X_lstm[lstm_split:]
                        y_lstm_train, y_lstm_val = y_lstm[:lstm_split], y_lstm[lstm_split:]
                        
                        lstm_model = self._create_lstm_model(self.input_shape, self.sequence_length)
                        
                        lstm_history = lstm_model.fit(
                            X_lstm_train, y_lstm_train,
                            epochs=30,  # LSTM için daha az epoch
                            batch_size=16,  # Daha küçük batch
                            validation_data=(X_lstm_val, y_lstm_val),
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        # LSTM performance
                        lstm_val_loss = min(lstm_history.history['val_loss'])
                        lstm_val_accuracy = max(lstm_history.history['val_accuracy'])
                        
                        self.models['lstm'] = {
                            'model': lstm_model,
                            'score': lstm_val_accuracy,
                            'loss': lstm_val_loss,
                            'type': 'lstm',
                            'sequence_length': self.sequence_length
                        }
                        
                        logging.debug(f"LSTM NN trained - Accuracy: {lstm_val_accuracy:.3f}, Loss: {lstm_val_loss:.3f}")
                
            except Exception as e:
                logging.warning(f"LSTM NN training failed: {e}")
            
            # Training başarılı mı?
            if self.models:
                self.is_trained = True
                logging.info(f"Neural Networks trained successfully: {list(self.models.keys())}")
                return True
            else:
                logging.warning("All Neural Network training attempts failed")
                return False
                
        except Exception as e:
            logging.error(f"Neural Network training error: {e}")
            return False
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Neural Network predictions (ensemble)"""
        if not self.is_trained or not self.models:
            raise ValueError("Neural Networks not trained")
        
        predictions = []
        weights = []
        
        try:
            # Dense model prediction
            if 'dense' in self.models:
                dense_pred = self.models['dense']['model'].predict(X, verbose=0)
                predictions.append(dense_pred.flatten())
                weights.append(self.models['dense']['score'])
            
            # LSTM model prediction
            if 'lstm' in self.models and X.shape[0] >= self.sequence_length:
                # LSTM için sequential data hazırla
                X_lstm = []
                for i in range(len(X)):
                    if i >= self.sequence_length - 1:
                        # Son sequence_length kadar veriyi al
                        if i < self.sequence_length - 1:
                            # Padding ile başlangıcı doldur
                            padded_sequence = np.vstack([
                                np.zeros((self.sequence_length - i - 1, X.shape[1])),
                                X[:i+1]
                            ])
                        else:
                            padded_sequence = X[i-self.sequence_length+1:i+1]
                        X_lstm.append(padded_sequence)
                    else:
                        # İlk sequence_length için padding
                        padded_sequence = np.vstack([
                            np.zeros((self.sequence_length - i - 1, X.shape[1])),
                            X[:i+1]
                        ])
                        X_lstm.append(padded_sequence)
                
                X_lstm = np.array(X_lstm)
                lstm_pred = self.models['lstm']['model'].predict(X_lstm, verbose=0)
                predictions.append(lstm_pred.flatten())
                weights.append(self.models['lstm']['score'])
            
            if not predictions:
                # Fallback
                return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
            
            # Weighted ensemble
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
            else:
                ensemble_pred = np.mean(predictions, axis=0)
            
            # Return as [negative_class_prob, positive_class_prob]
            negative_probs = 1 - ensemble_pred
            positive_probs = ensemble_pred
            
            return np.column_stack([negative_probs, positive_probs])
            
        except Exception as e:
            logging.error(f"Neural Network prediction error: {e}")
            # Fallback prediction
            fallback_prob = 0.5
            return np.column_stack([
                np.ones(len(X)) * (1 - fallback_prob), 
                np.ones(len(X)) * fallback_prob
            ])
    
    def get_model_info(self) -> Dict[str, any]:
        """Model bilgilerini döndür"""
        if not self.is_trained:
            return {'trained': False, 'models': []}
        
        model_info = {'trained': True, 'models': []}
        
        for name, model_data in self.models.items():
            info = {
                'name': name,
                'type': model_data['type'],
                'score': model_data['score'],
                'loss': model_data['loss']
            }
            model_info['models'].append(info)
        
        return model_info

class DynamicSLTPManager:
    """
    Dynamic Stop Loss / Take Profit Manager
    
    Özellikler:
    - ATR-based dynamic SL/TP hesaplama
    - Market regime'e göre adaptive multiplier'lar
    - AI confidence'a göre risk scaling
    - Volatility-aware adjustments
    - Real-time SL/TP güncelleme
    """
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        
        # Market regime bazlı multiplier'lar
        self.regime_multipliers = {
            MarketRegime.UPTREND: {
                'sl_base': 1.5,
                'tp_base': 3.0,
                'sl_confidence_factor': 1.2,  # High confidence'ta SL daha geniş
                'tp_confidence_factor': 1.1   # High confidence'ta TP daha aggressive
            },
            MarketRegime.SIDEWAYS_RANGING: {
                'sl_base': 1.2,
                'tp_base': 2.0,
                'sl_confidence_factor': 1.1,
                'tp_confidence_factor': 1.0
            },
            MarketRegime.HIGH_VOLATILITY: {
                'sl_base': 2.0,
                'tp_base': 2.5,
                'sl_confidence_factor': 1.3,  # Volatilitede daha geniş SL
                'tp_confidence_factor': 0.9   # Volatilitede daha conservative TP
            },
            MarketRegime.DOWNTREND: {
                'sl_base': 1.8,
                'tp_base': 2.2,
                'sl_confidence_factor': 1.0,
                'tp_confidence_factor': 0.8   # Downtrend'de conservative TP
            },
            MarketRegime.UNKNOWN: {
                'sl_base': 1.5,
                'tp_base': 2.5,
                'sl_confidence_factor': 1.0,
                'tp_confidence_factor': 1.0
            }
        }
        
        # Volatility scaling factors
        self.volatility_scaling = {
            'low': {'threshold': 1.5, 'sl_factor': 0.8, 'tp_factor': 1.2},
            'medium': {'threshold': 3.0, 'sl_factor': 1.0, 'tp_factor': 1.0},
            'high': {'threshold': 5.0, 'sl_factor': 1.3, 'tp_factor': 0.9},
            'extreme': {'threshold': 10.0, 'sl_factor': 1.5, 'tp_factor': 0.8}
        }
        
        # Güvenlik limitleri
        self.safety_limits = {
            'min_sl_distance_pct': 0.003,  # %0.3 minimum SL distance
            'max_sl_distance_pct': 0.05,   # %5 maximum SL distance
            'min_tp_distance_pct': 0.005,  # %0.5 minimum TP distance
            'max_tp_distance_pct': 0.15    # %15 maximum TP distance
        }
        
        logging.info("Dynamic SL/TP Manager initialized")
    
    def calculate_dynamic_sltp(self, 
                              entry_price: float, 
                              atr_percent: float, 
                              market_regime: MarketRegime, 
                              confidence: float,
                              funding_time_remaining: float = 240) -> Tuple[float, float]:
        """
        V15.7 - Calculates dynamic SL/TP with a built-in safety net for minimum stop distance.
        """
        try:
            if entry_price <= 0 or atr_percent <= 0:
                logging.error(f"Invalid inputs: entry_price={entry_price}, atr_percent={atr_percent}")
                return entry_price * 0.98, entry_price * 1.02  # Fallback

            # ... (metodun başındaki çarpan hesaplamaları aynı kalıyor) ...
            regime_config = self.regime_multipliers.get(market_regime, self.regime_multipliers[MarketRegime.UNKNOWN])
            base_sl_multiplier = regime_config['sl_base']
            base_tp_multiplier = regime_config['tp_base']
            confidence_normalized = max(0, min(1, confidence))
            sl_confidence_adjustment = 1 + (confidence_normalized - 0.5) * regime_config['sl_confidence_factor']
            tp_confidence_adjustment = 1 + (confidence_normalized - 0.5) * regime_config['tp_confidence_factor']
            volatility_category = self._categorize_volatility(atr_percent)
            vol_config = self.volatility_scaling[volatility_category]
            funding_factor = self._calculate_funding_factor(funding_time_remaining)
            final_sl_multiplier = (base_sl_multiplier * sl_confidence_adjustment * vol_config['sl_factor'] * funding_factor)
            final_tp_multiplier = (base_tp_multiplier * tp_confidence_adjustment * vol_config['tp_factor'] * funding_factor)
            
            # ===== YENİ GÜVENLİK AĞI MANTIĞI BURADA BAŞLIYOR =====

            # 1. SL mesafesini ATR'ye göre hesapla
            atr_based_sl_distance = entry_price * (atr_percent / 100) * final_sl_multiplier

            # 2. Her zaman olması gereken minimum SL mesafesini belirle
            # %1.5'lik sabit bir minimum mesafe + yüksek güvende ekstra %0.5
            min_sl_percent = 0.015 
            if confidence > 0.85:
                min_sl_percent = 0.020 # Yüksek güvende pozisyona daha fazla alan ver
            
            minimum_guaranteed_sl_distance = entry_price * min_sl_percent

            # 3. Bu iki mesafeden BÜYÜK olanı seç
            final_sl_distance = max(atr_based_sl_distance, minimum_guaranteed_sl_distance)
            
            # Take profit mesafesi aynı kalabilir, o genellikle daha cömerttir.
            tp_distance = entry_price * (atr_percent / 100) * final_tp_multiplier
            
            # ===== YENİ GÜVENLİK AĞI MANTIĞI BİTTİ =====

            stop_loss_price = entry_price - final_sl_distance
            take_profit_price = entry_price + tp_distance
            
            # Güvenlik limitleri (max SL/TP) hala geçerli
            stop_loss_price, take_profit_price = self._apply_safety_limits(
                entry_price, stop_loss_price, take_profit_price
            )
            
            sl_percent_for_log = (final_sl_distance / entry_price) * 100
            tp_percent_for_log = (tp_distance / entry_price) * 100

            log_message = (
                f"Dynamic SL/TP calculated: "
                f"Regime={market_regime.value}, Conf={confidence:.2f}, ATR={atr_percent:.1f}%, "
                f"SL={stop_loss_price:.4f} ({sl_percent_for_log:.2f}%), TP={take_profit_price:.4f} ({tp_percent_for_log:.2f}%)"
            )
            
            # Eğer güvenlik ağı devreye girdiyse bunu logla
            if final_sl_distance == minimum_guaranteed_sl_distance:
                log_message += " [Safety Net Activated for SL]"
            
            logging.info(log_message)
            
            return stop_loss_price, take_profit_price
            
        except Exception as e:
            logging.error(f"Dynamic SL/TP calculation failed: {e}")
            fallback_sl = entry_price * 0.98
            fallback_tp = entry_price * 1.03
            return fallback_sl, fallback_tp
    
    def _categorize_volatility(self, atr_percent: float) -> str:
        """ATR'a göre volatilite kategorisi belirle"""
        if atr_percent <= self.volatility_scaling['low']['threshold']:
            return 'low'
        elif atr_percent <= self.volatility_scaling['medium']['threshold']:
            return 'medium'
        elif atr_percent <= self.volatility_scaling['high']['threshold']:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_funding_factor(self, minutes_to_funding: float) -> float:
        """Funding zamanına göre risk faktörü"""
        if minutes_to_funding < 30:
            return 0.8  # Funding'e yakın, conservative ol
        elif minutes_to_funding < 60:
            return 0.9  # Biraz conservative
        elif minutes_to_funding > 180:
            return 1.1  # Funding'e uzak, biraz aggressive
        else:
            return 1.0  # Normal
    
    def _apply_safety_limits(self, entry_price: float, sl_price: float, tp_price: float) -> Tuple[float, float]:
        """Güvenlik limitlerini uygula"""
        try:
            # SL distance kontrolü
            sl_distance_pct = abs(entry_price - sl_price) / entry_price
            
            if sl_distance_pct < self.safety_limits['min_sl_distance_pct']:
                # Çok yakın SL, minimum distance'a çek
                sl_price = entry_price * (1 - self.safety_limits['min_sl_distance_pct'])
                logging.warning(f"SL too close, adjusted to minimum distance: {self.safety_limits['min_sl_distance_pct']:.1%}")
            
            elif sl_distance_pct > self.safety_limits['max_sl_distance_pct']:
                # Çok uzak SL, maximum distance'a çek
                sl_price = entry_price * (1 - self.safety_limits['max_sl_distance_pct'])
                logging.warning(f"SL too far, adjusted to maximum distance: {self.safety_limits['max_sl_distance_pct']:.1%}")
            
            # TP distance kontrolü
            tp_distance_pct = abs(tp_price - entry_price) / entry_price
            
            if tp_distance_pct < self.safety_limits['min_tp_distance_pct']:
                # Çok yakın TP, minimum distance'a çek
                tp_price = entry_price * (1 + self.safety_limits['min_tp_distance_pct'])
                logging.warning(f"TP too close, adjusted to minimum distance: {self.safety_limits['min_tp_distance_pct']:.1%}")
            
            elif tp_distance_pct > self.safety_limits['max_tp_distance_pct']:
                # Çok uzak TP, maximum distance'a çek
                tp_price = entry_price * (1 + self.safety_limits['max_tp_distance_pct'])
                logging.warning(f"TP too far, adjusted to maximum distance: {self.safety_limits['max_tp_distance_pct']:.1%}")
            
            return sl_price, tp_price
            
        except Exception as e:
            logging.error(f"Safety limits application failed: {e}")
            return sl_price, tp_price
    
    def should_update_sltp(self, 
                          trade: TradeMetrics, 
                          current_price: float, 
                          current_atr: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Mevcut pozisyon için SL/TP güncelleme gerekli mi kontrol et
        
        Returns:
            (should_update, new_sl_price, new_tp_price)
        """
        try:
            # Trailing stop zaten aktifse, onu bozmayalım
            if trade.trailing_stop_activated:
                return False, None, None
            
            # Mevcut market conditions'ı al
            current_regime = self._detect_current_regime(current_price, trade.entry_price)
            
            # Yeni SL/TP hesapla
            new_sl, new_tp = self.calculate_dynamic_sltp(
                entry_price=trade.entry_price,
                atr_percent=current_atr,
                market_regime=current_regime,
                confidence=trade.confidence,
                funding_time_remaining=120  # Default 2 hours
            )
            
            # Güncelleme gerekli mi?
            sl_improvement = self._calculate_sl_improvement(trade.stop_loss, new_sl, trade.entry_price)
            tp_improvement = self._calculate_tp_improvement(trade.take_profit, new_tp, trade.entry_price)
            
            # En az %10 improvement varsa güncelle
            should_update = sl_improvement > 0.1 or tp_improvement > 0.1
            
            if should_update:
                logging.info(
                    f"[{trade.symbol}] SL/TP update recommended: "
                    f"SL improvement: {sl_improvement:.1%}, "
                    f"TP improvement: {tp_improvement:.1%}"
                )
                return True, new_sl, new_tp
            
            return False, None, None
            
        except Exception as e:
            logging.error(f"SL/TP update check failed: {e}")
            return False, None, None
    
    def _detect_current_regime(self, current_price: float, entry_price: float) -> MarketRegime:
        """Mevcut fiyat hareketine göre basit regime detection"""
        price_change_pct = (current_price - entry_price) / entry_price
        
        if price_change_pct > 0.02:  # %2+ yukarı
            return MarketRegime.UPTREND
        elif price_change_pct < -0.02:  # %2+ aşağı
            return MarketRegime.DOWNTREND
        else:
            return MarketRegime.SIDEWAYS_RANGING
    
    def _calculate_sl_improvement(self, old_sl: float, new_sl: float, entry_price: float) -> float:
        """SL improvement'ı hesapla (pozitif değer = improvement)"""
        if old_sl <= 0 or new_sl <= 0 or entry_price <= 0:
            return 0.0
        
        old_distance = abs(entry_price - old_sl) / entry_price
        new_distance = abs(entry_price - new_sl) / entry_price
        
        # LONG pozisyon için: daha yüksek SL = improvement
        if new_sl > old_sl:
            return (new_distance - old_distance) / old_distance  # Relative improvement
        else:
            return 0.0
    
    def _calculate_tp_improvement(self, old_tp: float, new_tp: float, entry_price: float) -> float:
        """TP improvement'ı hesapla (pozitif değer = improvement)"""
        if old_tp <= 0 or new_tp <= 0 or entry_price <= 0:
            return 0.0
        
        old_distance = abs(old_tp - entry_price) / entry_price
        new_distance = abs(new_tp - entry_price) / entry_price
        
        # LONG pozisyon için: daha yüksek TP = improvement
        if new_tp > old_tp:
            return (new_distance - old_distance) / old_distance  # Relative improvement
        else:
            return 0.0
    
    def get_sltp_info(self, trade: TradeMetrics) -> Dict[str, any]:
        """SL/TP bilgilerini döndür (monitoring için)"""
        try:
            entry_price = trade.entry_price
            sl_distance_pct = abs(entry_price - trade.stop_loss) / entry_price * 100
            tp_distance_pct = abs(trade.take_profit - entry_price) / entry_price * 100
            
            return {
                'symbol': trade.symbol,
                'entry_price': entry_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'sl_distance_pct': sl_distance_pct,
                'tp_distance_pct': tp_distance_pct,
                'trailing_active': trade.trailing_stop_activated,
                'market_regime': trade.market_regime,
                'confidence': trade.confidence,
                'volatility': trade.volatility
            }
            
        except Exception as e:
            logging.error(f"SL/TP info extraction failed: {e}")
            return {'error': str(e)}

# ===== V13 GÜNCELLEMESİ - Gelişmiş Piyasa Rejimi Tespiti =====
class EnhancedMarketRegimeDetector:
    @staticmethod
    @safe_dataframe_operations
    def detect_regime(df: pd.DataFrame) -> MarketRegime:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 50:
                return MarketRegime.UNKNOWN

            required_cols = ['close', 'atr_percent']
            if any(col not in df.columns for col in required_cols):
                return MarketRegime.UNKNOWN
            
            df_copy = df[required_cols].dropna()
            if len(df_copy) < 50:
                return MarketRegime.UNKNOWN

            last_close = df_copy['close'].iloc[-1]
            last_atr = df_copy['atr_percent'].iloc[-1]

            # Hem kısa (hızlı) hem de uzun (yavaş) vadeli trendi ölç
            ema20 = ta.trend.ema_indicator(df_copy['close'], window=20)
            sma50 = ta.trend.sma_indicator(df_copy['close'], window=50)

            if ema20 is None or sma50 is None or ema20.dropna().empty or sma50.dropna().empty:
                return MarketRegime.UNKNOWN
            
            last_ema20 = ema20.dropna().iloc[-1]
            last_sma50 = sma50.dropna().iloc[-1]
            
            # 1. Yüksek Volatilite her şeyden önce gelir
            if last_atr > 4.5:
                return MarketRegime.HIGH_VOLATILITY

            # 2. Güçlü Düşüş Trendi Tespiti (en önemli filtre)
            if last_ema20 < last_sma50:
                return MarketRegime.DOWNTREND

            # 3. Güçlü Yükseliş Trendi Tespiti
            if last_close > last_ema20 and last_close > last_sma50:
                return MarketRegime.UPTREND
            
            # 4. Diğer tüm durumlar "Yatay/Kararsız" kabul edilir
            return MarketRegime.SIDEWAYS_RANGING

        except Exception as e:
            logging.error(f"Error in detect_regime: {e}", exc_info=True)
            return MarketRegime.UNKNOWN

# ===== V9.2 GÜNCELLEMESİ - Portföy Korelasyon Analizi =====
class EnhancedPortfolioRiskManager:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        # Fiyat geçmişlerini tekrar tekrar API'den çekmemek için bir önbellek
        self.price_history_cache = {}
        self.cache_expiry = timedelta(minutes=30)

    async def check_correlation_risk(self, new_symbol: str) -> bool:
        """
        V14 - Merkezi config objesini kullanarak çalışır.
        Kendisiyle korelasyonunu kontrol etme hatası düzeltildi.
        """
        if not config.ENABLE_CORRELATION_FILTER or not self.bot.active_positions:
            return True

        logging.debug(f"[{new_symbol}] Checking correlation risk against {list(self.bot.active_positions.keys())}...")
        
        new_symbol_prices = await self._get_price_history(new_symbol)
        if new_symbol_prices is None or new_symbol_prices.empty:
            logging.warning(f"[{new_symbol}] Could not get price history for correlation check. Allowing trade.")
            return True

        for active_symbol in self.bot.active_positions.keys():
            # ===== YENİ EKLENEN KONTROL BURADA =====
            # Yeni adayı kendisiyle karşılaştırmayı engelle
            if new_symbol == active_symbol:
                continue
            # =======================================

            active_symbol_prices = await self._get_price_history(active_symbol)
            if active_symbol_prices is None or active_symbol_prices.empty:
                continue

            combined_df = pd.concat([new_symbol_prices, active_symbol_prices], axis=1).dropna()
            if len(combined_df) < 20: continue
            
            correlation = combined_df.iloc[:, 0].corr(combined_df.iloc[:, 1])
            logging.debug(f"[{new_symbol}] Correlation with {active_symbol}: {correlation:.2f}")

            if correlation > config.MAX_CORRELATION_THRESHOLD:
                rejection_msg = (
                    f"⚠️ Trade Rejected: High Correlation\n"
                    f"Symbol: <b>{new_symbol}</b>\n"
                    f"Reason: Correlation with <b>{active_symbol}</b> is <b>{correlation:.2f}</b>, "
                    f"which is above the threshold of <b>{config.MAX_CORRELATION_THRESHOLD}</b>."
                )
                logging.warning(rejection_msg.replace('\n', ' ').replace('<b>', '').replace('</b>', ''))
                # Telegram'a bu mesajın gönderilmesi isteğe bağlı, loglamak yeterli olabilir.
                # await self.bot.send_telegram(rejection_msg) 
                return False
        
        return True

    async def _get_price_history(self, symbol: str) -> Optional[pd.Series]:
        # ... Bu fonksiyonun içeriği aynı kalacak ...
        now = datetime.now(IST_TIMEZONE)
        if symbol in self.price_history_cache and (now - self.price_history_cache[symbol]['timestamp']) < self.cache_expiry:
            return self.price_history_cache[symbol]['prices']

        klines = await self.bot._api_request_with_retry(
            'GET', '/fapi/v1/klines', 
            {'symbol': symbol, 'interval': '5m', 'limit': 100}
        )
        if klines:
            prices = pd.to_numeric(pd.DataFrame(klines)[4])
            self.price_history_cache[symbol] = {'prices': prices, 'timestamp': now}
            return prices
            
        logging.warning(f"Could not fetch price history for {symbol}")
        return None

# Sildiğiniz üç sınıfın yerine bu yeni sınıfı yapıştırın

class EnsemblePredictor:
    """
    Unified Ensemble Predictor (V15.3 Refactored)
    - Combines Traditional ML (XGB, LGB) and Neural Networks.
    - Intelligently decides whether to use Neural Networks based on data and availability.
    - Performs dynamic, performance-based weighting for all models.
    - Manages its own training lifecycle and provides detailed info.
    """
    def __init__(self, use_neural_networks: bool = True):
        self.traditional_models: Dict[str, Any] = {}
        self.neural_network: Optional[NeuralNetworkPredictor] = None
        self.model_weights: Dict[str, float] = {}
        
        self.is_trained: bool = False
        self.can_use_nn: bool = use_neural_networks and TENSORFLOW_AVAILABLE
        self.nn_was_trained: bool = False
        
        self.training_time: Optional[timedelta] = None
        self.model_scores: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Trains the entire ensemble: traditional models first, then optionally Neural Networks."""
        if len(X) < 50 or y.sum() < 5:
            logging.warning("Insufficient data for training. Skipping.")
            return False

        training_start = time.time()
        
        # --- 1. Train Traditional Models ---
        self._fit_traditional_models(X, y)
        if not self.traditional_models:
            logging.error("Failed to train any traditional models. Aborting ensemble training.")
            return False

        # --- 2. Train Neural Network (if conditions are met) ---
        if self.can_use_nn and len(X) >= 100:
            self.neural_network = NeuralNetworkPredictor()
            if self.neural_network.fit(X, y):
                self.nn_was_trained = True
                nn_info = self.neural_network.get_model_info()
                if nn_info.get('trained') and nn_info.get('models'):
                    best_nn_score = max(m['score'] for m in nn_info['models'])
                    # Give NN a slightly lower initial score to prevent overfitting dominance
                    self.model_scores['neural_network'] = best_nn_score * 0.85
        
        # --- 3. Calculate Final Weights ---
        self._calculate_weights()
        if not self.model_weights:
            logging.error("Failed to calculate model weights.")
            return False

        self.is_trained = True
        self.training_time = timedelta(seconds=time.time() - training_start)
        
        logging.info(f"🚀 Ensemble training complete in {self.training_time.total_seconds():.1f}s. NN Trained: {self.nn_was_trained}")
        logging.info(f"📊 Final Weights: {self.model_weights}")
        return True

    def _fit_traditional_models(self, X: np.ndarray, y: np.ndarray):
        """Trains XGBoost and LightGBM models."""
        models_to_train = {
            'xgb': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0),
            'lgb': LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1, force_row_wise=True)
        }
        
        successful_models = {}
        for name, model in models_to_train.items():
            try:
                # Use a small validation set to get a score
                split_idx = int(len(X) * 0.85)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                model.fit(X_train, y_train)
                
                # Use precision score on validation set
                if len(X_val) > 0:
                    preds = model.predict(X_val)
                    score = precision_score(y_val, preds, zero_division=0.0)
                    self.model_scores[name] = max(score, 0.1) # Ensure a minimum score
                    successful_models[name] = model
            except Exception as e:
                logging.warning(f"Failed to train traditional model {name}: {e}")
        
        self.traditional_models = successful_models
        
    def _calculate_weights(self):
        """Calculates model weights based on their scores."""
        if not self.model_scores:
            return

        total_score = sum(self.model_scores.values())
        if total_score <= 0:
            # Fallback to equal weights
            num_models = len(self.model_scores)
            self.model_weights = {name: 1.0 / num_models for name in self.model_scores.keys()}
            return

        # Normalize scores to get weights
        self.model_weights = {name: score / total_score for name, score in self.model_scores.items()}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generates a weighted prediction from all trained models."""
        if not self.is_trained:
            raise ValueError("EnsemblePredictor is not trained.")

        all_predictions = []
        all_weights = []
        
        # --- Get predictions from traditional models ---
        for name, model in self.traditional_models.items():
            if name in self.model_weights:
                try:
                    preds = model.predict_proba(X)[:, 1]
                    all_predictions.append(preds)
                    all_weights.append(self.model_weights[name])
                except Exception as e:
                    logging.warning(f"Prediction failed for {name}: {e}")

        # --- Get predictions from Neural Network ---
        if self.nn_was_trained and 'neural_network' in self.model_weights:
            try:
                nn_preds = self.neural_network.predict_proba(X)[:, 1]
                all_predictions.append(nn_preds)
                all_weights.append(self.model_weights['neural_network'])
            except Exception as e:
                logging.warning(f"Prediction failed for Neural Network: {e}")

        if not all_predictions:
            # Emergency fallback if all models fail
            logging.error("All models failed to predict. Returning 0.5 probability.")
            fallback_prob = np.full(X.shape[0], 0.5)
            return np.column_stack([1 - fallback_prob, fallback_prob])

        # --- Combine predictions with weights ---
        # Re-normalize weights for the models that successfully predicted
        total_weight = sum(all_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in all_weights]
            final_pred = np.average(all_predictions, axis=0, weights=normalized_weights)
        else:
            final_pred = np.mean(all_predictions, axis=0) # Fallback if all weights are zero

        final_pred = np.clip(final_pred, 0.01, 0.99) # Clip to avoid extreme values
        return np.column_stack([1 - final_pred, final_pred])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Returns a dictionary with comprehensive info about the ensemble."""
        info = {
            'is_trained': self.is_trained,
            'training_time_seconds': self.training_time.total_seconds() if self.training_time else 0,
            'nn_was_trained': self.nn_was_trained,
            'model_scores': self.model_scores,
            'model_weights': self.model_weights,
            'total_models': len(self.model_weights),
        }
        if self.nn_was_trained and self.neural_network:
            info['neural_network_details'] = self.neural_network.get_model_info()
        return info

# ===== V13 - Canlı Kontrol Kulesi için TelegramCommandHandler =====
class TelegramCommandHandler:
    def __init__(self, bot):
        self.bot = bot
        self.commands = {
            '/exit_emergency': self.cmd_exit_emergency,
            '/status': self.cmd_status,
            '/help': self.cmd_help,
            '/positions': self.cmd_positions,
            '/analyze': self.cmd_analyze,
            '/config': self.cmd_config,
            '/set': self.cmd_set,
            '/memory': self.cmd_memory,
            '/force_sync': self.cmd_force_sync,
            '/emergency_cleanup': self.cmd_force_sync,
            # '/memory_report': self.cmd_memory_report,      # ← Bu satırı ekleyin
            # '/memory_health': self.cmd_memory_health,      # ← Bu satırı ekleyin
            '/database': self.cmd_database,
            '/force_cleanup': self.cmd_force_cleanup,
            '/health': self.cmd_health,
            '/performance': self.cmd_performance,
            '/errors': self.cmd_errors,
            '/recovery': self.cmd_recovery,
            '/emergency': self.cmd_emergency,
            '/circuits': self.cmd_circuits,
            '/shutdown': self.cmd_shutdown,
            '/ai': self.cmd_ai_status,
            '/ai_details': self.cmd_ai_details,
            '/debug': self.cmd_debug,
            '/fr': self.cmd_fr,
        }

    async def cmd_exit_emergency(self):
        """Force exit emergency mode"""
        try:
            if self.bot.error_recovery.emergency_mode:
                await self.bot.error_recovery.exit_emergency_mode()
                await self.bot.send_telegram("✅ <b>Emergency mode manually deactivated</b>")
            else:
                await self.bot.send_telegram("ℹ️ Emergency mode is not active")
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error exiting emergency mode: {e}")

    async def cmd_database(self):
        """Database durumunu gösteren komut"""
        try:
            db_info = await self.bot.db_manager.get_database_info()
            
            if 'error' in db_info:
                await self.bot.send_telegram(f"❌ Database error: {db_info['error']}")
                return
            
            # Cleanup bilgilerini hesapla
            next_cleanup = db_info['next_cleanup']
            time_to_cleanup = next_cleanup - datetime.now(IST_TIMEZONE)
            days_to_cleanup = max(0, time_to_cleanup.days)
            
            message = (
                f"🗃️ <b>Database Status</b>\n\n"
                f"<b>Total Trades:</b> {db_info['total_trades']:,}\n"
                f"<b>Recent (30d):</b> {db_info['recent_trades_30d']:,}\n"
                f"<b>File Size:</b> {db_info['file_size_mb']:.1f}MB\n\n"
                f"<b>Cleanup Info:</b>\n"
                f"• Last: {db_info['last_cleanup'].strftime('%Y-%m-%d %H:%M')}\n"
                f"• Next: in {days_to_cleanup} days"
            )
            
            if db_info['file_size_mb'] > 50:
                message += "\n\n⚠️ <b>Large database file!</b>"
                
            if days_to_cleanup <= 1:
                message += "\n\n🔧 <b>Cleanup due soon</b>"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting database status: {e}")

    async def cmd_cleanup_db(self):
        """Manuel database cleanup komutu"""
        try:
            await self.bot.send_telegram("🗃️ Starting database cleanup...")
            
            await self.bot.db_manager.cleanup_old_trades(days_to_keep=90)
            db_info = await self.bot.db_manager.get_database_info()
            
            message = (
                f"✅ <b>Database Cleanup Completed</b>\n\n"
                f"<b>Total Trades:</b> {db_info['total_trades']:,}\n"
                f"<b>File Size:</b> {db_info['file_size_mb']:.1f}MB"
            )
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Database cleanup failed: {e}") 

    async def cmd_debug(self):
        """Geçici hata ayıklama komutu."""
        try:
            message = "🕵️‍♂️ <b>DEBUG REPORT</b> 🕵️‍♂️\n\n"
            
            # 1. Trade history'de veri var mı kontrol et
            if not self.bot.trade_history:
                message += "❌ Trade history is EMPTY. No data loaded from database.\n"
                await self.bot.send_telegram(message)
                return

            message += f"✅ Trade history contains {len(self.bot.trade_history)} items.\n\n"
            
            # 2. Sadece kapanmış son işlemi bul
            last_closed_trade = None
            for trade in reversed(self.bot.trade_history):
                if trade.exit_time:
                    last_closed_trade = trade
                    break
            
            if not last_closed_trade:
                message += "❌ No CLOSED trades found in history.\n"
                await self.bot.send_telegram(message)
                return

            message += "🔍 Analyzing last closed trade:\n"
            
            # 3. Son işlemin verilerini analiz et
            trade = last_closed_trade
            message += f"   - Symbol: {trade.symbol}\n"
            
            # exit_time'ın tipini ve değerini kontrol et
            exit_time = trade.exit_time
            message += f"   - Exit Time Value: `{exit_time}`\n"
            message += f"   - Exit Time Type: `{type(exit_time)}`\n"

            # 4. Saat dilimi kontrolü yap
            if isinstance(exit_time, datetime):
                if exit_time.tzinfo is None:
                    message += "   - ⚠️ **Timezone: NAIVE (Missing timezone info!)**\n"
                else:
                    message += f"   - ✅ Timezone: `{exit_time.tzinfo}` (OK)\n"
                
                # 5. /status komutundaki filtrelemeyi simüle et
                try:
                    today_ist = datetime.now(IST_TIMEZONE).date()
                    trade_date = exit_time.astimezone(IST_TIMEZONE).date()
                    is_today = (trade_date == today_ist)
                    
                    message += "\n🔬 Simulating filter:\n"
                    message += f"   - Today (IST): `{today_ist}`\n"
                    message += f"   - Trade Date (IST): `{trade_date}`\n"
                    message += f"   - Match? **{is_today}**\n"
                except Exception as e:
                    message += f"   - ❌ **Filter simulation FAILED:** `{e}`\n"
            else:
                message += "   - ❌ **Exit time is not a datetime object!**\n"
            
            await self.bot.send_telegram(message)

        except Exception as e:
            await self.bot.send_telegram(f"❌ Debug command failed: {e}")

    async def cmd_memory(self):
        """Memory durumunu gösteren komut"""
        try:
            status = self.bot.get_memory_status()
            
            if 'error' in status:
                await self.bot.send_telegram(f"❌ Memory status error: {status['error']}")
                return
            
            message = (
                f"💾 <b>Memory Status</b>\n\n"
                f"<b>RAM Usage:</b> {status['memory_mb']:.1f}MB\n"
                f"<b>Memory %:</b> {status['memory_percentage']:.1f}%\n"
                f"<b>Available:</b> {status['available_mb']:.1f}MB\n\n"
                f"<b>Cache Stats:</b>\n"
                f"• Trades: {status['trade_history_count']}/{status['trade_history_limit']}\n"
                f"• Models: {status['models_cached']}/{status['models_limit']}\n"
                f"• Active Positions: {status['active_positions']}\n"
                f"• Cooldowns: {status['cooldown_symbols']}"
            )
            
            if status['memory_mb'] > 400:
                message += "\n\n⚠️ <b>High memory usage!</b>"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting memory status: {e}")

    async def handle_command(self, message_text: str):
        parts = message_text.strip().lower().split(' ')
        command, args = parts[0], parts[1:]
        if command in self.commands:
            await self.commands[command](args) if command == '/set' else await self.commands[command]()
        else:
            await self.cmd_help()

    async def cmd_help(self):
        help_text = (
            "🤖 <b>FR Hunter V15 - Commands</b>\n\n"
            "<b>📊 Trading & Status:</b>\n"
            "<code>/status</code> - Bot status & performance\n"
            "<code>/positions</code> - Active positions\n"
            "<code>/analyze</code> - Performance analysis\n"
            "<code>/fr</code> - Scan for FR opportunities & AI scores\n\n"  # <-- YENİ SATIR EKLENDİ
            "<b>⚙️ Configuration:</b>\n"
            "<code>/config</code> - Current settings\n"
            "<code>/set [key] [value]</code> - Change setting\n\n"
            "<b>🔧 System Monitoring:</b>\n"
            "<code>/health</code> - System health status\n"
            "<code>/performance</code> - Performance metrics\n"
            "<code>/memory</code> - Basic memory usage\n"
            "<code>/memory_report</code> - Detailed memory report\n"
            "<code>/memory_health</code> - Memory health score\n"
            "<code>/database</code> - Database status\n"
            "<code>/errors</code> - Recent errors\n\n"
            "<b>🧠 AI System:</b>\n"
            "<code>/ai</code> - AI models & features status\n"
            "<code>/ai_details</code> - Detailed AI model info\n\n"
            "<b>🚨 Error Recovery:</b>\n"
            "<code>/recovery</code> - Recovery system status\n"
            "<code>/emergency</code> - Emergency mode info\n"
            "<code>/circuits</code> - Circuit breaker status\n"
            "<code>/shutdown</code> - Graceful shutdown\n\n"
            "<b>🎯 Configuration Examples:</b>\n"
            "<code>/set ai_confidence_threshold 0.55</code>\n"
            "<code>/set FR_LONG_THRESHOLD -0.05</code>\n"
            "<code>/set MAX_ACTIVE_TRADES 3</code>\n"
            "<code>/set ENABLE_DYNAMIC_SETTINGS false</code>\n"
            "<code>/set MAX_LOSS_PER_TRADE_USD 20.0</code>\n"
            "<code>/set COOLDOWN_PERIOD_MINUTES 15</code>\n"
            "<code>/set ENABLE_TRAILING_STOP true</code>\n"
            "<code>/set ENABLE_PUMP_FILTER false</code>\n\n"
            "<i>💡 Use /config to see all current settings</i>"
        )
        await self.bot.send_telegram(help_text)

    # YENİ /config KOMUTU
    async def cmd_config(self):
        c = self.bot
        message = (
            f"⚙️ <b>CURRENT LIVE SETTINGS</b>\n\n"
            f"<b>FR Threshold:</b> <code>{config.FR_LONG_THRESHOLD}</code>\n"
            f"<b>AI Confidence:</b> <code>{config.AI_CONFIDENCE_THRESHOLD:.2f}</code>\n"
            f"<b>Max Active Trades:</b> <code>{config.MAX_ACTIVE_TRADES}</code>\n"
            f"<b>Max Loss / Trade:</b> <code>${config.MAX_LOSS_PER_TRADE_USD}</code>\n"
            f"<b>Cooldown (min):</b> <code>{config.COOLDOWN_PERIOD_MINUTES}</code>\n\n"
            f"<i>Use /set [key] [value] to change.</i>"
        )
        await self.bot.send_telegram(message)

    # YENİ /set KOMUTU
    async def cmd_set(self, args: List[str]):
        if len(args) != 2:
            await self.bot.send_telegram("❌ Invalid format. Use: <code>/set [key] [value]</code>")
            return
        
        key_to_set, new_value_str = args[0], args[1]
        
        # Global config objesinin set_value metodunu çağır
        success = config.set_value(key_to_set, new_value_str)

        if success:
            # Güncellenen değeri yine global config'den oku
            updated_value = getattr(config, key_to_set.upper())
            await self.bot.send_telegram(f"✅ Setting updated: <b>{key_to_set.upper()}</b> is now <b>{updated_value}</b>")
            await self.cmd_config()
        else:
            await self.bot.send_telegram(f"❌ Failed to update '<code>{key_to_set}</code>'.")
 
    # YENİ /fr KOMUTU

    async def cmd_fr(self):
        """
        Funding rate taramasını, ana döngüyü BLOKLAMADAN bir arka plan görevi
        olarak başlatır ve kullanıcıya anında yanıt verir.
        """
        try:
            await self.bot.send_telegram("⏳ <b>Funding Rate Taraması Başlatıldı...</b>\nSonuçlar hazır olduğunda ayrı bir mesaj olarak gönderilecektir.")
            asyncio.create_task(self._fr_background_worker())
        except Exception as e:
            logging.error(f"Failed to start /fr background task: {e}")
            await self.bot.send_telegram("❌ Tarama görevi başlatılamadı.")

    # YENİ ARKA PLAN GÖREVİ
    async def _fr_background_worker(self):
        """
        Performs a fast FR scan and reports the raw candidates directly to the user,
        without performing any AI analysis. This provides an instant snapshot of opportunities.
        """
        try:
            # Botun ana hızlı tarama fonksiyonunu kullan
            # Bu, tek bir API çağrısıyla tüm piyasayı tarar.
            all_funding_data = await self.bot._api_request_with_retry('GET', '/fapi/v1/premiumIndex')
            if not all_funding_data:
                await self.bot.send_telegram("❌ FR verileri çekilemedi.")
                return

            candidates = []
            for item in all_funding_data:
                try:
                    if not isinstance(item, dict) or not item.get('symbol', '').endswith('USDT'):
                        continue
                    
                    symbol = item['symbol']
                    current_fr = float(item.get('lastFundingRate', 0)) * 100

                    # Kullanıcının görmesi için daha geniş bir aralık kullanalım.
                    # Bu, ana botun işlem filtresinden farklı olabilir.
                    if -5.0 <= current_fr < -0.08:
                        candidates.append({'symbol': symbol, 'fr': current_fr})
                except (ValueError, KeyError):
                    continue

            if not candidates:
                await self.bot.send_telegram("ℹ️ Belirtilen aralıkta (-0.08% ile -5.00%) uygun FR bulunamadı.")
                return

            # Adayları FR'ye göre sırala
            candidates.sort(key=lambda x: x['fr'])
            
            # Mesajı oluştur
            message = "✅ <b>Hızlı FR Taraması Sonuçları</b> ✅\n\n<pre>"
            message += f"{'Symbol':<15} {'FR Rate':>12}\n"
            message += "-" * 28 + "\n"
            for c in candidates:
                fr_str = f"{c['fr']:.3f}%"
                message += f"{c['symbol']:<15} {fr_str:>12}\n"
            message += f"</pre>\n----------------------------------------\n"
            message += f"⏰ {datetime.now(IST_TIMEZONE).strftime('%H:%M')} | Toplam: {len(candidates)}"

            await self.bot.send_telegram(message)

        except Exception as e:
            logging.error(f"Error in /fr background worker: {e}", exc_info=True)
            await self.bot.send_telegram(f"❌ FR taraması sırasında bir hata oluştu.")
 
    # MEMORY
    async def cmd_memory_report(self):
        """Enhanced memory report command"""
        try:
            await self.bot.send_telegram("📊 Generating detailed memory report...")
            report = await self.bot.memory_manager.generate_memory_report()
            await self.bot.send_telegram(report)
        except Exception as e:
            logging.error(f"Error generating memory report: {e}")
            await self.bot.send_telegram(f"❌ Error generating memory report: {e}")

    async def cmd_memory_health(self):
        """Memory health score command"""
        try:
            health_score = self.bot.memory_manager.get_memory_health_score()
            stats = self.bot.memory_manager.get_detailed_memory_stats()
            
            if 'error' in stats:
                await self.bot.send_telegram(f"❌ Memory health check error: {stats['error']}")
                return
            
            zone_emoji = {
                'green': '🟢',
                'yellow': '🟡', 
                'orange': '🟠',
                'red': '🔴'
            }
            
            message = (
                f"{zone_emoji.get(stats['current_zone'], '⚪')} <b>Memory Health Check</b>\n\n"
                f"<b>Health Score:</b> {health_score:.0f}/100\n"
                f"<b>Current Zone:</b> {stats['current_zone'].upper()}\n"
                f"<b>Memory Usage:</b> {stats['memory_percent']:.1f}% ({stats['rss_mb']:.1f}MB)\n"
                f"<b>Memory Limit:</b> {stats['max_memory_mb']}MB\n"
                f"<b>Cache Memory:</b> {stats['cache_memory']['total_cache_mb']:.1f}MB\n\n"
                f"<b>Zone Thresholds:</b>\n"
                f"🟢 Green: < {stats['zone_thresholds']['yellow']:.0f}MB\n"
                f"🟡 Yellow: {stats['zone_thresholds']['yellow']:.0f}MB - {stats['zone_thresholds']['orange']:.0f}MB\n"
                f"🟠 Orange: {stats['zone_thresholds']['orange']:.0f}MB - {stats['zone_thresholds']['red']:.0f}MB\n"
                f"🔴 Red: > {stats['zone_thresholds']['red']:.0f}MB"
            )
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            logging.error(f"Error checking memory health: {e}")
            await self.bot.send_telegram(f"❌ Error checking memory health: {e}")


    async def cmd_positions(self):
        if not self.bot.active_positions:
            await self.bot.send_telegram("📭 No active positions.")
            return

        message = "📊 <b>ACTIVE POSITIONS</b>\n"
        for symbol, trade in self.bot.active_positions.items():
            try:
                current_price = await self.bot.get_current_price(symbol)
                if current_price is None:
                    current_price = trade.entry_price # Fiyat alınamazsa son bilineni kullan
                
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                pnl_usd = pnl_pct * trade.quantity * trade.entry_price
                
                ts_status = "(Trailing)" if trade.trailing_stop_activated else ""
                
                message += (
                    f"\n💎 <b>{symbol}</b>\n"
                    f"   <b>P&L:</b> ${pnl_usd:+.2f} ({pnl_pct:+.2%})\n"
                    f"   <b>Entry:</b> ${trade.entry_price:.4f}\n"
                    f"   <b>Current:</b> ${current_price:.4f}\n"
                    f"   <b>SL:</b> ${trade.stop_loss:.4f} {ts_status}"
                )
            except Exception as e:
                logging.error(f"Error processing position {symbol} for /positions: {e}")
                message += f"\n💎 <b>{symbol}</b> - Error fetching details."
        
        await self.bot.send_telegram(message)

    async def cmd_status(self):
        """V15.3 - Provides a clean and comprehensive status report."""
        try:
            active_pos_count = len(self.bot.active_positions)
            
            balance_str = ""
            if config.TRADING_MODE == TradingMode.LIVE:
                live_balance = await self.bot.get_futures_balance()
                balance_str = f"► <b>Balance:</b> ${live_balance:,.2f} (Live)" if live_balance is not None else "► <b>Balance:</b> Error"
            else:
                balance_str = f"► <b>Balance:</b> ${self.bot.paper_balance:,.2f} (Paper)"

            today_ist = datetime.now(IST_TIMEZONE).date()
            daily_stats = await self.bot.db_manager.get_stats_for_date(today_ist)
            
            daily_pnl = 0.0
            daily_wins = 0
            daily_losses = 0
            total_daily_trades = 0

            if daily_stats:
                daily_pnl = daily_stats.get('total_pnl', 0.0)
                daily_wins = daily_stats.get('winning_trades', 0)
                total_daily_trades = daily_stats.get('total_trades', 0)
                daily_losses = total_daily_trades - daily_wins

            pnl_emoji = "🟢" if daily_pnl > 0 else ("🔴" if daily_pnl < 0 else "⚪️")
            
            message = (
                f"📊 <b>FR HUNTER V15.3 STATUS</b>\n"
                f"----------------------------------------\n"
                f"<b><u>CORE</u></b>\n"
                f"► <b>Mode:</b> {config.TRADING_MODE.value.upper()}\n"
                f"{balance_str}\n"
                f"► <b>Active Trades:</b> {active_pos_count} / {config.MAX_ACTIVE_TRADES}\n\n"
                
                f"<b><u>TODAY'S PERFORMANCE</u></b>\n"
                f"{pnl_emoji} <b>Daily P&L:</b> ${daily_pnl:,.2f}\n"
                f"► <b>Trades Closed:</b> {total_daily_trades} ({daily_wins} W / {daily_losses} L)\n\n"

                f"<b><u>CURRENT SETTINGS</u></b>\n"
                f"► <b>FR Threshold:</b> {config.FR_LONG_THRESHOLD}%\n"
                f"► <b>AI Confidence:</b> {config.AI_CONFIDENCE_THRESHOLD * 100:.0f}%"
            )
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            logging.error(f"Error in cmd_status: {e}", exc_info=True)
            await self.bot.send_telegram("❌ Error retrieving status.")
            
    # YENİ ANALİZ METODU
    async def cmd_analyze(self):
        await self.bot.send_telegram("🔬 Generating performance analysis...")
        try:
            report = await self.generate_analysis_report()
            await self.bot.send_telegram(f"<pre>{report}</pre>")
        except Exception as e:
            logging.error(f"Error generating analysis report: {e}", exc_info=True)
            await self.bot.send_telegram("❌ Failed to generate analysis report.")
            
    async def generate_analysis_report(self) -> str:
        """V12.1 - Saat dilimi (timezone) hatası düzeltildi."""
        
        # --- DÜZELTME BURADA ---
        # Rapor için başlangıç ve bitiş tarihlerini oluştururken,
        # onlara hangi saat diliminde olduklarını söylüyoruz (IST_TIMEZONE).
        end_date = datetime.now(IST_TIMEZONE)
        start_date = end_date - timedelta(days=config.ANALYSIS_REPORT_DAYS)
        
        trades = [t for t in self.bot.trade_history if t.exit_time and start_date <= t.exit_time.astimezone(IST_TIMEZONE) <= end_date]
        
        if not trades:
            return f"No closed trades found in the last {config.ANALYSIS_REPORT_DAYS} days."


        # Rapor oluşturma kısmı (bu kısım aynı kalabilir)
        df = pd.DataFrame([t.__dict__ for t in trades])
        # entry_time'ı da timezone-aware yapalım
        df['entry_time'] = pd.to_datetime(df['entry_time']).dt.tz_convert(IST_TIMEZONE)
        df['hour'] = df['entry_time'].apply(lambda x: x.hour)
        df['pnl_is_positive'] = df['pnl'] > 0

        # Saatlik Analiz
        # pd.cut için saat dilimini kaldırmamız gerekebilir
        hourly_analysis = df.groupby(pd.cut(df['hour'], bins=[-1, 3, 7, 11, 15, 19, 23], labels=['00-04', '04-08', '08-12', '12-16', '16-20', '20-24'])).agg(
            total_pnl=('pnl', 'sum'),
            trade_count=('pnl', 'size'),
            win_rate=('pnl_is_positive', 'mean')
        ).reset_index()
        
        # Rejim Analizi
        regime_analysis = df.groupby('market_regime').agg(
            total_pnl=('pnl', 'sum'),
            trade_count=('pnl', 'size'),
            win_rate=('pnl_is_positive', 'mean')
        ).reset_index()
        
        # Raporu oluştur
        report = f"📊 PERFORMANCE ANALYSIS (Last {config.ANALYSIS_REPORT_DAYS} Days) 📊\n\n"
        report += "--- BY TRADING HOUR (UTC+3) ---\n"
        for _, row in hourly_analysis.iterrows():
            report += f"• {str(row['hour']):>7}: ${row['total_pnl']:>7.2f} ({row['trade_count']}T, {row['win_rate']:.0%} WR)\n"
        
        report += "\n--- BY MARKET REGIME ---\n"
        for _, row in regime_analysis.iterrows():
            report += f"• {row['market_regime']:>10}: ${row['total_pnl']:>7.2f} ({row['trade_count']}T, {row['win_rate']:.0%} WR)\n"

        return report

    async def cmd_health(self):
        """Sistem sağlığını gösteren komut"""
        try:
            health_data = self.bot.health_monitor.check_system_health()
            
            uptime_hours = health_data['uptime_hours']
            uptime_days = int(uptime_hours // 24)
            uptime_hours_remaining = int(uptime_hours % 24)
            uptime_minutes = int((uptime_hours % 1) * 60)
            
            message = (
                f"{health_data['status_emoji']} <b>System Health - {health_data['status']}</b>\n\n"
                f"<b>Health Score:</b> {health_data['health_score']}/100\n"
                f"<b>Uptime:</b> {uptime_days}d {uptime_hours_remaining}h {uptime_minutes}m\n"
                f"<b>Error Rate:</b> {health_data['error_rate_per_hour']:.1f}/hour\n"
                f"<b>API Success:</b> {health_data['api_success_rate']:.1f}%\n"
                f"<b>Total Errors:</b> {health_data['total_errors']}\n"
                f"<b>Recent Errors:</b> {health_data['recent_errors']} (last hour)"
            )
            
            # Durum bazlı ek bilgiler
            if health_data['health_score'] < 70:
                message += "\n\n⚠️ <b>Recommendations:</b>"
                if health_data['error_rate_per_hour'] > 5:
                    message += "\n• High error rate detected"
                if health_data['api_success_rate'] < 90:
                    message += "\n• API issues detected"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting health status: {e}")

    async def cmd_performance(self):
        """Performans özetini gösteren komut"""
        try:
            perf_data = self.bot.health_monitor.get_performance_summary()
            
            message = (
                f"📊 <b>Performance Summary</b>\n\n"
                f"<b>Uptime:</b> {perf_data['uptime_days']}d {perf_data['uptime_hours']}h {perf_data['uptime_minutes']}m\n\n"
                f"<b>Today's Activity:</b>\n"
            )
            
            today_metrics = perf_data['today_metrics']
            if today_metrics:
                if 'trades_opened' in today_metrics:
                    message += f"• Trades Opened: {today_metrics['trades_opened']}\n"
                if 'trades_closed' in today_metrics:
                    message += f"• Trades Closed: {today_metrics['trades_closed']}\n"
                if 'api_calls_total' in today_metrics:
                    message += f"• API Calls: {today_metrics['api_calls_total']}\n"
                if 'telegram_messages_sent' in today_metrics:
                    message += f"• Messages Sent: {today_metrics['telegram_messages_sent']}\n"
            else:
                message += "• No activity recorded today\n"
            
            message += f"\n<b>Total Stats:</b>\n"
            total_metrics = perf_data['total_metrics']
            message += f"• Total Trades Opened: {total_metrics['trades_opened']}\n"
            message += f"• Total Trades Closed: {total_metrics['trades_closed']}\n"
            message += f"• Total API Calls: {total_metrics['api_calls_total']}\n"
            
            # Error breakdown
            error_breakdown = perf_data['error_breakdown']
            if error_breakdown['total_errors'] > 0:
                message += f"\n<b>Error Breakdown:</b>\n"
                message += f"• API Errors: {error_breakdown['api_errors']}\n"
                message += f"• Telegram Errors: {error_breakdown['telegram_errors']}\n"
                message += f"• Other Errors: {error_breakdown['other_errors']}\n"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting performance data: {e}")

    async def cmd_errors(self):
        """Son hataları gösteren komut"""
        try:
            recent_errors = list(self.bot.health_monitor.last_errors)
            
            if not recent_errors:
                await self.bot.send_telegram("✅ <b>No Recent Errors</b>\nSystem is running clean!")
                return
            
            # Son 10 hatayı göster
            recent_errors.reverse()  # En yeni önce
            errors_to_show = recent_errors[:10]
            
            message = f"🚨 <b>Recent Errors ({len(errors_to_show)}/{len(recent_errors)})</b>\n\n"
            
            for i, error in enumerate(errors_to_show, 1):
                timestamp = error['timestamp'].strftime('%H:%M:%S')
                error_type = error['type']
                source = error['source']
                message_text = error['message'][:80] + "..." if len(error['message']) > 80 else error['message']
                
                message += f"<b>{i}.</b> [{timestamp}] <code>{error_type}</code>\n"
                message += f"   Source: {source}\n"
                message += f"   {message_text}\n\n"
            
            if len(recent_errors) > 10:
                message += f"<i>Showing last 10 of {len(recent_errors)} total errors</i>"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting error log: {e}")

    async def cmd_recovery(self):
        """Error recovery durumunu gösteren komut"""
        try:
            recovery = self.bot.error_recovery
            
            message = f"🔧 <b>Error Recovery Status</b>\n\n"
            
            # Emergency mode durumu
            if recovery.emergency_mode:
                duration = datetime.now(IST_TIMEZONE) - recovery.emergency_start_time
                hours = int(duration.total_seconds() // 3600)
                minutes = int((duration.total_seconds() % 3600) // 60)
                message += f"🆘 <b>EMERGENCY MODE ACTIVE</b>\n"
                message += f"Duration: {hours}h {minutes}m\n\n"
            else:
                message += f"✅ Normal operation mode\n\n"
            
            # Recovery attempts
            if recovery.recovery_attempts:
                message += f"<b>Recent Recovery Attempts:</b>\n"
                for key, attempts in recovery.recovery_attempts.items():
                    if attempts > 0:
                        message += f"• {key}: {attempts} attempts\n"
                message += "\n"
            else:
                message += f"<b>No recent recovery attempts</b>\n\n"
            
            # Emergency triggers status
            try:
                triggers = recovery.should_trigger_emergency()
                active_triggers = [t for t, active in triggers.items() if active]
                
                if active_triggers:
                    message += f"⚠️ <b>Active Emergency Triggers:</b>\n"
                    for trigger in active_triggers:
                        message += f"• {trigger.replace('_', ' ').title()}\n"
                else:
                    message += f"✅ No emergency triggers active\n"
            except Exception as e:
                message += f"❌ Error checking triggers: {e}\n"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting recovery status: {e}")

    async def cmd_force_cleanup(self):
        """Force cleanup stuck positions"""
        try:
            stuck_positions = []
            async with self.bot._positions_lock:
                for symbol, trade in list(self.bot.active_positions.items()):
                    if hasattr(trade, '_closing_in_progress') and trade._closing_in_progress:
                        # Check if stuck for more than 5 minutes
                        if hasattr(trade, '_close_start_time'):
                            stuck_time = datetime.now(IST_TIMEZONE) - trade._close_start_time
                            if stuck_time.total_seconds() > 300:  # 5 minutes
                                stuck_positions.append(symbol)
                
                # Force remove stuck positions
                for symbol in stuck_positions:
                    del self.bot.active_positions[symbol]
                    logging.warning(f"Force removed stuck position: {symbol}")
            
            if stuck_positions:
                await self.bot.send_telegram(f"🧹 <b>Force Cleanup</b>\nRemoved stuck positions: {', '.join(stuck_positions)}")
            else:
                await self.bot.send_telegram("✅ No stuck positions found")
                
        except Exception as e:
            await self.bot.send_telegram(f"❌ Force cleanup error: {e}")

    async def cmd_force_sync(self):
        """Force sync positions with exchange"""
        try:
            await self.bot.send_telegram("🔄 <b>Force Position Sync</b>\nStarting emergency cleanup...")
            
            # Get live positions from exchange
            position_risk_data = await self.bot._api_request_with_retry('GET', '/fapi/v2/positionRisk', signed=True)
            
            if not position_risk_data:
                await self.bot.send_telegram("❌ Could not fetch exchange positions")
                return
            
            # Find symbols with zero position on exchange
            exchange_positions = {}
            for pos in position_risk_data:
                symbol = pos['symbol']
                position_amount = float(pos.get('positionAmt', '0'))
                exchange_positions[symbol] = position_amount
            
            # Force remove positions that are closed on exchange
            removed_positions = []
            async with self.bot._positions_lock:
                for symbol in list(self.bot.active_positions.keys()):
                    exchange_amount = exchange_positions.get(symbol, 0)
                    
                    if exchange_amount == 0:  # Position closed on exchange
                        trade = self.bot.active_positions[symbol]
                        del self.bot.active_positions[symbol]
                        removed_positions.append(symbol)
                        
                        # Clear any close flags
                        if hasattr(trade, '_closing_in_progress'):
                            delattr(trade, '_closing_in_progress')
                        
                        logging.warning(f"Force removed stuck position: {symbol}")
            
            if removed_positions:
                message = f"✅ <b>Emergency Cleanup Complete</b>\n\nRemoved stuck positions:\n"
                for symbol in removed_positions:
                    message += f"• {symbol}\n"
                message += f"\nActive positions now: {len(self.bot.active_positions)}"
            else:
                message = "ℹ️ No stuck positions found to remove"
                
            await self.bot.send_telegram(message)
            
        except Exception as e:
            logging.error(f"Force sync error: {e}")
            await self.bot.send_telegram(f"❌ Force sync failed: {e}")

    async def cmd_emergency(self):
        """Emergency mode kontrolü ve yönetimi"""
        try:
            recovery = self.bot.error_recovery
            
            if recovery.emergency_mode:
                # Emergency mode çıkış kontrolü
                can_exit = recovery.should_exit_emergency_mode()
                duration = datetime.now(IST_TIMEZONE) - recovery.emergency_start_time
                
                message = (
                    f"🆘 <b>Emergency Mode Active</b>\n\n"
                    f"<b>Duration:</b> {int(duration.total_seconds() // 60)} minutes\n"
                    f"<b>Can exit:</b> {'✅ Yes' if can_exit else '❌ No'}\n\n"
                )
                
                if can_exit:
                    message += f"<i>System appears stable. Emergency mode can be deactivated.</i>"
                else:
                    message += f"<i>System still unstable. Continue monitoring.</i>"
                    
            else:
                # Emergency triggers kontrolü
                triggers = recovery.should_trigger_emergency()
                active_triggers = [t for t, active in triggers.items() if active]
                
                message = f"✅ <b>Normal Operation Mode</b>\n\n"
                
                if active_triggers:
                    message += f"⚠️ <b>Warning - Emergency triggers detected:</b>\n"
                    for trigger in active_triggers:
                        message += f"• {trigger.replace('_', ' ').title()}\n"
                    message += f"\n<i>Emergency mode may activate automatically.</i>"
                else:
                    message += f"<b>All systems normal</b>\n"
                    message += f"No emergency triggers detected."
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error checking emergency status: {e}")

    async def cmd_circuits(self):
        """Circuit breaker durumlarını göster"""
        try:
            recovery = self.bot.error_recovery
            circuit_breakers = recovery.circuit_breakers
            
            if not circuit_breakers:
                await self.bot.send_telegram("🔌 <b>Circuit Breakers</b>\n\n✅ All circuits closed (normal operation)")
                return
            
            message = f"🔌 <b>Circuit Breaker Status</b>\n\n"
            
            for service, breaker in circuit_breakers.items():
                status_emoji = "🔴" if breaker['circuit_open'] else "🟢"
                status_text = "OPEN (Blocked)" if breaker['circuit_open'] else "CLOSED (Normal)"
                
                message += f"{status_emoji} <b>{service.upper()}</b>\n"
                message += f"   Status: {status_text}\n"
                message += f"   Failures: {breaker['failures']}\n"
                
                if breaker['circuit_open']:
                    time_remaining = recovery.circuit_breaker_config[service]['timeout'] - int((datetime.now(IST_TIMEZONE) - breaker['circuit_open_time']).total_seconds())
                    if time_remaining > 0:
                        message += f"   Reopens in: {time_remaining}s\n"
                    else:
                        message += f"   Ready to reopen\n"
                
                message += "\n"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting circuit breaker status: {e}")

    async def cmd_shutdown(self):
        """Graceful shutdown komutu (acil durum için)"""
        try:
            # Güvenlik sorusu
            confirmation_message = (
                f"⚠️ <b>SHUTDOWN CONFIRMATION</b>\n\n"
                f"This will initiate graceful shutdown.\n"
                f"Current active positions: {len(self.bot.active_positions)}\n\n"
                f"Type '<code>CONFIRM SHUTDOWN</code>' to proceed."
            )
            
            await self.bot.send_telegram(confirmation_message)
            
            # Not: Gerçek implementasyonda, bir confirmation sistemi eklenebilir
            # Şimdilik sadece uyarı gösteriyoruz
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error in shutdown command: {e}")

    async def cmd_ai_status(self):
        """AI sistem durumunu gösteren komut"""
        try:
            message = f"🧠 <b>AI System Status</b>\n\n"
            
            # TensorFlow durumu
            tf_available = globals().get('TENSORFLOW_AVAILABLE', False)
            message += f"<b>🔧 Infrastructure:</b>\n"
            message += f"• TensorFlow: <code>{'✅ Available' if tf_available else '❌ Not Available'}</code>\n"
            
            # Model durumu
            model_count = len(self.bot.ensemble_models)
            scaler_count = len(self.bot.scalers)
            message += f"\n<b>📊 Models:</b>\n"
            message += f"• Trained Models: <code>{model_count}</code>\n"
            message += f"• Scalers Available: <code>{scaler_count}</code>\n"
            
            # Feature durumu
            feature_count = len(self.bot.selected_features) if self.bot.selected_features else 0
            message += f"• Selected Features: <code>{feature_count}</code>\n"
            
            # Trading durumu
            balance = await self.bot.get_futures_balance()
            message += f"\n<b>💰 Trading:</b>\n"
            message += f"• Balance: <code>${balance:.2f}</code>\n"
            message += f"• Active Positions: <code>{len(self.bot.active_positions)}</code>\n"
            
            # RELAX mode countdown (eğer defensive mode'daysa)
            try:
                current_hour = datetime.now(IST_TIMEZONE).hour
                if config.DEFENSIVE_MODE_START_HOUR <= current_hour or current_hour < config.RELAX_MODE_START_HOUR:
                    # Defensive mode'dayız, RELAX mode'a ne kadar kaldı?
                    if current_hour >= config.DEFENSIVE_MODE_START_HOUR:
                        # Gece yarısından sonra RELAX mode
                        hours_to_relax = (24 - current_hour) + config.RELAX_MODE_START_HOUR
                    else:
                        # Sabah RELAX mode'a kadar
                        hours_to_relax = config.RELAX_MODE_START_HOUR - current_hour
                    
                    if hours_to_relax <= 3:
                        message += f"\n⏰ <b>RELAX Mode in {hours_to_relax}h</b>\n"
                        message += f"<i>(Neural Networks will be very active!)</i>\n"
            except:
                pass
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting AI status: {e}")

    async def cmd_ai_details(self):
        """Belirli bir sembol için AI model detayları"""
        try:
            if not self.bot.ensemble_models:
                await self.bot.send_telegram("📭 No AI models currently trained")
                return
            
            # En son trained olan modeli seç
            latest_symbol = list(self.bot.ensemble_models.keys())[-1]
            model = self.bot.ensemble_models[latest_symbol]
            
            message = f"🔍 <b>AI Model Details: {latest_symbol}</b>\n\n"
            
            if hasattr(model, 'get_model_info'):
                try:
                    model_info = model.get_model_info()
                    
                    # Model type
                    if hasattr(model, 'neural_network'):
                        message += f"<b>Type:</b> Advanced Ensemble\n"
                        message += f"<b>Total Models:</b> {model_info.get('total_models', 0)}\n"
                        message += f"<b>Traditional Models:</b> {', '.join(model_info.get('traditional_models', []))}\n"
                        
                        # Neural Network detayları
                        if model_info.get('neural_network_trained', False):
                            nn_details = model_info.get('neural_network_details', {})
                            if nn_details.get('trained', False):
                                nn_models = nn_details.get('models', [])
                                message += f"<b>Neural Networks:</b>\n"
                                for nn_model in nn_models:
                                    name = nn_model.get('name', 'Unknown')
                                    score = nn_model.get('score', 0)
                                    message += f"  • {name.upper()}: {score:.3f}\n"
                        else:
                            message += f"<b>Neural Networks:</b> Not trained\n"
                        
                        # Weights
                        weights = model_info.get('model_weights', {})
                        if weights:
                            message += f"\n<b>Model Weights:</b>\n"
                            for name, weight in weights.items():
                                message += f"  • {name}: {weight:.3f}\n"
                        
                        # Performance history
                        avg_confidence = model_info.get('avg_recent_confidence', 0)
                        if avg_confidence > 0:
                            message += f"\n<b>Avg Recent Confidence:</b> {avg_confidence:.3f}\n"
                        
                    else:
                        message += f"<b>Type:</b> Legacy Ensemble\n"
                        
                    # Last update
                    if latest_symbol in self.bot.last_model_update:
                        last_update = self.bot.last_model_update[latest_symbol]
                        message += f"<b>Last Update:</b> {last_update.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    
                except Exception as detail_error:
                    message += f"❌ Error getting model details: {detail_error}\n"
            else:
                message += f"<b>Type:</b> Legacy (no detailed info available)\n"
            
            # Features used
            if self.bot.selected_features:
                message += f"\n<b>Features Used ({len(self.bot.selected_features)}):</b>\n"
                # İlk 10 feature'ı göster
                for i, feature in enumerate(self.bot.selected_features[:10]):
                    message += f"  {i+1}. <code>{feature}</code>\n"
                
                if len(self.bot.selected_features) > 10:
                    message += f"  ... and {len(self.bot.selected_features) - 10} more\n"
            
            await self.bot.send_telegram(message)
            
        except Exception as e:
            await self.bot.send_telegram(f"❌ Error getting AI details: {e}")

    async def cmd_memory_report(self):
        """Enhanced memory report command"""
        try:
            await self.bot.send_telegram("📊 Generating detailed memory report...")
            
            # Check if EnhancedMemoryManager has the method
            if hasattr(self.bot.memory_manager, 'generate_memory_report'):
                report = await self.bot.memory_manager.generate_memory_report()
                await self.bot.send_telegram(report)
            else:
                # Fallback for basic memory manager
                memory_stats = self.bot.memory_manager.get_memory_usage()
                fallback_report = f"""📊 <b>Basic Memory Report</b>
                💾 <b>Memory Usage:</b>
                - Current: {memory_stats.get('rss_mb', 0):.1f}MB
                - System %: {memory_stats.get('percentage', 0):.1f}%
                - Available: {memory_stats.get('available_mb', 0):.0f}MB

                ℹ️ <i>Enhanced memory reporting requires restart</i>"""
                await self.bot.send_telegram(fallback_report)
                
        except Exception as e:
            logging.error(f"Error generating memory report: {e}")
            await self.bot.send_telegram(f"❌ Error generating memory report: {e}")

    async def cmd_memory_health(self):
        """Memory health score command"""
        try:
            # Check if EnhancedMemoryManager is available
            if hasattr(self.bot.memory_manager, 'get_memory_health_score') and hasattr(self.bot.memory_manager, 'get_detailed_memory_stats'):
                health_score = self.bot.memory_manager.get_memory_health_score()
                stats = self.bot.memory_manager.get_detailed_memory_stats()
                
                if 'error' in stats:
                    await self.bot.send_telegram(f"❌ Memory health check error: {stats['error']}")
                    return
                
                zone_emoji = {
                    'green': '🟢',
                    'yellow': '🟡', 
                    'orange': '🟠',
                    'red': '🔴'
                }
                
                message = (
                    f"{zone_emoji.get(stats['current_zone'], '⚪')} <b>Memory Health Check</b>\n\n"
                    f"<b>Health Score:</b> {health_score:.0f}/100\n"
                    f"<b>Current Zone:</b> {stats['current_zone'].upper()}\n"
                    f"<b>Memory Usage:</b> {stats['memory_percent']:.1f}% ({stats['rss_mb']:.1f}MB)\n"
                    f"<b>Memory Limit:</b> {stats['max_memory_mb']}MB\n"
                    f"<b>Available System:</b> {stats['available_system_mb']:.0f}MB\n\n"
                    f"<b>Zone Thresholds:</b>\n"
                    f"🟢 Green: < {stats['zone_thresholds']['yellow']:.0f}MB\n"
                    f"🟡 Yellow: {stats['zone_thresholds']['yellow']:.0f}-{stats['zone_thresholds']['orange']:.0f}MB\n"
                    f"🟠 Orange: {stats['zone_thresholds']['orange']:.0f}-{stats['zone_thresholds']['red']:.0f}MB\n"
                    f"🔴 Red: > {stats['zone_thresholds']['red']:.0f}MB"
                )
                
                await self.bot.send_telegram(message)
                
            else:
                # Fallback for basic memory manager
                memory_stats = self.bot.memory_manager.get_memory_usage()
                
                # Simple health calculation
                rss_mb = memory_stats.get('rss_mb', 0)
                if rss_mb < 300:
                    health_emoji = "🟢"
                    health_status = "EXCELLENT"
                    health_score = 95
                elif rss_mb < 400:
                    health_emoji = "🟡"
                    health_status = "GOOD"
                    health_score = 80
                elif rss_mb < 450:
                    health_emoji = "🟠"
                    health_status = "WARNING"
                    health_score = 60
                else:
                    health_emoji = "🔴"
                    health_status = "CRITICAL"
                    health_score = 30
                
                message = (
                    f"{health_emoji} <b>Basic Memory Health</b>\n\n"
                    f"<b>Status:</b> {health_status}\n"
                    f"<b>Health Score:</b> {health_score}/100\n"
                    f"<b>Memory Usage:</b> {rss_mb:.1f}MB\n"
                    f"<b>System %:</b> {memory_stats.get('percentage', 0):.1f}%\n"
                    f"<b>Available:</b> {memory_stats.get('available_mb', 0):.0f}MB\n\n"
                    f"ℹ️ <i>Enhanced features require restart</i>"
                )
                
                await self.bot.send_telegram(message)
                
        except Exception as e:
            logging.error(f"Error checking memory health: {e}")
            await self.bot.send_telegram(f"❌ Error checking memory health: {e}")


# ===== MAIN TRADING CLASS (V13) =====
class FRHunterV12:
    def __init__(self, telegram_token: str, chat_id: str):
        # Race condition koruması için lock'lar
        self.scan_offset = 0
        self._positions_lock = asyncio.Lock()  # Active positions için
        self._balance_lock = asyncio.Lock()    # Balance operations için
        self._model_lock = asyncio.Lock()      # Model training için
        self._telegram_lock = asyncio.Lock()   # Telegram messaging için

        # ===== EKSİK SATIRLAR BURAYA EKLENDİ =====
        # Memory management
        self.memory_manager = EnhancedMemoryManager(max_memory_mb=512, bot_instance=self)
        # System health monitoring
        self.health_monitor = SystemHealthMonitor()
        logging.info("System health monitoring initialized")
        # Error recovery and emergency management
        self.error_recovery = ErrorRecoveryManager(self)
        logging.info("Error recovery system initialized")
        # ==========================================

        # Model kullanım takibi (LRU benzeri)
        self._model_usage_order = deque()
        self._cache_access_times = {}

        self.trade_history = deque(maxlen=config.MAX_TRADE_HISTORY_SIZE)
        self._cache_max_age = timedelta(hours=config.MODEL_CACHE_MAX_AGE_HOURS)

        logging.info("Memory management initialized")

        # Advanced AI features
        self.feature_engineer = AdvancedFeatureEngineer()
        self.selected_features: Dict[str, List[str]] = {} # Her sembol için kendi özellik listesi olacak
        self.feature_importance_history = {}  # Feature importance tracking

        logging.info("Advanced AI feature engineering initialized")

        # Rate limiting için
        self._telegram_last_call = 0
        self._api_last_call = 0

        logging.info("Thread-safety locks initialized")
        # Ayarlar artık tamamen global 'config' objesinden yönetilecek.
        self.session: Optional[aiohttp.ClientSession] = None
        # self.ensemble_models: Dict[str, EnsemblePredictor] = {}
        # self.scalers: Dict[str, StandardScaler] = {}
        # ✅ NEW: Smart cache
        self.model_cache = SmartModelCache(
            max_size=config.MAX_MODEL_CACHE_SIZE,  # Add to config: 25
            max_memory_mb=config.MAX_MODEL_MEMORY_MB  # Add to config: 200
        )
        
        # Keep for backward compatibility / transition
        self.ensemble_models = {}  # Will be phased out
        self.scalers = {}          # Will be phased out
        
        self.symbol_info: Dict[str, SymbolInfo] = {}
        self.active_positions: Dict[str, TradeMetrics] = {}
        self.all_symbols: List[str] = []
        self.trade_history = deque(maxlen=config.MAX_TRADE_HISTORY_SIZE) # config'den okuyor
        self.paper_balance = config.PAPER_BALANCE
        self.cooldown_symbols: Dict[str, datetime] = {}
        
        self.rate_limiter = APIRateLimiter()
        self.db_manager = DatabaseManager()
        self.risk_manager = EnhancedPortfolioRiskManager(self)
        self.command_handler = TelegramCommandHandler(self)
        self.dynamic_sltp_manager = DynamicSLTPManager(self)
        self.last_auto_tune_time = datetime.now(IST_TIMEZONE)
        self.last_model_update: Dict[str, datetime] = {}
        self.model_update_interval = timedelta(hours=config.MODEL_UPDATE_HOURS)
        
        self.telegram_offset = 0
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        
        self.required_features = []

        # Her tarama döngüsü için istatistikleri sıfırdan başlat
        self.funnel_stats = SignalFunnelStats()


    # ==================================================================
    # ADIM 3: YENİ YARDIMCI FONKSİYONU BURAYA EKLEYİN
    # ==================================================================
    def _log_funnel_report(self):
        """Sinyal hunisinin detaylı raporunu loglar."""
        stats = self.funnel_stats
        report = "\n--- [SİNYAL AKIŞI RAPORU] ---\n"
        report += "--------------------------------------------------\n"
        report += f"| {'Aşama':<25} | {'Girdi':>7} | {'Çıktı':>7} |\n"
        report += "--------------------------------------------------\n"
        report += f"| {'Hızlı FR Ön Tarama':<25} | {stats.total_symbols:>7} | {len(stats.initial_fr_candidates):>7} |\n"
        report += f"| {'Zamanlama & Cooldown':<25} | {len(stats.initial_fr_candidates):>7} | {len(stats.after_timing_cooldown):>7} |\n"
        report += f"| {'Korelasyon Filtresi':<25} | {len(stats.after_timing_cooldown):>7} | {len(stats.after_correlation):>7} |\n"
        report += f"| {'Derin Analiz & AI':<25} | {len(stats.after_correlation):>7} | {len(stats.after_deep_analysis):>7} |\n"
        report += "--------------------------------------------------\n"
        report += f"| Sonuç: {len(stats.after_deep_analysis)} Fırsat Bulundu\n"
        
        if stats.reasons_for_rejection:
            report += "|\n| Reddedilme Nedenleri:\n"
            for reason, count in stats.reasons_for_rejection.most_common():
                report += f"| - {reason:<30}: {count}\n"
        report += "--------------------------------------------------\n"
        logging.info(report)

    async def _fast_fr_pre_scan(self) -> List[str]:
        """
        Performs a rapid, bulk scan of all symbols to find candidates based only on Funding Rate.
        This is extremely lightweight and uses a single API call.
        """
        logging.info("⚡ Performing fast, bulk FR pre-scan for all symbols...")
        
        try:
            all_funding_data = await self._api_request_with_retry('GET', '/fapi/v1/premiumIndex')
            if not all_funding_data:
                logging.error("Failed to fetch bulk premium index for fast scan.")
                return []

            candidates = []
            # Güvenlik için aşırı negatif değerleri filtrelemek üzere bir alt sınır belirleyelim
            EXTREME_FR_LOWER_BOUND = -5.0 

            for item in all_funding_data:
                try:
                    if not isinstance(item, dict) or not item.get('symbol', '').endswith('USDT'):
                        continue

                    symbol = item['symbol']
                    current_fr = float(item.get('lastFundingRate', 0)) * 100

                    # ===== DÜZELTİLMİŞ FİLTRELEME MANTIĞI =====
                    # Koşul: FR, alt sınırdan büyük OLMALI VE bizim eşiğimizden küçük veya eşit OLMALI.
                    # Örnek (RELAX): FR, -5.0'dan büyük VE -0.12'den küçük veya eşit olmalı.
                    # -0.007 bu koşulu sağlamaz. -0.15 sağlar.
                    if EXTREME_FR_LOWER_BOUND < current_fr <= config.FR_LONG_THRESHOLD:
                        candidates.append(symbol)
                        
                except (ValueError, KeyError):
                    continue
            
            logging.info(f"⚡ Fast scan complete. Found {len(candidates)} potential candidates matching FR <= {config.FR_LONG_THRESHOLD}%.")
            return candidates
            
        except Exception as e:
            logging.error(f"Critical error during fast FR pre-scan: {e}", exc_info=True)
            return []

    def set_value(self, key: str, value: str) -> bool:
        """Telegram'dan gelen komutla bir ayarı canlı olarak günceller."""
        key = key.upper()
        if hasattr(self, key):
            try:
                attr_type = type(getattr(self, key))
                if attr_type == bool:
                    new_value = value.lower() in ['true', '1', 't', 'y', 'yes']
                else:
                    new_value = attr_type(value)
                setattr(self, key, new_value)
                logging.info(f"CONFIG UPDATED via command: {key} set to {new_value}")
                return True
            except (ValueError, TypeError):
                logging.error(f"Failed to set config {key}: Invalid value type for '{value}'")
                return False
        return False

    async def initialize(self):
        """V14.1 - Sadece teknik başlatma yapar, mesaj göndermez."""
        await self._sync_server_time()
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        await self.update_and_filter_symbols()
        await self.load_trade_history()
        
        logging.info("Initializing... Attempting to sync open positions with exchange.")
        await self.sync_open_positions()
        logging.info(f"Synchronization complete. Found {len(self.active_positions)} active position(s).")
        
        # Başlangıç mesajı artık run() içinde, ilk güncellemeden sonra gönderilecek.
        logging.info("FR Hunter V14 initialized successfully. Waiting for first run cycle to send status.")

    async def sync_open_positions(self):
        """
        V15 - Bot başladığında, açık pozisyonları ve onlara bağlı
        SL/TP emirlerini borsadan çekerek tam senkronizasyon sağlar.
        """
        if config.TRADING_MODE != TradingMode.LIVE:
            return

        logging.info("Attempting to sync open positions and their SL/TP orders...")
        try:
            # 1. Borsadaki tüm açık pozisyonları al
            position_risk_data = await self._api_request_with_retry('GET', '/fapi/v2/positionRisk', signed=True)
            if not position_risk_data: return

            # 2. Borsadaki tüm açık emirleri al (SL/TP'leri bulmak için)
            open_orders_data = await self._api_request_with_retry('GET', '/fapi/v1/openOrders', signed=True)
            orders_by_symbol = {}
            if open_orders_data:
                for order in open_orders_data:
                    symbol = order['symbol']
                    if symbol not in orders_by_symbol:
                        orders_by_symbol[symbol] = []
                    orders_by_symbol[symbol].append(order)

            # 3. Pozisyonları ve emirleri eşleştirerek senkronize et
            for pos in position_risk_data:
                position_amount = float(pos.get('positionAmt', '0'))
                
                if position_amount > 0: # Sadece LONG pozisyonları dikkate al
                    symbol = pos['symbol']
                    if symbol not in self.active_positions:
                        logging.warning(f"Found an open LONG position for {symbol}. Restoring to memory...")
                        
                        entry_price = float(pos.get('entryPrice', '0'))
                        quantity = abs(position_amount)
                        if entry_price == 0 or quantity == 0: continue

                        # Bu sembole ait SL ve TP emirlerini bul
                        sl_order, tp_order = None, None
                        symbol_orders = orders_by_symbol.get(symbol, [])
                        for order in symbol_orders:
                            if order.get('type') == 'STOP_MARKET':
                                sl_order = order
                            elif order.get('type') == 'TAKE_PROFIT_MARKET':
                                tp_order = order
                        
                        recreated_trade = TradeMetrics(
                            symbol=symbol,
                            entry_time=datetime.now(IST_TIMEZONE) - timedelta(minutes=1),
                            entry_price=entry_price,
                            quantity=quantity,
                            confidence=0.75, # Geri yüklenen işlem için varsayılan değer
                            funding_rate=0.0,
                            stop_loss=float(sl_order['stopPrice']) if sl_order else entry_price * 0.98,
                            take_profit=float(tp_order['stopPrice']) if tp_order else entry_price * 1.05,
                            sl_order_id=sl_order.get('orderId') if sl_order else None,
                            tp_order_id=tp_order.get('orderId') if tp_order else None,
                            volatility=2.0 # Varsayılan değer, sonradan güncellenebilir
                        )
                        
                        self.active_positions[symbol] = recreated_trade
                        logging.info(f"Successfully synced {symbol} with SL Order ID: {recreated_trade.sl_order_id} and TP Order ID: {recreated_trade.tp_order_id}")

                elif position_amount < 0:
                    logging.info(f"Found an external SHORT position for {pos['symbol']}. Bot will ignore it.")

        except Exception as e:
            logging.error(f"Critical error during position synchronization: {e}", exc_info=True)

    # === MEMORY CLEANUP FONKSİYONLARI - close() FONKSİYONUNDAN ÖNCE EKLEYİN ===
    
    def _cleanup_old_data(self) -> int:
        """Eski verileri temizle ve temizlenen item sayısını döndür"""
        now = datetime.now(IST_TIMEZONE)
        cleaned_items = 0
        
        try:
            # 1. Model cache cleanup (LRU benzeri)
            if len(self.ensemble_models) > config.MAX_MODEL_CACHE_SIZE:
                # En az kullanılan modelleri temizle
                while len(self.ensemble_models) > config.MAX_MODEL_CACHE_SIZE and self._model_usage_order:
                    oldest_symbol = self._model_usage_order.popleft()
                    if oldest_symbol in self.ensemble_models:
                        del self.ensemble_models[oldest_symbol]
                        if oldest_symbol in self.scalers:
                            del self.scalers[oldest_symbol]
                        if oldest_symbol in self.last_model_update:
                            del self.last_model_update[oldest_symbol]
                        cleaned_items += 1
                
                if cleaned_items > 0:
                    logging.debug(f"Cleaned {cleaned_items} old models from cache")
            
            # 2. Price cache cleanup (EnhancedPortfolioRiskManager içinde)
            if hasattr(self, 'risk_manager') and hasattr(self.risk_manager, 'price_history_cache'):
                cache = self.risk_manager.price_history_cache
                symbols_to_remove = []
                
                for symbol, data in cache.items():
                    if (now - data['timestamp']) > self._cache_max_age:
                        symbols_to_remove.append(symbol)
                
                for symbol in symbols_to_remove:
                    del cache[symbol]
                    cleaned_items += len(symbols_to_remove)
                
                if symbols_to_remove:
                    logging.debug(f"Cleaned {len(symbols_to_remove)} expired price cache entries")
            
            # 3. Cooldown symbols cleanup
            expired_cooldowns = []
            for symbol, expiry_time in self.cooldown_symbols.items():
                if now > expiry_time:
                    expired_cooldowns.append(symbol)
            
            for symbol in expired_cooldowns:
                del self.cooldown_symbols[symbol]
                cleaned_items += len(expired_cooldowns)
            
            if expired_cooldowns:
                logging.debug(f"Cleaned {len(expired_cooldowns)} expired cooldowns")
            
            # 4. Cache access times cleanup
            old_access_times = []
            for symbol, access_time in self._cache_access_times.items():
                if (now - access_time) > self._cache_max_age:
                    old_access_times.append(symbol)
            
            for symbol in old_access_times:
                del self._cache_access_times[symbol]
            
            return cleaned_items
            
        except Exception as e:
            logging.error(f"Error during data cleanup: {e}")
            return 0

    def _track_model_usage(self, symbol: str):
        """Model kullanımını takip et (LRU için)"""
        try:
            # Eğer symbol zaten varsa, onu listeden çıkar
            if symbol in self._model_usage_order:
                # deque'da remove() yavaş olabilir, ama cache boyutu küçük
                temp_list = list(self._model_usage_order)
                temp_list.remove(symbol)
                self._model_usage_order = deque(temp_list)
            
            # En sona ekle (en son kullanılan)
            self._model_usage_order.append(symbol)
            self._cache_access_times[symbol] = datetime.now(IST_TIMEZONE)
            
        except Exception as e:
            logging.error(f"Error tracking model usage: {e}")

    async def _periodic_memory_check(self):
        """V15.4 - UNIFIED Memory Management"""
        try:
            # Enhanced manager varsa onu kullan
            if hasattr(self.memory_manager, 'should_run_check') and hasattr(self.memory_manager, 'proactive_memory_check'):
                # Enhanced Memory Manager
                if not self.memory_manager.should_run_check():
                    return
                
                result = await self.memory_manager.proactive_memory_check()
                
                if result.get('memory_freed_mb', 0) > 0:
                    logging.info(f"Enhanced memory check: {result['zone']} zone, freed {result['memory_freed_mb']:.1f}MB")
                
                # Periodic reports
                current_hour = datetime.now(IST_TIMEZONE).hour
                last_report_hour = getattr(self, '_last_memory_report_hour', -1)
                
                if (result['zone'] in ['orange', 'red'] and 
                    current_hour != last_report_hour and 
                    current_hour % 2 == 0):
                    
                    try:
                        if hasattr(self.memory_manager, 'generate_memory_report'):
                            report = await self.memory_manager.generate_memory_report()
                            await self.send_telegram(report)
                            self._last_memory_report_hour = current_hour
                    except Exception as report_error:
                        logging.error(f"Failed to send memory report: {report_error}")
                
                # Emergency protocol
                if (result['zone'] == 'red' and 
                    result.get('memory_freed_mb', 0) < 5.0 and 
                    result.get('effectiveness_score', 0) < 0.3):
                    
                    logging.critical("Memory emergency protocols insufficient")
                    try:
                        await self.send_telegram(
                            "🆘 <b>CRITICAL MEMORY WARNING</b>\n"
                            "Emergency cleanup was insufficient.\n"
                            "System monitoring required!"
                        )
                    except:
                        pass
            else:
                # Basic Memory Manager - Simple check
                loop_count = getattr(self, '_loop_count', 0) + 1
                self._loop_count = loop_count
                
                if loop_count % 50 == 0:
                    try:
                        memory_stats = self.memory_manager.get_memory_usage()
                        if memory_stats.get('rss_mb', 0) > 400:
                            logging.warning(f"High memory usage: {memory_stats['rss_mb']:.1f}MB")
                            self._cleanup_old_data()
                            gc.collect()
                    except Exception as e:
                        logging.error(f"Basic memory check failed: {e}")
                        
        except Exception as e:
            logging.error(f"Critical error in memory check: {e}", exc_info=True)
            self.health_monitor.record_error("MEMORY_CHECK_CRITICAL", str(e), "memory_manager")

    def get_memory_status(self) -> Dict[str, any]:
        """Memory durumunu döndür (Telegram komutları için) - UNIFIED VERSION"""
        try:
            # Enhanced Memory Manager metodlarını kullan
            if hasattr(self.memory_manager, 'get_detailed_memory_stats'):
                # Enhanced manager var
                detailed_stats = self.memory_manager.get_detailed_memory_stats()
                
                if 'error' in detailed_stats:
                    # Fallback to basic stats
                    basic_stats = self.memory_manager.get_memory_usage()
                    return {
                        'memory_mb': basic_stats.get('rss_mb', 0),
                        'memory_percentage': basic_stats.get('percentage', 0),
                        'available_mb': basic_stats.get('available_mb', 0),
                        'trade_history_count': len(self.trade_history),
                        'trade_history_limit': config.MAX_TRADE_HISTORY_SIZE,
                        'models_cached': len(self.ensemble_models),
                        'models_limit': config.MAX_MODEL_CACHE_SIZE, # <-- DÜZELTME
                        'active_positions': len(self.active_positions),
                        'cooldown_symbols': len(self.cooldown_symbols),
                        'manager_type': 'enhanced_fallback'
                    }
                
                # Enhanced stats available
                return {
                    'memory_mb': detailed_stats.get('rss_mb', 0),
                    'memory_percentage': detailed_stats.get('memory_percent', 0),
                    'available_mb': detailed_stats.get('available_system_mb', 0),
                    'trade_history_count': len(self.trade_history),
                    'trade_history_limit': config.MAX_TRADE_HISTORY_SIZE,
                    'models_cached': len(self.ensemble_models),
                    'models_limit': config.MAX_MODEL_CACHE_SIZE, # <-- DÜZELTME
                    'active_positions': len(self.active_positions),
                    'cooldown_symbols': len(self.cooldown_symbols),
                    'current_zone': detailed_stats.get('current_zone', 'unknown'),
                    'cache_memory_mb': detailed_stats.get('cache_memory', {}).get('total_cache_mb', 0),
                    'manager_type': 'enhanced'
                }
            else:
                # Basic manager (artık kullanılmıyor ama güvenlik için düzeltelim)
                basic_stats = self.memory_manager.get_memory_usage()
                return {
                    'memory_mb': basic_stats.get('rss_mb', 0),
                    'memory_percentage': basic_stats.get('percentage', 0),
                    'available_mb': basic_stats.get('available_mb', 0),
                    'trade_history_count': len(self.trade_history),
                    'trade_history_limit': config.MAX_TRADE_HISTORY_SIZE,
                    'models_cached': len(self.ensemble_models),
                    'models_limit': config.MAX_MODEL_CACHE_SIZE, # <-- DÜZELTME
                    'active_positions': len(self.active_positions),
                    'cooldown_symbols': len(self.cooldown_symbols),
                    'manager_type': 'basic'
                }
                
        except Exception as e:
            logging.error(f"Error getting memory status: {e}")
            return {
                'error': str(e),
                'manager_type': 'error'
            }
 # === CRITICAL ERROR HANDLING METODLARI - close() FONKSİYONUNDAN ÖNCE EKLEYİN ===
    
    async def handle_critical_api_error(self, error: Exception, operation: str, attempt: int = 1):
        """API kritik hatalarını yönet"""
        error_msg = f"API critical error during {operation}: {str(error)}"
        logging.error(error_msg)
        
        try:
            await self.error_recovery.handle_critical_error(
                error_type="API_CRITICAL_ERROR",
                error_details=f"{operation}: {str(error)}",
                source=f"api_{operation}_attempt_{attempt}"
            )
        except Exception as recovery_error:
            logging.critical(f"Error recovery failed: {recovery_error}")

    async def handle_critical_telegram_error(self, error: Exception, operation: str):
        """Telegram kritik hatalarını yönet"""
        error_msg = f"Telegram critical error during {operation}: {str(error)}"
        logging.error(error_msg)
        
        try:
            await self.error_recovery.handle_critical_error(
                error_type="TELEGRAM_CRITICAL_ERROR",
                error_details=f"{operation}: {str(error)}",
                source=f"telegram_{operation}"
            )
        except Exception as recovery_error:
            logging.critical(f"Telegram error recovery failed: {recovery_error}")

    async def handle_critical_memory_error(self, error: Exception, operation: str):
        """Memory kritik hatalarını yönet"""
        error_msg = f"Memory critical error during {operation}: {str(error)}"
        logging.error(error_msg)
        
        try:
            await self.error_recovery.handle_critical_error(
                error_type="MEMORY_CRITICAL_ERROR",
                error_details=f"{operation}: {str(error)}",
                source=f"memory_{operation}"
            )
        except Exception as recovery_error:
            logging.critical(f"Memory error recovery failed: {recovery_error}")

    async def handle_critical_database_error(self, error: Exception, operation: str):
        """Database kritik hatalarını yönet"""
        error_msg = f"Database critical error during {operation}: {str(error)}"
        logging.error(error_msg)
        
        try:
            await self.error_recovery.handle_critical_error(
                error_type="DATABASE_CRITICAL_ERROR",
                error_details=f"{operation}: {str(error)}",
                source=f"database_{operation}"
            )
        except Exception as recovery_error:
            logging.critical(f"Database error recovery failed: {recovery_error}")

    async def safe_operation_wrapper(self, operation_func, operation_name: str, *args, **kwargs):
        """Operasyonları güvenli wrapper ile çalıştır"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                result = await operation_func(*args, **kwargs)
                return result
                
            except aiohttp.ClientError as e:
                await self.handle_critical_api_error(e, operation_name, attempt + 1)
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except asyncio.TimeoutError as e:
                await self.handle_critical_api_error(e, f"{operation_name}_timeout", attempt + 1)
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except MemoryError as e:
                await self.handle_critical_memory_error(e, operation_name)
                if attempt == max_retries - 1:
                    raise
                # Memory error için daha uzun bekle
                await asyncio.sleep(10)
                
            except Exception as e:
                # Genel kritik hata
                error_msg = f"Critical error in {operation_name}: {str(e)}"
                logging.critical(error_msg)
                
                try:
                    await self.error_recovery.handle_critical_error(
                        error_type="GENERAL_CRITICAL_ERROR",
                        error_details=f"{operation_name}: {str(e)}",
                        source=operation_name
                    )
                except:
                    pass
                
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        return None

    def check_service_availability(self, service: str) -> bool:
        """Service'in kullanılabilir olup olmadığını kontrol et"""
        return self.error_recovery.is_service_available(service)

    async def graceful_shutdown(self, reason: str = "Unknown"):
        """Graceful shutdown - tüm pozisyonları güvenli şekilde kapat"""
        logging.critical(f"Initiating graceful shutdown. Reason: {reason}")
        
        try:
            # Emergency notification gönder
            await self.send_telegram(
                f"🔴 <b>SYSTEM SHUTDOWN</b>\n\n"
                f"<b>Reason:</b> {reason}\n"
                f"<b>Active positions:</b> {len(self.active_positions)}\n"
                f"<b>Action:</b> Monitoring positions only"
            )
        except:
            logging.error("Failed to send shutdown notification")
        
        # Aktif pozisyonları sadece takip et, yeni trade alma
        shutdown_start = datetime.now(IST_TIMEZONE)
        max_shutdown_time = timedelta(minutes=30)  # Max 30 dakika bekle
        
        while self.active_positions and (datetime.now(IST_TIMEZONE) - shutdown_start) < max_shutdown_time:
            try:
                logging.info(f"Shutdown: Monitoring {len(self.active_positions)} active positions...")
                await self.manage_active_positions()
                await asyncio.sleep(30)  # 30 saniye bekle
            except Exception as e:
                logging.error(f"Error during shutdown monitoring: {e}")
                break
        
        # Son durum raporu
        if self.active_positions:
            try:
                final_report = (
                    f"🔴 <b>Shutdown Complete</b>\n\n"
                    f"<b>Remaining positions:</b> {len(self.active_positions)}\n"
                    f"<b>Symbols:</b> {', '.join(self.active_positions.keys())}\n\n"
                    f"<i>Positions left for manual monitoring</i>"
                )
                await self.send_telegram(final_report)
            except:
                logging.error("Failed to send final shutdown report")
        
        # Cleanup
        try:
            await self.save_performance_data()
            if self.session:
                await self.session.close()
        except Exception as e:
            logging.error(f"Error during shutdown cleanup: {e}")
        
        logging.critical("Graceful shutdown completed")   

    async def close(self):
        if self.session: await self.session.close()
        await self.save_performance_data()
        logging.info("System shutdown completed.")

    @safe_async_operation
    async def send_telegram(self, message: str):
        """V15.1 - Health tracking ile gelişmiş Telegram mesaj gönderimi"""
        
        # ===== HEALTH MONITORING =====
        self.health_monitor.record_metric('telegram_messages_sent')
        
        # ===== MESAJ HAZIRLIĞI =====
        if len(message) > 4096:
            logging.warning(f"Message too long ({len(message)} chars). Truncating.")
            message = message[:4090] + "..."
        
        # ===== RETRY LOGIC =====
        for attempt in range(3):
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                params = {
                    "chat_id": self.chat_id, 
                    "text": message, 
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                }
                
                timeout = aiohttp.ClientTimeout(total=10)
                async with self.session.post(url, json=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
                        
                    elif resp.status == 429:
                        # Rate limit
                        retry_after = int(resp.headers.get('Retry-After', 30))
                        error_msg = f"Telegram rate limit hit. Retry after: {retry_after}s"
                        logging.warning(error_msg)
                        self.health_monitor.record_error("TELEGRAM_RATE_LIMIT", error_msg, f"attempt_{attempt+1}")
                        
                        await asyncio.sleep(retry_after)
                        continue
                        
                    else:
                        error_text = await resp.text()
                        error_msg = f"Telegram error {resp.status}: {error_text[:100]}"
                        logging.error(error_msg)
                        self.health_monitor.record_error("TELEGRAM_HTTP_ERROR", error_msg, f"attempt_{attempt+1}")
                        
            except asyncio.TimeoutError:
                error_msg = f"Telegram timeout (attempt {attempt + 1})"
                logging.warning(error_msg)
                self.health_monitor.record_error("TELEGRAM_TIMEOUT", error_msg, f"attempt_{attempt+1}")
                
            except Exception as e:
                error_msg = f"Telegram exception: {str(e)[:100]}"
                logging.error(error_msg)
                self.health_monitor.record_error("TELEGRAM_EXCEPTION", error_msg, f"attempt_{attempt+1}")
            
            # Retry öncesi bekleme
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
        
        # Tüm denemeler başarısız
        final_error = "Failed to send Telegram message after all retries"
        logging.error(final_error)
        self.health_monitor.record_error("TELEGRAM_ALL_RETRIES_FAILED", final_error, "final")
        return False

    async def _api_request_with_retry(self, method: str, path: str, params: Optional[Dict] = None, signed: bool = False, max_retries: int = 4):
        """V15.1 - Health tracking ile gelişmiş API istekleri"""
        params = params or {}
        
        # ===== HEALTH MONITORING =====
        self.health_monitor.record_metric('api_calls_total')
        
        # ===== RATE LIMITING KONTROLLERİ =====
        now = time.time()
        if hasattr(self, '_api_last_call'):
            time_since_last = now - self._api_last_call
            min_interval = 0.1
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
        
        self._api_last_call = time.time()
        
        # Klines için ekstra bekleme
        if '/klines' in path:
            await asyncio.sleep(0.2)
        
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(
                    total=30 + (attempt * 10),
                    connect=10, 
                    sock_read=20 + (attempt * 5)
                )

                if signed:
                    # Server time offset düzeltmesi
                    server_time_offset = getattr(self, '_server_time_offset', 0)
                    params['timestamp'] = int((time.time() + server_time_offset) * 1000)
                    query_string = urllib.parse.urlencode(params, True)
                    params['signature'] = hmac.new(
                        BINANCE_API_SECRET.encode('utf-8'),
                        msg=query_string.encode('utf-8'),
                        digestmod=hashlib.sha256
                    ).hexdigest()
                
                url = BASE_URL + path
                headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
                
                async with self.session.request(
                    method.upper(), url, params=params, headers=headers, timeout=timeout
                ) as resp:
                    
                    # Weight tracking
                    used_weight = resp.headers.get('X-MBX-USED-WEIGHT-1M')
                    if used_weight and int(used_weight) > 1000:
                        logging.warning(f"High API weight usage: {used_weight}/1200")
                        self.health_monitor.record_metric('memory_warnings')
                    
                    # ===== BAŞARILI YANIT =====
                    if 200 <= resp.status < 300:
                        return await resp.json()

                    # ===== HATA DURUMLARI =====
                    error_text = await resp.text()
                    
                    # Rate limit hataları
                    if resp.status in [429, 418]:
                        retry_after = int(resp.headers.get('Retry-After', 60))
                        
                        if 'Too many requests' in error_text:
                            wait_time = max(retry_after, 60)
                            error_msg = f"IP Rate limit hit on {path}"
                            logging.warning(f"{error_msg}. Waiting {wait_time}s")
                            self.health_monitor.record_error("API_RATE_LIMIT", error_msg, f"attempt_{attempt+1}")
                        else:
                            wait_time = retry_after
                            error_msg = f"General rate limit on {path}"
                            logging.warning(f"{error_msg}. Waiting {wait_time}s")
                            self.health_monitor.record_error("API_RATE_LIMIT_GENERAL", error_msg, f"attempt_{attempt+1}")
                        
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # IP Ban
                    if resp.status == 403 and 'banned' in error_text.lower():
                        critical_error = f"IP BANNED by Binance! Path: {path}"
                        logging.critical(critical_error)
                        self.health_monitor.record_error("API_IP_BAN", critical_error, "binance")
                        
                        try:
                            await self.send_telegram("🚨 <b>CRITICAL</b>: IP banned by Binance!")
                        except:
                            pass
                        
                        raise Exception("IP banned by Binance")
                    
                    # Server errors
                    if 500 <= resp.status < 600:
                        wait_time = min(2 ** attempt, 30)
                        error_msg = f"Server error {resp.status} on {path}: {error_text[:100]}"
                        logging.warning(f"{error_msg}. Retrying in {wait_time}s")
                        self.health_monitor.record_error("API_SERVER_ERROR", error_msg, f"attempt_{attempt+1}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Client errors
                    else:
                        error_msg = f"Client error {resp.status} on {path}: {error_text[:100]}"
                        logging.error(error_msg)
                        self.health_monitor.record_error("API_CLIENT_ERROR", error_msg, f"final_attempt")
                        # API call failed sayacını artır
                        self.health_monitor.record_metric('api_calls_failed')
                        return None

            except asyncio.TimeoutError:
                wait_time = 5 + (attempt * 2)
                error_msg = f"API timeout on {path} (attempt {attempt + 1})"
                logging.warning(f"{error_msg}. Retrying in {wait_time}s")
                self.health_monitor.record_error("API_TIMEOUT", error_msg, f"attempt_{attempt+1}")
                
                if attempt == max_retries - 1:
                    logging.error(f"API request failed after all retries: {path}")
                    self.health_monitor.record_metric('api_calls_failed')
                    return None
                    
                await asyncio.sleep(wait_time)

            except Exception as e:
                wait_time = 5 + attempt
                error_msg = f"API request exception on {path}: {str(e)[:100]}"
                logging.error(error_msg)
                self.health_monitor.record_error("API_EXCEPTION", error_msg, f"attempt_{attempt+1}")
                
                if attempt == max_retries - 1:
                    self.health_monitor.record_metric('api_calls_failed')
                    return None
                    
                await asyncio.sleep(wait_time)
        
        # Tüm denemeler başarısız
        final_error = f"API request for {path} failed after all {max_retries} retries"
        logging.error(final_error)
        self.health_monitor.record_error("API_ALL_RETRIES_FAILED", final_error, "final")
        self.health_monitor.record_metric('api_calls_failed')
        return None

    async def _sync_server_time(self):
        """Binance server time ile senkronize ol"""
        try:
            # DOĞRUDAN API çağrısı (existing session yok henüz)
            url = BASE_URL + '/fapi/v1/time'
            
            # Temporary session oluştur
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(url) as resp:
                    if resp.status == 200:
                        response = await resp.json()
                        if response and 'serverTime' in response:
                            server_time = response['serverTime'] / 1000  # milliseconds to seconds
                            local_time = time.time()
                            self._server_time_offset = server_time - local_time
                            
                            logging.info(f"✅ Server time synchronized. Offset: {self._server_time_offset:.3f}s")
                            return True
                        else:
                            logging.error(f"Invalid server time response: {response}")
                    else:
                        logging.error(f"Server time sync failed: HTTP {resp.status}")
                        
        except Exception as e:
            logging.error(f"Failed to sync server time: {e}")
            
        # Fallback
        self._server_time_offset = 0
        logging.warning("Using zero time offset as fallback")
        return False

    def is_optimal_funding_window(self, next_funding_time: int) -> bool:
        """
        V14 - Merkezi config objesini kullanarak çalışır.
        """
        try:
            current_time_ms = int(time.time() * 1000)
            time_to_funding_min = (next_funding_time - current_time_ms) / 60000
            
            # Ayarları global 'config' objesinden oku
            if not (config.MIN_TIME_BEFORE_FUNDING <= time_to_funding_min <= config.MAX_TIME_BEFORE_FUNDING):
                logging.debug(f"Timing rejected (BEFORE): {time_to_funding_min:.1f} min")
                return False

            funding_interval_min = 240
            time_since_last_funding_min = funding_interval_min - time_to_funding_min
            
            # Ayarı global 'config' objesinden oku
            if time_since_last_funding_min < config.WAIT_TIME_AFTER_FUNDING:
                logging.debug(f"Timing rejected (AFTER): {time_since_last_funding_min:.1f} min passed")
                return False

            return True
            
        except Exception as e:
            logging.error(f"Zamanlama analizi sırasında kritik hata: {e}")
            return False

    async def get_futures_balance(self) -> float:
        """
        V9.11 - Binance vadeli işlemler hesabındaki USDT bakiyesini çeker.
        """
        try:
            # /fapi/v2/balance endpoint'ine imzalı bir istek gönder
            res = await self._api_request_with_retry('GET', '/fapi/v2/balance', signed=True)
            if res:
                # Gelen yanıttaki tüm varlıklar arasından USDT'yi bul
                for asset in res:
                    if asset.get('asset') == 'USDT':
                        # 'availableBalance' anlık olarak kullanılabilir bakiyedir.
                        balance = float(asset.get('availableBalance', 0))
                        logging.info(f"Successfully fetched live futures balance: ${balance:,.2f}")
                        return balance
            # Eğer USDT bulunamazsa veya hata olursa, varsayılan küçük bir değer döndür
            logging.warning("Could not fetch USDT balance from futures account. Using fallback value.")
            return 100.0 # Güvenlik için düşük bir varsayılan değer

        except Exception as e:
            logging.error(f"Error fetching futures balance: {e}")
            return 100.0

    def calculate_position_size(self, balance: float, confidence: float, market_regime: MarketRegime, entry_price: float, stop_loss_price: float) -> float:
        """V15.1 - Tüm division by zero riskleri giderildi ve güvenlik kontrolleri eklendi"""
        
        # ===== GİRİŞ PARAMETRELERİNİ KONTROL ET =====
        if balance <= 0:
            logging.error(f"Invalid balance: {balance}")
            return 0.0
        
        if entry_price <= 0:
            logging.error(f"Invalid entry price: {entry_price}")
            return 0.0
        
        if stop_loss_price <= 0:
            logging.error(f"Invalid stop loss price: {stop_loss_price}")
            return 0.0
        
        if not (0 <= confidence <= 1):
            logging.error(f"Invalid confidence: {confidence}")
            return 0.0
        
        # ===== STOP LOSS MESAFESİ KONTROLÜ =====
        stop_loss_distance = entry_price - stop_loss_price
        if stop_loss_distance <= 0:
            logging.error(f"Invalid stop-loss distance ({stop_loss_distance}). SL must be below entry price for LONG positions.")
            return 0.0
        
        # Minimum mesafe kontrolü (çok küçük SL mesafesi riskli)
        min_sl_distance_pct = 0.005  # %0.5 minimum
        min_sl_distance = entry_price * min_sl_distance_pct
        if stop_loss_distance < min_sl_distance:
            logging.warning(f"Stop loss too close to entry. Adjusting from {stop_loss_distance:.4f} to {min_sl_distance:.4f}")
            stop_loss_distance = min_sl_distance
        
        # ===== PİYASA REJİMİNE GÖRE RİSK ÇARPANI =====
        risk_multiplier = 1.0  # Varsayılan (Uptrend için)
        if market_regime == MarketRegime.SIDEWAYS_RANGING:
            risk_multiplier = 0.5  # Riski %50 azalt
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            risk_multiplier = 0.4  # Riski %60 azalt
        elif market_regime == MarketRegime.DOWNTREND:
            risk_multiplier = 0.3  # Riski %70 azalt (güvenlik için)
        
        # ===== GÜVENLİ NORMALIZED CONFIDENCE HESAPLAMASI =====
        try:
            if confidence > 0.5:
                normalized_confidence = (confidence - 0.5) / 0.5
            else:
                normalized_confidence = 0
            normalized_confidence = max(0, normalized_confidence)  # Negatif olmasını önle
        except Exception as e:
            logging.warning(f"Error calculating normalized confidence: {e}")
            normalized_confidence = 0
        
        # ===== DİNAMİK RİSK HESAPLAMASI =====
        try:
            # Base risk + extra risk based on confidence
            dynamic_risk_pct = config.BASE_RISK_PERCENT + (config.MAX_RISK_PERCENT - config.BASE_RISK_PERCENT) * normalized_confidence
            
            # Market regime ile ayarla
            adjusted_risk_pct = dynamic_risk_pct * risk_multiplier
            
            # Güvenlik sınırları
            min_risk = config.BASE_RISK_PERCENT * 0.5
            max_risk = config.MAX_RISK_PERCENT * 1.5
            adjusted_risk_pct = max(min_risk, min(adjusted_risk_pct, max_risk))
            
        except Exception as e:
            logging.error(f"Error in risk calculation: {e}")
            adjusted_risk_pct = config.BASE_RISK_PERCENT  # Fallback
        
        # ===== RİSK MİKTARINI HESAPLA =====
        try:
            # Yüzdesel risk miktarı
            risk_amount_by_percentage = balance * adjusted_risk_pct
            
            # İki limiti karşılaştır: yüzdesel vs sabit dolar
            final_risk_in_usd = min(risk_amount_by_percentage, config.MAX_LOSS_PER_TRADE_USD)
            
            # Final risk sıfır veya negatif olmasın
            if final_risk_in_usd <= 0:
                logging.error("Calculated risk amount is zero or negative")
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating risk amounts: {e}")
            return 0.0
        
        # ===== GÜVENLİ QUANTITY VE POSİTİON VALUE HESAPLAMASI =====
        try:
            # Division by zero korumalı quantity hesabı
            if stop_loss_distance <= 0:
                logging.error("Stop loss distance is zero or negative")
                return 0.0
            
            quantity = final_risk_in_usd / stop_loss_distance
            if quantity <= 0:
                logging.error("Calculated quantity is zero or negative")
                return 0.0
            
            # Position value hesabı
            position_value = quantity * entry_price
            if position_value <= 0:
                logging.error("Calculated position value is zero or negative")
                return 0.0
            
            # ===== GÜVENLİK LİMİTLERİ =====
            # Pozisyon değeri bakiyenin yarısından fazla olmamalı
            max_position_value = balance * 0.5
            if position_value > max_position_value:
                logging.warning(f"Position value ({position_value:.2f}) exceeds safety limit. Capping at {max_position_value:.2f}")
                position_value = max_position_value
            
            # Minimum pozisyon değeri kontrolü
            min_position_value = 5  # $5 minimum
            if position_value < min_position_value:
                logging.warning(f"Position value too small: ${position_value:.2f}")
                return 0.0
            
            logging.info(f"Position Size Calc: Regime={market_regime.value} (Risk x{risk_multiplier:.2f}) -> Final Position Value=${position_value:,.2f}")
            return position_value
            
        except ZeroDivisionError as e:
            logging.error(f"Division by zero in position calculation: {e}")
            return 0.0
        except Exception as e:
            logging.error(f"Unexpected error in position size calculation: {e}")
            return 0.0

    async def execute_trade(self, signal: SignalData):
        """
        V15.2 - Race condition tamamen giderildi
        
        ÖNCEKİ SORUN:
        - Lock alınıp validation yapılıyor
        - Lock bırakılıp işlemler yapılıyor  
        - Bu sırada başka thread aynı symbol/limit için devam edebiliyor
        
        YENİ ÇÖZÜM:
        - Reservation pattern ile atomic validation
        - Exception safety ile cleanup
        - Tüm kritik bölümler lock altında
        """
        
        # ===== PHASE 1: ATOMIC VALIDATION + RESERVATION =====
        async with self._positions_lock:
            # Kontrol 1: Bu symbol için pozisyon var mı?
            if signal.symbol in self.active_positions:
                logging.warning(f"[{signal.symbol}] Position already exists. Rejecting duplicate.")
                return False
            
            # Kontrol 2: Max pozisyon limitine ulaştık mı?
            current_count = len(self.active_positions)
            if current_count >= config.MAX_ACTIVE_TRADES:
                logging.info(f"[{signal.symbol}] Max trades reached ({current_count}/{config.MAX_ACTIVE_TRADES})")
                return False
            
            # ✅ KRITIK: Slot'ı hemen rezerve et (placeholder ile)
            # Bu sayede başka thread aynı symbol için veya max limit kontrolünde
            # bu reserved slot'ı görecek ve devam edemeyecek
            self.active_positions[signal.symbol] = "RESERVED"
            logging.debug(f"[{signal.symbol}] Position slot reserved ({current_count + 1}/{config.MAX_ACTIVE_TRADES})")
        
        # ===== PHASE 2: SAFE OPERATIONS (LOCK DIŞINDA) =====
        try:
            # Exchange'de pozisyon kontrolü (canlı mod)
            if config.TRADING_MODE == TradingMode.LIVE:
                existing_position = await self._check_existing_position_on_exchange(signal.symbol)
                if existing_position:
                    logging.error(f"[{signal.symbol}] Position already exists on exchange!")
                    raise Exception("Position exists on exchange")
            
            # Balance kontrolü (lock ile korunmalı)
            async with self._balance_lock:
                if config.TRADING_MODE == TradingMode.LIVE:
                    current_balance = await self.get_futures_balance()
                    if current_balance is None or current_balance <= 0:
                        raise Exception(f"Invalid balance: {current_balance}")
                else:
                    current_balance = self.paper_balance
                    if current_balance <= 5:  # Minimum $5
                        raise Exception(f"Insufficient paper balance: {current_balance}")
            
            # Dynamic SL/TP hesaplama
            time_to_funding = float('inf')
            if signal.next_funding_time:
                time_to_funding = (signal.next_funding_time - int(time.time() * 1000)) / 60000
            
            try:
                stop_loss_price, take_profit_price = self.dynamic_sltp_manager.calculate_dynamic_sltp(
                    entry_price=signal.current_price,
                    atr_percent=signal.volatility, 
                    market_regime=signal.market_regime,
                    confidence=signal.confidence,
                    funding_time_remaining=time_to_funding
                )
            except Exception as sltp_error:
                logging.error(f"[{signal.symbol}] SL/TP calculation failed: {sltp_error}")
                # Fallback hesaplama
                stop_loss_price = signal.current_price * 0.98   # %2 SL
                take_profit_price = signal.current_price * 1.04  # %4 TP
            
            # Position size hesaplama
            position_value = self.calculate_position_size(
                balance=current_balance,
                confidence=signal.confidence,
                market_regime=signal.market_regime, 
                entry_price=signal.current_price,
                stop_loss_price=stop_loss_price
            )
            
            if position_value <= 0:
                raise Exception("Invalid position size calculated")
            
            # Symbol info kontrolü
            symbol_info = self.symbol_info.get(signal.symbol)
            if not symbol_info:
                raise Exception("Symbol info not found")
            
            min_required = max(symbol_info.min_notional, 5.0)  # En az $5 veya symbol min_notional
            if position_value < min_required:
                logging.warning(f"Position size adjusted: ${position_value:.2f} -> ${min_required:.2f}")
                position_value = min_required
            
            # Quantity hesaplama
            quantity = round(position_value / signal.current_price, symbol_info.quantity_precision)
            if quantity <= 0:
                raise Exception("Invalid quantity calculated")
            
            # Trade object oluştur
            current_hour = datetime.now(IST_TIMEZONE).hour
            bot_mode = "RELAX" if config.RELAX_MODE_START_HOUR <= current_hour < config.DEFENSIVE_MODE_START_HOUR else "DEFENSIVE"
            
            trade = TradeMetrics(
                symbol=signal.symbol,
                entry_time=datetime.now(IST_TIMEZONE),
                entry_price=signal.current_price,
                quantity=quantity,
                confidence=signal.confidence,
                funding_rate=signal.funding_rate,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                volatility=signal.volatility,
                market_regime=signal.market_regime.value,
                smart_sl_activated=(abs(signal.current_price - stop_loss_price) > signal.current_price * 0.025),
                bot_mode_on_entry=bot_mode,
                time_to_funding_on_entry=time_to_funding,
                highest_price_seen=signal.current_price
            )
            
            # ===== EXCHANGE ORDER PLACEMENT (CANLΙ MOD) =====
            sl_order_id, tp_order_id = None, None
            
            if config.TRADING_MODE == TradingMode.LIVE:
                # Entry order
                entry_params = {
                    'symbol': signal.symbol,
                    'side': 'BUY', 
                    'type': 'MARKET',
                    'quantity': quantity
                }
                
                entry_response = await self._api_request_with_retry(
                    'POST', '/fapi/v1/order',
                    params=entry_params,
                    signed=True
                )
                
                if not (entry_response and entry_response.get('orderId')):
                    raise Exception(f"Entry order failed: {entry_response}")
                
                logging.info(f"[{signal.symbol}] Entry order placed: {entry_response['orderId']}")
                
                # SL/TP orders (paralel)
                price_precision = symbol_info.price_precision
                sl_price_str = f"{stop_loss_price:.{price_precision}f}"
                tp_price_str = f"{take_profit_price:.{price_precision}f}"
                
                sl_params = {
                    'symbol': signal.symbol,
                    'side': 'SELL',
                    'type': 'STOP_MARKET', 
                    'quantity': quantity,
                    'stopPrice': sl_price_str,
                    'reduceOnly': 'true'
                }
                
                tp_params = {
                    'symbol': signal.symbol,
                    'side': 'SELL',
                    'type': 'TAKE_PROFIT_MARKET',
                    'quantity': quantity, 
                    'stopPrice': tp_price_str,
                    'reduceOnly': 'true'
                }
                
                # SL ve TP'yi paralel gönder
                sl_response, tp_response = await asyncio.gather(
                    self._api_request_with_retry('POST', '/fapi/v1/order', params=sl_params, signed=True),
                    self._api_request_with_retry('POST', '/fapi/v1/order', params=tp_params, signed=True),
                    return_exceptions=True
                )
                
                # SL order ID
                if isinstance(sl_response, dict) and sl_response.get('orderId'):
                    sl_order_id = sl_response['orderId']
                    logging.info(f"[{signal.symbol}] SL order placed: {sl_order_id}")
                else:
                    logging.error(f"[{signal.symbol}] SL order failed: {sl_response}")
                    
                # TP order ID  
                if isinstance(tp_response, dict) and tp_response.get('orderId'):
                    tp_order_id = tp_response['orderId']
                    logging.info(f"[{signal.symbol}] TP order placed: {tp_order_id}")
                else:
                    logging.error(f"[{signal.symbol}] TP order failed: {tp_response}")
            
            # Order ID'leri trade'e ekle
            trade.sl_order_id = sl_order_id
            trade.tp_order_id = tp_order_id
            
            # ===== PHASE 3: FINAL COMMIT (ATOMIC) =====
            async with self._positions_lock:
                # Son kontrol: Rezervasyon hala geçerli mi?
                if signal.symbol not in self.active_positions or self.active_positions[signal.symbol] != "RESERVED":
                    # Başka bir thread bizim rezervasyonumuzu değiştirmiş!
                    raise Exception("Reservation was overwritten by another thread")
                
                # ✅ Gerçek trade object'i kaydet
                self.active_positions[signal.symbol] = trade
                logging.info(f"[{signal.symbol}] Position successfully committed to memory")
            
            # Paper modda balance güncellemesi
            if config.TRADING_MODE == TradingMode.PAPER:
                async with self._balance_lock:
                    self.paper_balance -= position_value
                    logging.debug(f"[{signal.symbol}] Paper balance updated: ${self.paper_balance:.2f}")
            
            # ===== SUCCESS ACTIONS =====
            # Bildirim gönder
            try:
                await self.send_trade_notification(trade, "OPENED", position_value=position_value)
            except Exception as notification_error:
                # Bildirim hatası trade'i iptal etmemeli
                logging.error(f"[{signal.symbol}] Notification failed: {notification_error}")
            
            # Metrics
            self.health_monitor.record_metric('trades_opened')
            
            logging.info(f"[{signal.symbol}] ✅ TRADE EXECUTED: ${position_value:.2f}, Conf: {signal.confidence:.1%}")
            return True
            
        except Exception as e:
            # ===== PHASE 4: ERROR CLEANUP (ATOMIC) =====
            async with self._positions_lock:
                # Rezervasyonu temizle
                if signal.symbol in self.active_positions and self.active_positions[signal.symbol] == "RESERVED":
                    del self.active_positions[signal.symbol]
                    logging.debug(f"[{signal.symbol}] Reservation cleaned up after error")
            
            # Canlı modda emergency order cleanup
            if config.TRADING_MODE == TradingMode.LIVE:
                try:
                    # Eğer kısmi bir şeyler açılmışsa temizle
                    await self._api_request_with_retry(
                        'DELETE', '/fapi/v1/allOpenOrders',
                        params={'symbol': signal.symbol},
                        signed=True
                    )
                    logging.info(f"[{signal.symbol}] Emergency order cleanup sent")
                except Exception as cleanup_error:
                    logging.error(f"[{signal.symbol}] Emergency cleanup failed: {cleanup_error}")
            
            logging.error(f"[{signal.symbol}] ❌ TRADE EXECUTION FAILED: {e}")
            return False

    async def update_settings_based_on_time(self):
        """V14 - Merkezi config objesini doğru metod ismiyle günceller."""
        if not config.ENABLE_DYNAMIC_SETTINGS:
            return

        now_hour = datetime.now(IST_TIMEZONE).hour
        
        target_fr, target_ai, mode_name = (config.RELAX_FR_THRESHOLD, config.RELAX_AI_CONFIDENCE, "RELAX") \
            if config.RELAX_MODE_START_HOUR <= now_hour < config.DEFENSIVE_MODE_START_HOUR \
            else (config.DEFENSIVE_FR_THRESHOLD, config.DEFENSIVE_AI_CONFIDENCE, "DEFENSIVE")

        if config.FR_LONG_THRESHOLD != target_fr or config.AI_CONFIDENCE_THRESHOLD != target_ai:
            logging.info(f"TIME-BASED SETTINGS: Switching to {mode_name} mode.")
            
            # ===== BURASI DÜZELTİLDİ =====
            # 'update_live_setting' yerine doğru metod adı olan 'set_value' kullanılıyor.
            config.set_value('FR_LONG_THRESHOLD', str(target_fr))
            config.set_value('AI_CONFIDENCE_THRESHOLD', str(target_ai))
            
            await self.send_telegram(
                f"🕰️ <b>Dynamic Settings Update</b> 🕰️\n"
                f"Switched to <b>{mode_name} Mode</b>.\n\n"
                f"New FR Threshold: <code>{config.FR_LONG_THRESHOLD}</code>\n"
                f"New AI Confidence: <code>{config.AI_CONFIDENCE_THRESHOLD}</code>"
            )

    async def manage_active_positions(self):
        """
        V15.2 - Race condition tamamen giderildi
        
        ÖNCEKİ SORUNLAR:
        1. Dictionary iteration sırasında concurrent modification
        2. Position update'leri atomic değil
        3. close_position çakışmaları
        4. Trailing stop update conflicts
        
        YENİ ÇÖZÜM:
        1. Safe iteration ile snapshot
        2. Position-level locking
        3. State management ile duplicate close prevention
        4. Atomic updates
        """
        
        # ===== STEP 1: SAFE SNAPSHOT CREATION =====
        # Active positions'ların güvenli bir kopyasını al
        async with self._positions_lock:
            if not self.active_positions:
                return
            
            # Snapshot: symbol -> trade mapping (shallow copy yeterli)
            positions_snapshot = dict(self.active_positions)
            active_count = len(positions_snapshot)
        
        logging.info(f"--- Managing {active_count} active positions ---")
        
        # ===== STEP 2: EXCHANGE SYNC DATA =====
        # Borsadaki pozisyonları bir kez çek (tüm semboller için)
        live_positions = {}
        try:
            if config.TRADING_MODE == TradingMode.LIVE:
                position_risk_data = await self._api_request_with_retry('GET', '/fapi/v2/positionRisk', signed=True)
                if position_risk_data:
                    live_positions = {
                        pos['symbol']: float(pos.get('positionAmt', '0')) 
                        for pos in position_risk_data
                    }
        except Exception as e:
            logging.warning(f"Could not fetch live positions for sync: {e}")
            # Sync yapamıyorsak, mevcut logic ile devam et
            live_positions = {}
        
        # ===== STEP 3: PARALLEL POSITION PROCESSING =====
        # Her position için concurrent task oluştur (ama controlled concurrency)
        semaphore = asyncio.Semaphore(3)  # Max 3 position aynı anda işlensin
        tasks = []
        
        for symbol, trade in positions_snapshot.items():
            task = asyncio.create_task(
                self._process_single_position_safe(
                    symbol, trade, live_positions.get(symbol, 0.0), semaphore
                )
            )
            tasks.append(task)
        
        # Tüm position'ları paralel işle
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logging.error(f"Error in parallel position processing: {e}")

    async def _process_single_position_safe(self, symbol: str, trade: TradeMetrics,
                                        live_position_amt: float, semaphore: asyncio.Semaphore):
        """
        Tek bir position'ı güvenli şekilde işle (ENHANCED NOTIFICATION VERSION).
        """
        async with semaphore:
            try:
                # ===== STEP 1: POSITION VALIDATION =====
                async with self._positions_lock:
                    if symbol not in self.active_positions:
                        return
                    current_trade = self.active_positions.get(symbol)
                    if not isinstance(current_trade, TradeMetrics):
                        return
                
                # ===== STEP 2: EXCHANGE SYNC =====
                if config.TRADING_MODE == TradingMode.LIVE and live_position_amt is not None:
                    if live_position_amt == 0.0:
                        current_price = await self.get_current_price(symbol) or current_trade.entry_price
                        await self._close_position_safe(symbol, "synced_closed_on_exchange", current_price)
                        return
                    if live_position_amt < 0:
                        await self._remove_position_safe(symbol, "converted_to_short")
                        return

                # ===== STEP 3: CURRENT PRICE =====
                current_price = await self.get_current_price(symbol)
                if not current_price:
                    return

                # ===== STEP 4: ATOMIC UPDATE & VARIABLE DEFINITION =====
                needs_trailing_update = False
                new_sl_price = None

                async with self._positions_lock:
                    if symbol not in self.active_positions: return
                    live_trade = self.active_positions.get(symbol)
                    if not isinstance(live_trade, TradeMetrics): return
                    
                    pnl_pct = (current_price - live_trade.entry_price) / live_trade.entry_price
                    live_trade.max_profit = max(live_trade.max_profit, pnl_pct)
                    live_trade.max_drawdown = min(live_trade.max_drawdown, pnl_pct)
                    live_trade.highest_price_seen = max(live_trade.highest_price_seen, current_price)
                    
                    # Variables tanımla
                    current_sl = live_trade.stop_loss
                    current_tp = live_trade.take_profit
                    
                    if config.ENABLE_TRAILING_STOP and not live_trade.trailing_stop_activated and current_price > live_trade.entry_price * (1 + config.TRAILING_STOP_ACTIVATION_PERCENT):
                        live_trade.trailing_stop_activated = True
                        logging.info(f"[{symbol}] Trailing stop activated at ${current_price:.4f}")

                    if live_trade.trailing_stop_activated:
                        new_sl_candidate = live_trade.highest_price_seen * (1 - (live_trade.volatility / 100 * config.TRAILING_STOP_DISTANCE_ATR_MULTIPLIER))
                        if new_sl_candidate > current_sl and new_sl_candidate > live_trade.entry_price:
                            needs_trailing_update = True
                            new_sl_price = new_sl_candidate

                # ===== STEP 5: TRAILING STOP UPDATE =====
                if needs_trailing_update and new_sl_price:
                    success = await self._update_trailing_stop_safe(symbol, new_sl_price)
                    if success:
                        current_sl = new_sl_price

                # ===== STEP 5.5: ENHANCED POSITION NOTIFICATIONS =====
                try:
                    sl_distance_pct = abs(current_price - current_sl) / current_price
                    tp_distance_pct = abs(current_tp - current_price) / current_price
                    
                    # SL approaching (sadece bir kez gönder)
                    if sl_distance_pct < 0.01:
                        if not hasattr(live_trade, '_sl_warning_sent'):
                            logging.info(f"[{symbol}] 📊 SL approaching notification - Distance: {sl_distance_pct:.2%}")
                            await self.send_position_update_notification(symbol, "sl_approaching", current_price, live_trade)
                            live_trade._sl_warning_sent = True
                            logging.info(f"[{symbol}] ✅ SL approaching notification sent and flagged")
                        else:
                            logging.debug(f"[{symbol}] SL approaching notification already sent")
                    
                    # TP approaching (sadece bir kez gönder)
                    if tp_distance_pct < 0.01:
                        if not hasattr(live_trade, '_tp_warning_sent'):
                            logging.info(f"[{symbol}] 🎯 TP approaching notification - Distance: {tp_distance_pct:.2%}")
                            await self.send_position_update_notification(symbol, "tp_approaching", current_price, live_trade)
                            live_trade._tp_warning_sent = True
                            logging.info(f"[{symbol}] ✅ TP approaching notification sent and flagged")
                        else:
                            logging.debug(f"[{symbol}] TP approaching notification already sent")
                            
                except Exception as notif_error:
                    logging.error(f"[{symbol}] Notification error: {notif_error}", exc_info=True)

                # ===== STEP 6: EXIT CONDITIONS CHECK (enhanced logging) =====
                if current_price <= current_sl:
                    logging.info(f"[{symbol}] 🛑 STOP LOSS TRIGGERED: {current_price:.6f} <= {current_sl:.6f}")
                    await self._close_position_safe(symbol, "stop_loss_hit", current_price)
                elif current_price >= current_tp:
                    logging.info(f"[{symbol}] 🎯 TAKE PROFIT TRIGGERED: {current_price:.6f} >= {current_tp:.6f}")
                    await self._close_position_safe(symbol, "take_profit_hit", current_price)
                    
            except Exception as e:
                logging.error(f"[{symbol}] Error processing position: {e}", exc_info=True)
 
    async def _close_position_safe(self, symbol: str, reason: str, exit_price: float):
        """
        Race condition safe position closing - ENHANCED NOTIFICATION VERSION
        """
        # ===== STEP 1: ATOMIC CLOSE ATTEMPT =====
        async with self._positions_lock:
            if symbol not in self.active_positions:
                logging.debug(f"[{symbol}] Position already closed/removed")
                return False
            
            trade = self.active_positions[symbol]
            
            # Type validation
            if not isinstance(trade, TradeMetrics):
                logging.warning(f"[{symbol}] Invalid position type during close: {type(trade)}")
                del self.active_positions[symbol]
                return False
            
            # Check if already being closed
            if hasattr(trade, '_closing_in_progress') and trade._closing_in_progress:
                # Check if stuck
                if hasattr(trade, '_close_start_time'):
                    stuck_time = datetime.now() - trade._close_start_time
                    if stuck_time.total_seconds() > 300:  # 5 minutes stuck
                        logging.error(f"[{symbol}] Position stuck for {stuck_time.total_seconds():.0f}s, force removing")
                        del self.active_positions[symbol]
                        return True
                
                logging.info(f"[{symbol}] Close already in progress by another thread")
                return False
            
            # Mark as closing
            trade._closing_in_progress = True
            trade._close_start_time = datetime.now()
            logging.info(f"[{symbol}] Close process started: {reason}")
        
        # ===== STEP 2: ENHANCED CLOSE NOTIFICATIONS =====
        try:
            # ===== DÜZELTME: SL/TP hit bildirimlerini garantile =====
            if reason == "stop_loss_hit":
                logging.info(f"[{symbol}] 🛑 STOP LOSS HIT - Sending notification")
                try:
                    await self.send_position_update_notification(
                        symbol, "sl_hit", exit_price, trade
                    )
                    logging.info(f"[{symbol}] ✅ Stop loss notification sent successfully")
                except Exception as notif_error:
                    logging.error(f"[{symbol}] ❌ Stop loss notification failed: {notif_error}")
            
            elif reason == "take_profit_hit":
                logging.info(f"[{symbol}] 🎯 TAKE PROFIT HIT - Sending notification")
                try:
                    await self.send_position_update_notification(
                        symbol, "tp_hit", exit_price, trade
                    )
                    logging.info(f"[{symbol}] ✅ Take profit notification sent successfully")
                except Exception as notif_error:
                    logging.error(f"[{symbol}] ❌ Take profit notification failed: {notif_error}")

            # ===== STEP 3: CLEAN ORDER BOOK =====
            # Canlı modda tüm açık emirleri iptal et
            if config.TRADING_MODE == TradingMode.LIVE:
                try:
                    orders_cancelled = await self.cancel_all_orders_for_symbol(symbol)
                    
                    if not orders_cancelled:
                        logging.warning(f"[{symbol}] Some orders may not have been cancelled")
                        
                        # Yedek: Spesifik SL/TP emirlerini iptal et
                        if hasattr(trade, 'sl_order_id') and trade.sl_order_id:
                            try:
                                await self._api_request_with_retry(
                                    'DELETE', '/fapi/v1/order',
                                    params={'symbol': symbol, 'orderId': trade.sl_order_id},
                                    signed=True
                                )
                            except:
                                pass
                                
                        if hasattr(trade, 'tp_order_id') and trade.tp_order_id:
                            try:
                                await self._api_request_with_retry(
                                    'DELETE', '/fapi/v1/order',
                                    params={'symbol': symbol, 'orderId': trade.tp_order_id},
                                    signed=True
                                )
                            except:
                                pass
                                
                except Exception as e:
                    logging.error(f"[{symbol}] Order cleanup failed: {e}")

            # ===== STEP 4: NORMAL CLOSE PROCESS =====
            success = await self.close_position(trade, reason, exit_price)
            return success
            
        except Exception as e:
            # ===== STEP 5: EMERGENCY CLEANUP =====
            async with self._positions_lock:
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                    logging.error(f"[{symbol}] EMERGENCY: Position force removed due to close error")
                    
                    try:
                        await self.send_telegram(
                            f"⚠️ <b>Emergency Position Cleanup</b>\n"
                            f"Symbol: {symbol}\n"
                            f"Reason: Close process failed\n"
                            f"Error: {str(e)[:100]}"
                        )
                    except:
                        pass
            
            return False

    async def _remove_position_safe(self, symbol: str, reason: str):
        """
        Position'ı güvenli şekilde memory'den kaldır (kapatmadan)
        """
        async with self._positions_lock:
            if symbol in self.active_positions:
                del self.active_positions[symbol]
                logging.info(f"[{symbol}] Position removed from memory: {reason}")
                
                # Notification gönder
                try:
                    await self.send_telegram(f"ℹ️ <b>{symbol}</b> position removed: {reason.replace('_', ' ').title()}")
                except Exception as e:
                    logging.error(f"Failed to send removal notification: {e}")

    async def _update_trailing_stop_safe(self, symbol: str, new_sl_price: float) -> bool:
        """
        Trailing stop'ı güvenli şekilde güncelle
        
        Returns:
            bool: Update başarılı ise True
        """
        try:
            # Get current order info
            async with self._positions_lock:
                if symbol not in self.active_positions:
                    return False
                
                trade = self.active_positions[symbol]
                if not isinstance(trade, TradeMetrics):
                    return False
                
                current_sl_order_id = trade.sl_order_id
                quantity = trade.quantity
            
            # Canlı modda exchange'de güncelle
            if config.TRADING_MODE == TradingMode.LIVE and current_sl_order_id:
                return await self.update_sl_on_exchange(trade, new_sl_price)
            
            # Paper modda sadece memory'yi güncelle
            elif config.TRADING_MODE == TradingMode.PAPER:
                async with self._positions_lock:
                    if symbol in self.active_positions and isinstance(self.active_positions[symbol], TradeMetrics):
                        self.active_positions[symbol].stop_loss = new_sl_price
                        logging.info(f"[{symbol}] Paper trailing SL updated: ${new_sl_price:.4f}")
                        return True
            
            return False
            
        except Exception as e:
            logging.error(f"[{symbol}] Trailing stop update failed: {e}")
            return False

    async def update_sl_on_exchange(self, trade: TradeMetrics, new_sl_price: float):
        """
        V10.6 - Borsadaki STOP_MARKET emrini günceller (iptal edip yeniden göndererek).
        """
        # Eğer SL emrinin ID'sini bilmiyorsak, güncelleme yapamayız.
        if not trade.sl_order_id:
            logging.warning(f"[{trade.symbol}] Cannot update SL on exchange: Missing SL Order ID.")
            return

        # Fiyatı, sembolün hassasiyetine göre doğru formatta string'e çevir
        price_precision = self.symbol_info[trade.symbol].price_precision
        new_sl_price_str = f"{new_sl_price:.{price_precision}f}"

        # 1. Eski SL emrini iptal et
        logging.info(f"[{trade.symbol}] Attempting to cancel old SL order: {trade.sl_order_id}")
        cancel_params = {'symbol': trade.symbol, 'orderId': trade.sl_order_id}
        cancel_response = await self._api_request_with_retry('DELETE', '/fapi/v1/order', params=cancel_params, signed=True)
        
        # İptal işlemi her zaman başarılı olmayabilir, özellikle emir zaten dolduysa.
        # Bu yüzden hatayı loglayıp devam ediyoruz.
        if not (cancel_response and cancel_response.get('orderId') == str(trade.sl_order_id)):
            logging.warning(f"[{trade.symbol}] Could not confirm cancellation of old SL order {trade.sl_order_id}. It might have been filled. Response: {cancel_response}")
            # Eski emri iptal edemezsek, yenisini göndermeyi denememek daha güvenli olabilir.
            # Ancak bazen "Order does not exist" hatası alırız ki bu da sorun değil.
            # Bu yüzden devam etmeyi deneyelim.

        # 2. Yeni SL emrini gönder
        logging.info(f"[{trade.symbol}] Attempting to place new Trailing SL order at ${new_sl_price_str}")
        new_sl_params = {
            'symbol': trade.symbol, 'side': 'SELL', 'type': 'STOP_MARKET',
            'quantity': trade.quantity, 'stopPrice': new_sl_price_str, 'reduceOnly': 'true'
        }
        new_sl_response = await self._api_request_with_retry('POST', '/fapi/v1/order', params=new_sl_params, signed=True)
        
        if new_sl_response and new_sl_response.get('orderId'):
            # Yeni bilgileri trade objesinde ve hafızada güncelle
            trade.stop_loss = new_sl_price
            trade.sl_order_id = new_sl_response.get('orderId')
            logging.info(f"[{trade.symbol}] ✅ Trailing SL successfully updated on exchange. New SL Order ID: {trade.sl_order_id}")
        else:
            logging.error(f"[{trade.symbol}] ❌ FAILED to place new Trailing SL order on exchange after cancellation! Response: {new_sl_response}")
            # Yeni SL emri konulamazsa, bu pozisyon artık borsada SL koruması olmadan kalmış olabilir.
            # Bu kritik bir durumdur ve bildirilmelidir.
            await self.send_telegram(f"🚨 <b>CRITICAL WARNING</b> 🚨\nFailed to set new Trailing SL for <b>{trade.symbol}</b>. Position may be unprotected!") 

    async def cancel_all_orders_for_symbol(self, symbol: str) -> bool:
        """
        Bir sembol için TÜM açık emirleri güvenli şekilde iptal eder
        """
        try:
            # Önce açık emirleri listele
            open_orders = await self._api_request_with_retry(
                'GET', '/fapi/v1/openOrders',
                params={'symbol': symbol},
                signed=True
            )
        
            if not open_orders:
                logging.info(f"[{symbol}] No open orders to cancel")
                return True
        
            logging.info(f"[{symbol}] Found {len(open_orders)} open orders to cancel")
        
            # Her emri tek tek iptal et
            cancelled_count = 0
            for order in open_orders:
                try:
                    cancel_result = await self._api_request_with_retry(
                        'DELETE', '/fapi/v1/order',
                        params={
                            'symbol': symbol,
                            'orderId': order['orderId']
                        },
                        signed=True
                    )
                
                    if cancel_result:
                        cancelled_count += 1
                        logging.info(f"[{symbol}] Cancelled order {order['orderId']} (type: {order['type']})")
                    
                except Exception as e:
                    logging.error(f"[{symbol}] Failed to cancel order {order['orderId']}: {e}")
        
            # Toplu iptal komutu da gönder (ekstra güvenlik)
            try:
                await self._api_request_with_retry(
                    'DELETE', '/fapi/v1/allOpenOrders',
                    params={'symbol': symbol},
                    signed=True
                )
            except:
                pass
        
            logging.info(f"[{symbol}] Successfully cancelled {cancelled_count}/{len(open_orders)} orders")
            return cancelled_count == len(open_orders)
        
        except Exception as e:
            logging.error(f"[{symbol}] Error in cancel_all_orders: {e}")
            return False

    async def close_position(self, trade: TradeMetrics, reason: str, exit_price: float):
        """
        Finalizes a trade, calculates PNL, and ensures it's recorded.
        This is the final step after all exchange operations are complete.
        """
        symbol = trade.symbol
        try:
            # PNL'i keşfet veya hesapla
            calculated_pnl = (exit_price - trade.entry_price) * trade.quantity
            final_pnl = calculated_pnl

            if config.TRADING_MODE == TradingMode.LIVE:
                discovered_pnl = await self._discover_realized_pnl_safe(symbol, trade.entry_time)
                if discovered_pnl is not None:
                    final_pnl = discovered_pnl
                    logging.info(f"[{symbol}] Realized PNL from exchange: ${final_pnl:.4f}")
                else:
                    logging.warning(f"[{symbol}] Could not discover PNL, using calculated value: ${final_pnl:.4f}")

            # Trade nesnesini sonlandır
            trade.exit_time = datetime.now(IST_TIMEZONE)
            trade.exit_price = exit_price
            trade.pnl = final_pnl
            trade.exit_reason = reason
            
            # Kağıt bakiye güncellemesi
            if config.TRADING_MODE == TradingMode.PAPER:
                async with self._balance_lock:
                    position_return = (trade.quantity * trade.entry_price) + final_pnl
                    self.paper_balance += position_return
                    logging.info(f"[{symbol}] Paper balance updated to ${self.paper_balance:.2f}")

        finally:
            # Bu blok, yukarıda bir hata olsa bile ÇALIŞIR
            
            # 1. Trade'i geçmişe ekle
            self.trade_history.append(trade)
            self.health_monitor.record_metric('trades_closed')
            
            # 2. Veritabanına kaydet
            await self.db_manager.save_trade(trade)
            
            # 3. Cooldown'u ayarla
            if config.ENABLE_COOLDOWN:
                cooldown_end_time = datetime.now(IST_TIMEZONE) + timedelta(minutes=config.COOLDOWN_PERIOD_MINUTES)
                self.cooldown_symbols[symbol] = cooldown_end_time

            # ===== YENİ EKLENEN SATIR BURADA =====
            # 4. Korelasyon önbelleğinden bu sembolü temizle!
            if symbol in self.risk_manager.price_history_cache:
                del self.risk_manager.price_history_cache[symbol]
                logging.info(f"[{symbol}] Removed from correlation price cache after closing.")
            # =======================================

            # 5. Bildirim gönder (Artık 5. adım)
            await self.send_trade_notification(trade, "CLOSED")
            logging.info(f"[{symbol}] ✅ POSITION CLOSED & RECORDED: {reason} | P&L: ${trade.pnl or 0:.2f}")

        return True
    
    async def _discover_realized_pnl_safe(self, symbol: str, entry_time: datetime, max_attempts: int = 4) -> Optional[float]:
        """
        Borsadan gerçekleşen PNL'i güvenli şekilde keşfet
        
        Args:
            symbol: Trading symbol
            entry_time: Position entry time
            max_attempts: Maximum discovery attempts
            
        Returns:
            Optional[float]: Discovered PNL or None if not found
        """
        
        for attempt in range(max_attempts):
            try:
                # Start time'ı entry time'dan başlat
                start_time_ms = int(entry_time.timestamp() * 1000)
                
                params = {
                    'symbol': symbol,
                    'startTime': start_time_ms,
                    'limit': 50
                }
                
                user_trades = await self._api_request_with_retry(
                    'GET', '/fapi/v1/userTrades', 
                    params=params, 
                    signed=True
                )
                
                if not user_trades:
                    logging.debug(f"[{symbol}] No user trades found (attempt {attempt + 1})")
                    continue
                
                # Position kapatışıyla ilgili PNL'leri topla
                realized_pnls = []
                for trade_record in reversed(user_trades):  # En yeni'den başla
                    realized_pnl = float(trade_record.get('realizedPnl', '0'))
                    if realized_pnl != 0:
                        realized_pnls.append(realized_pnl)
                
                if realized_pnls:
                    # Birden fazla PNL varsa (partial fills) topla
                    total_realized_pnl = sum(realized_pnls)
                    logging.info(f"[{symbol}] PNL discovered on attempt {attempt + 1}: ${total_realized_pnl:.4f}")
                    return total_realized_pnl
                
                # PNL bulunamadı, bekle ve tekrar dene
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                    logging.debug(f"[{symbol}] PNL not found, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                logging.error(f"[{symbol}] PNL discovery attempt {attempt + 1} failed: {e}")
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep((attempt + 1) * 1.5)  # Progressive backoff
        
        logging.warning(f"[{symbol}] PNL discovery failed after all attempts")
        return None



        """
        Position'ın mevcut durumunu thread-safe şekilde al
        
        Returns:
            Dict with position info or None if not found
        """
        try:
            # No async context needed for read-only check
            if symbol in self.active_positions:
                trade = self.active_positions[symbol]
                
                if isinstance(trade, TradeMetrics):
                    return {
                        'symbol': symbol,
                        'entry_price': trade.entry_price,
                        'quantity': trade.quantity,
                        'exit_time': trade.exit_time,
                        'closing_in_progress': hasattr(trade, '_closing_in_progress') and trade._closing_in_progress,
                        'is_valid': True
                    }
                else:
                    return {
                        'symbol': symbol,
                        'type': str(type(trade)),
                        'is_valid': False
                    }
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting position state for {symbol}: {e}")
            return None

    
    # ===== HELPER FONKSİYONLARI (Class içinde doğru yerde) =====
    
    async def _check_existing_position_on_exchange(self, symbol: str) -> bool:
        """Exchange'de bu symbol için pozisyon var mı kontrol et"""
        try:
            pos_risk = await self._api_request_with_retry(
                'GET', '/fapi/v2/positionRisk',
                {'symbol': symbol},
                signed=True
            )
                
            if pos_risk and isinstance(pos_risk, list) and len(pos_risk) > 0:
                position_amount = float(pos_risk[0].get('positionAmt', '0'))
                return position_amount != 0
                
            return False
                
        except Exception as e:
            logging.error(f"[{symbol}] Error checking existing position: {e}")
            # Hata durumunda güvenli olmak için True döndür
            return True

    async def force_remove_position_safe(self, symbol: str, reason: str = "force_cleanup"):
        """
        Acil durumlarda position'ı güvenli şekilde zorla kaldır
        
        Bu method sadece error recovery için kullanılmalı
        Normal close için close_position kullanın
        """
        async with self._positions_lock:
            if symbol in self.active_positions:
                trade = self.active_positions[symbol]
                del self.active_positions[symbol]
                
                logging.warning(f"[{symbol}] Position force removed: {reason}")
                
                # Emergency notification
                try:
                    await self.send_telegram(f"🚨 <b>Emergency Position Removal</b>\n<b>{symbol}</b>: {reason}")
                except:
                    pass
                
                return True
        
        return False

    def get_position_state_safe(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Position'ın mevcut durumunu thread-safe şekilde al
        
        Returns:
            Dict with position info or None if not found
        """
        try:
            # No async context needed for read-only check
            if symbol in self.active_positions:
                trade = self.active_positions[symbol]
                
                if isinstance(trade, TradeMetrics):
                    return {
                        'symbol': symbol,
                        'entry_price': trade.entry_price,
                        'quantity': trade.quantity,
                        'exit_time': trade.exit_time,
                        'closing_in_progress': hasattr(trade, '_closing_in_progress') and trade._closing_in_progress,
                        'is_valid': True
                    }
                else:
                    return {
                        'symbol': symbol,
                        'type': str(type(trade)),
                        'is_valid': False
                    }
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting position state for {symbol}: {e}")
            return None

    async def auto_tune_parameters(self):
        """V14 - Merkezi config objesini kullanarak çalışır."""
        
        # Ayarları global 'config' objesinden oku
        if not config.ENABLE_AUTO_PARAM_TUNING:
            return
            
        now = datetime.now(IST_TIMEZONE)
        
        # Ayarları global 'config' objesinden oku
        if (now - self.last_auto_tune_time.astimezone(IST_TIMEZONE)) < timedelta(hours=config.AUTO_TUNE_INTERVAL_HOURS):
            return

        logging.info("🔬 Running auto-parameter tuning check...")
        self.last_auto_tune_time = now

        recent_trades = [
            t for t in self.trade_history 
            if t.exit_time and (now - t.exit_time.astimezone(IST_TIMEZONE)) < timedelta(days=7)
        ]
        
        # Ayarı global 'config' objesinden oku
        if len(recent_trades) < config.AUTO_TUNE_MIN_TRADES:
            logging.info(f"Auto-tune skipped: Not enough recent trades ({len(recent_trades)}/{config.AUTO_TUNE_MIN_TRADES})")
            return

        wins = sum(1 for t in recent_trades if t.pnl is not None and t.pnl > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0
        
        adjustment = 0.0
        if win_rate > 0.65:
            # Ayarı global 'config' objesinden oku
            adjustment = -config.AUTO_TUNE_STEP 
        elif win_rate < 0.45:
            # Ayarı global 'config' objesinden oku
            adjustment = config.AUTO_TUNE_STEP
        
        if adjustment != 0.0:
            # Ayarı değiştirmek için global 'config' objesinin metodunu çağır
            config.adjust_confidence_threshold(adjustment)
            
            await self.send_telegram(
                f"⚙️ <b>Auto-Tuning Activated</b>\n"
                f"New AI Confidence: <b>{config.AI_CONFIDENCE_THRESHOLD:.3f}</b>\n" # Değeri yine config'den oku
                f"(Based on {win_rate:.1%} win rate from last {len(recent_trades)} trades)"
            )
        else:
            logging.info(f"Auto-tune: No adjustment needed. Win rate ({win_rate:.1%}) is within the stable range (45%-65%).")

    async def run(self):
        """V15.1 - Memory management ile ana döngü"""
        
        # ===== BAŞLANGIÇ PROSEDÜRÜ =====
        await self.initialize()
        await self.update_settings_based_on_time()
        logging.info("Sending initial status with memory monitoring...")
        await self.command_handler.cmd_status()

        # ===== ANA DÖNGÜ =====
        while True:
            try:
                start_time = time.time()

                try:
                    # ❌ GEÇİCİ: Emergency mode kontrollerini kapat
                    """
                    # Emergency mode'daysa sadece temel işlemleri yap
                    if self.error_recovery.emergency_mode:
                        logging.info("Emergency mode active - limited operations only")
                        
                        # Sadece pozisyon yönetimi yap
                        await self.manage_active_positions()
                        
                        # Emergency mode'dan çıkış kontrolü
                        if self.error_recovery.should_exit_emergency_mode():
                            await self.error_recovery.exit_emergency_mode()
                        
                        # Emergency mode'dayken daha uzun bekle
                        await asyncio.sleep(60)
                        continue  # Normal operations'ları atla
                    
                    # Emergency mode tetikleyicilerini kontrol et
                    emergency_triggers = self.error_recovery.should_trigger_emergency()
                    if any(emergency_triggers.values()):
                        await self.error_recovery.activate_emergency_mode(emergency_triggers)
                        continue  # Bu cycle'ı atla
                    """
                            
                except Exception as emergency_error:
                    logging.error(f"Emergency mode check failed: {emergency_error}")
                
                # ===== NORMAL OPERATIONS =====
                await self.poll_telegram_messages()
                await self.update_settings_based_on_time()
                await self.manage_active_positions()
                await self.auto_tune_parameters()
                await self.scan_markets()
                
                # ===== MEMORY MONITORING (BASİT VERSİYON) =====
                # Enhanced memory manager yerine basit kontrol
                try:
                    # Her 50 döngüde bir basit memory kontrolü
                    loop_count = getattr(self, '_loop_count', 0) + 1
                    self._loop_count = loop_count
                    
                    if loop_count % 50 == 0:
                        try:
                            memory_stats = self.memory_manager.get_memory_usage()
                            if memory_stats['rss_mb'] > 400:  # 400MB üzerinde uyar
                                logging.warning(f"High memory usage: {memory_stats['rss_mb']:.1f}MB")
                                # Basit cleanup
                                self._cleanup_old_data()
                                gc.collect()
                        except Exception as e:
                            logging.error(f"Simple memory check failed: {e}")
                            
                except Exception as memory_error:
                    logging.error(f"Memory monitoring error: {memory_error}")
                
                # ===== LOOP TIMING =====
                loop_duration = time.time() - start_time
                sleep_time = max(0, config.MARKET_SCAN_INTERVAL - loop_duration)
                
                if sleep_time == 0:
                    logging.warning(f"Loop took longer than interval: {loop_duration:.2f}s")
                
                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logging.info("Shutdown requested by user.")
                break
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)  # Kısa bekleme

    async def poll_telegram_messages(self):
        """Polls Telegram for new commands."""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            params = {'offset': self.telegram_offset, 'timeout': 1, 'allowed_updates': ['message']}
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('ok') and data.get('result'):
                        for update in data['result']:
                            self.telegram_offset = update['update_id'] + 1
                            if 'message' in update and str(update['message']['chat']['id']) == self.chat_id:
                                await self.command_handler.handle_command(update['message'].get('text', ''))
        except Exception:
            # Polling hatalarını sessizce geç, logları kirletmesin
            pass  
 
    # ==================================================================
    # ADIM 4: scan_markets FONKSİYONUNU TAMAMEN DEĞİŞTİRİN
    # ==================================================================
    async def scan_markets(self):
        """
        V36 - Signal Funnel Edition.
        Adayları en ucuzdan en pahalıya doğru filtreler ve her adımı loglar.
        """
        if len(self.active_positions) >= config.MAX_ACTIVE_TRADES:
            logging.info(f"Max active trades ({config.MAX_ACTIVE_TRADES}) reached. Skipping scan.")
            return

        # Her tarama için istatistikleri sıfırla
        self.funnel_stats = SignalFunnelStats(total_symbols=len(self.all_symbols))

        # --- ADIM 1: HIZLI FR ÖN TARAMASI (En Ucuz Filtre) ---
        candidates = await self._fast_fr_pre_scan()
        if not candidates:
            logging.info("Signal Funnel: No symbols matched FR criteria in fast scan.")
            self._log_funnel_report()
            return
        self.funnel_stats.initial_fr_candidates = list(candidates)

        # --- ADIM 2: ZAMANLAMA VE COOLDOWN FİLTRESİ (Çok Ucuz) ---
        passed_timing_filter = []
        funding_data_map = {item['symbol']: item for item in await self._api_request_with_retry('GET', '/fapi/v1/premiumIndex') or []}

        for symbol in candidates:
            # Cooldown kontrolü
            if config.ENABLE_COOLDOWN and symbol in self.cooldown_symbols and datetime.now(IST_TIMEZONE) < self.cooldown_symbols[symbol]:
                self.funnel_stats.reasons_for_rejection['cooldown_active'] += 1
                continue
            
            # Zamanlama kontrolü
            funding_info = funding_data_map.get(symbol)
            if not funding_info or not self.is_optimal_funding_window(int(funding_info.get('nextFundingTime', 0))):
                self.funnel_stats.reasons_for_rejection['bad_timing_window'] += 1
                continue
            
            passed_timing_filter.append(symbol)
        self.funnel_stats.after_timing_cooldown = list(passed_timing_filter)

        # --- ADIM 3: KORELASYON FİLTRESİ (Orta Maliyetli) ---
        passed_correlation_filter = []
        if config.ENABLE_CORRELATION_FILTER and self.active_positions:
             for symbol in passed_timing_filter:
                if await self.risk_manager.check_correlation_risk(symbol):
                    passed_correlation_filter.append(symbol)
                else:
                    self.funnel_stats.reasons_for_rejection['high_correlation'] += 1
        else:
            passed_correlation_filter = list(passed_timing_filter)
        self.funnel_stats.after_correlation = list(passed_correlation_filter)
        
        # --- ADIM 4: DERİN ANALİZ & AI (En Pahalı Filtre) ---
        # Sadece önceki tüm filtrelerden geçen az sayıdaki sembolü analiz et
        if not passed_correlation_filter:
            logging.info("Signal Funnel: No candidates left after preliminary filters.")
            self._log_funnel_report()
            return

        logging.info(f"Signal Funnel: Starting deep AI analysis for {len(passed_correlation_filter)} candidates...")

        opportunities = []
        for symbol in passed_correlation_filter:
            try:
                if symbol in self.active_positions: continue
                
                # generate_trading_signal artık reddetme nedenini de döndürebilir
                signal, reason = await self.generate_trading_signal(symbol)
                if signal:
                    opportunities.append(signal)
                elif reason:
                    self.funnel_stats.reasons_for_rejection[reason] += 1

            except Exception as e:
                logging.error(f"Error during deep analysis of {symbol}: {e}")
                self.funnel_stats.reasons_for_rejection['deep_analysis_error'] += 1
                continue
        
        self.funnel_stats.after_deep_analysis = list(opportunities)
        self._log_funnel_report() # Son raporu logla

        # --- ADIM 5: İŞLEM YAPMA ---
        if not opportunities:
            return

        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        available_slots = config.MAX_ACTIVE_TRADES - len(self.active_positions)
        if available_slots <= 0: return

        selected_opportunities = opportunities[:available_slots]
        for signal in selected_opportunities:
            await self.execute_trade(signal)
            
            
    # ==================================================================
    # ADIM 5: analyze_symbol ve generate_trading_signal FONKSİYONLARINI DEĞİŞTİRİN
    # ==================================================================
    @safe_async_operation
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Sadece bir sembol için derinlemesine analiz yapar ve gerekli tüm verileri döndürür.
        FİLTRELEME YAPMAZ, sadece veri toplar ve hazırlar.
        """
        funding_info = await self._api_request_with_retry('GET', '/fapi/v1/premiumIndex', {'symbol': symbol})
        if not funding_info: return None

        klines = await self._api_request_with_retry('GET', '/fapi/v1/klines', {'symbol': symbol, 'interval': '5m', 'limit': 150})
        if not klines or len(klines) < 100: return None

        # ===== DÜZELTME BURADA YAPILDI =====
        # Sadece ilk 6 sütunu al ve onlara isim ver
        df = pd.DataFrame(klines, dtype=float).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        current_fr = float(funding_info.get('lastFundingRate', 0)) * 100
        df_featured = await self.feature_engineer.create_features(df.copy(), current_fr)
        if df_featured.empty: return None

        market_regime = EnhancedMarketRegimeDetector.detect_regime(df_featured)

        return {
            'symbol': symbol,
            'funding_rate': current_fr,
            'df_featured': df_featured,
            'market_regime': market_regime,
            'next_funding_time': funding_info.get('nextFundingTime', 0)
        }

    async def generate_trading_signal(self, symbol: str) -> Tuple[Optional[SignalData], Optional[str]]:
        """
        Bir sembolü analiz eder ve bir alım sinyali veya reddetme nedeni döndürür.
        Returns: (SignalData, rejection_reason)
        """
        analysis = await self.analyze_symbol(symbol)
        if not analysis:
            return None, "analysis_failed"

        df_featured = analysis['df_featured']
        current_fr = analysis['funding_rate']
        market_regime = analysis['market_regime']

        # Model ve AI Confidence
        if not await self.get_or_train_ensemble_model(symbol):
            return None, "model_training_failed"
        
        confidence = 0.0
        try:
            # ===== DÜZELTME BURADA YAPILDI =====
            if symbol in self.scalers and symbol in self.selected_features:
                model_features = self.selected_features[symbol]
                
                # Gerekli özelliklerin DataFrame'de olup olmadığını kontrol et
                if not all(feature in df_featured.columns for feature in model_features):
                    logging.error(f"[{symbol}] Prediction skipped. Missing features for model.")
                    return None, "missing_features"
                
                latest_features = df_featured[model_features].iloc[-1:].values
                features_scaled = self.scalers[symbol].transform(latest_features)
                confidence = self.ensemble_models[symbol].predict_proba(features_scaled)[0][1]
            else:
                # Bu durum, model eğitimi başarılı olsa da scaler veya feature listesi bulunamadığında oluşur.
                logging.warning(f"[{symbol}] Scaler or features not found after model training.")
                return None, "scaler_or_features_missing"

        except Exception as e:
            logging.error(f"[{symbol}] Prediction failed: {e}", exc_info=True)
            return None, "prediction_error"

        # Gelişmiş Güven Filtresi
        if current_fr <= -1.0:
            required_confidence = 0.60
        elif market_regime == MarketRegime.DOWNTREND:
            required_confidence = config.AI_CONFIDENCE_THRESHOLD + 0.15
        else:
            required_confidence = config.AI_CONFIDENCE_THRESHOLD

        if confidence < required_confidence:
            logging.debug(f"[{symbol}] AI confidence {confidence:.2f} < {required_confidence:.2f}")
            return None, "ai_confidence_too_low"

        # Pump Filtresi
        if config.ENABLE_PUMP_FILTER:
            sensitivity = config.PUMP_SENSITIVITY_UPTREND if market_regime == MarketRegime.UPTREND else config.PUMP_SENSITIVITY_RANGING
            true_range = df_featured['high'] - df_featured['low']
            recent_avg = true_range.tail(config.PUMP_FILTER_RECENT_CANDLES).mean()
            lookback_avg = true_range.iloc[-(config.PUMP_FILTER_LOOKBACK_CANDLES + config.PUMP_FILTER_RECENT_CANDLES):-config.PUMP_FILTER_RECENT_CANDLES].mean()
            if lookback_avg > 0 and (recent_avg / lookback_avg) > sensitivity:
                return None, "pump_filter_triggered"
        
        # Sinyal Onaylandı
        logging.info(f"[{symbol}] ✅ SIGNAL APPROVED: FR={current_fr:.3f}%, Conf={confidence:.1%}, Regime={market_regime.value}")
        return SignalData(
            symbol=symbol, side='BUY', confidence=confidence, funding_rate=current_fr,
            current_price=df_featured['close'].iloc[-1], volatility=df_featured['atr_percent'].iloc[-1],
            market_regime=market_regime, next_funding_time=analysis.get('next_funding_time')
        ), None
    
    # Orijinal koddan gelen diğer yardımcı fonksiyonlar
    async def update_and_filter_symbols(self):
        logging.info("Updating symbol list...")
        data = await self._api_request_with_retry('GET', '/fapi/v1/exchangeInfo')
        if not data: return
        self.all_symbols = []
        for s_data in data.get('symbols', []):
            if s_data['symbol'].endswith('USDT') and s_data['status'] == 'TRADING' and s_data['symbol'] not in BLACKLISTED_SYMBOLS:
                # minNotional filtresini bul ve SymbolInfo'ya ekle
                min_notional_filter = next((f for f in s_data.get('filters', []) if f.get('filterType') == 'MIN_NOTIONAL'), None)
                min_notional_value = float(min_notional_filter.get('notional', '5.0')) if min_notional_filter else 5.0

                self.all_symbols.append(s_data['symbol'])
                self.symbol_info[s_data['symbol']] = SymbolInfo(
                    symbol=s_data['symbol'], 
                    price_precision=s_data['pricePrecision'], 
                    quantity_precision=s_data['quantityPrecision'],
                    min_notional=min_notional_value # Yeni alanı ata
                )
        logging.info(f"Symbol update complete: {len(self.all_symbols)} symbols")

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Sembolün mevcut fiyatını al"""
        try:
            ticker = await self._api_request_with_retry('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
            return float(ticker['price']) if ticker and 'price' in ticker else None
        except Exception as e:
            logging.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def get_or_train_ensemble_model(self, symbol: str) -> bool:
        """
        V15.3 Refactored - Uses the unified EnsemblePredictor for non-blocking training.
        """
        now = datetime.now(IST_TIMEZONE)
        
        # Check if a recent and valid model already exists
        if symbol in self.ensemble_models:
            self._track_model_usage(symbol)
            if (now - self.last_model_update.get(symbol, now)) < self.model_update_interval:
                return True

        try:
            # Memory check before training
            memory_stats = self.memory_manager.get_memory_usage()
            if memory_stats.get('rss_mb', 0) > self.memory_manager.max_memory_mb * 0.9:
                logging.warning(f"High memory usage ({memory_stats['rss_mb']:.1f}MB). Forcing cleanup before model training.")
                self._cleanup_old_data()
                self.memory_manager.force_cleanup()

            # Data fetching and preparation
            klines = await self._api_request_with_retry('GET', '/fapi/v1/klines', {'symbol': symbol, 'interval': '5m', 'limit': 500})
            if not klines or len(klines) < config.MIN_TRAINING_SAMPLES: return False
            
            # ===== DÜZELTME BURADA YAPILDI =====
            # Sadece ilk 6 sütunu al ve onlara isim ver
            df = pd.DataFrame(klines, dtype=float).iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            # Feature Engineering
            df_featured = await self.feature_engineer.create_features(df.copy(), 0.0)
            if df_featured.empty: return False

            # Target Creation
            df_featured['target'] = (df_featured['high'].rolling(config.N_FUTURE_CANDLES).max().shift(-config.N_FUTURE_CANDLES) > df_featured['close'] * 1.005).astype(int)
            df_train = df_featured.dropna()
            
            if len(df_train) < config.MIN_TRAINING_SAMPLES or df_train['target'].sum() < 3:
                logging.warning(f"[{symbol}] Insufficient training data after feature engineering.")
                return False

            # Feature Selection and Scaling
            controllable_features = [
                'open', 'high', 'low', 'close', 'volume', 'fundingRate', 'rsi_14', 
                'macd_diff', 'atr_percent', 'bb_width', 'momentum_5', 'momentum_10', 
                'momentum_20', 'ema_5_20_diff', 'volume_ratio'
            ]
            
            available_features_for_selection = [f for f in controllable_features if f in df_train.columns]

            self.selected_features[symbol] = self.feature_engineer.select_best_features(
                df_train[available_features_for_selection], 
                df_train['target'], 
                max_features=40
            )
            
            X = df_train[self.selected_features[symbol]].values
            y = df_train['target'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # --- Non-blocking Training Execution ---
            loop = asyncio.get_running_loop()
            ensemble = EnsemblePredictor()

            logging.info(f"[{symbol}] 🚀 Handing off AI training to executor...")
            
            training_success = await loop.run_in_executor(
                None, ensemble.fit, X_scaled, y
            )

            if not training_success:
                logging.error(f"[{symbol}] ❌ AI Ensemble training failed in background.")
                return False

            # --- Update Cache with the new model ---
            async with self._model_lock:
                self.ensemble_models[symbol] = ensemble
                self.scalers[symbol] = scaler
                self.last_model_update[symbol] = now
                self._track_model_usage(symbol)

            logging.info(f"[{symbol}] ✅ Model training complete. Cache updated.")
            return True
            
        except Exception as e:
            logging.error(f"Model training wrapper failed for {symbol}: {e}", exc_info=True)
            return False

    async def get_current_price(self, symbol: str) -> Optional[float]:
        ticker = await self._api_request_with_retry('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
        return float(ticker['price']) if ticker and 'price' in ticker else None

    async def load_trade_history(self):
        """V15.4 - Gelişmiş Teşhis ve Güvenli Veri Yorumlama ile Trade History Yükleme"""
        logging.info("Attempting to load trade history from database...")
        try:
            recent_trades_data = await self.db_manager.load_recent_trades(days=config.ANALYSIS_REPORT_DAYS)
            
            if not recent_trades_data:
                logging.warning("No recent trade data found in database. History will remain empty.")
                return

            logging.info(f"Found {len(recent_trades_data)} trade records in DB. Recreating objects...")
            loaded_count = 0
            
            for trade_data in recent_trades_data:
                try:
                    # GÜVENLİ ZAMAN DAMGASI DÖNÜŞÜMÜ
                    entry_time_str = str(trade_data['entry_time'])
                    entry_time = datetime.fromisoformat(entry_time_str.replace(" ", "T"))

                    exit_time = None
                    exit_time_str = trade_data.get('exit_time')
                    if exit_time_str:
                         exit_time = datetime.fromisoformat(str(exit_time_str).replace(" ", "T"))
                    
                    if entry_time.tzinfo is None: entry_time = IST_TIMEZONE.localize(entry_time)
                    if exit_time and exit_time.tzinfo is None: exit_time = IST_TIMEZONE.localize(exit_time)

                    recreated_trade = TradeMetrics(
                        symbol=trade_data['symbol'],
                        entry_time=entry_time,
                        exit_time=exit_time,
                        # Diğer tüm alanları float() ile güvenli hale getir
                        entry_price=float(trade_data['entry_price']),
                        quantity=float(trade_data['quantity']),
                        confidence=float(trade_data['confidence']),
                        funding_rate=float(trade_data['funding_rate']),
                        stop_loss=float(trade_data['stop_loss']),
                        take_profit=float(trade_data['take_profit']),
                        exit_price=float(trade_data['exit_price']) if trade_data.get('exit_price') is not None else None,
                        pnl=float(trade_data['pnl']) if trade_data.get('pnl') is not None else None,
                        exit_reason=trade_data.get('exit_reason'),
                        max_profit=float(trade_data.get('max_profit', 0)),
                        max_drawdown=float(trade_data.get('max_drawdown', 0)),
                        market_regime=trade_data.get('market_regime'),
                        volatility=float(trade_data.get('volatility', 0))
                    )
                    
                    self.trade_history.append(recreated_trade)
                    loaded_count += 1
                except Exception as e:
                    logging.error(f"Failed to recreate trade object. Error: {e}. Data: {trade_data}")
                    continue
            
            logging.info(f"Successfully loaded {loaded_count} trades. Total in history: {len(self.trade_history)}")
        except Exception as e:
            logging.critical(f"A critical error occurred while loading trade history: {e}", exc_info=True)

    async def save_performance_data(self):
        # Orijinal kodunuzdaki kaydetme mantığı burada olmalı
        pass

    async def send_trade_notification(self, trade: TradeMetrics, action: str, position_value: Optional[float] = None):
        """V11.1 - 'Analist Notu' (Strategy Rationale) ile zenginleştirilmiş bildirimler gönderir."""
        
        # ===== POZİSYON AÇMA BİLDİRİMİ =====
        if action == "OPENED":
            title = "🚀 POSITION LONG (V11.1)"
            
            # Countdown bilgisini formatla
            countdown_str = "N/A"
            if trade.time_to_funding_on_entry is not None:
                hours = int(trade.time_to_funding_on_entry // 60)
                minutes = int(trade.time_to_funding_on_entry % 60)
                countdown_str = f"{hours}h {minutes}m"

            # 'Analist Notu' bölümünü oluştur
            rationale = (
                f"<b><u>STRATEGY RATIONALE</u></b>\n"
                f"► <b>Regime:</b> {trade.market_regime}\n"
                f"► <b>Confidence:</b> {trade.confidence:.1%}\n"
                f"► <b>FR:</b> {trade.funding_rate:.3f}%\n"
                f"► <b>Volatility (ATR):</b> {trade.volatility:.2f}%\n"
                f"► <b>Smart SL:</b> {'Active' if trade.smart_sl_activated else 'Standard'}\n"
                f"► <b>Countdown:</b> {countdown_str}\n"
                f"► <b>Bot Mode:</b> {trade.bot_mode_on_entry}"
            )

            # Ana mesajı oluştur
            main_info = (
                f"💎 <b>Symbol:</b> {trade.symbol}\n"
                f"💰 <b>Entry:</b> ${trade.entry_price:.4f}\n"
                f"🛑 <b>SL:</b> ${trade.stop_loss:.4f}\n"
                f"🎯 <b>TP:</b> ${trade.take_profit:.4f}\n"
                f"💰 <b>Margin:</b> ${position_value:,.2f}"
            )
            
            # Tam mesajı birleştir
            message = f"<b>{title}</b>\n\n{main_info}\n----------------------------------------\n{rationale}"

        # ===== POZİSYON KAPATMA BİLDİRİMİ =====
        elif action == "CLOSED":
            pnl_emoji = "🟢" if trade.pnl is not None and trade.pnl > 0 else "🔴"
            holding_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time and trade.entry_time else 0
            message = (
                f"{pnl_emoji} <b>POSITION CLOSED</b>\n\n"
                f"💎 <b>Symbol:</b> {trade.symbol}\n"
                f"💰 <b>Exit:</b> ${trade.exit_price or 0:.4f}\n"
                f"📊 <b>P&L:</b> ${trade.pnl or 0:,.2f}\n"
                f"🎯 <b>Reason:</b> {trade.exit_reason.replace('_', ' ').title()}\n"
                f"⏰ <b>Duration:</b> {holding_time:.1f}h"
            )
        else:
            return

        await self.send_telegram(message)

    async def send_position_update_notification(self, symbol: str, update_type: str, 
                                          current_price: float, trade: TradeMetrics,
                                          pnl_usd: float = None, pnl_pct: float = None):
        """
        Pozisyon güncellemeleri için detaylı bildirim gönderir
        """
        try:
            # Update tipine göre emoji ve başlık
            if update_type == "sl_approaching":
                emoji = "⚠️"
                title = "STOP LOSS YAKLAŞIYOR"
            elif update_type == "sl_hit":
                emoji = "🛑"
                title = "STOP LOSS TETİKLENDİ"
            elif update_type == "tp_approaching":
                emoji = "🎯"
                title = "HEDEF YAKLAŞIYOR"
            elif update_type == "tp_hit":
                emoji = "✅"
                title = "HEDEF ULAŞILDI"
            elif update_type == "trailing_activated":
                emoji = "📈"
                title = "TRAILING STOP AKTİF"
            elif update_type == "trailing_updated":
                emoji = "🔄"
                title = "TRAILING STOP GÜNCELLENDİ"
            else:
                emoji = "📊"
                title = "POZİSYON GÜNCELLEMESİ"
            
            # PNL hesapla
            if pnl_usd is None and trade.entry_price:
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                pnl_usd = pnl_pct * trade.quantity * trade.entry_price
            
            pnl_emoji = "🟢" if pnl_usd > 0 else "🔴"
            
            # Mesaj oluştur
            message = f"{emoji} <b>{title}</b>\n\n"
            message += f"💎 <b>Symbol:</b> {symbol}\n"
            message += f"💰 <b>Current:</b> ${current_price:.4f}\n"
            message += f"📊 <b>Entry:</b> ${trade.entry_price:.4f}\n"
            message += f"{pnl_emoji} <b>P&L:</b> ${pnl_usd:+.2f} ({pnl_pct:+.2%})\n"
            
            # Özel bilgiler
            if update_type in ["sl_hit", "sl_approaching"]:
                message += f"🛑 <b>Stop Loss:</b> ${trade.stop_loss:.4f}\n"
                distance = abs(current_price - trade.stop_loss) / current_price * 100
                message += f"📏 <b>Distance:</b> {distance:.2f}%\n"
                
            elif update_type in ["tp_hit", "tp_approaching"]:
                message += f"🎯 <b>Take Profit:</b> ${trade.take_profit:.4f}\n"
                distance = abs(trade.take_profit - current_price) / current_price * 100
                message += f"📏 <b>Distance:</b> {distance:.2f}%\n"
                
            elif update_type == "trailing_updated":
                message += f"🔄 <b>New SL:</b> ${trade.stop_loss:.4f}\n"
                message += f"📈 <b>Highest:</b> ${trade.highest_price_seen:.4f}\n"
            
            # Holding time
            if trade.entry_time:
                holding_time = (datetime.now(IST_TIMEZONE) - trade.entry_time).total_seconds() / 3600
                message += f"⏰ <b>Duration:</b> {holding_time:.1f}h"
            
            await self.send_telegram(message)
            
        except Exception as e:
            logging.error(f"Failed to send position update notification: {e}")

# ===== ANA ÇALIŞTIRMA BLOĞU =====
if __name__ == "__main__":
    try:
        # ===== BURASI DÜZELTİLDİ =====
        trader = FRHunterV12(TELEGRAM_TOKEN, CHAT_ID)
        # ==========================
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    except Exception as e:
        logging.critical(f"Failed to start: {e}", exc_info=True)
        sys.exit(1)