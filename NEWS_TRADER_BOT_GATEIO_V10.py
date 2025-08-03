# -*- coding: utf-8 -*-
# NEWS_TRADER_BOT_GATEIO_V07.py
# Tüm hatalar, tekrarlar ve mantıksal çelişkiler giderilmiş son versiyon.

import asyncio
import random
import json
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import threading
import time
from collections import deque
import os
# import requests
import sqlite3
import pandas as pd
import numpy as np
import ta

from dotenv import load_dotenv
from textblob import TextBlob
import ccxt
try:
    from telethon import TelegramClient, events
    from telethon.errors import FloodWaitError # <<<--- BU SATIRI EKLEYİN
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False
load_dotenv()

# --- Logging Kurulumu ---
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO').upper())
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if not TELETHON_AVAILABLE:
    logger.warning("Telethon kütüphanesi kurulu değil, Telegram kanalları dinlenemeyecek.")


# --- DATA CLASSES ---

class TradingDatabase:
    def __init__(self, db_path="aenews_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Database tablolarını oluşturur."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Trades tablosu
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        size REAL NOT NULL,
                        pnl_usd REAL,
                        pnl_percent REAL,
                        open_timestamp TEXT NOT NULL,
                        close_timestamp TEXT,
                        close_reason TEXT,
                        exchange TEXT NOT NULL,
                        news_source TEXT,
                        news_title TEXT,
                        sentiment_score REAL,
                        impact_level TEXT,
                        volatility_atr REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        confidence REAL,
                        status TEXT DEFAULT 'OPEN',
                        user_login TEXT DEFAULT 'ahmetakyurek'
                    )
                ''')
                
                # News tablosu
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        content TEXT,
                        source TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        sentiment_score REAL,
                        impact_level TEXT,
                        coins_mentioned TEXT,
                        url TEXT,
                        processed BOOLEAN DEFAULT 1,
                        trades_opened INTEGER DEFAULT 0,
                        user_login TEXT DEFAULT 'ahmetakyurek'
                    )
                ''')
                
                # Performance metrics tablosu
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        starting_balance REAL,
                        ending_balance REAL,
                        daily_pnl REAL,
                        trades_count INTEGER,
                        wins INTEGER,
                        losses INTEGER,
                        win_rate REAL,
                        best_trade REAL,
                        worst_trade REAL,
                        user_login TEXT DEFAULT 'ahmetakyurek'
                    )
                ''')
                
                conn.commit()
                logger.info("✅ SQLite Database başarıyla başlatıldı: aenews_bot.db")
                
        except Exception as e:
            logger.error(f"❌ Database başlatma hatası: {e}")

    def save_trade(self, position, news_item=None):
        """Yeni trade'i database'e kaydeder."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades (
                        symbol, side, entry_price, size, open_timestamp,
                        exchange, news_source, news_title, sentiment_score,
                        impact_level, volatility_atr, stop_loss, take_profit,
                        confidence, status, user_login
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position.symbol, position.side, position.entry_price,
                    position.size, position.timestamp.isoformat(),
                    position.exchange,
                    news_item.source if news_item else None,
                    news_item.title if news_item else None,
                    news_item.sentiment_score if news_item else None,
                    news_item.impact_level if news_item else None,
                    position.volatility, position.stop_loss, position.take_profit,
                    getattr(position, 'confidence', 1.0), 'OPEN', 'ahmetakyurek'
                ))
                
                trade_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"✅ Trade DB'ye kaydedildi: ID {trade_id}, {position.symbol}")
                return trade_id
                
        except Exception as e:
            logger.error(f"❌ Trade kaydetme hatası: {e}")
            return None

    def close_trade(self, symbol, exit_price, close_reason):
        """Trade'i kapat ve database'i güncelle."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Açık trade'i bul
                cursor.execute('''
                    SELECT id, entry_price, size, side FROM trades 
                    WHERE symbol = ? AND status = 'OPEN' AND user_login = ?
                    ORDER BY open_timestamp DESC LIMIT 1
                ''', (symbol, 'ahmetakyurek'))
                
                result = cursor.fetchone()
                if not result:
                    logger.info(f"ℹ️ Kapatılacak açık trade bulunamadı: {symbol}")
                    return False
                
                trade_id, entry_price, size, side = result
                
                # PnL hesapla
                if side.upper() == 'BUY':
                    pnl_usd = (exit_price - entry_price) * size
                else:
                    pnl_usd = (entry_price - exit_price) * size
                
                pnl_percent = (pnl_usd / (entry_price * size)) * 100 if entry_price * size > 0 else 0
                
                # Trade'i güncelle
                cursor.execute('''
                    UPDATE trades SET 
                        exit_price = ?, pnl_usd = ?, pnl_percent = ?,
                        close_timestamp = ?, close_reason = ?, status = 'CLOSED'
                    WHERE id = ?
                ''', (
                    exit_price, pnl_usd, pnl_percent,
                    datetime.now().isoformat(), close_reason, trade_id
                ))
                
                conn.commit()
                logger.info(f"✅ Trade DB'de kapatıldı: ID {trade_id}, PnL: ${pnl_usd:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Trade kapatma hatası: {e}")
            return False

    def save_news(self, news, trades_opened=0):
        """Haber kaydını database'e ekler."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO news (
                        title, content, source, timestamp, sentiment_score,
                        impact_level, coins_mentioned, url, processed, trades_opened,
                        user_login
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    news.title, getattr(news, 'content', ''), news.source,
                    news.timestamp.isoformat(), news.sentiment_score,
                    news.impact_level, ','.join(news.coins_mentioned),
                    getattr(news, 'url', ''), True, trades_opened, 'ahmetakyurek'
                ))
                
                conn.commit()
                logger.debug(f"📰 Haber DB'ye kaydedildi: {news.title[:30]}...")
                
        except Exception as e:
            logger.error(f"❌ Haber kaydetme hatası: {e}")

    def get_performance_stats(self):
        """Database'den detaylı performans istatistikleri alır."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Temel istatistikler
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl_usd > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN pnl_usd <= 0 THEN 1 END) as losing_trades,
                        COALESCE(SUM(pnl_usd), 0) as total_pnl,
                        COALESCE(AVG(CASE WHEN pnl_usd > 0 THEN pnl_usd END), 0) as avg_win,
                        COALESCE(AVG(CASE WHEN pnl_usd <= 0 THEN pnl_usd END), 0) as avg_loss,
                        COALESCE(MAX(pnl_usd), 0) as best_trade,
                        COALESCE(MIN(pnl_usd), 0) as worst_trade,
                        MAX(close_timestamp) as last_trade_date
                    FROM trades 
                    WHERE status = 'CLOSED' AND user_login = ?
                ''', ('ahmetakyurek',))
                
                result = cursor.fetchone()
                if not result or result[0] == 0:
                    return {
                        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                        'total_pnl_usd': 0, 'avg_win_amount': 0, 'avg_loss_amount': 0,
                        'best_trade_pnl': 0, 'worst_trade_pnl': 0, 'win_rate_pct': 0,
                        'last_trade_date': 'N/A', 'trading_days': 0
                    }
                
                stats = {
                    'total_trades': result[0],
                    'winning_trades': result[1],
                    'losing_trades': result[2],
                    'total_pnl_usd': result[3],
                    'avg_win_amount': result[4],
                    'avg_loss_amount': result[5],
                    'best_trade_pnl': result[6],
                    'worst_trade_pnl': result[7],
                    'last_trade_date': result[8][:10] if result[8] else 'N/A'
                }
                
                # Win rate hesapla
                stats['win_rate_pct'] = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
                
                # Trading günleri hesapla
                cursor.execute('''
                    SELECT COUNT(DISTINCT DATE(close_timestamp)) 
                    FROM trades 
                    WHERE status = 'CLOSED' AND user_login = ?
                ''', ('ahmetakyurek',))
                
                trading_days = cursor.fetchone()[0] or 1
                stats['trading_days'] = trading_days
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Performans istatistikleri alınamadı: {e}")
            return {}

    def get_open_positions_count(self):
        """Açık pozisyon sayısını döndürür."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM trades 
                    WHERE status = 'OPEN' AND user_login = ?
                ''', ('ahmetakyurek',))
                return cursor.fetchone()[0] or 0
        except Exception as e:
            logger.error(f"❌ Açık pozisyon sayısı alınamadı: {e}")
            return 0

@dataclass
class NewsItem:
    title: str; content: str; source: str; timestamp: datetime; sentiment_score: float; coins_mentioned: List[str]; impact_level: str; url: str = ""

@dataclass
class TradeSignal:
    symbol: str
    action: str
    confidence: float
    expected_impact: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float = 0.0 # <<<--- EKSİK OLAN SATIRI BURAYA EKLİYORUZ
    volatility: float = 0.0

@dataclass
class Position:
    # Gerekli alanlar
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    timestamp: datetime
    exchange: str  # <<<--- Doğru yer burası
    
    # Opsiyonel alanlar
    volatility: float = 0.0
    trailing_stop: float = 0.0
    trailing_stop_activated: bool = False
    highest_price_seen: float = 0.0
    lowest_price_seen: float = float('inf')
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    created_timestamp: Optional[datetime] = None  # For grace period tracking
    opening_order_id: Optional[str] = None  # For order verification
    news_item: Optional[NewsItem] = None

# --- CONFIG MANAGER ---
class ConfigManager:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = {}
        self.load_config()

    def load_config(self):
        """Yapılandırma dosyasını yükler."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"✅ Yapılandırma dosyası başarıyla yüklendi: {self.config_path}")
        except FileNotFoundError:
            logger.critical(f"❌ KRİTİK HATA: Yapılandırma dosyası bulunamadı: {self.config_path}")
            raise
        except json.JSONDecodeError:
            logger.critical(f"❌ KRİTİK HATA: Yapılandırma dosyası hatalı formatta: {self.config_path}")
            raise
        except Exception as e:
            logger.critical(f"❌ KRİTİK HATA: Yapılandırma dosyası yüklenemedi: {e}")
            raise

    def get(self, key_path: str, default=None):
        """'trading_strategy.min_confidence' gibi bir yoldan değeri alır."""
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Yapılandırmada '{key_path}' bulunamadı, varsayılan değer ({default}) kullanılıyor.")
            return default

# --- CORE CLASSES ---

class TelegramNotifier:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.loop = bot_instance.loop # <<<--- YENİ: Ana botun event loop'unu alıyoruz
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.is_configured = bool(self.bot_token and self.chat_id)
        
        # aiohttp session'ını başlat
        self._session = aiohttp.ClientSession() # <<<--- YENİ: Asenkron session oluşturuyoruz

        if not self.is_configured:
            logger.warning("Telegram Bot Token veya Chat ID eksik. Bildirimler pasif.")
        else:
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    async def send_message(self, message: str, use_header: bool = True):
        """
        Telegram'a formatlanmış bir mesajı asenkron olarak gönderir.
        """
        if not self.is_configured: return
        
        header = "🤖 <b>AEnews Bot</b>\n━━━━━━━━━━━━━━━━━━━━━\n\n"
        full_message = (header + message) if use_header else message
        
        payload = {
            'chat_id': self.chat_id, 
            'text': full_message, 
            'parse_mode': 'HTML', 
            'disable_web_page_preview': True
        }

        try:
            # requests.post yerine aiohttp.ClientSession.post kullanıyoruz
            async with self._session.post(self.base_url, json=payload, timeout=10) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"Telegram'a mesaj gönderilemedi. Durum: {response.status}, Yanıt: {response_text}")
        except asyncio.TimeoutError:
             logger.error("Telegram'a istek gönderilirken zaman aşımı (timeout) hatası.")
        except aiohttp.ClientError as e:
            logger.error(f"Telegram'a istek gönderilirken aiohttp hatası: {e}")
        except Exception as e: 
            logger.error(f"Telegram'a istek gönderilirken beklenmedik hata: {e}")

    async def close_session(self):
        """Uygulama kapanırken aiohttp session'ını güvenle kapatır."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def notify_trade_opened(self, position: Position, signal: TradeSignal, news_title: str):
        """Hem canlı hem de kağıt işlem açılışlarını bildirir."""
        mode_text = "PAPER" if self.bot.paper_mode else "CANLI"
        exchange_name = position.exchange.upper()
        direction_emoji = "🟢📈" if position.side == "BUY" else "🔴📉"
        sl_pct = ((position.stop_loss / position.entry_price - 1) * 100) if position.entry_price > 0 else 0
        tp_pct = ((position.take_profit / position.entry_price - 1) * 100) if position.entry_price > 0 else 0
        
        message = (
            f"💰 <b>{mode_text} İŞLEM AÇILDI</b> {direction_emoji}\n\n"
            f"<b>Borsa:</b> {exchange_name}\n"
            f"<b>Coin:</b> {position.symbol}\n"
            f"<b>Yön:</b> {position.side}\n"
            f"<b>Giriş Fiyatı:</b> ${position.entry_price:,.4f}\n"
            f"<b>Miktar:</b> {position.size:,.4f}\n"
            f"<b>Stop Loss:</b> ${position.stop_loss:,.4f} ({sl_pct:+.2f}%)\n"
            f"<b>Take Profit:</b> ${position.take_profit:,.4f} ({tp_pct:+.2f}%)\n"
            f"<b>Güven Skoru:</b> {signal.confidence:.2f}\n\n"
            f"📰 <b>Tetikleyen Haber:</b> {news_title[:100]}..."
        )
        await self.send_message(message)

    async def notify_trade_closed(self, position: Position, reason: str):
        """Hem canlı hem de kağıt işlem kapanışlarını bildirir."""
        mode_text = "PAPER" if self.bot.paper_mode else "CANLI"
        pnl_emoji = "✅" if position.pnl >= 0 else "❌"
        try: 
            pnl_percent = (position.pnl / (position.entry_price * position.size)) * 100
        except ZeroDivisionError: 
            pnl_percent = 0
            
        message = (
            f"💰 {pnl_emoji} <b>{mode_text} İŞLEM KAPANDI</b>\n\n"
            f"<b>Coin:</b> {position.symbol}\n"
            f"<b>Giriş:</b> ${position.entry_price:,.4f}\n"
            f"<b>Çıkış:</b> ${position.current_price:,.4f}\n"
            f"<b>PnL:</b> ${position.pnl:,.2f} ({pnl_percent:+.2f}%)\n"
            f"<b>Sebep:</b> {reason.replace('_', ' ').title()}"
        )
        await self.send_message(message)

    async def notify_signal_rejected(self, news_title: str, coin: str, reason: str):
        """Bir sinyalin neden reddedildiğini bildirir."""
        reason_map = {
            "confidence_low": ("📉 Düşük Güven", "Sinyal güven skoru minimum eşiğin altında."),
            "no_coins": ("🚫 Coin Yok", "Haber metninde işlem yapılabilecek bir coin bulunamadı."),
            "price_error": ("💸 Fiyat Hatası", "Coin için güncel fiyat alınamadı."),
            "risk_manager_decline": ("🛡️ Risk Engeli", "Risk yönetimi yeni pozisyon açılmasına izin vermedi."),
        }
        emoji, reason_text = reason_map.get(reason, ("❌ Reddedildi", reason))
        message = (
            f"<b>{emoji}</b>\n\n"
            f"💰 <b>Coin:</b> {coin}\n"
            f"📰 <b>Haber:</b> {news_title[:80]}...\n"
            f"🚫 <b>Sebep:</b> {reason_text}"
        )
        await self.send_message(message)

    async def notify_raw_news(self, news: NewsItem):
        """İşleme girilip girilmediğine bakılmaksızın önemli haberleri bildirir."""
        impact_emoji = {"ULTRA_HIGH": "🔥🔥🔥", "HIGH": "🔥🔥", "MEDIUM": "🔥", "LOW": "⚡"}
        if news.sentiment_score > 0.1: sentiment_emoji = "🟢 LONG"
        elif news.sentiment_score < -0.1: sentiment_emoji = "🔴 SHORT"
        else: sentiment_emoji = "⚪️ NÖTR"
        coins_text = ", ".join(news.coins_mentioned)
        
        message = (
            f"🔔 <b>HABER BİLDİRİMİ</b> {impact_emoji.get(news.impact_level, '⚡')}\n\n"
            f"<b>Kaynak:</b> {news.source}\n"
            f"<b>Coin(ler):</b> {coins_text}\n"
            f"<b>Impact:</b> {news.impact_level}\n"
            f"<b>Sentiment:</b> {news.sentiment_score:.2f} {sentiment_emoji}\n\n"
            f"📰 <b>Başlık:</b> {news.title}"
        )
        if news.url: 
            message += f"\n\n<a href='{news.url}'>Haberin Kaynağına Git</a>"
        
        # Bu mesajda botun ana başlığını kullanmıyoruz.
        await self.send_message(message, use_header=False)

class CsvLogger:
    def __init__(self, filename="trade_history.csv"):
        self.filename = filename
        self.file_exists = os.path.exists(self.filename)
        self.lock = asyncio.Lock()  # Asenkron yazma işlemleri için kilit
        self._write_header_if_needed()

    def _write_header_if_needed(self):
        """Dosya yoksa başlık satırını yazar."""
        if not self.file_exists:
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                header = [
                    "close_timestamp_utc", "symbol", "side", "duration_minutes",
                    "entry_price", "exit_price", "size", "pnl_usd", "pnl_percent",
                    "close_reason", "triggering_news_source", "triggering_news_title",
                    "sentiment_score", "impact_level", "volatility_atr"
                ]
                f.write(",".join(header) + "\n")

    async def log_trade(self, position: Position, reason: str, news_item: Optional[NewsItem] = None):
        """Kapanan bir işlemi CSV dosyasına asenkron olarak yazar."""
        async with self.lock:
            try:
                now_utc = datetime.utcnow()
                duration = (now_utc - position.timestamp).total_seconds() / 60
                
                pnl_percent = (position.pnl / (position.entry_price * position.size)) * 100 if position.entry_price * position.size != 0 else 0

                # Safe handling of news_item attributes
                news_source = getattr(news_item, 'source', 'N/A') if news_item else 'N/A'
                news_title = getattr(news_item, 'title', 'N/A') if news_item else 'N/A'
                if news_title != 'N/A':
                    news_title = news_title.replace(",", ";")  # Replace commas for CSV
                sentiment = getattr(news_item, 'sentiment_score', 0.0) if news_item else 0.0
                impact = getattr(news_item, 'impact_level', 'N/A') if news_item else 'N/A'
                
                row = [
                    now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    position.symbol,
                    position.side,
                    f"{duration:.2f}",
                    f"{position.entry_price:.6f}",
                    f"{position.current_price:.6f}",
                    f"{position.size:.6f}",
                    f"{position.pnl:.4f}",
                    f"{pnl_percent:.4f}",
                    reason,
                    news_source,
                    news_title,
                    f"{sentiment:.4f}",
                    impact,
                    f"{position.volatility:.6f}"
                ]
                
                with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                    f.write(",".join(row) + "\n")
                
                logger.info(f"💾 İşlem geçmişe kaydedildi: {position.symbol}")

            except Exception as e:
                logger.error(f"CSV'ye işlem kaydı sırasında hata: {e}")
                # Fallback: log essential info only
                try:
                    self.log_error_to_csv(position, reason, str(e))
                except:
                    pass  # Silent fail for logging errors

    def log_error_to_csv(self, position: Position, reason: str, error: str):
        """Fallback logging method for when regular logging fails"""
        try:
            fallback_data = {
                'symbol': str(position.symbol) if hasattr(position, 'symbol') else 'UNKNOWN',
                'side': str(position.side) if hasattr(position, 'side') else 'UNKNOWN',
                'error': str(error),
                'reason': str(reason),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with open('trade_errors.csv', 'a', newline='', encoding='utf-8') as f:
                if not hasattr(self, '_error_header_written'):
                    f.write("timestamp,symbol,side,reason,error\n")
                    self._error_header_written = True
                f.write(f"{fallback_data['timestamp']},{fallback_data['symbol']},{fallback_data['side']},{fallback_data['reason']},{fallback_data['error']}\n")
        except:
            pass  # Silent fail for error logging

class PaperTradingEngine:
    def __init__(self, bot_instance, initial_balance: float = 10000):
        self.bot = bot_instance
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.start_time = datetime.now()

    def execute_trade(self, signal: TradeSignal, current_price: float, news_item: NewsItem) -> Optional[Position]:
        try:
            full_symbol = signal.symbol
            coin_symbol = full_symbol.split('/')[0]

            if any(p.symbol.split('/')[0] == coin_symbol for p in self.positions.values()):
                logger.info(f"[{coin_symbol}] için zaten açık bir pozisyon var. Yeni işlem atlanıyor.")
                return None
            
            # Pozisyon büyüklüğünü (miktarı) risk kurallarına göre hesapla
            risk_per_trade = self.current_balance * 0.02
            price_diff_per_unit = abs(signal.entry_price - signal.stop_loss)
            if price_diff_per_unit <= 0: return None
            
            position_size = risk_per_trade / price_diff_per_unit
            if (position_size * current_price) < 5.0: # Minimum 5 dolarlık pozisyon
                logger.warning(f"[{coin_symbol}] Hesaplanan pozisyon büyüklüğü çok küçük. İşlem atlanıyor.")
                return None
            
            current_time = datetime.utcnow()
            position = Position(
                symbol=full_symbol, exchange="paper", side=signal.action,
                size=position_size, entry_price=current_price, current_price=current_price,
                stop_loss=signal.stop_loss, take_profit=signal.take_profit, pnl=0.0,
                timestamp=current_time, news_item=news_item, volatility=signal.volatility,
                trailing_stop=signal.stop_loss, highest_price_seen=current_price, lowest_price_seen=current_price,
                created_timestamp=current_time, opening_order_id=None  # Paper trading doesn't have real orders
            )
            
            self.positions[coin_symbol] = position
            self.total_trades += 1
            
            logger.info(f"✅ PAPER TRADE AÇILDI: {signal.action} {position_size:.4f} {coin_symbol} @ {current_price:.4f}")
            return position
            
        except Exception as e:
            logger.error(f"Paper trade açma hatası: {e}", exc_info=True)
            return None

    def update_positions(self, current_prices: Dict[str, float]):
        """Pozisyonları günceller ve gerekirse kapatır."""
        positions_to_close = []
        
        for key, position in list(self.positions.items()):
            current_price = current_prices.get(position.symbol)
            if not current_price: 
                continue
                
            position.current_price = current_price
            
            if position.side == "BUY":
                position.pnl = (current_price - position.entry_price) * position.size
                position.highest_price_seen = max(position.highest_price_seen, current_price)
            else: # SELL
                position.pnl = (position.entry_price - current_price) * position.size
                position.lowest_price_seen = min(position.lowest_price_seen, current_price)
            
            # Paper mod için basit SL/TP kontrolü
            should_close, reason = self._check_close_conditions(position)
            if should_close:
                positions_to_close.append((key, position, reason))
        
        # ✅ KRITIK: Kapatılacak pozisyonları gerçekten kapat
        for key, pos, reason in positions_to_close:
            self.close_position(key, pos, reason)
        
        return positions_to_close  # İsteğe bağlı: logging için döndür

    def _check_close_conditions(self, position: Position) -> Tuple[bool, str]:
        price = position.current_price
        if position.side == "BUY":
            if price >= position.take_profit: return True, "TAKE_PROFIT"
            if price <= position.stop_loss: return True, "STOP_LOSS"
        else: # SELL
            if price <= position.take_profit: return True, "TAKE_PROFIT"
            if price >= position.stop_loss: return True, "STOP_LOSS"
        return False, ""

    def close_position(self, position_key: str, position: Position, reason: str):
        self.current_balance += position.pnl
        self.total_pnl += position.pnl
        self.daily_pnl += position.pnl
        if position.pnl > 0: self.winning_trades += 1
        
        if self.current_balance > self.peak_balance: self.peak_balance = self.current_balance
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        if position_key in self.positions: del self.positions[position_key]
        
        logger.info(f"PAPER TRADE KAPANDI: {position.symbol} PnL: ${position.pnl:.2f} - Sebep: {reason}")
        
        if hasattr(self.bot, 'csv_logger'):
            future = asyncio.run_coroutine_threadsafe(
                self.bot.csv_logger.log_trade(position, reason, position.news_item),
                self.bot.loop
            )
            try: future.result(timeout=5)
            except Exception as e: logger.error(f"CSV loglama hatası (thread-safe): {e}")

    def get_performance_stats(self) -> Dict:
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        return {
            'initial_balance': self.initial_balance, 'current_balance': self.current_balance,
            'total_pnl': self.total_pnl, 'total_return_pct': total_return,
            'total_trades': self.total_trades, 'winning_trades': self.winning_trades,
            'win_rate_pct': win_rate, 'max_drawdown_pct': self.max_drawdown * 100,
            'open_positions': len(self.positions),
            'days_running': max((datetime.now() - self.start_time).days, 1),
            'daily_pnl': self.daily_pnl
        }

# --- NewsAnalyzer SINIFININ TAMAMI ---

class NewsAnalyzer:
    def __init__(self):
        # En son konuştuğumuz, en kapsamlı ve doğru pattern listeleri
        self.ultra_positive_patterns = {
            'mainnet launch': 1.0, 'sec approves': 1.0, 'etf approved': 1.0, 
            'coinbase listing': 1.0, 'binance listing': 1.0, 'binance futures will launch': 1.0,
            'institutional adoption': 1.0, 'acquires': 0.8, 'acquisition': 0.8, 
            'stake in': 0.8, 'billion stake': 0.9, 'million stake': 0.85,
            'binance alpha': 1.0, 'available on binance alpha': 1.0, 'blackrock': 1.0
        }
        self.ultra_negative_patterns = {
            'delisting': -1.0, 'hacked': -1.0, 'sec investigation': -1.0, 
            'sec lawsuit': -1.0, 'rug pull': -1.0, 'monitoring tag': -1.0
        }
        self.high_positive_patterns = {
            'partnership': 0.8, 'integration': 0.8, 'adoption': 0.8, 
            'listing': 0.8, 'funding round': 0.8, 'rally': 0.8, 'token burn': 0.8, 
            'upgrade': 0.8, 'launch': 0.8
        }
        self.high_negative_patterns = {
            'exploit': -0.8, 'lawsuit': -0.8, 'fine': -0.8, 'bearish': -0.8, 
            'panic sell': -0.8, 'network halt': -0.8
        }
        self.medium_positive_patterns = {
            'positive': 0.5, 'bullish': 0.5, 'innovative': 0.5, 
            'airdrop': 0.5, 'staking': 0.5, 'testnet': 0.5
        }
        self.medium_negative_patterns = {
            'concern': -0.5, 'risk': -0.5, 'warning': -0.5, 'decline': -0.5, 
            'fud': -0.5, 'delay': -0.7, 'postponed': -0.8, 'suspended': -0.8, 
            'halted': -0.8, 'bug': -0.5, 'issue': -0.5
        }
        self.intensity_modifiers = {'major': 1.3, 'massive': 1.3, 'significant': 1.2, 'minor': 0.7, 'slight': 0.5}
        self.negation_words = ['not', 'no', 'never', 'without', 'failed to', 'unable to']
        
    def analyze_sentiment(self, text: str) -> float:
        """Haber metninin duygu skorunu, kelime listelerine göre ağırlıklı olarak hesaplar."""
        text_lower = text.lower()
        total_sentiment, pattern_matches = 0.0, 0
        all_patterns = [
            self.ultra_positive_patterns, self.ultra_negative_patterns, 
            self.high_positive_patterns, self.high_negative_patterns, 
            self.medium_positive_patterns, self.medium_negative_patterns
        ]
        
        for pattern_dict in all_patterns:
            for pattern, weight in pattern_dict.items():
                if pattern in text_lower:
                    is_negated = any(neg in text_lower[:text_lower.find(pattern)][-20:] for neg in self.negation_words)
                    if is_negated: weight *= -0.5
                    
                    intensity = 1.0
                    for modifier, multiplier in self.intensity_modifiers.items():
                        if modifier in text_lower: 
                            intensity = multiplier
                            break
                    
                    total_sentiment += weight * intensity
                    pattern_matches += 1
        
        if pattern_matches == 0: 
            return TextBlob(text).sentiment.polarity * 0.3
        else: 
            return max(-1.0, min(1.0, total_sentiment / pattern_matches))

    def calculate_impact_level(self, sentiment: float, coins: List[str], content: str) -> str:
        """Haberin etki seviyesini (ULTRA_HIGH, HIGH, vb.) belirler."""
        impact_score = abs(sentiment)
        content_lower = content.lower()
        
        if any(p in content_lower for p in ['delisting', 'hacked', 'sec approves', 'etf approved', 'listing', 'futures will launch']): 
            impact_score += 0.5
        elif any(p in content_lower for p in ['partnership', 'lawsuit', 'exploit', 'funding']): 
            impact_score += 0.3
            
        if len(coins) > 1: 
            impact_score += 0.1
            
        if any(e in content_lower for e in ['binance', 'coinbase', 'sec', 'blackrock']): 
            impact_score += 0.2
            
        if impact_score >= 1.0: return "ULTRA_HIGH"
        if impact_score >= 0.7: return "HIGH"
        if impact_score >= 0.4: return "MEDIUM"
        return "LOW"

    @staticmethod
    def calculate_atr(klines: List) -> Optional[float]:
        """Verilen klines verisinden son ATR değerini (fiyat cinsinden) hesaplar."""
        # --- YENİ GÜVENLİK KONTROLÜ ---
        if not klines: # klines None ise veya boş bir liste ise
            return None
        # -----------------------------
            
        try:
            if len(klines) < 20: return None
            
            df = pd.DataFrame(klines).iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            for col in ['high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Eksik veri varsa None dön
            if df.isnull().values.any():
                logger.warning("ATR hesaplama sırasında klines verisinde eksik değerler bulundu.")
                return None

            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            if atr.dropna().empty: return None # <<<--- ATR hesaplanamazsa None dön

            last_atr_value = atr.dropna().iloc[-1]
            
            # ARTIK YÜZDE HESABI YOK, DOĞRUDAN ATR DEĞERİNİ DÖNDÜRÜYORUZ
            return float(last_atr_value)

        except Exception as e:
            logger.error(f"ATR hesaplama hatası: {e}", exc_info=True)
            return None # <<<--- Hata durumunda None dön

    @staticmethod
    def detect_regime(klines: List) -> str:
        """Verilen klines verisine göre piyasa rejimini belirler."""
        try:
            if len(klines) < 50: return "Unknown"
            
            df = pd.DataFrame(klines).iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            close_prices = pd.to_numeric(df['close'], errors='coerce').dropna()
            if len(close_prices) < 50: return "Unknown"
            
            sma50 = ta.trend.sma_indicator(close_prices, window=50)
            if sma50.dropna().empty: return "Unknown"
            
            last_price = close_prices.iloc[-1]
            last_sma50 = sma50.dropna().iloc[-1]
            
            if last_price > last_sma50 * 1.015:
                return "Uptrend"
            elif last_price < last_sma50 * 0.985:
                return "Downtrend"
            else:
                return "Ranging"
        except Exception as e:
            logger.warning(f"Piyasa rejimi belirlenirken hata oluştu: {e}")
            return "Unknown"

class MarketManager:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.coin_list = set()
        self.aliases = {}
        self.last_update = None
        self.update_interval = timedelta(days=7).total_seconds()
        self.cache_file = f"{self.exchange.id}_coins.json" if self.exchange else "market_coins.json"
        self.load_aliases()
        self.load_coin_list()
    
    def load_aliases(self):
        try:
            alias_path = 'coin_aliases.json'
            if os.path.exists(alias_path):
                with open(alias_path, 'r') as f: self.aliases = json.load(f)
                logger.info(f"{len(self.aliases)} adet coin takma adı (alias) yüklendi.")
            else:
                logger.warning(f"'{alias_path}' dosyası bulunamadı. Alias sistemi devre dışı.")
        except Exception as e:
            logger.error(f"Coin alias dosyası okunurken hata oluştu: {e}")

    def load_coin_list(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f: cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data.get('last_update', '2000-01-01'))
                if (datetime.now() - cache_time).total_seconds() < self.update_interval:
                    self.coin_list = set(cache_data.get('coins', []))
                    self.last_update = cache_time
                    logger.info(f"{self.exchange.id.upper()} coin listesi cache'den yüklendi: {len(self.coin_list)} coin")
                    return
            self.update_from_exchange()
        except Exception as e:
            logger.error(f"Coin listesi yükleme hatası: {e}", exc_info=True)
            self.coin_list = {'BTC', 'ETH', 'SOL'}

    async def update_from_exchange(self): # <<<--- 'async' EKLENDİ
        """Gate.io'dan perpetual futures coin listesini günceller."""
        try:
            if not self.exchange: 
                return
            
            logger.info(f"🔄 {self.exchange.id.upper()}'dan coin listesi güncelleniyor...")
            
            # load_markets senkron bir işlem olduğu için, onu asenkron bir programda
            # güvenli bir şekilde çalıştırmak için run_in_executor kullanmalıyız.
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.exchange.load_markets, 
                True # Force reload
            )
            
            markets = self.exchange.markets
            coins = set()
            
            for symbol, market in markets.items():
                if (market.get('swap', False) and 
                    market.get('settle') == 'USDT' and 
                    market.get('active', True)):
                    
                    base_currency = market.get('base')
                    if base_currency:
                        coins.add(base_currency)

            self.coin_list = coins
            logger.info(f"✅ {self.exchange.id.upper()}'dan {len(coins)} adet perpetual futures coini güncellendi.")
            
            self.save_to_cache()
            
        except Exception as e:
            logger.error(f"{self.exchange.id.upper()} coin listesi güncelleme hatası: {e}", exc_info=True)

    def save_to_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({'coins': list(self.coin_list), 'last_update': datetime.now().isoformat()}, f, indent=2)
            logger.info(f"Coin listesi cache'e kaydedildi: {self.cache_file}")
        except Exception as e: logger.error(f"Cache kaydetme hatası: {e}")

    def is_valid_coin(self, coin: str) -> bool:
        coin_upper = coin.upper()
        if coin_upper in self.coin_list: return True
        if coin_upper in self.aliases and self.aliases[coin_upper] in self.coin_list: return True
        return False

    def resolve_alias(self, coin: str) -> str:
        coin_upper = coin.upper()
        return self.aliases.get(coin_upper, coin_upper)

class TelegramChannelCollector:
    def __init__(self, bot_instance, config: ConfigManager):
        self.bot = bot_instance
        self.analyzer = bot_instance.analyzer
        self.config = config
        
        self.api_id = os.getenv('TELEGRAM_API_ID')
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.phone = os.getenv('TELEGRAM_PHONE')
        
        self.channel_configs = self.config.get('telegram_channels.channel_configs', {})
        self.channels_to_listen = list(self.channel_configs.keys()) # <<< YENİ: Dinlenecek kanalları baştan belirle
        
        if not self.channels_to_listen:
            logger.warning("config.json dosyasında dinlenecek kanal bulunamadı.")
        
        logger.info(f"Telegram dinleyici {len(self.channels_to_listen)} kanal için yapılandırıldı: {self.channels_to_listen}")

        self.client = None
        self.is_running = False
 
    async def start(self):
        """Telethon client'ını başlatır ve kanalları dinler (FloodWait ve Test Kanalı Korumalı)."""
        if not TELETHON_AVAILABLE or not all([self.api_id, self.api_hash]):
            logger.warning("Telethon credentials eksik, Telegram dinleyicisi başlatılmıyor.")
            return

        try:
            self.client = TelegramClient('aenews_session', self.api_id, self.api_hash)
            logger.info("Telethon client oluşturuldu. Oturum açılıyor...")
            await self.client.start(phone=self.phone)
            logger.info("Oturum başarıyla açıldı.")

            # --- Adım 1: Sağlam ve Akıllı Kanal Doğrulama ---
            logger.info("Kanallar akıllı bekleme modu ile doğrulanıyor...")
            valid_channels_for_listener = []
            channels_to_check = list(self.channels_to_listen)
            
            while channels_to_check:
                channel = channels_to_check.pop(0)
                try:
                    await asyncio.sleep(5)
                    entity = await self.client.get_entity(channel)
                    valid_channels_for_listener.append(entity)
                    logger.info(f"✅ Kanal doğrulandı: @{channel}")

                except FloodWaitError as e:
                    wait_time = e.seconds
                    logger.warning(f"⏳ FloodWait hatası! Telegram {wait_time} saniye beklememizi istiyor. Bekleniyor...")
                    channels_to_check.append(channel) # Başarısız kanalı sona ekle
                    await asyncio.sleep(wait_time + 15) # Güvenlik payı ile bekle
                    
                except Exception as e:
                    logger.error(f"❌ Kanal doğrulanırken bilinmeyen hata: @{channel}, Hata: {e}")

            if not valid_channels_for_listener:
                logger.critical("Hiçbir geçerli kanal bulunamadı! Telegram dinleyicisi başlatılamıyor.")
                return

            logger.info(f"Toplam {len(valid_channels_for_listener)} kanal dinlemeye alınıyor.")

            # --- Adım 2: Gelen Mesajları İşleyecek Fonksiyonu Tanımlama ---
            @self.client.on(events.NewMessage(chats=valid_channels_for_listener))
            async def message_handler(event):
                try:
                    message_text = getattr(event.message, 'message', '')
                    if not message_text:
                        return
                    
                    # Kullanıcı adını al ve kontrol et
                    entity = await event.get_chat()
                    username = getattr(entity, 'username', None)
                    if not username:
                        return

                    # Manuel test kanalından gelen giden mesajları kabul et
                    if event.out:
                        channel_config = self.channel_configs.get(username, {})
                        if channel_config.get('type') != 'MANUAL_TEST':
                            return
                        logger.info(f"✅ Manuel test kanalından giden mesaj kabul edildi: @{username}")
                    
                    # Komutları işle
                    if message_text.startswith('/'):
                        channel_config = self.channel_configs.get(username, {})
                        if channel_config.get('type') == 'MAIN_BOT_CHANNEL':
                             await self.bot.command_handler.execute_command(message_text.strip())
                        return
                    
                    # Haber mesajlarını işle
                    await self.process_message(event, username)

                except Exception as e:
                    logger.error(f"Mesaj handler içinde kritik hata: {e}", exc_info=True)
            
            # --- Adım 3: Botu Aktif Olarak Dinleme Moduna Geçirme ---
            logger.info("Telethon dinleyici mesaj bekliyor...")
            self.is_running = True
            await self.client.run_until_disconnected()

        except FloodWaitError as e:
            # Bu, client.start() sırasında olabilecek nadir bir durumdur.
            wait_time = e.seconds
            logger.critical(f"Kritik başlangıç FloodWait hatası! Program {wait_time} saniye sonra tekrar denenebilir. Hata: {e}")
            await self.bot.telegram.send_message(f"🚨 <b>Telegram FloodWait Hatası!</b>\n\nBot, Telegram tarafından {wait_time} saniye boyunca engellendi.")
        except Exception as e:
            logger.critical(f"Telethon client başlatma hatası: {e}", exc_info=True)

    def detect_special_message_type(self, text: str) -> dict:
        """Özel mesaj tiplerini tanır (listing, delisting, vb.)"""
        text_lower = text.lower()
        
        # BWE Coinbase Listing Pattern
        if 'coinbase listing' in text_lower or 'coinbase lists' in text_lower:
            return {
                'type': 'COINBASE_LISTING',
                'sentiment': 0.9,
                'confidence_boost': 0.3
            }
        
        # BWE özel haberler
        if any(keyword in text_lower for keyword in ['allocate', 'partnership', 'acquires', 'acquisition']):
            return {
                'type': 'BWE_MAJOR_NEWS',
                'sentiment': 0.7,
                'confidence_boost': 0.2
            }
        
        # Binance Futures Launch Pattern
        if 'binance futures will launch' in text_lower or 'futures will launch' in text_lower:
            return {
                'type': 'BINANCE_FUTURES_LAUNCH',
                'sentiment': 1.0,
                'confidence_boost': 0.3
            }
        
        # Binance Listing Pattern
        if 'binance' in text_lower and ('listing' in text_lower or 'listed' in text_lower):
            return {
                'type': 'BINANCE_LISTING', 
                'sentiment': 0.9,
                'confidence_boost': 0.2
            }
        
        # Delisting Pattern
        if 'delist' in text_lower or 'delisting' in text_lower:
            return {
                'type': 'DELISTING',
                'sentiment': -1.0,
                'confidence_boost': 0.4
            }
        
        return {'type': 'GENERAL', 'sentiment': 0.0, 'confidence_boost': 0.0}

    async def process_message(self, event, username: str):
        try:
            channel_info = self.channel_configs[username]
            message_text = event.message.message
            if not message_text: return
            
            logger.info(f"✅ Telegram Mesajı Alındı [{username}]: {message_text[:80]}...")
            
            # ✅ ACİL DÜZELTME: BOT BİLDİRİMLERİNİ YOKSAY
            if any(keyword in message_text for keyword in [
                "🚀 AEnews Trading vFINAL",
                "💰 CANLI İŞLEM", 
                "💰 ❌ CANLI İŞLE",
                "💰 ✅ CANLI İŞLE", 
                "🚫 Blacklist",
                "🛡️ İşlem Redde",
                "🎯 TAKE PROFIT",
                "📈", "📉", 
                "━━━━━━━━━━━━━━━━━━━━━",
                "🔔 AE HABER BİLDİRİMİ",
                "⚠️ Duplicate Pozisyon"
            ]):
                logger.info(f"🔄 Bot bildirimi atlandı: {message_text[:30]}...")
                return
            
            # BWEnews için özel işleme
            if username == 'BWEnews':
                # BWE formatını tanı: "⚠️BWENEWS:" veya "BWENEWS:" ile başlayan mesajlar
                if '⚠️BWENEWS:' in message_text or 'BWENEWS:' in message_text:
                    # BWE mesajını temizle
                    clean_message = message_text.replace('⚠️BWENEWS:', '').replace('BWENEWS:', '').strip()
                    # Çince kısmı varsa sadece İngilizce kısmını al
                    if '⚠️方程式新闻:' in message_text:
                        clean_message = clean_message.split('⚠️方程式新闻:')[0].strip()
                    message_text = clean_message
                    logger.info(f"BWE mesajı temizlendi: {message_text[:80]}...")
            
            # Özel mesaj tipi tespiti
            special_msg_info = self.detect_special_message_type(message_text)
            
            ham_adaylar = self.extract_potential_coins(message_text)
            if not ham_adaylar:
                logger.warning("Mesajda coin adayı bulunamadı.")
                return

            # Sentiment hesaplama - özel mesaj tipi varsa öncelik ver
            if special_msg_info['type'] != 'GENERAL':
                sentiment = special_msg_info['sentiment']
                logger.info(f"Özel mesaj tipi tespit edildi: {special_msg_info['type']}, Sentiment: {sentiment}")
            else:
                sentiment = channel_info.get('sentiment', 0.0)
                if sentiment == 0.0:
                    sentiment = self.analyzer.analyze_sentiment(message_text)

            # Impact hesaplama
            impact = self.analyzer.calculate_impact_level(sentiment, ham_adaylar, message_text)
            
            # Özel mesaj tipi için impact boost
            if special_msg_info['type'] == 'BINANCE_FUTURES_LAUNCH' and impact == 'LOW':
                impact = 'HIGH'
                logger.info(f"Binance Futures Launch tespit edildi, impact LOW'dan HIGH'a yükseltildi")

            news_item = NewsItem(
                title=f"[{channel_info['type']}] {message_text[:100]}", 
                content=message_text,
                source=f"Telegram/{username}", 
                timestamp=datetime.now(),
                sentiment_score=sentiment, 
                coins_mentioned=ham_adaylar,
                impact_level=impact, 
                url=f"https://t.me/{username}/{event.message.id}"
            )
            
            await self.bot.process_news(news_item)
            
        except Exception as e:
            logger.error(f"Telegram mesajı işlenirken hata oluştu ({username}): {e}", exc_info=True)

    def extract_potential_coins(self, text: str) -> List[str]:
        """
        Geliştirilmiş coin extraction - öncelik sırasına göre
        """
        text_upper = text.upper()
        potential_coins = []
        
        # ✅ ÖNCELİK 1: $SYMBOL formatı (en güvenilir)
        dollar_coins = re.findall(r'\$([A-Z0-9]{2,10})', text_upper)
        for coin in dollar_coins:
            if coin not in potential_coins:
                potential_coins.append(coin)
        
        # ✅ ÖNCELİK 2: COIN/USDT formatı
        usdt_pairs = re.findall(r'\b([A-Z0-9]{2,8})/USDT?\b', text_upper)
        for coin in usdt_pairs:
            if coin not in potential_coins:
                potential_coins.append(coin)
        
        # ✅ ÖNCELİK 3: #HASHTAG formatı
        hashtag_coins = re.findall(r'#([A-Z0-9]{2,8})\b', text_upper)
        for coin in hashtag_coins:
            if coin not in potential_coins:
                potential_coins.append(coin)
        
        # ✅ ÖNCELİK 4: Parantez içi coinler - SADECE 2-4 HARF
        parenthesis_matches = re.findall(r'\(([^)]+)\)', text_upper)
        for match in parenthesis_matches:
            parts = re.split(r'[,\s/&]+', match)
            for part in parts:
                p = part.strip()
                if 2 <= len(p) <= 4 and p.isalpha() and p not in potential_coins:  # Sadece harf, 2-4 karakter
                    potential_coins.append(p)
        
        # ✅ ÖNCELİK 5: Büyük harfli kelimeler - ÇOK KISITLI
        # SADECE $ ile başlayan cümlelerde veya çok spesifik durumlarda
        if '$' in text_upper:  # $ varsa büyük harflileri de kontrol et
            word_candidates = re.findall(r'\b([A-Z]{3,6})\b', text_upper)
            
            # ÇOOK GENİŞLETİLMİŞ EXCLUDEd_WORDS LİSTESİ
            excluded_words = {
                # Temel İngilizce kelimeler
                'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'FROM', 'WILL', 
                'HAS', 'NEW', 'MORE', 'HERO', 'INTO', 'ADDED', 'PROJECTS', 'THIS', 'THAT',
                'WITH', 'THEY', 'HAVE', 'BEEN', 'THERE', 'THEIR', 'WHAT', 'WERE', 'SAID',
                'EACH', 'WHICH', 'WOULD', 'THESE', 'FIRST', 'AFTER', 'BACK', 'OTHER',
                'MANY', 'THAN', 'THEN', 'THEM', 'WELL', 'WERE', 'BEEN', 'HAVE', 'THEIR',
                'SAID', 'EACH', 'WHICH', 'WOULD', 'THERE', 'THEM', 'BEEN', 'MANY', 'WHO',
                'ITS', 'NOW', 'FIND', 'LONG', 'DOWN', 'DAY', 'GET', 'HAS', 'HIS', 'HER',
                'YOUR', 'MAY', 'VERY', 'THROUGH', 'JUST', 'FORM', 'MUCH', 'GREAT', 'THINK',
                'SAY', 'HELP', 'LOW', 'LINE', 'TURN', 'CAUSE', 'SAME', 'MEAN', 'DIFFER',
                'MOVE', 'RIGHT', 'BOY', 'OLD', 'TOO', 'DOES', 'TELL', 'SENTENCE', 'SET',
                'THREE', 'WANT', 'AIR', 'WELL', 'ALSO', 'PLAY', 'SMALL', 'END', 'PUT',
                'HOME', 'READ', 'HAND', 'PORT', 'LARGE', 'SPELL', 'ADD', 'EVEN', 'LAND',
                'HERE', 'MUST', 'BIG', 'HIGH', 'SUCH', 'FOLLOW', 'ACT', 'WHY', 'ASK',
                'MEN', 'CHANGE', 'WENT', 'LIGHT', 'KIND', 'OFF', 'NEED', 'HOUSE', 'PICTURE',
                'TRY', 'AGAIN', 'ANIMAL', 'POINT', 'MOTHER', 'WORLD', 'NEAR', 'BUILD',
                'SELF', 'EARTH', 'FATHER', 'HEAD', 'STAND', 'OWN', 'PAGE', 'SHOULD',
                'COUNTRY', 'FOUND', 'ANSWER', 'SCHOOL', 'GROW', 'STUDY', 'STILL', 'LEARN',
                'PLANT', 'COVER', 'FOOD', 'SUN', 'FOUR', 'BETWEEN', 'STATE', 'KEEP',
                'EYE', 'NEVER', 'LAST', 'LET', 'THOUGHT', 'CITY', 'TREE', 'CROSS',
                'FARM', 'HARD', 'START', 'MIGHT', 'STORY', 'SAW', 'FAR', 'SEA', 'DRAW',
                'LEFT', 'LATE', 'RUN', 'WHILE', 'PRESS', 'CLOSE', 'NIGHT', 'REAL',
                'LIFE', 'FEW', 'NORTH', 'OPEN', 'SEEM', 'TOGETHER', 'NEXT', 'WHITE',
                'CHILDREN', 'BEGIN', 'GOT', 'WALK', 'EXAMPLE', 'EASE', 'PAPER', 'GROUP',
                'ALWAYS', 'MUSIC', 'THOSE', 'BOTH', 'MARK', 'OFTEN', 'LETTER', 'UNTIL',
                'MILE', 'RIVER', 'CAR', 'FEET', 'CARE', 'SECOND', 'BOOK', 'CARRY',
                'TOOK', 'SCIENCE', 'EAT', 'ROOM', 'FRIEND', 'BEGAN', 'IDEA', 'FISH',
                'MOUNTAIN', 'STOP', 'ONCE', 'BASE', 'HEAR', 'HORSE', 'CUT', 'SURE',
                'WATCH', 'COLOR', 'FACE', 'WOOD', 'MAIN', 'ENOUGH', 'PLAIN', 'GIRL',
                'USUAL', 'YOUNG', 'READY', 'ABOVE', 'EVER', 'RED', 'LIST', 'THOUGH',
                'FEEL', 'TALK', 'BIRD', 'SOON', 'BODY', 'DOG', 'FAMILY', 'DIRECT',
                'LEAVE', 'SONG', 'MEASURE', 'DOOR', 'PRODUCT', 'BLACK', 'SHORT', 'NUMERAL',
                'CLASS', 'WIND', 'QUESTION', 'HAPPEN', 'COMPLETE', 'SHIP', 'AREA', 'HALF',
                'ROCK', 'ORDER', 'FIRE', 'SOUTH', 'PROBLEM', 'PIECE', 'TOLD', 'KNEW',
                'PASS', 'SINCE', 'TOP', 'WHOLE', 'KING', 'SPACE', 'HEARD', 'BEST',
                'HOUR', 'BETTER', 'DURING', 'HUNDRED', 'FIVE', 'REMEMBER', 'STEP',
                'EARLY', 'HOLD', 'WEST', 'GROUND', 'INTEREST', 'REACH', 'FAST', 'VERB',
                'SING', 'LISTEN', 'SIX', 'TABLE', 'TRAVEL', 'LESS', 'MORNING', 'TEN',
                'SIMPLE', 'SEVERAL', 'VOWEL', 'TOWARD', 'WAR', 'LAY', 'AGAINST', 'PATTERN',
                'SLOW', 'CENTER', 'PERSON', 'MONEY', 'SERVE', 'APPEAR', 'ROAD', 'MAP',
                'RAIN', 'RULE', 'GOVERN', 'PULL', 'COLD', 'NOTICE', 'VOICE', 'UNIT',
                'POWER', 'TOWN', 'FINE', 'CERTAIN', 'FLY', 'FALL', 'LEAD', 'CRY',
                'DARK', 'MACHINE', 'NOTE', 'WAIT', 'PLAN', 'FIGURE', 'STAR', 'BOX',
                'NOUN', 'FIELD', 'REST', 'CORRECT', 'ABLE', 'POUND', 'DONE', 'BEAUTY',
                'DRIVE', 'STOOD', 'CONTAIN', 'FRONT', 'TEACH', 'WEEK', 'FINAL', 'GAVE',
                'GREEN', 'QUICK', 'DEVELOP', 'OCEAN', 'WARM', 'FREE', 'MINUTE', 'STRONG',
                'SPECIAL', 'MIND', 'BEHIND', 'CLEAR', 'TAIL', 'PRODUCE', 'FACT', 'STREET',
                'INCH', 'MULTIPLY', 'NOTHING', 'COURSE', 'STAY', 'WHEEL', 'FULL', 'FORCE',
                'BLUE', 'OBJECT', 'DECIDE', 'SURFACE', 'DEEP', 'MOON', 'ISLAND', 'FOOT',
                'SYSTEM', 'BUSY', 'TEST', 'RECORD', 'BOAT', 'COMMON', 'GOLD', 'POSSIBLE',
                'PLANE', 'STEAD', 'DRY', 'WONDER', 'LAUGH', 'THOUSANDS', 'AGO', 'RAN',
                'CHECK', 'GAME', 'SHAPE', 'EQUATE', 'MISS', 'BROUGHT', 'HEAT', 'SNOW',
                'TIRE', 'BRING', 'YES', 'DISTANT', 'FILL', 'EAST', 'PAINT', 'LANGUAGE',
                'AMONG', 'UNIT', 'POWER', 'TOWN', 'FINE', 'CERTAIN', 'FLY', 'FALL',
                'LEAD', 'CRY', 'DARK', 'MACHINE', 'NOTE', 'WAIT', 'PLAN', 'FIGURE',
                'STAR', 'BOX', 'NOUN', 'FIELD', 'REST', 'CORRECT', 'ABLE', 'POUND',

                # Türkçe kelimeler
                'VE', 'BU', 'DA', 'DE', 'İLE', 'BİR', 'İÇİN', 'VAR', 'OLAN', 'OLARAK',
                'KADAR', 'DAHA', 'SONRA', 'ÖNCE', 'ANCAK', 'BILE', 'HALA', 'ARTIK',
                'BUNA', 'ŞUNA', 'KENDI', 'KERE', 'ZAMAN', 'YER', 'GİBİ', 'BÖYLE',
                'ŞÖYLE', 'NASIL', 'NEDEN', 'NIYE', 'NIÇIN', 'HANGISI', 'HANGİ',
                'KİM', 'KIMSE', 'HIÇBIR', 'HIÇKIMSE', 'HERKES', 'HERHANGI', 'BAŞKA',
                'BAŞKASI', 'OTOMATIK', 'EŞLEŞTIRME', 'YANLIŞ', 'OLABILIR', 'KAYNAK',
                'TÜRKÇE', 'İNGILIZCE', 'MARJINLI', 'SÜREKLİ', 'SÖZLEŞME', 'BAŞLATACAK',
                'SÜREKLI', 'SÖZLEŞMELERINI', 'BAŞLATACAK', 'MARKETCAP', 'MILLION',
                'OLMADĞINA', 'KARA', 'VERİYOR', 'NORMAL', 'NASIL', 'YAPIYOR',
                'YAPMAK', 'YAPMAYA', 'YAPMALI', 'YAPMAMALI', 'YAPMAYA', 'YAPMAMAK',
                'YAPMAKLA', 'YAPMAKTAN', 'YAPMAKLA', 'YAPMAKTAN', 'YAPMAKTA',
                'YAPMAKTAYIM', 'YAPMAKTAYIZ', 'YAPMAKTALAR', 'YAPMAKTADIR',
                'YAPMAKTADIRIM', 'YAPMAKTADIRLAR', 'YAPMAKTADIRIZ', 'YAPMAKTADIR',

                # Kripto ve Borsa Terimleri
                'SPOT', 'USD', 'USDT', 'USDS', 'BSC', 'ETH', 'BTC', 'CEO', 'API', 'DAO', 'IDO',
                'BYBIT', 'BINANCE', 'KUCOIN', 'GATE', 'OKX', 'GEMINI', 'HUOBI', 'MEXC',
                'TRADING', 'EXCHANGE', 'MARKET', 'WALLET', 'CHAIN', 'BRIDGE', 'SWAP',
                'DELIST', 'LISTED', 'LISTING', 'LIST', 'PAIRS', 'PAIR', 'FUTURES', 'MARGIN', 'LEVERAGE', 
                'DEPOSIT', 'WITHDRAW', 'NETWORK', 'UPGRADE', 'HARD', 'FORK', 'AIRDROP', 'STAKE', 'SQUARE', 'PAD', 
                'SUPPORT', 'WILL', 'TAG', 'SEED', 'ZONE', 'LAUNCHPAD', 'LAUNCHPOOL', 'BUSD', 'HODLER',
                'TOKEN', 'TOKENS', 'REWARD', 'REWARDS', 'COMPLETE', 'TASKS', 'LAUNCH', 'CONTRACTS',
                'MARGINED', 'PERPETUAL', 'FUTURES', 'SÖZLEŞME', 'MARJINLI', 'BAŞLATACAK',
                'SÜREKLI', 'SÖZLEŞMELERINI', 'CONTRACTS', 'PERPETUAL', 'MARGINED', 'USDS',
                'MILLIONS', 'BILLION', 'MARKETCAP', 'MILLION', 'BILLION', 'TRILLION',
                'MARKET', 'CAP', 'VOLUME', 'PRICE', 'CHANGE', 'PERCENT', 'HOURLY',
                'DAILY', 'WEEKLY', 'MONTHLY', 'YEARLY', 'ANALYSIS', 'TECHNICAL', 'FUNDAMENTAL',
                'RESISTANCE', 'SUPPORT', 'BREAKOUT', 'BREAKDOWN', 'BULLISH', 'BEARISH',
                'PUMP', 'DUMP', 'MOON', 'LAMBO', 'HODL', 'FOMO', 'DYOR', 'REKT',
                'WHALE', 'DIAMOND', 'HANDS', 'PAPER', 'HANDS', 'BAGGAGE', 'BAGGING',
                'SHILL', 'SHILLING', 'ALTCOIN', 'ALTCOINS', 'MEME', 'COIN', 'COINS',
                'CRYPTO', 'CRYPTOCURRENCY', 'BLOCKCHAIN', 'DEFI', 'DAPP', 'DAPPS',
                'SMART', 'CONTRACT', 'CONTRACTS', 'PROTOCOL', 'PROTOCOLS', 'GOVERNANCE',
                'STAKING', 'YIELD', 'FARMING', 'LIQUIDITY', 'POOL', 'POOLS', 'MINING',
                'PROOF', 'STAKE', 'WORK', 'CONSENSUS', 'ALGORITHM', 'HASH', 'HASHING',
                'NONCE', 'DIFFICULTY', 'ADJUSTMENT', 'HALVING', 'HALVINGS', 'INFLATION',
                'DEFLATION', 'SUPPLY', 'CIRCULATING', 'TOTAL', 'MAX', 'MAXIMUM',
                'MINIMUM', 'LOCKED', 'UNLOCKED', 'VESTING', 'CLIFF', 'LINEAR', 'UNLOCK',
                'LOCKUP', 'PERIOD', 'DURATION', 'MATURITY', 'EXPIRY', 'EXPIRATION',

                # Platform ve Proje İsimleri
                'ALPHA', 'BETA', 'GAMMA', 'DELTA', 'TERRA', 'MINA', 'LABS', 'FINANCE', 'CAPITAL',
                'VENTURES', 'PROTOCOL', 'GAMES', 'STUDIO', 'FOUNDATION', 'INSTITUTE',
                'RESEARCH', 'DEVELOPMENT', 'INNOVATION', 'TECHNOLOGY', 'SOLUTIONS',
                'SERVICES', 'SYSTEMS', 'NETWORKS', 'INFRASTRUCTURE', 'PLATFORM',
                'PLATFORMS', 'ECOSYSTEM', 'ECOSYSTEMS', 'COMMUNITY', 'COMMUNITIES',
                'PARTNERSHIP', 'PARTNERSHIPS', 'COLLABORATION', 'COLLABORATIONS',
                'INTEGRATION', 'INTEGRATIONS', 'ADOPTION', 'ADOPTIONS', 'MAINSTREAM',
                'ENTERPRISE', 'ENTERPRISES', 'CORPORATE', 'CORPORATIONS', 'INSTITUTIONAL',
                'INSTITUTIONS', 'RETAIL', 'INDIVIDUAL', 'INDIVIDUALS', 'INVESTOR',
                'INVESTORS', 'TRADER', 'TRADERS', 'DEVELOPER', 'DEVELOPERS',
                'BUILDER', 'BUILDERS', 'CREATOR', 'CREATORS', 'FOUNDER', 'FOUNDERS',
                'TEAM', 'TEAMS', 'MEMBER', 'MEMBERS', 'STAFF', 'EMPLOYEE', 'EMPLOYEES',
                'ADVISOR', 'ADVISORS', 'BOARD', 'DIRECTOR', 'DIRECTORS', 'EXECUTIVE',
                'EXECUTIVES', 'MANAGER', 'MANAGERS', 'LEAD', 'LEADS', 'HEAD', 'HEADS',
                'CHIEF', 'OFFICER', 'OFFICERS', 'PRESIDENT', 'PRESIDENTS', 'CHAIRMAN',
                'CHAIRMEN', 'CHAIRWOMAN', 'CHAIRWOMEN', 'CHAIRPERSON', 'CHAIRPEOPLE',

                # Web ve teknik terimler
                'HTTP', 'HTTPS', 'WWW', 'COM', 'NET', 'ORG', 'IO', 'CO', 'ME', 'AI',
                'TECH', 'BLOG', 'NEWS', 'MEDIUM', 'TWITTER', 'TELEGRAM', 'DISCORD',
                'REDDIT', 'YOUTUBE', 'FACEBOOK', 'INSTAGRAM', 'LINKEDIN', 'GITHUB',
                'GITLAB', 'STACKOVERFLOW', 'GOOGLE', 'APPLE', 'MICROSOFT', 'AMAZON',
                'NETFLIX', 'SPOTIFY', 'UBER', 'AIRBNB', 'PAYPAL', 'VISA', 'MASTERCARD',
                'AMEX', 'AMERICAN', 'EXPRESS', 'JPMORGAN', 'CHASE', 'BANK', 'AMERICA',
                'GOLDMAN', 'SACHS', 'MORGAN', 'STANLEY', 'BLACKROCK', 'VANGUARD',
                'FIDELITY', 'SCHWAB', 'MERRILL', 'LYNCH', 'WELLS', 'FARGO', 'CITIGROUP',
                'CITI', 'HSBC', 'BARCLAYS', 'DEUTSCHE', 'SANTANDER', 'CREDIT', 'SUISSE',
                'MIZUHO', 'NOMURA', 'MITSUBISHI', 'SUMITOMO', 'TOKYO', 'OSAKA', 'NIKKEI',
                'HANG', 'SENG', 'SHANGHAI', 'SHENZHEN', 'KOSPI', 'KOSDAQ', 'NIFTY',
                'SENSEX', 'FTSE', 'DAX', 'CAC', 'IBEX', 'FTSE', 'MIB', 'STOXX',
                'NASDAQ', 'NYSE', 'AMEX', 'CBOE', 'CME', 'CBOT', 'NYMEX', 'COMEX',
                'ICE', 'EUREX', 'LIFFE', 'MATIF', 'MEFF', 'BVMF', 'BOVESPA', 'BMV',
                'TSX', 'LSE', 'FSE', 'XETRA', 'EURONEXT', 'SIX', 'SWISS', 'EXCHANGE',
                'AUSTRALIAN', 'SECURITIES', 'HONG', 'KONG', 'SINGAPORE', 'THAILAND',
                'MALAYSIA', 'INDONESIA', 'PHILIPPINES', 'VIETNAM', 'INDIA', 'PAKISTAN',
                'BANGLADESH', 'SRI', 'LANKA', 'MYANMAR', 'CAMBODIA', 'LAOS', 'BRUNEI',
                'CHINA', 'JAPAN', 'SOUTH', 'KOREA', 'NORTH', 'TAIWAN', 'MACAU',
                'MONGOLIA', 'KAZAKHSTAN', 'KYRGYZSTAN', 'TAJIKISTAN', 'TURKMENISTAN',
                'UZBEKISTAN', 'AFGHANISTAN', 'IRAN', 'IRAQ', 'SYRIA', 'LEBANON',
                'ISRAEL', 'PALESTINE', 'JORDAN', 'SAUDI', 'ARABIA', 'KUWAIT', 'BAHRAIN',
                'QATAR', 'EMIRATES', 'OMAN', 'YEMEN', 'TURKEY', 'GREECE', 'CYPRUS',
                'BULGARIA', 'ROMANIA', 'MOLDOVA', 'UKRAINE', 'BELARUS', 'RUSSIA',
                'GEORGIA', 'ARMENIA', 'AZERBAIJAN', 'ESTONIA', 'LATVIA', 'LITHUANIA',
                'POLAND', 'CZECH', 'SLOVAKIA', 'HUNGARY', 'SLOVENIA', 'CROATIA',
                'BOSNIA', 'HERZEGOVINA', 'SERBIA', 'MONTENEGRO', 'ALBANIA', 'MACEDONIA',
                'KOSOVO', 'AUSTRIA', 'SWITZERLAND', 'LIECHTENSTEIN', 'GERMANY',
                'NETHERLANDS', 'BELGIUM', 'LUXEMBOURG', 'FRANCE', 'MONACO', 'ITALY',
                'VATICAN', 'SAN', 'MARINO', 'SPAIN', 'ANDORRA', 'PORTUGAL', 'GIBRALTAR',
                'MALTA', 'IRELAND', 'UNITED', 'KINGDOM', 'SCOTLAND', 'WALES',
                'NORTHERN', 'ICELAND', 'FAROE', 'ISLANDS', 'GREENLAND', 'DENMARK',
                'SWEDEN', 'NORWAY', 'FINLAND', 'ÅLAND', 'SVALBARD', 'JAN', 'MAYEN',
                'CANADA', 'UNITED', 'STATES', 'AMERICA', 'MEXICO', 'GUATEMALA',
                'BELIZE', 'HONDURAS', 'SALVADOR', 'NICARAGUA', 'COSTA', 'RICA',
                'PANAMA', 'CUBA', 'JAMAICA', 'HAITI', 'DOMINICAN', 'REPUBLIC',
                'PUERTO', 'RICO', 'VIRGIN', 'BAHAMAS', 'BARBADOS', 'TRINIDAD',
                'TOBAGO', 'GRENADA', 'LUCIA', 'VINCENT', 'GRENADINES', 'DOMINICA',
                'ANTIGUA', 'BARBUDA', 'KITTS', 'NEVIS', 'MONTSERRAT', 'ANGUILLA',
                'TURKS', 'CAICOS', 'CAYMAN', 'BERMUDA', 'COLOMBIA', 'VENEZUELA',
                'GUYANA', 'SURINAME', 'FRENCH', 'GUIANA', 'BRAZIL', 'ECUADOR',
                'PERU', 'BOLIVIA', 'PARAGUAY', 'URUGUAY', 'ARGENTINA', 'CHILE',
                'FALKLAND', 'SOUTH', 'GEORGIA', 'SANDWICH', 'ANTARCTICA', 'MOROCCO',
                'ALGERIA', 'TUNISIA', 'LIBYA', 'EGYPT', 'SUDAN', 'SOUTH', 'CHAD',
                'CENTRAL', 'AFRICAN', 'CAMEROON', 'EQUATORIAL', 'GUINEA', 'GABON',
                'CONGO', 'DEMOCRATIC', 'ANGOLA', 'ZAMBIA', 'MALAWI', 'MOZAMBIQUE',
                'ZIMBABWE', 'BOTSWANA', 'NAMIBIA', 'AFRICA', 'LESOTHO', 'SWAZILAND',
                'MADAGASCAR', 'MAURITIUS', 'SEYCHELLES', 'COMOROS', 'MAYOTTE',
                'REUNION', 'RODRIGUES', 'MAURITANIA', 'SENEGAL', 'GAMBIA',
                'GUINEA', 'BISSAU', 'CAPE', 'VERDE', 'SIERRA', 'LEONE', 'LIBERIA',
                'IVORY', 'COAST', 'GHANA', 'TOGO', 'BENIN', 'NIGERIA', 'NIGER',
                'BURKINA', 'FASO', 'MALI', 'GUINEA', 'BISSAU', 'SENEGAL', 'MAURITANIA',
                'WESTERN', 'SAHARA', 'CANARY', 'MADEIRA', 'AZORES', 'CAPE', 'VERDE',
                'SAINT', 'HELENA', 'ASCENSION', 'TRISTAN', 'CUNHA', 'AUSTRALIA',
                'NEW', 'ZEALAND', 'PAPUA', 'GUINEA', 'SOLOMON', 'VANUATU',
                'CALEDONIA', 'FIJI', 'TONGA', 'SAMOA', 'TUVALU', 'KIRIBATI',
                'NAURU', 'MARSHALL', 'MICRONESIA', 'PALAU', 'GUAM', 'NORTHERN',
                'MARIANA', 'AMERICAN', 'COOK', 'NIUE', 'TOKELAU', 'WALLIS',
                'FUTUNA', 'PITCAIRN', 'NORFOLK', 'CHRISTMAS', 'COCOS', 'KEELING',
                'HEARD', 'MCDONALD', 'MACQUARIE', 'ANTARCTICA', 'ROSS', 'DEPENDENCY',
                'PETER', 'QUEEN', 'MAUD', 'LAND', 'MARIE', 'BYRD', 'ELLSWORTH',
                'PALMER', 'PENINSULA', 'GRAHAM', 'ALEXANDER', 'THURSTON', 'BERKNER',
                'RONNE', 'FILCHNER', 'WEDDELL', 'LARSEN', 'GEORGE', 'KING', 'SOUTH',
                'SHETLAND', 'ORKNEY', 'ELEPHANT', 'CLARENCE', 'GIBBS', 'HOPE',
                'TRINITY', 'JOINVILLE', 'DUNDEE', 'VEGA', 'JAMES', 'ROSS', 'SNOW',
                'HILL', 'LIVINGSTON', 'DECEPTION', 'HALF', 'MOON', 'GREENWICH',
                'ROBERT', 'NELSON', 'RUGGED', 'SMITH', 'LOW', 'BRABANT', 'ANVERS',
                'WIENCKE', 'BOOTH', 'RENAUD', 'DOUMER', 'WANDEL', 'ENTERPRISE',
                'DEUX', 'POINTS', 'USEFUL', 'GALINDEZ', 'WINTER', 'SKUA', 'HUMBLE',
                'BERTHELOT', 'PETERMANN', 'PLÉNEAU', 'HOVGAARD', 'KIEV', 'RASMUSSEN',
                'FORGE', 'CRULS', 'IRIZAR', 'JOUBIN', 'LAHILLE', 'LÉOPOLD', 'RIDLEY',
                'SCULLIN', 'YALOUR', 'ARGENTINE', 'WIENCKE', 'ANVERS', 'DREAM',
                'CUVERVILLE', 'DANCO', 'CHARLOTTE', 'PORTAL', 'GOURDON', 'LEMAIRE',
                'PETERMANN', 'PLÉNEAU', 'HOVGAARD', 'KIEV', 'RASMUSSEN', 'FORGE',
                'CRULS', 'IRIZAR', 'JOUBIN', 'LAHILLE', 'LÉOPOLD', 'RIDLEY',
                'SCULLIN', 'YALOUR', 'ARGENTINE', 'WIENCKE', 'ANVERS', 'DREAM',
                'CUVERVILLE', 'DANCO', 'CHARLOTTE', 'PORTAL', 'GOURDON', 'LEMAIRE',
                
                # Sayılar ve ölçü birimleri
                'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT',
                'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN',
                'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN', 'NINETEEN', 'TWENTY', 'THIRTY',
                'FORTY', 'FIFTY', 'SIXTY', 'SEVENTY', 'EIGHTY', 'NINETY', 'HUNDRED',
                'THOUSAND', 'MILLION', 'BILLION', 'TRILLION', 'QUADRILLION',
                'QUINTILLION', 'SEXTILLION', 'SEPTILLION', 'OCTILLION', 'NONILLION',
                'DECILLION', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH', 'SIXTH',
                'SEVENTH', 'EIGHTH', 'NINTH', 'TENTH', 'ELEVENTH', 'TWELFTH',
                'THIRTEENTH', 'FOURTEENTH', 'FIFTEENTH', 'SIXTEENTH', 'SEVENTEENTH',
                'EIGHTEENTH', 'NINETEENTH', 'TWENTIETH', 'THIRTIETH', 'FORTIETH',
                'FIFTIETH', 'SIXTIETH', 'SEVENTIETH', 'EIGHTIETH', 'NINETIETH',
                'HUNDREDTH', 'THOUSANDTH', 'MILLIONTH', 'BILLIONTH', 'TRILLIONTH',
                
                # Özel format sayılar
                '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000',
                '10000', '100000', '1000000', '10000000', '100000000', '1000000000',
                '1M', '2M', '3M', '4M', '5M', '10M', '25M', '50M', '100M', '250M',
                '500M', '1B', '2B', '3B', '4B', '5B', '10B', '25B', '50B', '100B',
                '250B', '500B', '1T', '2T', '3T', '4T', '5T', '10T', '25T', '50T'
            }
            
            for word in word_candidates:
                if (word not in excluded_words and 
                    len(word) >= 3 and 
                    word not in potential_coins and
                    len(potential_coins) < 3):  # Maksimum 3 coin
                    potential_coins.append(word)

        # Market manager ile doğrulama
        if hasattr(self.bot, 'market_manager'):
            validated_coins = []
            for coin in potential_coins:
                # Önce direkt kontrol et
                if self.bot.market_manager.is_valid_coin(coin):
                    validated_coins.append(coin)
                # Sonra alias kontrolü
                else:
                    resolved = self.bot.market_manager.resolve_alias(coin)
                    if resolved != coin and self.bot.market_manager.is_valid_coin(resolved):
                        validated_coins.append(resolved)
            
            # ✅ ÖNCELIK SIRASI: $ ile başlayanlar önce
            prioritized_coins = []
            dollar_mentioned = [c for c in validated_coins if f'${c}' in text_upper]
            other_coins = [c for c in validated_coins if c not in dollar_mentioned]
            
            prioritized_coins.extend(dollar_mentioned)
            prioritized_coins.extend(other_coins)
            
            logger.info(f"Coin extraction: {len(potential_coins)} aday → {len(prioritized_coins)} geçerli → Öncelik: {prioritized_coins[:3]}")
            return prioritized_coins[:2]  # Maksimum 2 coin döndür
        
        return potential_coins[:2]

    async def stop(self):
        """Telegram dinleyiciyi durdurur."""
        if self.client and self.client.is_connected():
            await self.client.disconnect()
            logger.info("✅ Telegram dinleyici durduruldu.")
        self.is_running = False

class TradingStrategy:
    def __init__(self, config: ConfigManager): # <<<--- DEĞİŞİKLİK
        self.config = config
        # Ayarları config'den çek
        self.min_confidence = self.config.get('trading_strategy.min_confidence', 0.3)
        self.atr_multipliers = self.config.get('trading_strategy.atr_multipliers', {})
        self.current_signal_side = None
        
        # ✅ EKSİK OLAN RISK PARAMS
        self.risk_params = {
            "LOW": {
                "confidence_threshold": 0.2,
                "position_size": 0.01,
                "stop_loss": 0.02,      # ✅ sl → stop_loss
                "take_profit": 0.03     # ✅ tp → take_profit
            },
            "MEDIUM": {
                "confidence_threshold": 0.3,
                "position_size": 0.02,
                "stop_loss": 0.03,      # ✅ sl → stop_loss
                "take_profit": 0.05     # ✅ tp → take_profit
            },
            "HIGH": {
                "confidence_threshold": 0.4,
                "position_size": 0.03,
                "stop_loss": 0.04,      # ✅ sl → stop_loss
                "take_profit": 0.08     # ✅ tp → take_profit
            }
        }

        # YENİ: ATR ÇARPANLARI
        # Bunlar ayarlanabilir değerlerdir. Örneğin HIGH impact bir haberde,
        # SL'i ATR'nin 2 katı uzağa, TP'yi ise 4 katı uzağa koyuyoruz.
        self.atr_multipliers = {
            "ULTRA_HIGH": {"sl": 2.0, "tp": 5.0},
            "HIGH":       {"sl": 2.5, "tp": 4.0},
            "MEDIUM":     {"sl": 3.0, "tp": 3.0},
            "LOW":        {"sl": 3.5, "tp": 2.0}
        }
                
    def generate_signal(self, news: NewsItem, current_price: float, volatility: float) -> Optional[TradeSignal]:
        if not news.coins_mentioned: 
            logger.info(f"❌ Sinyal üretilmedi: Geçerli coin bulunamadı")
            return None
        
        # Base confidence hesaplama
        base_confidence = abs(news.sentiment_score)
        
        # Impact bonusu
        impact_bonus = {"ULTRA_HIGH": 0.4, "HIGH": 0.25, "MEDIUM": 0.1, "LOW": 0.0}
        
        # Özel boost - eğer title'da futures launch varsa
        futures_launch_boost = 0.0
        if 'futures will launch' in news.title.lower() or 'binance futures' in news.title.lower():
            futures_launch_boost = 0.2
            logger.info(f"Futures Launch boost uygulandı: +{futures_launch_boost}")
        
        # Final confidence hesaplama
        confidence = min(base_confidence + impact_bonus.get(news.impact_level, 0) + futures_launch_boost, 1.0)
        
        if confidence < self.min_confidence:
            # ✅ DETAYLI REJECTİON BİLDİRİMİ
            rejection_reason = f"Düşük güven: {confidence:.2f} < {self.min_confidence}"
            logger.info(f"❌ Sinyal reddedildi: {rejection_reason}")
            
            # Rejection detaylarını hesapla
            rejection_details = (
                f"📉 <b>Sinyal Reddedildi</b>\n\n"
                f"<b>Coin:</b> {news.coins_mentioned[0]}\n"
                f"<b>Sebep:</b> {rejection_reason}\n"
                f"<b>Sentiment:</b> {news.sentiment_score:.2f}\n"
                f"<b>Impact:</b> {news.impact_level}\n"
                f"<b>Base Güven:</b> {base_confidence:.2f}\n"
                f"<b>Impact Bonusu:</b> +{impact_bonus.get(news.impact_level, 0):.2f}\n"
                f"<b>Final Güven:</b> {confidence:.2f}\n"
                f"<b>Minimum Gerekli:</b> {self.min_confidence:.2f}\n"
                f"<b>Haber:</b> {news.title[:50]}..."
            )
            
            # Bot instance'a erişim için - bu kısmı process_news'de handle edeceğiz
            return None
        
        action = "BUY" if news.sentiment_score > 0 else "SELL"
        params = self.risk_params.get(news.impact_level, self.risk_params["MEDIUM"])
        
        sl_price = current_price * (1 - params["stop_loss"]) if action == "BUY" else current_price * (1 + params["stop_loss"])
        tp_price = current_price * (1 + params["take_profit"]) if action == "BUY" else current_price * (1 - params["take_profit"])
        
        logger.info(f"✅ Sinyal üretildi: {action} | Güven: {confidence:.2f} | Impact: {news.impact_level}")
        
        return TradeSignal(
            symbol="", action=action, confidence=confidence, expected_impact=news.impact_level,
            entry_price=current_price, stop_loss=sl_price, take_profit=tp_price, position_size=0,
            volatility=volatility
        )

class ProductionLogger:
    """Production ortamı için optimize edilmiş logging"""
    
    @staticmethod
    def setup_production_logging():
        """Production için logging seviyelerini ayarlar"""
        
        # Environment'tan log level al
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        is_debug = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
        # Ana logger'ı konfigüre et
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level))
        
        # Handler'ları temizle
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Production handler ekle
        handler = logging.StreamHandler()
        
        if is_debug:
            # Debug mode: Detaylı loglar
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            # Production mode: Sadece önemli bilgiler
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

class RiskManager:
    def __init__(self, config: ConfigManager, initial_balance: float): # <<<--- DEĞİŞİKLİK
        self.config = config
        self.initial_balance = initial_balance
        # Ayarları config'den çek
        self.max_open_positions = self.config.get('risk_management.general.max_open_positions', 5)
        self.daily_loss_limit_pct = self.config.get('risk_management.general.daily_loss_limit_pct', 0.25)
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
    def _check_daily_reset(self):
        if datetime.now().date() > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = datetime.now().date()
            logger.info("Günlük PnL sıfırlandı.")
    
    def can_open_position(self, open_positions_count: int) -> bool:
        """Pozisyon açma iznini kontrol eder - BASİT VERSİYON (2 parametre)."""
        self._check_daily_reset()
        
        # Maksimum pozisyon kontrolü
        if open_positions_count >= self.max_open_positions:
            logger.warning(f"Maksimum açık pozisyon limitine ({self.max_open_positions}) ulaşıldı.")
            return False
        
        # Günlük zarar limiti kontrolü (şimdilik devre dışı)
        # daily_loss_limit_amount = self.initial_balance * self.daily_loss_limit_pct
        # if self.daily_pnl <= -daily_loss_limit_amount:
        #     logger.warning(f"Günlük zarar limitine (${-daily_loss_limit_amount:.2f}) ulaşıldı.")
        #     return False
        
        return True

class PortfolioRiskManager:
    def __init__(self, bot_instance, config: ConfigManager): # <<<--- DEĞİŞİKLİK
        self.bot = bot_instance
        self.config = config
        self.price_history_cache = {}
        self.cache_expiry = timedelta(minutes=30)
        # Ayarları config'den çek
        self.enabled = self.config.get('risk_management.portfolio.enable_correlation_filter', True)
        self.threshold = self.config.get('risk_management.portfolio.max_correlation_threshold', 0.75)

    async def check_correlation_risk(self, new_symbol: str) -> bool:
        """Yeni bir sembolün, mevcut aktif pozisyonlarla olan fiyat korelasyonunu kontrol eder."""
        if not self.enabled:
            return True

        # Paper mode pozisyonları kontrol et
        if self.bot.paper_mode:
            active_positions = self.bot.paper_engine.positions
        else:
            active_positions = self.bot.live_positions

        if not active_positions:
            return True

        logger.info(f"[{new_symbol}] Korelasyon riski kontrol ediliyor...")
        
        new_symbol_prices = await self._get_price_history(new_symbol)
        if new_symbol_prices is None or new_symbol_prices.empty:
            logger.warning(f"[{new_symbol}] Korelasyon kontrolü için fiyat verisi alınamadı, işlem onaylandı.")
            return True

        # Current signal side'ı strategy'den al - GÜVENLİ
        current_signal_side = getattr(self.bot.strategy, 'current_signal_side', None)
        if not current_signal_side:
            logger.warning("Current signal side bulunamadı, korelasyon kontrolü atlanıyor")
            return True

        for active_key, position in active_positions.items():
            # Sadece aynı yöndeki pozisyonlarla korelasyonu kontrol et
            if position.side != current_signal_side:
                continue

            active_symbol = position.symbol
            active_symbol_prices = await self._get_price_history(active_symbol)
            if active_symbol_prices is None or active_symbol_prices.empty:
                continue

            # Korelasyon hesaplama
            combined_df = pd.concat([new_symbol_prices, active_symbol_prices], axis=1).dropna()
            if len(combined_df) < 20: 
                continue
            
            correlation = combined_df.iloc[:, 0].corr(combined_df.iloc[:, 1])
            
            logger.info(f"[{new_symbol}] ile [{active_symbol}] arasındaki korelasyon: {correlation:.2f}")

            if correlation > self.threshold:
                rejection_msg = (
                    f"🛡️ <b>İşlem Reddedildi (Yüksek Korelasyon)</b>\n\n"
                    f"<b>Coin:</b> {new_symbol.replace('/USDT:USDT', '')}\n"
                    f"<b>Sebep:</b> Açık olan <b>{active_symbol.replace('/USDT:USDT', '')}</b> pozisyonu ile korelasyonu "
                    f"(<b>{correlation:.2f}</b>), belirlenen eşiğin (<b>{self.threshold}</b>) üzerinde."
                )
                await self.bot.telegram.send_message(rejection_msg)
                return False
        
        return True

    async def _get_price_history(self, symbol: str) -> Optional[pd.Series]:
        """Fiyat geçmişini API'den çeker veya önbellekten alır."""
        now = datetime.now()
        if symbol in self.price_history_cache and (now - self.price_history_cache[symbol]['timestamp']) < self.cache_expiry:
            return self.price_history_cache[symbol]['prices']

        try:
            klines = await self.bot.loop.run_in_executor(None, lambda: self.bot.exchange.fetch_ohlcv(symbol, '15m', limit=100))
            if klines:
                prices = pd.to_numeric(pd.DataFrame(klines)[4])  # Kapanış fiyatları
                self.price_history_cache[symbol] = {'prices': prices, 'timestamp': now}
                return prices
        except Exception as e:
            logger.warning(f"Korelasyon için fiyat geçmişi alınamadı ({symbol}): {e}")
        return None

class CommandHandler:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.commands = {
            '/stats': self.get_stats,
            '/positions': self.get_positions,
            '/balance': self.get_balance,
            '/report': self.get_performance_report,  # ✅ YENİ KOMUT
            '/help': self.get_help
        }

    async def get_performance_report(self) -> str:
        """
        Detaylı performans raporu oluşturur - CSV analizi + canlı veriler
        """
        try:
            # ✅ DATABASE'DEN İSTATİSTİKLER AL
            if self.bot.database:
                trade_stats = await self.bot.loop.run_in_executor(
                    None, 
                    self.bot.database.get_performance_stats
                )
                open_positions = await self.bot.loop.run_in_executor(
                    None,
                    self.bot.database.get_open_positions_count
                )
            else:
                # Fallback: CSV analizi
                trade_stats = self._analyze_trade_history_detailed()
                open_positions = len(self.bot.live_positions)
            
            # Canlı bakiye bilgisi
            if self.bot.paper_mode:
                current_balance = self.bot.paper_engine.current_balance
                initial_balance = self.bot.paper_engine.initial_balance
                mode_text = "PAPER"
                exchange_text = "PAPER TRADING"
            else:
                current_balance = await self.bot.get_futures_balance()
                initial_balance = float(os.getenv('INITIAL_BALANCE', '50'))
                mode_text = "LIVE"
                exchange_text = self.bot.exchange.id.upper()
            
            # Return hesaplama
            if current_balance and initial_balance > 0:
                total_return_usd = current_balance - initial_balance
                total_return_pct = (total_return_usd / initial_balance) * 100
            else:
                total_return_usd = trade_stats.get('total_pnl_usd', 0)
                total_return_pct = 0
            
            # Açık pozisyon sayısı
            open_positions = len(self.bot.paper_engine.positions) if self.bot.paper_mode else len(self.bot.live_positions)
            
            # Profit Factor hesaplama
            total_wins = trade_stats.get('total_win_amount', 0)
            total_losses = abs(trade_stats.get('total_loss_amount', 0))
            profit_factor = round(total_wins / total_losses, 2) if total_losses > 0 else "∞"
            
            # Return emoji
            return_emoji = "📈" if total_return_usd >= 0 else "📉"
            pnl_emoji = "💚" if total_return_usd >= 0 else "💔"
            
            # Risk/Reward Ratio
            avg_win = trade_stats.get('avg_win_amount', 0)
            avg_loss = abs(trade_stats.get('avg_loss_amount', 0))
            risk_reward = round(avg_win / avg_loss, 2) if avg_loss > 0 else "∞"
            
            # Streak hesaplama
            current_streak = self._calculate_current_streak()
            max_win_streak = trade_stats.get('max_win_streak', 0)
            max_loss_streak = trade_stats.get('max_loss_streak', 0)
            
            # Ana rapor
            report = f"""📊 <b>PERFORMANCE REPORT ({mode_text})</b>
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            💼 <b>BALANCE & RETURNS</b>
            💰 <b>Current Balance:</b> ${current_balance:,.2f}
            {return_emoji} <b>Total Return:</b> ${total_return_usd:+,.2f} ({total_return_pct:+.2f}%)
            🏛️ <b>Exchange:</b> {exchange_text}

            📈 <b>TRADING STATISTICS</b>
            📊 <b>Total Trades:</b> {trade_stats.get('total_trades', 0)}
            🎯 <b>Win Rate:</b> {trade_stats.get('win_rate_pct', 0):.1f}% ({trade_stats.get('winning_trades', 0)}/{trade_stats.get('total_trades', 0)})
            💚 <b>Avg Win:</b> ${trade_stats.get('avg_win_amount', 0):.2f}
            💔 <b>Avg Loss:</b> ${trade_stats.get('avg_loss_amount', 0):.2f}
            ⚖️ <b>Profit Factor:</b> {profit_factor}
            🎲 <b>Risk/Reward:</b> 1:{risk_reward}

            🔥 <b>STREAKS & PERFORMANCE</b>
            🔄 <b>Current Streak:</b> {current_streak}
            📈 <b>Best Win Streak:</b> {max_win_streak}
            📉 <b>Worst Loss Streak:</b> {max_loss_streak}
            💎 <b>Best Trade:</b> ${trade_stats.get('best_trade_pnl', 0):+.2f}
            💸 <b>Worst Trade:</b> ${trade_stats.get('worst_trade_pnl', 0):+.2f}

            📍 <b>CURRENT STATUS</b>
            🔄 <b>Open Positions:</b> {open_positions}
            📅 <b>Last Trade:</b> {trade_stats.get('last_trade_date', 'N/A')}
            ⏱️ <b>Total Days:</b> {trade_stats.get('trading_days', 1)}

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📊 <b>Report Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"""

            return report
            
        except Exception as e:
            logger.error(f"Performance raporu oluşturulurken hata: {e}")
            return "❌ <b>Rapor Hatası</b>\n\nPerformans raporu oluşturulamadı. Lütfen daha sonra tekrar deneyin."

    def _analyze_trade_history_detailed(self) -> Dict:
        """CSV dosyasından detaylı trade analizi yapar."""
        stats = {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate_pct': 0,
            'total_pnl_usd': 0, 'avg_win_amount': 0, 'avg_loss_amount': 0,
            'best_trade_pnl': 0, 'worst_trade_pnl': 0, 'total_win_amount': 0, 'total_loss_amount': 0,
            'max_win_streak': 0, 'max_loss_streak': 0, 'last_trade_date': 'N/A', 'trading_days': 1
        }
        
        filepath = "trade_history.csv"
        if not os.path.exists(filepath):
            return stats

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                return stats

            # Temel hesaplamalar
            stats['total_trades'] = len(df)
            winning_trades = df[df['pnl_usd'] > 0]
            losing_trades = df[df['pnl_usd'] <= 0]
            
            stats['winning_trades'] = len(winning_trades)
            stats['losing_trades'] = len(losing_trades)
            stats['win_rate_pct'] = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
            
            # PnL hesaplamaları
            stats['total_pnl_usd'] = df['pnl_usd'].sum()
            stats['best_trade_pnl'] = df['pnl_usd'].max()
            stats['worst_trade_pnl'] = df['pnl_usd'].min()
            
            # Kazanan/Kaybeden ortalamalar
            if not winning_trades.empty:
                stats['avg_win_amount'] = winning_trades['pnl_usd'].mean()
                stats['total_win_amount'] = winning_trades['pnl_usd'].sum()
            
            if not losing_trades.empty:
                stats['avg_loss_amount'] = losing_trades['pnl_usd'].mean()
                stats['total_loss_amount'] = losing_trades['pnl_usd'].sum()
            
            # Streak hesaplama
            streaks = self._calculate_streaks(df['pnl_usd'].tolist())
            stats['max_win_streak'] = streaks['max_win_streak']
            stats['max_loss_streak'] = streaks['max_loss_streak']
            
            # Son trade tarihi
            if 'close_timestamp_utc' in df.columns:
                stats['last_trade_date'] = df['close_timestamp_utc'].iloc[-1][:10]  # Sadece tarih
            
            # Trading günleri
            if 'close_timestamp_utc' in df.columns:
                dates = pd.to_datetime(df['close_timestamp_utc']).dt.date
                stats['trading_days'] = len(dates.unique())

            return stats

        except Exception as e:
            logger.error(f"CSV analiz hatası: {e}")
            return stats

    def _calculate_streaks(self, pnl_list: List[float]) -> Dict:
        """Kazanma/kaybetme streak'lerini hesaplar."""
        if not pnl_list:
            return {'max_win_streak': 0, 'max_loss_streak': 0}
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for pnl in pnl_list:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {'max_win_streak': max_win_streak, 'max_loss_streak': max_loss_streak}

    def _calculate_current_streak(self) -> str:
        """Mevcut streak'i hesaplar."""
        filepath = "trade_history.csv"
        if not os.path.exists(filepath):
            return "0"
        
        try:
            df = pd.read_csv(filepath)
            if df.empty or len(df) < 2:
                return "0"
            
            # Son 10 trade'i al
            recent_trades = df.tail(10)['pnl_usd'].tolist()
            
            current_streak = 0
            last_pnl = recent_trades[-1]
            
            # Sondan başlayarak streak hesapla
            for pnl in reversed(recent_trades):
                if (last_pnl > 0 and pnl > 0) or (last_pnl <= 0 and pnl <= 0):
                    current_streak += 1
                else:
                    break
            
            streak_type = "W" if last_pnl > 0 else "L"
            return f"{current_streak}{streak_type}"
            
        except Exception as e:
            logger.error(f"Current streak hesaplama hatası: {e}")
            return "0"

    async def execute_command(self, text: str):
        command_word = text.strip().lower().split(' ')[0]
        command_function = self.commands.get(command_word)
        
        if command_function:
            response_message = await command_function()
        else:
            response_message = self.get_help()
            
        await self.bot.telegram.send_message(response_message)

    def get_help(self) -> str:
        """Yardım mesajı oluşturur."""
        return (
            "🤖 <b>AEnews Bot Komutları</b>\n\n"
            "<code>/report</code> - 📊 Detaylı performans raporu\n"
            "<code>/stats</code> - 📈 Hızlı performans özeti\n"
            "<code>/positions</code> - 📍 Açık pozisyonları listeler\n"
            "<code>/balance</code> - 💰 Anlık bakiyeyi gösterir\n"
            "<code>/help</code> - ❓ Bu yardım mesajını gösterir"
        )

    def _analyze_trade_history(self) -> Dict:
        """trade_history.csv dosyasını analiz eder ve performans istatistikleri döndürür."""
        stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0,
            'total_pnl_usd': 0,
            'average_pnl_usd': 0,
            'best_trade_pnl': 0,
            'worst_trade_pnl': -1,
            'most_profitable_coin': 'N/A',
            'most_losing_coin': 'N/A'
        }
        
        filepath = "trade_history.csv"
        if not os.path.exists(filepath):
            return stats # Dosya yoksa boş istatistik döndür

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                return stats # Dosya boşsa boş istatistik döndür

            # Temel Hesaplamalar
            stats['total_trades'] = len(df)
            stats['winning_trades'] = df[df['pnl_usd'] > 0].shape[0]
            stats['losing_trades'] = df[df['pnl_usd'] <= 0].shape[0]
            if stats['total_trades'] > 0:
                stats['win_rate_pct'] = (stats['winning_trades'] / stats['total_trades']) * 100
            
            stats['total_pnl_usd'] = df['pnl_usd'].sum()
            stats['average_pnl_usd'] = df['pnl_usd'].mean()
            stats['best_trade_pnl'] = df['pnl_usd'].max()
            stats['worst_trade_pnl'] = df['pnl_usd'].min()

            # Coin Bazlı Analiz
            coin_pnl = df.groupby('symbol')['pnl_usd'].sum()
            if not coin_pnl.empty:
                stats['most_profitable_coin'] = coin_pnl.idxmax()
                stats['most_losing_coin'] = coin_pnl.idxmin()

            return stats

        except Exception as e:
            logger.error(f"Trade geçmişi analiz edilirken hata: {e}")
            return stats # Hata durumunda boş istatistik döndür

    async def get_balance(self) -> str:
        """Anlık bakiye durumunu raporlar."""
        message = "💰 <b>BAKİYE DURUMU</b>\n"
        
        if self.bot.paper_mode:
            balance = self.bot.paper_engine.current_balance
            message += f"\n► <b>PAPER MODE:</b> ${balance:,.2f} USDT"
        else:
            balance = await self.bot.get_futures_balance()
            if balance is not None:
                message += f"\n► <b>{self.bot.exchange.id.upper()}:</b> ${balance:,.2f} USDT"
            else:
                message += f"\n► <b>{self.bot.exchange.id.upper()}:</b> Bakiye alınamadı."
        
        return message

    async def get_positions(self) -> str:
        """Açık pozisyonları listeler."""
        active_positions = self.bot.paper_engine.positions if self.bot.paper_mode else self.bot.live_positions
        if not active_positions:
            return "✅ Şu anda açık pozisyon yok."

        message = f"📊 <b>AÇIK POZİSYONLAR ({len(active_positions)} adet)</b>\n"
        
        for position_key, position in active_positions.items():
            try:
                # Canlı modda anlık PnL hesapla
                if not self.bot.paper_mode:
                    current_price = await self.bot.get_current_price(position.symbol)
                    if current_price:
                        position.current_price = current_price
                        if position.side == 'BUY':
                            position.pnl = (current_price - position.entry_price) * position.size
                        else:
                            position.pnl = (position.entry_price - current_price) * position.size
                
                pnl_percent = (position.pnl / (position.entry_price * position.size)) * 100 if position.entry_price > 0 and position.size > 0 else 0
                pnl_emoji = "🟢" if position.pnl >= 0 else "🔴"
                ts_status = " (TS Aktif)" if position.trailing_stop_activated else ""

                message += (
                    f"\n<b>{pnl_emoji} {position.symbol} ({position.side})</b>\n"
                    f"  - <b>Giriş:</b> ${position.entry_price:,.4f}\n"
                    f"  - <b>Anlık:</b> ${position.current_price:,.4f}\n"
                    f"  - <b>PnL:</b> ${position.pnl:+.2f} ({pnl_percent:+.2f}%)\n"
                    f"  - <b>SL:</b> ${position.stop_loss:,.4f}{ts_status}"
                )
            except Exception as e:
                logger.error(f"Pozisyon raporu oluşturulurken hata: {position_key}, {e}")
        
        return message

    async def get_stats(self) -> str:
        """Botun genel performans istatistiklerini raporlar."""
        
        mode_text = "PAPER" if self.bot.paper_mode else "LIVE"
        message = f"📊 <b>PERFORMANS İSTATİSTİKLERİ ({mode_text})</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━━\n"

        if self.bot.paper_mode:
            # Paper mod için PaperTradingEngine'den veri al
            stats = self.bot.paper_engine.get_performance_stats()
            
            message += (
                f"<b>Başlangıç Bakiye:</b> ${stats.get('initial_balance', 0):,.2f}\n"
                f"<b>Güncel Bakiye:</b> ${stats.get('current_balance', 0):,.2f}\n"
                f"<b>Toplam PnL:</b> ${stats.get('total_pnl', 0):,.2f} ({stats.get('total_return_pct', 0):+.2f}%)\n"
                f"<b>Max Drawdown:</b> {stats.get('max_drawdown_pct', 0):.2f}%\n\n"
            )
            message += (
                f"<b>Toplam İşlem:</b> {stats.get('total_trades', 0)}\n"
                f"<b>Kazanan İşlem:</b> {stats.get('winning_trades', 0)}\n"
                f"<b>Kazanma Oranı:</b> {stats.get('win_rate_pct', 0):.2f}%\n\n"
            )
            message += f"<b>Açık Pozisyon:</b> {stats.get('open_positions', 0)}\n"

        else:
            # Canlı mod için CSV dosyasını analiz et
            stats = self._analyze_trade_history()
            
            balance_info = await self.get_balance() # Mevcut bakiye bilgisini al
            
            pnl_emoji = "✅" if stats.get('total_pnl_usd', 0) >= 0 else "❌"

            message += (
                f"{balance_info}\n\n" # Bakiye durumunu ekle
                f"<b>--- Geçmiş Performans (Tüm Zamanlar) ---</b>\n"
                f"<b>Toplam İşlem:</b> {stats.get('total_trades', 0)}\n"
                f"<b>Kazanan / Kaybeden:</b> {stats.get('winning_trades', 0)} / {stats.get('losing_trades', 0)}\n"
                f"<b>Kazanma Oranı:</b> {stats.get('win_rate_pct', 0):.2f}%\n\n"
                
                f"<b>Toplam Net PnL:</b> {pnl_emoji} ${stats.get('total_pnl_usd', 0):,.2f}\n"
                f"<b>Ortalama PnL/İşlem:</b> ${stats.get('average_pnl_usd', 0):,.2f}\n\n"
                
                f"<b>En İyi İşlem:</b> 🟢 ${stats.get('best_trade_pnl', 0):,.2f}\n"
                f"<b>En Kötü İşlem:</b> 🔴 ${stats.get('worst_trade_pnl', 0):,.2f}\n\n"
                
                f"<b>En Kârlı Coin:</b> {stats.get('most_profitable_coin', 'N/A')}\n"
                f"<b>En Zararlı Coin:</b> {stats.get('most_losing_coin', 'N/A')}\n\n"
            )
            message += f"<b>Açık Pozisyon:</b> {len(self.bot.live_positions)}\n"
            
        message += "━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"<i>Rapor Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return message

class CryptoNewsBot:
    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.is_running = False
        self.live_positions: Dict[str, Position] = {}
        self.symbol_info_cache: Dict[str, Dict] = {}
        self.recent_news = deque(maxlen=200)
        self.processing_lock = asyncio.Lock()
        self.currently_processing = set()
        self.exchange_api_semaphore = asyncio.Semaphore(5)

        self.config = ConfigManager()
        self.telegram = TelegramNotifier(self)
        self.analyzer = NewsAnalyzer()
        self.exchange = self._setup_exchange()
        
        if not self.exchange:
            return
            
        self.market_manager = MarketManager(self.exchange)
        self.strategy = TradingStrategy(self.config)
        self.portfolio_risk_manager = PortfolioRiskManager(self, self.config)
        self.telegram_collector = TelegramChannelCollector(self, self.config)
        self.command_handler = CommandHandler(self)
        self.csv_logger = CsvLogger()

        self.paper_mode = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        initial_balance = float(os.getenv('INITIAL_BALANCE', '50'))
        self.paper_engine = PaperTradingEngine(self, initial_balance)
        self.risk_manager = RiskManager(self.config, initial_balance)

        # ✅ DATABASE BAŞLAT
        try:
            self.database = TradingDatabase()
            logger.info("✅ SQLite Database entegrasyonu tamamlandı")
        except Exception as e:
            logger.error(f"❌ Database başlatma hatası: {e}")
            self.database = None
        
        # Adım 5: Veri Kaydedici
        self.csv_logger = CsvLogger()

    def _setup_exchange(self):
        """
        .env dosyasındaki EXCHANGE değişkenine göre doğru borsa nesnesini kurar.
        """
        try:
            exchange_name = os.getenv('EXCHANGE', 'gateio').lower()
            api_key = os.getenv(f'{exchange_name.upper()}_API_KEY')
            secret_key = os.getenv(f'{exchange_name.upper()}_SECRET_KEY')

            if not api_key or not secret_key:
                logger.critical(f"❌ {exchange_name.upper()} için API anahtarları .env dosyasında eksik.")
                return None

            config = {
                'apiKey': api_key, 'secret': secret_key, 
                'options': {
                    # --- BU SATIR TÜM SORUNU ÇÖZECEK ---
                    'defaultType': 'swap', 
                    # ------------------------------------
                    'createMarketBuyOrderRequiresPrice': False
                },
                'enableRateLimit': True, 
                'sandbox': os.getenv(f'{exchange_name.upper()}_SANDBOX', 'false').lower() == 'true'
            }
            
            if exchange_name == 'gateio': 
                config['options']['settle'] = 'usdt'
            
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class(config)
            
            logger.info(f"✅ {exchange.id.upper()} borsasına başarıyla bağlanıldı.")
            return exchange

        except Exception as e:
            logger.critical(f"❌ Borsa örneği oluşturulamadı: {e}", exc_info=True)
            return None

    async def _validate_exchange_connection(self) -> bool:
        logger.info("Borsa bağlantısı doğrulanıyor...")
        try:
            async with self.exchange_api_semaphore:
                await self.get_futures_balance()
            logger.info("✅ Borsa bağlantısı başarıyla doğrulandı.")
            return True
        except ccxt.AuthenticationError as e:
            logger.critical(f"❌ Borsa Doğrulama Hatası: API anahtarları geçersiz. Hata: {e}")
            await self.telegram.send_message("🚨 <b>API Anahtar Hatası!</b>\n\nLütfen .env dosyasındaki API anahtarlarınızın doğru olduğundan emin olun.")
            return False
        except ccxt.PermissionDenied as e:
            logger.critical(f"❌ Borsa Doğrulama Hatası: API anahtarının izni yok. Hata: {e}")
            await self.telegram.send_message("🚨 <b>API İzin Hatası!</b>\n\nAPI anahtarınızın 'Vadeli İşlemler' ve 'Spot' için 'Okuma ve Yazma' izinlerine sahip olduğundan emin olun.")
            return False
        except ccxt.NetworkError as e:
            logger.critical(f"❌ Borsa Doğrulama Hatası: Borsa API'sine ulaşılamıyor. Hata: {e}")
            await self.telegram.send_message("🚨 <b>Ağ Hatası!</b>\n\nBorsa sunucularına ulaşılamıyor. İnternet bağlantınızı kontrol edin.")
            return False
        except Exception as e:
            logger.critical(f"❌ Borsa Doğrulama Hatası: Bilinmeyen hata. Hata: {e}")
            await self.telegram.send_message(f"🚨 <b>Bilinmeyen Borsa Hatası!</b>\n\nBağlantı sırasında bir hata oluştu: {str(e)}")
            return False

    async def start(self):
        """Botun tüm asenkron görevlerini başlatır ve ana döngüyü çalıştırır."""
        if not self.exchange:
            error_message = "❌ <b>BOT BAŞLATILAMADI</b>\n\nBorsa bağlantısı kurulamadı. Lütfen .env dosyasındaki API anahtarlarını kontrol edin."
            logger.critical(error_message.replace("<b>", "").replace("</b>", ""))
            await self.telegram.send_message(error_message)
            return

        self.is_running = True
        
        await self.market_manager.update_from_exchange()

        mode_text = "Paper Trading" if self.paper_mode else "Live Trading"
        balance_str = f"${self.paper_engine.initial_balance:,.2f}"
        if not self.paper_mode:
            live_balance = await self.get_futures_balance()
            balance_str = f"${live_balance:,.2f}" if live_balance is not None else "Sorgulanamadı"

        startup_message = (
            f"🚀 <b>BOT BAŞLATILDI</b>\n\n"
            f"📝 <b>Mod:</b> {mode_text}\n💰 <b>Anlık Bakiye:</b> {balance_str}\n"
            f"🏛️ <b>Borsa:</b> {self.exchange.id.upper()}\n"
            f"📡 <b>Kanal Sayısı:</b> {len(self.telegram_collector.channels_to_listen)}\n"
            f"✅ <b>Bot aktif ve haber bekliyor...</b>"
        )
        await self.telegram.send_message(startup_message)

        # ARKA PLAN GÖREVLERİNİ BAŞLAT
        logger.info("Arka plan görevleri (pozisyon yönetimi) başlatılıyor...")
        
        # Mevcut periyodik güncelleme görevi
        periodic_updates_task = self.loop.create_task(self.periodic_updates())
        logger.info("✅ Periyodik güncelleme görevi başlatıldı")
        
        # YENİ: Manuel Stop Loss takip sistemi (sadece live modda)
        manual_sl_task = None
        if not self.paper_mode:
            logger.info("🛡️ Manuel Stop Loss takip sistemi başlatılıyor...")
            manual_sl_task = self.loop.create_task(self.manual_stop_loss_monitor())
            logger.info("✅ Manuel Stop Loss takip sistemi başlatıldı")
        else:
            logger.info("📝 Paper modda manuel SL takip sistemi gerekmiyor")

        logger.info("Telegram dinleyici başlatılıyor ve ana kontrol ona devrediliyor...")
        
        # Ana Telegram dinleyicisini başlat (bu blocking operation)
        try:
            await self.telegram_collector.start()
        except KeyboardInterrupt:
            logger.info("❌ Kullanıcı tarafından durduruldu (Ctrl+C)")
        except Exception as e:
            logger.error(f"❌ Telegram dinleyici kritik hatası: {e}")
        finally:
            # Program durduğunda tüm görevleri temizle
            logger.info("Bot durduruluyor. Arka plan görevleri iptal ediliyor...")
            
            # Tüm arka plan görevlerini iptal et
            try:
                periodic_updates_task.cancel()
                logger.info("✅ Periyodik güncelleme görevi iptal edildi")
                
                if manual_sl_task and not manual_sl_task.cancelled():
                    manual_sl_task.cancel()
                    logger.info("✅ Manuel Stop Loss takip sistemi iptal edildi")
                
                # Görevlerin temizlenmesini bekle
                tasks_to_wait = [periodic_updates_task]
                if manual_sl_task:
                    tasks_to_wait.append(manual_sl_task)
                    
                # İptal edilen görevlerin exception'larını yakalayarak bekle
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)
                logger.info("✅ Tüm arka plan görevleri temizlendi")
                
            except Exception as e:
                logger.error(f"❌ Görev temizleme sırasında hata: {e}")
            
            self.is_running = False
            logger.info("🛑 Bot tamamen durduruldu")
    
    async def stop(self):
        """Botu ve tüm alt servisleri güvenli bir şekilde durdurur."""
        if not self.is_running: return
        self.is_running = False
        logger.info("Bot durduruluyor...")
        
        await self.telegram_collector.stop()
        await self.telegram.close_session()
        
        # Raporlama
        if self.paper_mode and self.paper_engine.total_trades > 0:
            stats = self.paper_engine.get_performance_stats()
            await self.telegram.notify_performance_report(stats)
        
        await self.telegram.send_message("🛑 <b>Bot Durduruldu</b>")
        logger.info("Bot başarıyla durduruldu.")

    async def periodic_updates(self):
        """Periyodik güncelleme - düzeltilmiş versiyon"""
        last_market_update_hour = -1 
        
        while self.is_running:
            try:
                # Pozisyon güncellemeleri
                if self.paper_mode:
                    if self.paper_engine.positions:
                        await self.update_paper_positions()  # ✅ Kendi metodunu kullan
                elif self.live_positions:
                    await self.update_live_positions()
                
                # Market listesi güncelleme
                current_hour = datetime.now().hour
                if current_hour != last_market_update_hour:
                    if current_hour % 6 == 0:
                        await self.market_manager.update_from_exchange()
                    last_market_update_hour = current_hour
                
                await asyncio.sleep(15)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periyodik güncelleme hatası: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def create_conditional_order(self, symbol: str, side: str, order_type: str, amount: float, trigger_price: float):
        """Gate.io için düzeltilmiş koşullu emir fonksiyonu - Fiyat formatı düzeltildi."""
        exchange = self.exchange
        try:
            order_side = 'sell' if side.upper() == 'BUY' else 'buy'
            
            # ✅ FİYAT FORMATINI DÜZELTELİM
            formatted_trigger_price = float(exchange.price_to_precision(symbol, trigger_price))
            
            # Gate.io için özel stop emir parametreleri
            params = {
                'settle': 'usdt',
                'reduceOnly': True,
                'stopPrice': formatted_trigger_price  # ✅ Float formatında gönder
            }
            
            # Stop limit emri kullan (daha güvenilir)
            response = await self.loop.run_in_executor(
                None,
                lambda: exchange.create_order(
                    symbol=symbol,
                    type='stop_limit',  # ✅ stop_limit kullan
                    side=order_side,
                    amount=amount,
                    price=formatted_trigger_price,  # ✅ Limit fiyatı da ekle
                    params=params
                )
            )
            
            order_id = response.get('id')
            if order_id:
                logger.info(f"✅ [{symbol}] {order_type} emri başarıyla gönderildi: ID {order_id}")
                return {'id': order_id, 'info': response}
            else:
                raise Exception(f"Emir ID'si alınamadı: {response}")
                
        except Exception as e:
            logger.error(f"❌ [{symbol}] {order_type} emri HATASI: {e}")
            return None

    async def get_balance(self) -> float:
        """
        Gate.io futures bakiyesini (USDT) döner.
        """
        exchange = self.exchange
        try:
            balance = await self.loop.run_in_executor(
                None,
                lambda: exchange.fetch_balance({'settle': 'usdt'})
            )
            total = balance.get('total', {}).get('USDT', 0.0)
            logger.info(f"💰 Bakiye alındı: ${total:.2f}")
            return float(total)
        except Exception as e:
            logger.error(f"❌ Bakiye alınamadı: {e}")
            return 0.0

    async def process_news(self, news: NewsItem):
        """Haberleri alır, filtreler ve her geçerli coin için işlem sürecini başlatır."""
        logger.info(f"Haber işleme adımları başlatıldı: '{news.title}'")

        # ✅ İNTERAKTİF HABER BİLDİRİMİ - HER HABER İÇİN
        await self.telegram.send_message(
            f"📰 <b>Haber Alındı</b>\n\n"
            f"<b>Kaynak:</b> {news.source}\n"
            f"<b>Impact:</b> {news.impact_level}\n"
            f"<b>Sentiment:</b> {news.sentiment_score:.2f}\n"
            f"<b>Coinler:</b> {', '.join(news.coins_mentioned) if news.coins_mentioned else 'Bulunamadı'}\n"
            f"<b>Haber:</b> {news.title[:80]}..."
        )

        # ✅ BOT BİLDİRİM KONTROLÜ
        if any(keyword in news.title for keyword in [
            "🚀 AEnews Trading vFINAL",
            "💰 CANLI İŞLEM", 
            "🚫 Blacklist",
            "🛡️ İşlem Redde",
            "🔔 AE HABER BİLDİRİMİ"
        ]):
            logger.info(f"🔄 Bot bildirimi atlandı: {news.title[:30]}...")
            return

        # ✅ DUPLICATE HABER KONTROLÜ
        news_fingerprint = f"{news.source}_{news.title[:100]}_{sorted(news.coins_mentioned)}"
        current_time = datetime.now()
        
        # Eski haberleri temizle (5 dakikadan eski)
        for recent_fingerprint, timestamp in list(self.recent_news):
            if (current_time - timestamp).total_seconds() > 300:  # 5 dakika
                self.recent_news.remove((recent_fingerprint, timestamp))
        
        # Aynı haber daha önce işlendi mi?
        if any(news_fingerprint == fp for fp, _ in self.recent_news):
            logger.info(f"🔄 Duplicate haber atlandı: {news.title[:50]}...")
            await self.telegram.send_message("🔄 <b>Duplicate Haber:</b> Bu haber daha önce işlendi")
            return
        
        self.recent_news.append((news_fingerprint, current_time))

        if not news.coins_mentioned:
            await self.telegram.send_message("❌ <b>İşlem Yapılmadı:</b> Mesajda geçerli coin bulunamadı")
            return

        # ✅ MARKET DOĞRULAMA ve ALIAS ÇEVİRME
        valid_actual_coins = []
        for candidate in news.coins_mentioned:
            if self.market_manager.is_valid_coin(candidate):
                actual_coin = self.market_manager.resolve_alias(candidate)
                if actual_coin not in valid_actual_coins:
                    valid_actual_coins.append(actual_coin)
        
        if not valid_actual_coins:
            await self.telegram.send_message(
                f"❌ <b>İşlem Yapılmadı:</b> Coin(ler) {self.exchange.id.upper()} borsasında bulunamadı\n"
                f"<b>Aranan Coinler:</b> {', '.join(news.coins_mentioned)}"
            )
            return
        
        news.coins_mentioned = valid_actual_coins
        
        # ✅ GEÇERLİ COİNLER BULUNDU BİLDİRİMİ
        await self.telegram.send_message(
            f"✅ <b>Analiz Başlatıldı</b>\n\n"
            f"<b>Geçerli Coinler:</b> {', '.join(valid_actual_coins)}\n"
            f"<b>İşlem Sayısı:</b> {len(valid_actual_coins)}\n"
            f"<b>Impact Level:</b> {news.impact_level}"
        )
        
        # ✅ HER GEÇERLİ COİN İÇİN İŞLEM DENEMESİ
        for coin in news.coins_mentioned:
            try:
                logger.info(f"--- '{coin}' için işlem kontrolü başlıyor ---")
                symbol = f"{coin.upper()}USDT"
                # symbol = self._convert_symbol_for_gateio(symbol)
                # Yeni kod:
                if self.exchange.id == 'gateio':
                    symbol = f"{coin}_USDT"
                else:
                    symbol = f"{coin}USDT"
                
                # ✅ İŞLEM BAŞLAMA BİLDİRİMİ
                await self.telegram.send_message(
                    f"🔍 <b>Analiz Ediliyor</b>\n\n"
                    f"<b>Coin:</b> {coin}\n"
                    f"<b>Symbol:</b> {symbol}\n"
                    f"<b>Durum:</b> Piyasa verisi alınıyor..."
                )
                
                # ✅ DUPLICATE POZİSYON KONTROLÜ
                if not self.paper_mode and symbol in self.live_positions:
                    logger.info(f"❌ [{symbol}] için zaten açık pozisyon var, yeni işlem iptal")
                    await self.telegram.send_message(
                        f"⚠️ <b>Duplicate Pozisyon Engeli</b>\n\n"
                        f"<b>Coin:</b> {coin}\n"
                        f"<b>Durum:</b> Zaten açık pozisyon var\n"
                        f"<b>Sembol:</b> {symbol}"
                    )
                    continue
                
                # ✅ MARKET VERİSİ AL
                market_data = await self.get_market_data(symbol)
                current_price, klines = market_data
                if not current_price or not klines:
                    await self.telegram.send_message(
                        f"❌ <b>Veri Hatası</b>\n\n"
                        f"<b>Coin:</b> {coin}\n"
                        f"<b>Sorun:</b> Fiyat verisi alınamadı\n"
                        f"<b>Symbol:</b> {symbol}"
                    )
                    continue
                
                # ✅ SİNYAL ÜRET
                volatility = self.analyzer.calculate_atr(klines)
                signal = self.strategy.generate_signal(news, current_price, volatility)
                
                if not signal:
                    # ✅ DETAYLI REJECTİON BİLDİRİMİ
                    base_confidence = abs(news.sentiment_score)
                    impact_bonus = {"ULTRA_HIGH": 0.4, "HIGH": 0.25, "MEDIUM": 0.1, "LOW": 0.0}
                    final_confidence = base_confidence + impact_bonus.get(news.impact_level, 0)
                    
                    await self.telegram.send_message(
                        f"📉 <b>Sinyal Reddedildi</b>\n\n"
                        f"<b>Coin:</b> {coin}\n"
                        f"<b>Fiyat:</b> ${current_price:,.4f}\n"
                        f"<b>Sebep:</b> Düşük güven skoru\n"
                        f"<b>Sentiment:</b> {news.sentiment_score:.2f}\n"
                        f"<b>Impact:</b> {news.impact_level}\n"
                        f"<b>Final Güven:</b> {final_confidence:.2f}\n"
                        f"<b>Minimum:</b> {self.strategy.min_confidence:.2f}\n"
                        f"<b>Volatilite:</b> {volatility:.2f}%"
                    )
                    continue
                
                # ✅ SİNYAL BAŞARILI BİLDİRİMİ
                await self.telegram.send_message(
                    f"✅ <b>Sinyal Üretildi</b>\n\n"
                    f"<b>Coin:</b> {coin}\n"
                    f"<b>Yön:</b> {signal.action}\n"
                    f"<b>Güven:</b> {signal.confidence:.2f}\n"
                    f"<b>Giriş:</b> ${signal.entry_price:,.4f}\n"
                    f"<b>Stop Loss:</b> ${signal.stop_loss:,.4f}\n"
                    f"<b>Take Profit:</b> ${signal.take_profit:,.4f}\n"
                    f"<b>Durum:</b> Risk kontrolleri yapılıyor..."
                )
                
                # ✅ KORELASYON FİLTRESİ
                self.strategy.current_signal_side = signal.action
                is_safe_to_add = await self.portfolio_risk_manager.check_correlation_risk(symbol)
                if not is_safe_to_add:
                    continue

                signal.symbol = symbol
                signal.volatility = volatility
                
                # ✅ İŞLEM AÇ
                if self.paper_mode:
                    if self.risk_manager.can_open_position(len(self.paper_engine.positions)):
                        position = self.paper_engine.execute_trade(signal, current_price, news.title)
                        if position: 
                            await self.telegram.notify_live_trade_opened(position, signal, news.title)
                    else:
                        await self.telegram.send_message(
                            f"🛡️ <b>Risk Engeli</b>\n\n"
                            f"<b>Coin:</b> {coin}\n"
                            f"<b>Sebep:</b> Maksimum pozisyon limitine ulaşıldı"
                        )
                        break
                else: # Live Trading
                    if self.risk_manager.can_open_position(len(self.live_positions)):
                        await self.execute_live_trade(signal, news.title)
                    else:
                        await self.telegram.send_message(
                            f"🛡️ <b>Risk Engeli</b>\n\n"
                            f"<b>Coin:</b> {coin}\n"
                            f"<b>Sebep:</b> Maksimum pozisyon limitine ulaşıldı"
                        )
                        break
                            
            except Exception as e:
                await self.telegram.send_message(
                    f"❌ <b>İşlem Hatası</b>\n\n"
                    f"<b>Coin:</b> {coin}\n"
                    f"<b>Hata:</b> {str(e)[:100]}..."
                )
                logger.error(f"'{coin}' için işlem döngüsünde hata: {e}", exc_info=True)    

    async def attempt_trade(self, news: NewsItem, coin: str):
        symbol = f"{coin}/USDT:USDT"
        
        async with self.processing_lock:
            if symbol in self.currently_processing:
                logger.warning(f"[{symbol}] için zaten bir işlem süreci devam ediyor. Bu yeni sinyal atlandı.")
                return
            self.currently_processing.add(symbol)

        try:
            logger.info(f"--- '{coin}' için işlem kontrolü başlıyor (Kilit Aktif) ---")

            # ✅ BU SATIRLARI EKLE - BLACKLIST KONTROLÜ
            blacklist = self.config.get('trading_strategy.coin_blacklist', [])
            # ✅ DEBUG KODU EKLE
            logger.info(f"🔍 Telegram debug: is_configured={self.telegram.is_configured}")
            logger.info(f"🔍 Blacklist kontrol: {coin} in {blacklist} = {coin in blacklist}")

            if coin in blacklist:
                logger.info(f"🚫 {coin} blacklist'te olduğu için atlandı")
                
                # Debug: Telegram gönderimi öncesi
                logger.info("📤 Telegram bildirimi gönderiliyor...")
                
                await self.telegram.send_message(
                    f"🚫 <b>Blacklist Engeli</b>\n\n"
                    f"<b>Coin:</b> {coin}\n"
                    f"<b>Durum:</b> İşlem dışı bırakıldı"
                )
                
                # Debug: Telegram gönderimi sonrası
                logger.info("✅ Telegram bildirimi gönderildi")
                return
            # ✅ BLACKLIST KONTROLÜ BİTTİ

            # ADIM 2: PİYASA VERİSİNİ AL
            market_data = await self.get_market_data(symbol)
            if not market_data:
                await self.telegram.notify_signal_rejected(news.title, coin, "price_error")
                return # try...finally bloğu yine de çalışacak

            current_price, klines = market_data
            
            # ADIM 3: SİNYAL ÜRET
            volatility_atr = self.analyzer.calculate_atr(klines)
            if volatility_atr is None:
                logger.warning(f"[{symbol}] için ATR hesaplanamadı, alternatif kullanılıyor.")
                volatility_atr = current_price * 0.01  # Fiyatın %1'i

            signal = self.strategy.generate_signal(news, current_price, volatility_atr)
            if not signal:
                # Düşük güven skoru gibi nedenlerle sinyal üretilmedi.
                return
            signal.symbol = symbol

            # ADIM 4: FİLTRELERİ UYGULA
            open_pos_count = len(self.paper_engine.positions) if self.paper_mode else len(self.live_positions)
            if not self.risk_manager.can_open_position(open_pos_count):
                await self.telegram.notify_signal_rejected(news.title, coin, "risk_manager_decline")
                return

            if not await self.portfolio_risk_manager.check_correlation_risk(symbol):
                return

            # ADIM 5: İŞLEMİ GERÇEKLEŞTİR
            if self.paper_mode:
                pos = self.paper_engine.execute_trade(signal, current_price, news)
                if pos:
                    await self.telegram.notify_trade_opened(pos, signal, news.title)
            else:
                # execute_live_trade fonksiyonuna artık doğru formatlı sembol gidecek.
                await self.execute_live_trade(signal, news)
        
        finally:
            # --- ADIM 6: KİLİDİ SERBEST BIRAK ---
            # İşlem başarıyla tamamlansa da, bir hata nedeniyle yarıda kesilse de
            # bu "finally" bloğu her zaman çalışır.
            async with self.processing_lock:
                if symbol in self.currently_processing:
                    self.currently_processing.remove(symbol)
            logger.info(f"--- '{coin}' için işlem süreci tamamlandı (Kilit Serbest) ---")
            # ------------------------------------

    async def execute_live_trade(self, signal: TradeSignal, news_item: NewsItem):
        symbol = signal.symbol
        try:
            logger.info(f"--- CANLI İŞLEM BAŞLATILIYOR (V5 Mantığı): {signal.action} {symbol} ---")
            
            # Pozisyon büyüklüğünü V10'daki dinamik fonksiyonla hesapla
            amount = await self.calculate_live_position_size(symbol, signal.entry_price, signal.stop_loss)
            if not amount or amount <= 0:
                logger.warning(f"[{symbol}] Pozisyon büyüklüğü sıfır veya negatif. İşlem iptal.")
                return

            # ✅ MARGIN VALIDATION BEFORE TRADE
            margin_valid, margin_message = await self.validate_margin_before_trade(symbol, amount, signal.entry_price)
            if not margin_valid:
                logger.error(f"❌ [{symbol}] Margin validation failed: {margin_message}")
                await self.telegram.send_message(f"💸 <b>Margin Error</b>\n<b>Coin:</b> {symbol}\n<b>Reason:</b> {margin_message}")
                
                # Try with reduced position size (50% of original)
                reduced_amount = amount * 0.5
                margin_valid_reduced, margin_message_reduced = await self.validate_margin_before_trade(symbol, reduced_amount, signal.entry_price)
                if margin_valid_reduced:
                    logger.info(f"✅ [{symbol}] Using reduced position size: {reduced_amount}")
                    amount = reduced_amount
                else:
                    logger.error(f"❌ [{symbol}] Even reduced position size fails margin check. Skipping trade.")
                    return

            # ANA PİYASA EMRİNİ GÖNDER
            logger.info(f"[{symbol}] Ana piyasa emri gönderiliyor: {amount} kontrat")
            
            # Kaldıraç parametresini ekleyerek V10'daki hatayı çözmeye çalışalım
            params = {'settle': 'usdt', 'leverage': '10'}

            order = await self.loop.run_in_executor(
                None,
                lambda: self.exchange.create_market_order(symbol, signal.action.lower(), amount, params)
            )

            # Başarılı olursa, SL/TP kurma görevini başlat
            if order and order.get('filled'):
                filled_amount = float(order['filled'])
                entry_price = float(order.get('average', signal.entry_price))
                logger.info(f"✅ [{symbol}] ANA EMİR BAŞARILI: {filled_amount} @ ${entry_price:.6f}")
                
                # SL/TP kurma ve pozisyonu kaydetme işini ayrı bir fonksiyona devret
                await self.setup_position_protections(signal, order, news_item)
            else:
                logger.error(f"❌ [{symbol}] Ana emir gerçekleşmedi: {order}")

        except ccxt.InsufficientFunds as e:
            logger.error(f"💸 [{symbol}] Yetersiz bakiye: {e}")
            await self.telegram.send_message(f"💸 <b>Yetersiz Bakiye</b>\n<b>Coin:</b> {symbol}\n<code>{e}</code>")
        except Exception as e:
            logger.critical(f"❌ [{symbol}] execute_live_trade kritik hatası: {e}", exc_info=True)

    async def setup_position_protections(self, signal: TradeSignal, order: dict, news_item: NewsItem):
        symbol = signal.symbol
        filled_amount = float(order['filled'])
        entry_price = float(order.get('average', signal.entry_price))
        sl_order_id, tp_order_id = None, None
        
        try:
            # SL/TP emirlerini dene
            sl_order = await self.create_conditional_order(
                symbol=symbol, side=signal.action, order_type="STOP-LOSS",
                amount=filled_amount, trigger_price=signal.stop_loss
            )
            if sl_order: 
                sl_order_id = sl_order['id']
                logger.info(f"✅ [{symbol}] Stop Loss API ile kuruldu")

            tp_order = await self.create_conditional_order(
                symbol=symbol, side=signal.action, order_type="TAKE-PROFIT",
                amount=filled_amount, trigger_price=signal.take_profit
            )
            if tp_order: 
                tp_order_id = tp_order['id']
                logger.info(f"✅ [{symbol}] Take Profit API ile kuruldu")

            # API emirleri başarısızsa manuel takibi devreye sok
            if not sl_order_id:
                logger.warning(f"⚠️ [{symbol}] API Stop Loss başarısız, manuel takip devrede")
            
            # Pozisyonu her durumda kaydet (manuel takip varken acil kapatma yok)
            current_time = datetime.utcnow()
            position = Position(
                symbol=symbol, side=signal.action, size=filled_amount, entry_price=entry_price,
                current_price=entry_price, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                pnl=0.0, timestamp=current_time, exchange=self.exchange.id, volatility=signal.volatility,
                sl_order_id=sl_order_id, tp_order_id=tp_order_id, news_item=news_item,
                highest_price_seen=entry_price, lowest_price_seen=entry_price, trailing_stop=signal.stop_loss,
                created_timestamp=current_time, opening_order_id=order.get('id')
            )
            # ✅ DATABASE'E KAYDET
            if self.database:
                try:
                    trade_id = self.database.save_trade(position, news_item)
                    position.trade_id = trade_id  # Position'a trade ID'yi ekle
                except Exception as e:
                    logger.error(f"❌ Position database kaydı hatası: {e}")
            
            self.live_positions[symbol] = position
            await self.telegram.notify_trade_opened(position, signal, news_item.title)
            logger.info(f"✅ [{symbol}] Pozisyon kaydedildi, koruma sistemi aktif")

        except Exception as e:
            logger.error(f"❌ [{symbol}] setup_position_protections hatası: {e}")

    async def update_live_positions(self):
        """Açık pozisyonları yönetir ve hayalet pozisyonları temizler - 180 saniye grace period ile."""
        if not self.live_positions: 
            return
        
        try:
            exchange = self.exchange
            params = {}
            if exchange.id == 'gateio': 
                params = {'settle': 'usdt'}
            
            # API'den gerçek pozisyonları al
            positions_from_api = await self.loop.run_in_executor(None, exchange.fetch_positions, [], params)
            active_symbols_on_api = {pos['symbol'] for pos in positions_from_api if float(pos.get('contracts', 0)) > 0}
            
            # Kapatılan pozisyonları tespit et
            for position_key, position in list(self.live_positions.items()):
                
                # ✅ ENHANCED PHANTOM POSITION CHECK WITH GRACE PERIOD
                if position.symbol not in active_symbols_on_api:
                    # Check if grace period has passed (180 seconds = 3 minutes)
                    position_age = (datetime.utcnow() - position.created_timestamp).total_seconds() if position.created_timestamp else 0
                    grace_period = 180  # 3 minutes
                    
                    if position_age < grace_period:
                        logger.info(f"🕐 [{position.symbol}] Position not found in API but within grace period ({position_age:.0f}s < {grace_period}s)")
                        
                        # Try robust verification with exponential backoff
                        position_exists, verification_result = await self.check_position_with_strong_retry(
                            position.symbol, 
                            position.opening_order_id
                        )
                        
                        if position_exists:
                            logger.info(f"✅ [{position.symbol}] Position verified after retry: {verification_result}")
                            continue
                        else:
                            logger.warning(f"⚠️ [{position.symbol}] Position not confirmed after retries: {verification_result}")
                            # Continue to phantom position cleanup only if verification failed
                    else:
                        logger.info(f"⏰ [{position.symbol}] Grace period expired ({position_age:.0f}s >= {grace_period}s)")
                    
                    # Phantom position cleanup
                    logger.info(f"🔄 [{position.symbol}] Removing phantom position")
                    
                    # Son fiyatı al ve PnL hesapla
                    current_price = await self.get_current_price(position.symbol)
                    if current_price:
                        position.current_price = current_price
                        if position.side.upper() == "BUY":
                            position.pnl = (current_price - position.entry_price) * position.size
                        else:
                            position.pnl = (position.entry_price - current_price) * position.size
                    
                    # Pozisyonu kaldır ve bildir
                    del self.live_positions[position_key]
                    await self.telegram.notify_trade_closed(position, "PHANTOM_POSITION_CLEANUP")
                    await self.csv_logger.log_trade(position, "PHANTOM_POSITION_CLEANUP", position.news_item)
                    
                    logger.info(f"✅ [{position.symbol}] Phantom position cleaned up and logged")
                    continue

                # Normal pozisyon güncellemesi
                current_price = await self.get_current_price(position.symbol)
                if not current_price: 
                    continue
                    
                position.current_price = current_price
                await self.manage_trailing_stop(position)

        except Exception as e:
            logger.error(f"Canlı pozisyonları yönetirken hata: {e}", exc_info=True)

    async def manual_stop_loss_monitor(self):
        """
        Manuel stop loss takip sistemi - API emirleri çalışmazsa backup olarak çalışır.
        Her 2 saniyede bir tüm açık pozisyonları kontrol eder.
        """
        logger.info("🛡️ Manuel Stop Loss takip sistemi başlatıldı")
        
        while self.is_running:
            try:
                # Paper mode için çalışmaz, sadece live positions için
                if self.paper_mode or not self.live_positions:
                    await asyncio.sleep(5)
                    continue
                
                # Tüm açık pozisyonları kontrol et
                positions_to_close = []
                
                for position_key, position in list(self.live_positions.items()):
                    try:
                        # Güncel fiyatı al
                        current_price = await self.get_current_price(position.symbol)
                        if not current_price:
                            continue
                            
                        position.current_price = current_price
                        
                        # Stop loss tetiklenme kontrolü
                        should_close_sl = False
                        should_close_tp = False
                        
                        if position.side.upper() == "BUY":
                            # Long pozisyon kontrolleri
                            if current_price <= position.stop_loss:
                                should_close_sl = True
                            elif current_price >= position.take_profit and not position.trailing_stop_activated:
                                should_close_tp = True
                                
                        else:  # SHORT
                            # Short pozisyon kontrolleri
                            if current_price >= position.stop_loss:
                                should_close_sl = True
                            elif current_price <= position.take_profit and not position.trailing_stop_activated:
                                should_close_tp = True
                        
                        # Kapatılacak pozisyonları listeye ekle
                        if should_close_sl:
                            positions_to_close.append((position_key, position, "MANUAL_STOP_LOSS"))
                        elif should_close_tp:
                            # TP durumunda trailing stop aktifleştir
                            await self.activate_trailing_stop(position)
                            
                    except Exception as e:
                        logger.error(f"Manuel SL kontrol hatası ({position_key}): {e}")
                        continue
                
                # Belirlenen pozisyonları kapat
                for position_key, position, reason in positions_to_close:
                    await self.close_position_manually(position, reason)
                
                # 2 saniye bekle ve tekrar kontrol et
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                logger.info("🛡️ Manuel Stop Loss takip sistemi durduruluyor")
                break
            except Exception as e:
                logger.error(f"Manuel SL monitor kritik hatası: {e}")
                await asyncio.sleep(5)

    async def close_position_manually(self, position: Position, reason: str):
        """
        Pozisyonu manuel olarak kapatır - API emirleri çalışmadığında kullanılır.
        """
        symbol = position.symbol
        try:
            logger.info(f"🔄 [{symbol}] Manuel kapatma başlatılıyor: {reason}")
            
            # Kapatma emrinin yönünü belirle
            close_side = 'sell' if position.side.upper() == 'BUY' else 'buy'
            
            # Gate.io için parametreler
            params = {
                'settle': 'usdt', 
                'reduceOnly': True  # Pozisyonu kapatmak için
            }
            
            # Market emri ile pozisyonu kapat
            close_order = await self.loop.run_in_executor(
                None,
                lambda: self.exchange.create_market_order(
                    symbol, 
                    close_side, 
                    position.size, 
                    params=params
                )
            )
            
            if close_order and close_order.get('filled'):
                # PnL hesapla
                exit_price = float(close_order.get('average', position.current_price))

                 # ✅ DATABASE'DE KAPAT
                if self.database:
                    try:
                        await self.loop.run_in_executor(
                            None, 
                            self.database.close_trade, 
                            position.symbol, 
                            exit_price, 
                            reason
                        )
                    except Exception as e:
                        logger.error(f"❌ Database position kapatma hatası: {e}")
                
                if position.side.upper() == "BUY":
                    position.pnl = (exit_price - position.entry_price) * position.size
                else:
                    position.pnl = (position.entry_price - exit_price) * position.size
                
                position.current_price = exit_price
                
                # Pozisyonu listeden çıkar
                if symbol in self.live_positions:
                    del self.live_positions[symbol]
                
                # Bildirimleri gönder
                logger.info(f"✅ [{symbol}] Manuel kapatma başarılı: {reason} | PnL: ${position.pnl:.2f}")
                await self.telegram.notify_trade_closed(position, reason)
                await self.csv_logger.log_trade(position, reason, position.news_item)
                
            else:
                logger.error(f"❌ [{symbol}] Manuel kapatma emri gerçekleşmedi: {close_order}")
                
        except Exception as e:
            logger.error(f"❌ [{symbol}] Manuel pozisyon kapatma hatası: {e}")
            # Kritik hata durumunda telegram'a bildir
            await self.telegram.send_message(
                f"🚨 <b>Kritik: Manuel Kapatma Hatası!</b>\n"
                f"<b>Coin:</b> {symbol}\n"
                f"<b>Sebep:</b> {reason}\n"
                f"<b>Hata:</b> {str(e)[:100]}...\n"
                f"<b>Durum:</b> Manuel kontrol gerekiyor!"
            )

    async def get_futures_balance(self) -> Optional[float]:
        try:
            params = {'type': 'swap'}
            if self.exchange.id == 'gateio': 
                params = {'settle': 'usdt'}
            
            async with self.exchange_api_semaphore:
                balance_data = await self.loop.run_in_executor(None, lambda: self.exchange.fetch_balance(params=params))
            
            available_balance = balance_data.get('USDT', {}).get('free')
            if available_balance is not None:
                return float(available_balance)
            return None
        except Exception as e:
            logger.error(f"Bakiye alınamadı: {e}")
            return None

    async def validate_margin_before_trade(self, symbol: str, position_size: float, price: float) -> Tuple[bool, str]:
        """Validate sufficient margin before opening position"""
        try:
            balance = await self.get_futures_balance()
            if balance is None:
                logger.warning(f"Margin validation failed: Could not get balance for {symbol}")
                return False, "Could not get balance"
            
            free_usdt = balance
            
            required_margin = position_size * price * 0.1  # 10% margin requirement
            
            if free_usdt < required_margin:
                logger.warning(f"Insufficient margin for {symbol}: Need {required_margin:.2f} USDT, Have {free_usdt:.2f} USDT")
                return False, f"Insufficient margin"
                
            return True, "Margin OK"
        except Exception as e:
            logger.error(f"Margin check failed for {symbol}: {e}")
            return False, f"Margin check error: {e}"

    async def get_market_data(self, symbol: str) -> Optional[Tuple[float, List]]:
        try:
            async with self.exchange_api_semaphore:
                price_ticker, klines_data = await asyncio.gather(
                    self.loop.run_in_executor(None, lambda: self.exchange.fetch_ticker(symbol)),
                    self.loop.run_in_executor(None, lambda: self.exchange.fetch_ohlcv(symbol, '5m', limit=150))
                )
            
            if price_ticker and 'last' in price_ticker and klines_data:
                return price_ticker['last'], klines_data
            return None, None
        except Exception as e:
            logger.error(f"Piyasa verisi alınırken hata ({symbol}): {e}")
            return None

    async def calculate_live_position_size(self, symbol: str, entry_price: float, stop_loss_price: float) -> Optional[float]:
        """
        Dinamik risk yönetimi uygulayan, LIQUIDATE_IMMEDIATELY hatasını çözen son versiyon.
        """
        exchange = self.exchange
        try:
            # 1. PARAMETRELERİ AL
            risk_percent = float(os.getenv('RISK_PERCENTAGE', 0.02))
            min_position_value_usd = float(os.getenv('MIN_POSITION_VALUE_USD', 5.0)) # .env'den minimum pozisyon değerini al, yoksa 5$ kullan
            
            balance = await self.get_futures_balance()
            if balance is None or balance < min_position_value_usd:
                logger.warning(f"[{symbol}] Bakiye (${balance:.2f}), minimum pozisyon değerinin (${min_position_value_usd}) altında.")
                return None

            # 2. RİSK TABANLI POZİSYON BÜYÜKLÜĞÜNÜ HESAPLA (USDT CİNSİNDEN)
            risk_amount_usd = balance * risk_percent
            price_diff = abs(entry_price - stop_loss_price)

            if price_diff <= 0:
                logger.warning(f"[{symbol}] Stop mesafesi sıfır veya negatif. İşlem atlanıyor.")
                return None
            
            # Risk yüzdesine göre pozisyonun nominal (kaldıraçsız) değeri
            position_value_by_risk = (risk_amount_usd / price_diff) * entry_price

            # 3. DİNAMİK AYARLAMA: POZİSYON DEĞERİNİ KONTROL ET
            final_position_value_usd = max(position_value_by_risk, min_position_value_usd)

            # Eğer bakiye bu pozisyonu 1x kaldıraçla bile açmaya yetmiyorsa, maksimum bakiye ile aç
            if final_position_value_usd > balance:
                final_position_value_usd = balance * 0.95 # Güvenlik payı
                logger.warning(f"[{symbol}] Hesaplanan pozisyon değeri bakiyeyi aşıyor. Değer bakiye ile sınırlandırıldı: ${final_position_value_usd:.2f}")

            # 4. SON KONTRAKT SAYISINI HESAPLA
            contracts = final_position_value_usd / entry_price
            
            # 5. BORSANIN MİN/MAX LİMİTLERİNE GÖRE AYARLA
            market = exchange.market(symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 1.0)
            max_amount = market.get('limits', {}).get('amount', {}).get('max', 1000000.0)

            final_contracts = max(min_amount, min(contracts, max_amount))
            final_contracts_precise = exchange.amount_to_precision(symbol, final_contracts)

            if float(final_contracts_precise) < min_amount:
                logger.warning(f"⚠️ [{symbol}] Son miktar ({final_contracts_precise}) minimumun ({min_amount}) altında. İşlem atlanıyor.")
                return None
            
            final_risk = (price_diff * float(final_contracts_precise))
            logger.info(f"✅ [{symbol}] Pozisyon Büyüklüğü: {final_contracts_precise} kontrat (Nominal Değer: ${final_position_value_usd:.2f}, Risk: ~${final_risk:.2f})")
            return float(final_contracts_precise)

        except Exception as e:
            logger.error(f"❌ [{symbol}] Pozisyon büyüklüğü hesaplanamadı: {e}", exc_info=True)
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Bir sembolün sadece anlık fiyatını alır."""
        try:
            ticker = await self.loop.run_in_executor(None, lambda: self.exchange.fetch_ticker(symbol))
            return ticker.get('last')
        except Exception: return None

    async def check_order_status(self, order_id: str, symbol: str) -> Optional[dict]:
        """Check order status by order ID"""
        try:
            async with self.exchange_api_semaphore:
                order = await self.loop.run_in_executor(None, lambda: self.exchange.fetch_order(order_id, symbol))
            return order
        except Exception as e:
            logger.warning(f"Could not fetch order {order_id} for {symbol}: {e}")
            return None

    async def check_position_with_strong_retry(self, symbol: str, order_id: Optional[str] = None, max_retries: int = 5) -> Tuple[bool, str]:
        """Enhanced position checking with exponential backoff and order ID verification"""
        for attempt in range(max_retries):
            try:
                # Check positions via API
                params = {}
                if self.exchange.id == 'gateio': 
                    params = {'settle': 'usdt'}
                
                positions = await self.loop.run_in_executor(None, lambda: self.exchange.fetch_positions([], params))
                position_exists = any(float(pos.get('contracts', 0)) > 0 and pos['symbol'] == symbol for pos in positions)
                
                # If order_id provided, also check order history
                if order_id:
                    order_status = await self.check_order_status(order_id, symbol)
                    if order_status and order_status.get('status') == 'closed':
                        logger.info(f"Position confirmed via order {order_id} for {symbol}")
                        return True, "Position confirmed via order"
                
                if position_exists:
                    logger.info(f"Position found for {symbol} on attempt {attempt + 1}")
                    return True, "Position found"
                    
            except Exception as e:
                logger.warning(f"Position check attempt {attempt + 1} failed for {symbol}: {e}")
            
            # Exponential backoff: 2^attempt seconds
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry for {symbol}...")
                await asyncio.sleep(wait_time)
        
        logger.warning(f"Position not found for {symbol} after {max_retries} retries")
        return False, "Position not found after all retries"

    async def force_update_coin_list(self):
        """Coin listesini zorla güncelle"""
        try:
            logger.info("🔄 Coin listesi zorla güncelleniyor...")
            
            if self.market_manager:
                # Cache dosyasını sil
                import os
                cache_file = f"{self.exchange.id}_coins.json"
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info(f"🗑️ Cache dosyası silindi: {cache_file}")
                
                # Zorla güncelle
                self.market_manager.update_from_exchange()
                
                # DOGE kontrolü
                if self.market_manager.is_valid_coin('DOGE'):
                    logger.info("✅ DOGE artık geçerli!")
                else:
                    logger.warning("❌ DOGE hala geçersiz!")
                    
            else:
                logger.error("❌ Market manager bulunamadı!")
                
        except Exception as e:
            logger.error(f"❌ Zorla güncelleme hatası: {e}")
          
    async def modify_position_sl(self, position: Position, new_stop_price: float):
        """Bir pozisyonun stop-loss emrini borsada günceller."""
        if not position.sl_order_id:
            logger.warning(f"[{position.symbol}] SL emri güncellenemedi: SL order ID mevcut değil.")
            return

        symbol = position.symbol
        exchange = self.exchange

        try:
            # Fiyatı borsanın hassasiyetine göre formatla
            formatted_price = exchange.price_to_precision(symbol, new_stop_price)

            logger.info(f"⚙️ [{symbol}] Trailing Stop emri güncelleniyor: ${position.stop_loss:.4f} -> ${float(formatted_price):.4f}")
            
            # ccxt'nin evrensel edit_order metodunu kullan
            await self.loop.run_in_executor(None, 
                lambda: exchange.edit_order(
                    id=position.sl_order_id, 
                    symbol=symbol, 
                    type='stop', 
                    side='sell' if position.side == 'BUY' else 'buy', 
                    amount=position.size, 
                    price=formatted_price, 
                    params={'stopPrice': formatted_price, 'reduceOnly': True}
                )
            )
            
            # Başarılı olursa, pozisyon bilgilerini güncelle
            position.stop_loss = new_stop_price
            position.trailing_stop = new_stop_price
            
            await self.telegram.send_message(
                f"⚙️ <b>Trailing Stop Güncellendi</b>\n"
                f"<b>Coin:</b> {symbol}\n"
                f"<b>Yeni SL:</b> ${new_stop_price:,.4f}"
            )
        except ccxt.OrderNotFound:
             logger.error(f"[{symbol}] SL emri güncellenemedi: Emir ({position.sl_order_id}) borsada bulunamadı. Muhtemelen daha önce tetiklendi.")
             # Bu durumda pozisyonu senkronizasyon mekanizmasının kapatmasını bekleyebiliriz.
             position.sl_order_id = None # Artık geçersiz
        except Exception as e:
            logger.error(f"[{symbol}] SL emri güncellenirken hata oluştu: {e}", exc_info=True)

    async def manage_trailing_stop(self, position: Position):
        """
        Bir pozisyonun trailing stop (TS) mantığını yönetir.
        Gerekirse TS'yi aktive eder, günceller veya pozisyonu kapatır.
        """
        current_price = position.current_price
        
        # --- 1. Trailing Stop Aktivasyon Kontrolü ---
        # Fiyat orijinal TP'ye ulaştıysa ve TS henüz aktif değilse, aktifleştir.
        if not position.trailing_stop_activated:
            should_activate = (position.side == "BUY" and current_price >= position.take_profit) or \
                              (position.side == "SELL" and current_price <= position.take_profit)
            
            if should_activate:
                position.trailing_stop_activated = True
                position.highest_price_seen = current_price if position.side == "BUY" else position.entry_price
                position.lowest_price_seen = current_price if position.side == "SELL" else position.entry_price
                
                # Yeni TS seviyesini belirle (örn: %2 mesafe ile)
                new_ts_level = current_price * 0.98 if position.side == "BUY" else current_price * 1.02
                
                # Borsadaki SL emrini yeni TS seviyesine taşı
                await self.modify_position_sl(position, new_ts_level)
                
                await self.telegram.send_message(
                    f"🎯 <b>TP Hit & Trailing Aktif!</b>\n"
                    f"<b>Coin:</b> {position.symbol}\n"
                    f"<b>Yeni Trailing Stop:</b> ${new_ts_level:,.4f}\n"
                    f"🚀 Artık kâr koruma modunda!"
                )
                return # Bu döngü için işlem tamamlandı

        # --- 2. Aktif Trailing Stop'u Güncelleme ---
        if position.trailing_stop_activated:
            new_ts_level = None
            
            # LONG pozisyon için: Fiyat yeni zirve yaptıysa TS'yi yukarı çek
            if position.side == "BUY" and current_price > position.highest_price_seen:
                position.highest_price_seen = current_price
                new_ts_level = current_price * 0.98 # %2 mesafe
                if new_ts_level > position.trailing_stop: # Sadece yukarı taşınabilir
                    await self.modify_position_sl(position, new_ts_level)

            # SHORT pozisyon için: Fiyat yeni dip yaptıysa TS'yi aşağı çek
            if position.side == "SELL" and current_price < position.lowest_price_seen:
                position.lowest_price_seen = current_price
                new_ts_level = current_price * 1.02 # %2 mesafe
                if new_ts_level < position.trailing_stop: # Sadece aşağı taşınabilir
                    await self.modify_position_sl(position, new_ts_level)

    async def process_coin_from_news(self, coin: str, news: NewsItem):
        """Bir haberden gelen tek bir coini işler, analiz eder ve gerekirse işlem açar."""
        
        # ✅ EXCHANGE'E GÖRE SYMBOL FORMAT
        if self.exchange.id == 'gateio':
            symbol = f"{coin}_{self.exchange.options.get('settle', 'USDT')}"
        else:
            symbol = f"{coin}{self.exchange.options.get('settle', 'USDT')}"
        
        try:
            market_data = await self.get_market_data(symbol)
            if not market_data: return

            current_price, klines = market_data
            
            if not self.paper_mode:
                balance = await self.get_futures_balance()
                if not balance or balance < 5:
                    logger.warning(f"💸 Yetersiz bakiye: ${balance}")
                    return
                if current_price > 50000 and "BTC" in coin:
                    logger.info(f"💸 {coin} çok pahalı (${current_price:.0f}), atlanıyor")
                    return
                logger.info(f"✅ {coin} FİLTRELEME GEÇTİ: ${current_price:.4f}")

            volatility = self.analyzer.calculate_atr(klines)
            signal = self.strategy.generate_signal(news, current_price, volatility)
            if not signal: return
            
            signal.symbol = symbol
            logger.info(f"🎯 {coin} SİNYAL: {signal.action} @ ${current_price:.2f} | Güven: {signal.confidence:.2f}")
            
            open_pos_count = len(self.paper_engine.positions) if self.paper_mode else len(self.live_positions)
            if not self.risk_manager.can_open_position(open_pos_count):
                await self.telegram.notify_signal_rejected(news.title, coin, "risk_manager_decline")
                return
                    
            if not await self.portfolio_risk_manager.check_correlation_risk(symbol):
                return

            logger.info(f"💰 {coin} işlem açılıyor...")
            
            if self.paper_mode:
                pos = self.paper_engine.execute_trade(signal, current_price, news) # news_title yerine news gönder
                if pos: await self.telegram.notify_trade_opened(pos, signal, news.title)
            else:
                await self.execute_live_trade(signal, news)
            
        except Exception as e:
            logger.error(f"❌ {coin} işlem hatası: {e}", exc_info=True)

    def is_duplicate_news(self, news: NewsItem) -> bool:
        """BASİT duplicate kontrol - 10 satır"""
        try:
            if not news.coins_mentioned:
                return False
                
            now = datetime.now()
            
            # Her coin için son 3 dakikayı kontrol et
            for coin in news.coins_mentioned:
                if coin in self.recent_news:
                    last_time = self.recent_news[coin]
                    seconds_diff = (now - last_time).total_seconds()
                    
                    if seconds_diff < 180:  # 3 dakika
                        logger.info(f"🔄 {coin} duplicate (son: {seconds_diff:.0f}s)")
                        return True
                
                # Yeni kayıt
                self.recent_news[coin] = now
            
            # Eski kayıtları temizle (10 dakikadan eski)
            cutoff = now - timedelta(minutes=10)
            self.recent_news = {k: v for k, v in self.recent_news.items() if v > cutoff}
            
            return False
            
        except Exception as e:
            logger.debug(f"Duplicate kontrol hatası: {e}")
            return False 

    async def set_sl_tp_for_position(self, signal: TradeSignal, amount: float, news_title: str) -> Optional[Position]:
        """
        Önce ana emri gönderir, sonra SL/TP'yi ayarlar. Başarılı olursa Position nesnesi döndürür.
        Bu, 'FRHunter'dan gelen en güvenli yöntemdir.
        """
        symbol = signal.symbol
        exchange_id = self.exchange.id
        exchange = self.exchange
        
        # 1. Ana Pozisyon Açma Emri
        try:
            logger.info(f"Piyasa emri gönderiliyor: {signal.action} {amount:.4f} {symbol}")
            order = await self.loop.run_in_executor(
                None,
                lambda: exchange.create_market_order(symbol, signal.action.lower(), amount)
            )
            logger.info(f"✅ [{symbol}] ANA EMİR BAŞARILI: ID {order['id']}")
            
            # Gerçekleşen fiyat ve miktarı al
            entry_price = float(order.get('average', signal.entry_price))
            filled_amount = float(order.get('filled', amount))

        except Exception as e:
            logger.error(f"❌ [{symbol}] KRİTİK: ANA POZİSYON AÇMA BAŞARISIZ: {e}")
            await self.telegram.send_message(f"🔴 <b>Emir Hatası:</b> {symbol} pozisyonu AÇILAMADI.")
            return None # Pozisyon açılamadıysa devam etme

        # 2. SL/TP Emirlerini Ayarlama
        sl_order_id, tp_order_id = None, None
        try:
            # Emirleri ayrı ayrı, 'reduceOnly' parametresiyle gönder
            side = 'sell' if signal.action == 'BUY' else 'buy'
            params = {'reduceOnly': True}

            # Stop-Loss emri (Zorunlu)
            sl_order = await self.loop.run_in_executor(None, lambda: exchange.create_order(symbol, 'stop', side, filled_amount, price=signal.stop_loss, params={**params, 'stopPrice': signal.stop_loss}))
            sl_order_id = sl_order['id']
            logger.info(f"✅ [{symbol}] STOP-LOSS emri ayarlandı: ${signal.stop_loss:.4f}")

            # Take-Profit emri (Opsiyonel)
            try:
                tp_order = await self.loop.run_in_executor(None, lambda: exchange.create_order(symbol, 'take_profit', side, filled_amount, price=signal.take_profit, params={**params, 'stopPrice': signal.take_profit}))
                tp_order_id = tp_order['id']
                logger.info(f"✅ [{symbol}] TAKE-PROFIT emri ayarlandı: ${signal.take_profit:.4f}")
            except Exception as e:
                 logger.warning(f"⚠️ [{symbol}] TAKE-PROFIT ayarlanamadı (devam ediliyor): {e}")
        
        except Exception as e:
            logger.error(f"❌ [{symbol}] KRİTİK: STOP-LOSS AYARLANAMADI! Pozisyon riskli olabilir. Hata: {e}")
            await self.telegram.send_message(f"🚨 <b>GÜVENLİK UYARISI</b>\n\n<b>Coin:</b> {symbol}\n<b>Durum:</b> Pozisyon açıldı ama STOP-LOSS ayarlanamadı!\n<b>Risk:</b> Likidasyon riski var. Manuel kontrol edin.")
            # Stop-loss ayarlanamadıysa bile pozisyonu kaydetmeliyiz ki yönetebilelim.

        # 3. Başarılı Position Nesnesini Oluştur ve Döndür
        current_time = datetime.utcnow()
        position = Position(
            symbol=symbol, side=signal.action, size=filled_amount, entry_price=entry_price,
            current_price=entry_price, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
            pnl=0.0, timestamp=current_time, exchange=exchange_id, volatility=signal.volatility,
            trailing_stop=signal.stop_loss, highest_price_seen=entry_price, lowest_price_seen=entry_price,
            sl_order_id=sl_order_id, tp_order_id=tp_order_id,
            created_timestamp=current_time, opening_order_id=order['id']
        )
        await self.telegram.notify_live_trade_opened(position, signal, news_title)
        return position

    def check_final_close_conditions(self, position: Position) -> tuple[bool, str]:
        """Final kapanma kontrolleri - sadece SL ve Trailing Stop"""
        try:
            current_price = position.current_price
            
            if position.side.upper() == "BUY":
                # Long pozisyon kontrolleri
                
                # STOP LOSS kontrolü
                if current_price <= position.stop_loss:
                    return True, "STOP_LOSS_HIT"
                
                # TRAİLİNG STOP kontrolü (aktifse)
                if position.trailing_stop_activated and current_price <= position.trailing_stop:
                    profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    logger.info(f"📈 [{position.symbol}] Trailing Stop Hit! Final kâr: {profit_pct:.2f}%")
                    return True, "TRAILING_STOP_HIT"
                    
            else:
                # Short pozisyon kontrolleri
                
                # STOP LOSS kontrolü
                if current_price >= position.stop_loss:
                    return True, "STOP_LOSS_HIT"
                
                # TRAİLİNG STOP kontrolü (aktifse)
                if position.trailing_stop_activated and current_price >= position.trailing_stop:
                    profit_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                    logger.info(f"📉 [{position.symbol}] SHORT Trailing Stop Hit! Final kâr: {profit_pct:.2f}%")
                    return True, "TRAILING_STOP_HIT"
            
        except Exception as e:
            logger.error(f"Final close kontrol hatası: {e}")
            return False, ""

    async def check_position_close_conditions(self, position: Position) -> tuple[bool, str]:
        """Pozisyonun kapanma koşullarını kontrol eder - TRAİLİNG STOP aktivasyonu ile"""
        try:
            current_price = position.current_price
            
            if position.side.upper() == "BUY":
                # Long pozisyon kontrolleri
                
                # ✅ TAKE PROFIT HIT KONTROLÜ
                if current_price >= position.take_profit:
                    # TP hit oldu! Trailing stop'u aktifleştir
                    if not position.trailing_stop_activated:
                        position.trailing_stop_activated = True
                        position.trailing_stop = current_price * 0.98  # %2 trailing
                        position.highest_price_seen = current_price
                        
                        logger.info(f"🎯 [{position.symbol}] TAKE PROFIT HIT! Trailing Stop aktifleştirildi: ${position.trailing_stop:.4f}")
                        
                        # Telegram bildirimi
                        trailing_msg = (
                            f"🎯 <b>TAKE PROFIT HIT - TRAİLİNG AKTIF</b>\n\n"
                            f"<b>Coin:</b> {position.symbol.replace('/USDT:USDT', '')}\n"
                            f"<b>TP Fiyat:</b> ${position.take_profit:.4f}\n"
                            f"<b>Güncel:</b> ${current_price:.4f}\n"
                            f"<b>Trailing Stop:</b> ${position.trailing_stop:.4f} (%2)\n\n"
                            f"📈 <b>Artık kâr koruma modunda!</b>\n"
                            f"Fiyat yükselirse trailing stop takip edecek"
                        )
                        await self.telegram.send_message(trailing_msg)
                    
                    # TP hit olduğunda pozisyonu kapatma, trailing stop devreye girsin
                    return False, ""
                
                # ✅ STOP LOSS KONTROLÜ
                if current_price <= position.stop_loss:
                    return True, "STOP_LOSS_HIT"
                
                # ✅ TRAİLİNG STOP KONTROLÜ
                if position.trailing_stop_activated and current_price <= position.trailing_stop:
                    # Kâr ile trailing stop'a takıldı
                    profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    logger.info(f"📈 [{position.symbol}] Trailing Stop Hit! Kâr: {profit_pct:.2f}%")
                    return True, "TRAILING_STOP_HIT"
                    
            else:
                # Short pozisyon kontrolleri
                
                # ✅ TAKE PROFIT HIT KONTROLÜ (Short için)
                if current_price <= position.take_profit:
                    # TP hit oldu! Trailing stop'u aktifleştir
                    if not position.trailing_stop_activated:
                        position.trailing_stop_activated = True
                        position.trailing_stop = current_price * 1.02  # %2 trailing
                        position.lowest_price_seen = current_price
                        
                        logger.info(f"🎯 [{position.symbol}] SHORT TAKE PROFIT HIT! Trailing Stop aktifleştirildi: ${position.trailing_stop:.4f}")
                        
                        # Telegram bildirimi
                        trailing_msg = (
                            f"🎯 <b>SHORT TP HIT - TRAİLİNG AKTIF</b>\n\n"
                            f"<b>Coin:</b> {position.symbol.replace('/USDT:USDT', '')}\n"
                            f"<b>TP Fiyat:</b> ${position.take_profit:.4f}\n"
                            f"<b>Güncel:</b> ${current_price:.4f}\n"
                            f"<b>Trailing Stop:</b> ${position.trailing_stop:.4f} (%2)\n\n"
                            f"📉 <b>Artık kâr koruma modunda!</b>\n"
                            f"Fiyat düşerse trailing stop takip edecek"
                        )
                        await self.telegram.send_message(trailing_msg)
                    
                    return False, ""
                
                # ✅ STOP LOSS KONTROLÜ (Short için)
                if current_price >= position.stop_loss:
                    return True, "STOP_LOSS_HIT"
                
                # ✅ TRAİLİNG STOP KONTROLÜ (Short için)
                if position.trailing_stop_activated and current_price >= position.trailing_stop:
                    profit_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                    logger.info(f"📉 [{position.symbol}] SHORT Trailing Stop Hit! Kâr: {profit_pct:.2f}%")
                    return True, "TRAILING_STOP_HIT"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Pozisyon close kontrol hatası: {e}")
            return False, ""

    async def activate_trailing_stop(self, position: Position):
        """Take Profit hit olduğunda trailing stop'u aktifleştir"""
        try:
            symbol = position.symbol
            current_price = position.current_price
            
            logger.info(f"🎯 [{symbol}] TAKE PROFIT HIT - Trailing Stop aktifleştiriliyor...")
            
            # ✅ 1. TAKE PROFIT ORDER'INI İPTAL ET
            if position.tp_order_id:
                try:
                    logger.info(f"[{symbol}] TP Order iptal ediliyor: {position.tp_order_id}")
                    
                    if hasattr(self.exchange, 'id') and self.exchange.id == 'gateio':
                        await self.loop.run_in_executor(None, lambda: 
                            self.exchange.cancel_order(position.tp_order_id, symbol, {'settle': 'usdt'})
                        )
                    else:
                        await self.loop.run_in_executor(None, lambda: 
                            self.exchange.cancel_order(position.tp_order_id, symbol)
                        )
                    
                    logger.info(f"✅ [{symbol}] TP Order başarıyla iptal edildi")
                    position.tp_order_id = None  # Temizle
                    
                except Exception as e:
                    logger.warning(f"⚠️ [{symbol}] TP Order iptal hatası (devam edilir): {e}")
            
            # ✅ 2. TRAİLİNG STOP'U AKTİFLEŞTİR
            position.trailing_stop_activated = True
            
            if position.side.upper() == "BUY":
                # Long pozisyon için
                position.trailing_stop = current_price * 0.98  # %2 trailing
                position.highest_price_seen = current_price
                
            else:
                # Short pozisyon için
                position.trailing_stop = current_price * 1.02  # %2 trailing
                position.lowest_price_seen = current_price
            
            # ✅ 3. KÂRI HESAPLA
            if position.side.upper() == "BUY":
                profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            else:
                profit_pct = ((position.entry_price - current_price) / position.entry_price) * 100
            
            logger.info(f"📈 [{symbol}] Trailing Stop aktif! Kâr: {profit_pct:.2f}% | Trailing: ${position.trailing_stop:.4f}")
            
            # ✅ 4. TELEGRAM BİLDİRİMİ
            side_emoji = "📈" if position.side.upper() == "BUY" else "📉"
            trailing_msg = (
                f"🎯 <b>TAKE PROFIT HIT - TRAİLİNG AKTİF</b> {side_emoji}\n\n"
                f"<b>Coin:</b> {position.symbol.replace('/USDT:USDT', '')}\n"
                f"<b>Hedef:</b> ${position.take_profit:.4f} ✅\n"
                f"<b>Güncel:</b> ${current_price:.4f}\n"
                f"<b>Kâr:</b> {profit_pct:.2f}%\n\n"
                f"🛡️ <b>Trailing Stop:</b> ${position.trailing_stop:.4f} (%2)\n"
                f"🚀 <b>Artık kâr koruma modunda!</b>\n\n"
                f"💡 Fiyat daha da yükselirse trailing stop takip edecek"
            )
            await self.telegram.send_message(trailing_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [{position.symbol}] Trailing stop aktivasyon hatası: {e}")
            return False 

    async def get_symbol_info(self, symbol: str, exchange_name: str) -> Optional[Dict]:
        """
        Belirtilen borsadan bir sembolün hassasiyet ve limit bilgilerini alır ve cache'ler.
        """
        cache_key = f"{exchange_name}_{symbol}"
        if cache_key in self.symbol_info_cache:
            return self.symbol_info_cache[cache_key]
        
        try:
            exchange = self.exchange_manager.get_exchange(exchange_name)
            if not exchange: return None

            await self.loop.run_in_executor(None, exchange.load_markets)
            market = exchange.market(symbol)
            
            min_notional = market.get('limits', {}).get('cost', {}).get('min', 5.0)
            
            info = {
                'price_precision': market.get('precision', {}).get('price'),
                'amount_precision': market.get('precision', {}).get('amount'),
                'min_notional': float(min_notional) if min_notional is not None else 5.0
            }
            
            # Eğer precision bilgisi çekilemezse, bu bir sorundur.
            if info['price_precision'] is None or info['amount_precision'] is None:
                logger.error(f"[{symbol}] için precision bilgisi borsadan çekilemedi!")
                return None

            self.symbol_info_cache[cache_key] = info
            return info
        except Exception as e:
            logger.error(f"'{symbol}' için sembol bilgisi alınamadı ({exchange_name}): {e}")
            return None

    async def test_manual_trade(self, coin_symbol="BTC", sentiment=0.8):
        """Manuel test için trade trigger'ı."""
        logger.info(f"🧪 MANUEL TEST: {coin_symbol} için trade test ediliyor...")
        
        # Fake news oluştur
        fake_news = NewsItem(
            title=f"[MANUAL_TEST] {coin_symbol} Test Trade Signal",
            content=f"{coin_symbol} manual test signal generated",
            source="MANUAL_TEST",
            timestamp=datetime.now(),
            sentiment_score=sentiment,
            coins_mentioned=[coin_symbol],
            impact_level="HIGH",
            url=""
        )
        
        await self.process_news(fake_news)

    async def emergency_position_reject(self, symbol: str, reason: str):
        """Acil durum pozisyon reddi - güvenlik için"""
        try:
            logger.error(f"🚨 [{symbol}] POZİSYON REDDEDİLDİ: {reason}")
            
            # Telegram'a kritik uyarı
            emergency_msg = (
                f"🚨 <b>POZİSYON GÜVENLİK REDDİ</b>\n\n"
                f"<b>Coin:</b> {symbol.replace('/USDT:USDT', '')}\n"
                f"<b>Sebep:</b> {reason}\n"
                f"<b>Durum:</b> Stop-Loss ayarlanamadığı için pozisyon açılmadı\n"
                f"<b>Risk:</b> Liquidation koruması olmayacaktı\n\n"
                f"✅ <b>GÜVENLİK ÖNCELİKLİ - Pozisyon iptal edildi</b>"
            )
            
            if hasattr(self, 'telegram') and self.telegram.is_configured:
                await self.telegram.send_message(emergency_msg)
            
            # Eğer yanlışlıkla market order açıldıysa acil kapat
            # (Bu durumda amount bilgimiz yok, manuel kontrol gerekir)
            
        except Exception as e:
            logger.error(f"Emergency position reject hatası: {e}")

    async def update_paper_positions(self):
        """Paper pozisyon güncelleme - basitleştirilmiş versiyon"""
        if not self.paper_engine.positions:
            return
            
        try:
            # Symbol'ları topla
            symbols_to_fetch = list(set(pos.symbol for pos in self.paper_engine.positions.values()))
            
            # Fiyatları al
            prices = await asyncio.gather(
                *[self.get_current_price(symbol) for symbol in symbols_to_fetch],
                return_exceptions=True
            )
            
            # Fiyat dictionary'si oluştur
            current_prices = {}
            for symbol, price in zip(symbols_to_fetch, prices):
                if not isinstance(price, Exception) and price is not None:
                    current_prices[symbol] = price
            
            # ✅ Paper engine kendi pozisyonlarını güncelleyip kapatacak
            closed_positions = self.paper_engine.update_positions(current_prices)
            
            # ✅ Sadece kapatılan pozisyonlar için Telegram bildirimi gönder
            for key, pos, reason in closed_positions:
                await self.telegram.notify_trade_closed(pos, reason)
                logger.info(f"📉 {pos.symbol.replace('/USDT:USDT', '')} kapatıldı: {reason} | PnL: ${pos.pnl:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Paper pozisyon güncelleme hatası: {e}")

    async def execute_command(self, command: str):
        """Komut çalıştırır ve sonucu Telegram'a gönderir."""
        try:
            logger.info(f"🤖 Komut çalıştırılıyor: '{command}'")
            
            # Komutu, TelegramNotifier'daki asenkron handle_command'a gönder
            response_message = await self.telegram.handle_command(command, self)
            
            # handle_command'dan gelen cevabı gönder
            await self.telegram.send_message(response_message)
            logger.info(f"✅ Komut başarıyla tamamlandı: {command}")
            
        except Exception as e:
            logger.error(f"❌ Komut çalıştırma hatası ({command}): {e}", exc_info=True)
            await self.telegram.send_message(f"❌ Komut işlenirken bir hata oluştu: {command}")

    def get_command_help(self):
        """Kullanılabilir komutları listeler (terminal için)."""
        print("\n🤖 KULLANILABILIR KOMUTLAR:")
        print("=" * 50)
        
        for cmd, desc in self.telegram.commands.items():
            print(f"{cmd:<12} - {desc}")
        
        print("\n💡 KULLANIM:")
        print("await bot.execute_command('/positions')")
        print("await bot.execute_command('/stats')")
        print("await bot.execute_command('/balance')")
        print("await bot.execute_command('/help')")
        print("=" * 50)

    async def force_test_trade(self, coin="BTC"):
        """Zorla test trade tetikle - GARANTILI ÇALIŞIR"""
        logger.info(f"🚨 ZORLA TEST: {coin} için garantili sinyal")
        
        # Garantili pozitif haber
        test_news = NewsItem(
            title=f"[FORCE_TEST] {coin} added to Binance alpha projects - GUARANTEED SIGNAL",
            content=f"{coin} added to binance alpha projects with major breakthrough partnership",
            source="FORCE_TEST",
            timestamp=datetime.now(),
            sentiment_score=1.0,  # Maksimum pozitif
            coins_mentioned=[coin],
            impact_level="ULTRA_HIGH",
            url=""
        )
        
        # Haberi işle
        await self.process_news(test_news)
        logger.info(f"✅ {coin} zorla test tamamlandı!")

async def main():
    """
    Botu başlatan ve güvenli bir şekilde çalışmasını sağlayan ana asenkron fonksiyon.
    """
    bot = None
    try:
        # Adım 1: Botun tüm bileşenlerini __init__ ile kur.
        bot = CryptoNewsBot()
        
        # Adım 2: Borsa bağlantısı __init__ sırasında başarılı oldu mu diye kontrol et.
        if not bot.exchange:
            logger.critical("Borsa nesnesi oluşturulamadı. Lütfen .env dosyasını kontrol edin.")
            # Hata mesajını __init__ içinde gönderemediğimiz için burada gönderiyoruz.
            await bot.telegram.send_message("❌ <b>BOT BAŞLATILAMADI</b>\n\nBorsa bağlantısı kurulamadı (nesne oluşturulamadı).")
            return # Programı sonlandır

        # Adım 3: Borsa bağlantısını doğrula.
        is_connection_valid = await bot._validate_exchange_connection()
        if not is_connection_valid:
            logger.critical("Borsa bağlantısı doğrulanamadı. Bot başlatılmıyor.")
            # Hata mesajı _validate_exchange_connection içinde zaten gönderildi.
            return # Programı sonlandır

        # Adım 4: Her şey yolundaysa, botun ana çalışma döngüsünü başlat.
        await bot.start()

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Bot kapatma sinyali alındı...")
    except Exception as e:
        logger.critical(f"Ana programda kritik bir hata oluştu: {e}", exc_info=True)
    finally:
        # Her durumda (başarılı kapanış veya hata), botu güvenli bir şekilde durdur.
        if bot and bot.is_running:
            await bot.stop()

if __name__ == "__main__":
    print("="*50)
    print("🚀 AEnews Trading Bot vFINAL Başlatılıyor...")
    print(f"📝 Paper Trading Modu: {os.getenv('PAPER_TRADING', 'true')}")
    print(f"💰 Başlangıç Bakiye: ${os.getenv('INITIAL_BALANCE', '10000')}")
    print("="*50)
    
    try:
        # Ana asenkron fonksiyonu çalıştır.
        asyncio.run(main())
    except KeyboardInterrupt:
        # Bu, asyncio.run'ın içindeki KeyboardInterrupt'ı yakalamak için.
        pass
    finally:
        print("Program kapatıldı. Hoşçakal!")