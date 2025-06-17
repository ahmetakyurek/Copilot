# -*- coding: utf-8 -*-
# AI-Powered QuantumTrade Bot (TEMA+SuperTrend, ML Destekli, Full Çalışır)

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
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
import itertools
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# ===== KONFİGÜRASYON =====
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# EN İYİ PARAMETRELERİ OTOMATİK KULLAN
from best_params_config import BEST_PARAMS

# API anahtarlarının yüklenip yüklenmediğini kontrol et
if not all([BINANCE_API_KEY, BINANCE_API_SECRET]):
    logging.critical("Binance API anahtarları .env dosyasında bulunamadı veya yüklenemedi!")
    # Gerekirse burada programdan çıkış yapılabilir veya kullanıcıya bilgi verilebilir.
    # exit() 

IST_TIMEZONE = pytz.timezone("Europe/Istanbul")
COOLDOWN_MINUTES = 60
MODEL_UPDATE_INTERVAL = timedelta(hours=6)
N_FUTURE_CANDLES = 3
BASE_URL = "https://fapi.binance.com"

# Piyasa Yönü İçin Konfigürasyon
MARKET_DIRECTION_SYMBOLS = ['ETHUSDT'] # Sadece ETH veya ['BTCUSDT', 'ETHUSDT']
MARKET_DIRECTION_INTERVAL_FOR_BOT = '1h' # Botun canlıda kullanacağı zaman dilimi
MARKET_DIRECTION_LOOKBACK_FOR_BOT = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def get_ml_market_direction(
    df, model=None, scaler=None, features=None, retrain=False,
    lookforward=None, threshold=None, prob_long=None, prob_short=None
):
    import os
    logging.warning(f"[ML_DIR][DEBUG] get_ml_market_direction çalıştı, df.shape={df.shape}, cwd={os.getcwd()}")
    logging.warning(f"[ML_DIR][DEBUG] FONKSİYON BAŞI: df.shape={df.shape}, cwd={os.getcwd()}")
    logging.info(f"[ML_DIR] Fonksiyon çağrıldı. Son kapanış: {df['close'].iloc[-1] if not df.empty else 'EMPTY'}, df shape: {df.shape}")

    # Varsayılan parametreleri BEST_PARAMS'ten al
    lookforward = lookforward if lookforward is not None else BEST_PARAMS.get('lookforward', 8)
    threshold = threshold if threshold is not None else BEST_PARAMS.get('threshold', 0.003)
    prob_long = prob_long if prob_long is not None else BEST_PARAMS.get('prob_long', 0.52)
    prob_short = prob_short if prob_short is not None else BEST_PARAMS.get('prob_short', 0.48)

    logging.info(f"[ML_DIR] Parametreler: lookforward={lookforward}, threshold={threshold}, prob_long={prob_long}, prob_short={prob_short}")

    df_feat = ml_feature_engineering(df)
    features = features or [
        "ema_diff", "ema_cross_up", "ema_cross_down",
        "rsi", "macd", "macd_signal", "macd_cross_up", "macd_cross_down",
        "bb_mavg", "bb_upper", "bb_lower", "bb_breakout",
        "momentum_10", "roc_10", "atr14", "atr_pct", "adx",
        "volume_change", "vol_spike"
    ]
    df_feat = df_feat.dropna(subset=features)
    df_feat['target'] = np.nan
    # --- Otomatik analiz için max_up ve min_down istatistikleri ---
    logging.warning(f"[ML_DIR][DEBUG] Target analiz bloğu başlıyor, df_feat.shape={df_feat.shape}, cwd={os.getcwd()}")
    max_up_list = []
    min_down_list = []
    target_analysis_records = []
    for idx in range(len(df_feat) - lookforward):
        future_prices = df_feat["close"].iloc[idx+1:idx+lookforward+1].values
        future_pct_chg = (future_prices - df_feat["close"].iloc[idx]) / df_feat["close"].iloc[idx]
        max_up = np.max(future_pct_chg)
        min_down = np.min(future_pct_chg)
        target_val = np.nan
        if max_up >= threshold:
            target_val = 1
        elif min_down <= -threshold:
            target_val = 0
        df_feat.loc[idx, 'target'] = target_val
        target_analysis_records.append({
            'timestamp': df_feat.index[idx],
            'close': df_feat['close'].iloc[idx],
            'max_up': max_up,
            'min_down': min_down,
            'target': target_val
        })
        max_up_list.append(max_up)
        min_down_list.append(min_down)

    # --- Detaylı target analizi CSV'ye kaydet ---
    import pandas as pd
    df_target_analysis = pd.DataFrame(target_analysis_records)
    logging.warning(f"[ML_DIR][DEBUG] CSV kaydı başlıyor, df_target_analysis.shape={df_target_analysis.shape}, cwd={os.getcwd()}")
    try:
        df_target_analysis.to_csv('target_analysis_ETHUSDT.csv', index=False)
        logging.warning(f"[ML_DIR][DEBUG] CSV başarıyla kaydedildi, cwd={os.getcwd()}")
        logging.info(f"[ML_DIR][ANALİZ] target_analysis_ETHUSDT.csv dosyasına kaydedildi. Dosya dizini: {os.getcwd()}\nİlk 5 satır:\n{df_target_analysis.head()}\nSon 5 satır:\n{df_target_analysis.tail()}")
    except Exception as e:
        logging.error(f"[ML_DIR][ANALİZ] target_analysis_ETHUSDT.csv kaydedilemedi! Hata: {e}, Dizin: {os.getcwd()}")

    # Target dağılımı ve özet istatistikler
    target_counts = df_target_analysis['target'].value_counts(dropna=False)
    logging.info(f"[ML_DIR][ANALİZ] Target dağılımı: {target_counts.to_dict()}")
    logging.info(f"[ML_DIR][ANALİZ] max_up: min={np.min(max_up_list):.4f}, max={np.max(max_up_list):.4f}, mean={np.mean(max_up_list):.4f}, std={np.std(max_up_list):.4f}, 25%={np.percentile(max_up_list,25):.4f}, 50%={np.percentile(max_up_list,50):.4f}, 75%={np.percentile(max_up_list,75):.4f}")
    logging.info(f"[ML_DIR][ANALİZ] min_down: min={np.min(min_down_list):.4f}, max={np.max(min_down_list):.4f}, mean={np.mean(min_down_list):.4f}, std={np.std(min_down_list):.4f}, 25%={np.percentile(min_down_list,25):.4f}, 50%={np.percentile(min_down_list,50):.4f}, 75%={np.percentile(min_down_list,75):.4f}")

    # --- Teşhis logları ---
    n_feat_nan = df_feat[features].isnull().sum().sum()
    n_target_nan = df_feat['target'].isnull().sum()
    logging.info(f"[ML_DIR][TEŞHİS] Feature'larda toplam {n_feat_nan} NaN, target'ta {n_target_nan} NaN var.")
    if n_target_nan < len(df_feat):
        logging.info(f"[ML_DIR][TEŞHİS] Target dağılımı (NaN olmayanlar): {df_feat['target'].value_counts().to_dict()}")
    else:
        logging.warning(f"[ML_DIR][TEŞHİS] Target'ın tamamı NaN!")
    # Eğitim verisine kalan satır sayısı
    df_train = df_feat.dropna(subset=features + ['target'])
    logging.info(f"[ML_DIR][TEŞHİS] Eğitim verisi kalan satır: {len(df_train)}")
    if len(df_train) < 40:
        logging.warning(f"[ML_DIR] Eğitim için yeterli veri yok! df_train.shape={df_train.shape}, input df.shape={df.shape}")
        return 'SHORT', None, None, None
    X = df_train[features]
    y = df_train['target']
    # Target dağılımını logla (yalnızca ilk 1 çağrıda - spam olmasın diye)
    if hasattr(get_ml_market_direction, "_logged_target_dist") is False:
        logging.info(f"[ML_DIR] Target dağılımı: {y.value_counts().to_dict()}")
        get_ml_market_direction._logged_target_dist = True
    if scaler is None or retrain:
        scaler = StandardScaler()
        scaler.fit(X.values)
    X_scaled = scaler.transform(X.values)
    if model is None or retrain:
        model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=0.5, eval_metric="logloss")
        model.fit(X_scaled, y)
    X_last = scaler.transform([df_feat[features].iloc[-1].values])
    pred_prob = model.predict_proba(X_last)[0][1]
    direction = 'LONG' if pred_prob >= 0.5 else 'SHORT'
    try:
        last_close = df['close'].iloc[-1] if not df.empty else 'EMPTY'
        logging.info(f"[ML_DIR] Sonuç: direction={direction}, pred_prob={pred_prob:.4f}, last_close={last_close}")
    except Exception as e:
        logging.error(f"[ML_DIR] Sonuç loglanırken hata: {e}")
    return direction, pred_prob, model, scaler

# ========== Klasik Skor Bazlı Market Direction Predictor ==========
def get_classic_market_direction(
    df,
    ema_fast=None, ema_slow=None, rsi_period=None, rsi_ob=None, rsi_os=None,
    macd_fast=None, macd_slow=None, macd_signal=None, vwap=None, score_threshold=None
):
    # Varsayılan parametreleri BEST_PARAMS'ten al, ama sadece parametre verilmezse!
    ema_fast = ema_fast if ema_fast is not None else BEST_PARAMS.get('ema_short_period', 5)
    ema_slow = ema_slow if ema_slow is not None else BEST_PARAMS.get('ema_long_period', 20)
    rsi_period = rsi_period if rsi_period is not None else BEST_PARAMS.get('rsi_period', 14)
    rsi_ob = rsi_ob if rsi_ob is not None else BEST_PARAMS.get('rsi_ob_threshold', 70)
    rsi_os = rsi_os if rsi_os is not None else BEST_PARAMS.get('rsi_os_threshold', 40)
    macd_fast = macd_fast if macd_fast is not None else BEST_PARAMS.get('macd_fast_period', 12)
    macd_slow = macd_slow if macd_slow is not None else BEST_PARAMS.get('macd_slow_period', 26)
    macd_signal = macd_signal if macd_signal is not None else BEST_PARAMS.get('macd_signal_period', 9)
    vwap = vwap if vwap is not None else BEST_PARAMS.get('use_vwap', True)
    score_threshold = score_threshold if score_threshold is not None else BEST_PARAMS.get('score_threshold', 1)

    close = df['close']
    ema_f = ta.trend.ema_indicator(close, window=ema_fast, fillna=False)
    ema_s = ta.trend.ema_indicator(close, window=ema_slow, fillna=False)
    rsi = ta.momentum.rsi(close, window=rsi_period, fillna=False)
    macd_diff = ta.trend.macd_diff(close, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal, fillna=False)
    vwap_val = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], fillna=False) if vwap else pd.Series([np.nan]*len(df))
    idx = -1
    long_score = 0
    short_score = 0

    # EMA ağırlığı
    if ema_f.iloc[idx] > ema_s.iloc[idx]:
        long_score += 1
    else:
        short_score += 1

    # RSI ağırlığı
    if rsi.iloc[idx] > rsi_ob:
        short_score += 2
        print(f"[RSI DETAY] RSI={rsi.iloc[idx]:.2f} > rsi_ob={rsi_ob} (Aşırı ALIM) -> short_score += 2")
    elif rsi.iloc[idx] < rsi_os:
        long_score += 2
        print(f"[RSI DETAY] RSI={rsi.iloc[idx]:.2f} < rsi_os={rsi_os} (Aşırı SATIM) -> long_score += 2")

    # MACD ağırlığı
    if macd_diff.iloc[idx] > 0:
        long_score += 1
    else:
        short_score += 1

    # VWAP ağırlığı
    if vwap and close.iloc[idx] > vwap_val.iloc[idx]:
        long_score += 1
    elif vwap and close.iloc[idx] < vwap_val.iloc[idx]:
        short_score += 1

    print(f"[AĞIRLIKLI SKOR] EMA: {ema_f.iloc[idx]:.2f}/{ema_s.iloc[idx]:.2f}, RSI: {rsi.iloc[idx]:.2f} (rsi_ob={rsi_ob}, rsi_os={rsi_os}), MACD: {macd_diff.iloc[idx]:.4f}, VWAP: {vwap_val.iloc[idx]:.2f}, Fiyat: {close.iloc[idx]:.2f} | long_score: {long_score}, short_score: {short_score}, threshold: {score_threshold}")

    if long_score >= score_threshold:
        return "LONG"
    else:
        return "SHORT"

# ========== GRID SEARCH BACKTEST RUNNER ==========
async def gridsearch_market_direction(
    df,
    method_list=['classic', 'ml'],
    grid_params=None,
    lookforward=4,
    threshold=0.0075,
    future_window_hours=4,
    freq='1h',
    verbose=1,
    csv_out="market_direction_gridsearch_results.csv"
):
    """
    df: DataFrame (ohlcv) - mümkünse en az 1000 bar
    method_list: ['classic', 'ml']
    grid_params: dict, ör: {'ema_fast': [5, 10], 'ema_slow': [20, 40], ...}
    lookforward: Kaç bar sonrası gerçek yön bakılacak (ML için)
    threshold: Yüzde olarak bar sonrası fiyat değişim eşiği (ML için)
    freq: Tahminler hangi sıklıkta (örn: '1h')
    """
    if grid_params is None:
        grid_params = {
            'ema_fast': [5, 10],
            'ema_slow': [20, 30],
            'rsi_period': [14],
            'rsi_ob': [70],
            'rsi_os': [30],
            'macd_fast': [12],
            'macd_slow': [26],
            'macd_signal': [9],
            'vwap': [True],
            'score_threshold': [3, 4],
            'prob_long': [0.52],
            'prob_short': [0.48],
        }
    param_keys = list(grid_params.keys())
    param_vals = list(grid_params.values())
    combinations = list(itertools.product(*param_vals))
    print(f"Toplam {len(combinations)} parametre kombinasyonu test edilecek.")

    results = []
    for combo in combinations:
        params = dict(zip(param_keys, combo))
        for method in method_list:
            # Her kombine ayar için
            dt_range = pd.date_range(df.index[0], df.index[-1], freq=freq)
            pred_list = []
            actual_list = []
            total_cnt = 0
            correct_cnt = 0
            for dt in dt_range:
                # Her tahmin anı için
                idxs = df.index <= dt
                df_slice = df[idxs].copy()
                if len(df_slice) < 60:
                    continue
                actual_dir = "SHORT"
                # --- Gelişmiş pencere ve fiyat değişimi hesaplama (Türkçe açıklamalı) ---
                interval_minutes = 15  # 15m için sabit
                try:
                    idx_current = df.index.get_loc(dt)
                    idx_future = idx_current + int(future_window_hours * 60 / interval_minutes)
                    if idx_future >= len(df):
                        continue  # Yeterli veri yoksa geç
                    price_start = df["open"].iloc[idx_current]
                    price_end = df["close"].iloc[idx_future]
                    price_chg = (price_end - price_start) / price_start
                except Exception as e:
                    print(f"Hata (pencere hesaplama): {e}")
                    continue

                if price_chg > threshold:
                    actual_dir = "LONG"
                elif price_chg < -threshold:
                    actual_dir = "SHORT"
                else:
                    continue  # Belirsiz/yatay hareketleri atla

                # İlk 10 tahmin için detaylı log
                if total_cnt < 10:
                    print(f"dt: {dt}, idx_current: {idx_current}, idx_future: {idx_future}")
                    print(f"price_start: {price_start}, price_end: {price_end}, price_chg: {price_chg:.4f}, actual_dir: {actual_dir}")

                if method == 'classic':
                    pred_dir = get_classic_market_direction(
                        df_slice,
                        ema_fast=params['ema_fast'],
                        ema_slow=params['ema_slow'],
                        rsi_period=params['rsi_period'],
                        rsi_ob=params['rsi_ob'],
                        rsi_os=params['rsi_os'],
                        macd_fast=params['macd_fast'],
                        macd_slow=params['macd_slow'],
                        macd_signal=params['macd_signal'],
                        vwap=params['vwap'],
                        score_threshold=params['score_threshold']
                    )
                else:
                    pred_dir, prob, model, scaler = get_ml_market_direction(
                        df_slice,
                        lookforward=lookforward,
                        threshold=threshold,
                        prob_long=params['prob_long'],
                        prob_short=params['prob_short']
                    )
                # İlk 10 tahmin için tahmin edilen ve gerçek yönü logla
                if total_cnt < 10:
                    print(f"TAHMİN LOGU | dt: {dt}, predicted_dir: {pred_dir}, actual_dir: {actual_dir}")
                pred_list.append(pred_dir)
                actual_list.append(actual_dir)
                total_cnt += 1
                if pred_dir == actual_dir:
                    correct_cnt += 1
            acc = (correct_cnt / total_cnt) * 100 if total_cnt > 0 else 0
            result = params.copy()
            result['method'] = method
            result['accuracy'] = acc
            result['total'] = total_cnt
            result['symbol'] = MARKET_DIRECTION_SYMBOLS[0]  # coin bilgisi
            result['interval'] = MARKET_DIRECTION_INTERVAL_FOR_BOT  # timeframe bilgisi
            results.append(result)
            if verbose:
                print(f"Method: {method} Params: {params} Accuracy: {acc:.2f}% (n={total_cnt})")
            
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by="accuracy", ascending=False)
            df_results.to_csv(csv_out, index=False)
            print(f"\nGrid search sonuçları {csv_out} dosyasına kaydedildi.")
            print(df_results.head(10).to_string())
            return df_results

def ml_feature_engineering(df):
    df = df.copy()
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema_diff"] = df["ema12"] - df["ema26"]
    df["ema_cross_up"] = ((df["ema12"] > df["ema26"]) & (df["ema12"].shift(1) <= df["ema26"].shift(1))).astype(int)
    df["ema_cross_down"] = ((df["ema12"] < df["ema26"]) & (df["ema12"].shift(1) >= df["ema26"].shift(1))).astype(int)
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_cross_up"] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
    df["macd_cross_down"] = ((macd < signal) & (macd.shift(1) >= signal.shift(1))).astype(int)
    df["bb_mavg"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mavg"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mavg"] - 2 * df["bb_std"]
    df["bb_breakout"] = (df["close"] > df["bb_upper"]).astype(int)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["roc_10"] = df["close"].pct_change(10)
    df["tr"] = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"]
    try:
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    except Exception:
        df["adx"] = np.nan
    df["volume_change"] = df["volume"].pct_change(5)
    df["vol_spike"] = (df["volume"] > df["volume"].rolling(20).mean() * 1.5).astype(int)
    return df

def sign_params(params):
    # API SECRET'ın varlığını kontrol et
    if BINANCE_API_SECRET is None:
        logging.critical("BINANCE_API_SECRET .env dosyasında bulunamadı veya yüklenemedi! İmzalama yapılamıyor.")
        raise ValueError("API Secret yüklenemedi.")
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

class SymbolInfo:
    def __init__(self, symbol):
        self.symbol = symbol
        self.price_precision = 4 # Varsayılan
        self.min_price = 0.0
        self.max_price = 0.0
        self.tick_size = 0.0001 # Varsayılan
        self.quantity_precision = 1 # Varsayılan
        self.min_quantity = 0.001 # Varsayılan
        self.max_quantity = 0.0
        self.step_size = 0.001 # Varsayılan
        self.min_notional = 5.0 # Varsayılan

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

    def add_trade(self, symbol, direction, entry, exit_price, quantity, pnl, pnl_reason='Unknown'): # 'pnl' parametresi EKLENDİ
        # pnl = (exit_price - entry) * quantity if direction == 'LONG' else (entry - exit_price) * quantity
        if not isinstance(pnl, (int, float)): # Gelen pnl'in tipini kontrol et
            logging.error(f"add_trade: PNL sayısal değil! Değer: {pnl}, Tip: {type(pnl)}. 0.0 olarak ayarlanıyor.")
            pnl = 0.0
        
        if abs(pnl) < 0.01 and pnl != 0:
            if pnl < 0: pnl = -0.01
            else: pnl = 0.01
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

        logging.info(f"STATISTICS: Adding trade: {trade_record}") # EKLENEN İŞLEMİ LOGLA
        self.trades.append(trade_record)
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
        logging.info(f"STATISTICS: All trades count: {len(self.trades)}") # TÜM İŞLEMLER
        logging.info(f"STATISTICS: Recent trades count (last {period_hours}h): {len(recent_trades)}") # SON 24 SAATLİK İŞLEMLER
        logging.info(f"STATISTICS: Recent trades list: {recent_trades}") # LİSTENİN İÇERİĞİ

        if not recent_trades:
            logging.warning("STATISTICS: No recent trades found for the report.")
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

    def save_trades(self, filepath="trade_history.joblib"):
        """İşlem geçmişini dosyaya kaydeder."""
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True) # Gerekirse dizin oluştur
            joblib.dump(self.trades, filepath)
            logging.info(f"İşlem geçmişi başarıyla '{filepath}' dosyasına kaydedildi.")
        except Exception as e:
            logging.error(f"İşlem geçmişi kaydedilirken hata oluştu: {e}", exc_info=True)

    def load_trades(self, filepath="trade_history.joblib"):
        """İşlem geçmişini dosyadan yükler."""
        if os.path.exists(filepath):
            try:
                loaded_trades = joblib.load(filepath)
                if isinstance(loaded_trades, list):
                    self.trades = loaded_trades
                    logging.info(f"İşlem geçmişi '{filepath}' dosyasından başarıyla yüklendi. Yüklenen işlem sayısı: {len(self.trades)}")
                    # Yüklendikten sonra genel metrikleri de yeniden hesaplayabiliriz (isteğe bağlı)
                    self._recalculate_overall_metrics()
                else:
                    logging.error(f"Yüklenen işlem geçmişi liste formatında değil: {filepath}")
            except Exception as e:
                logging.error(f"İşlem geçmişi yüklenirken hata oluştu: {e}", exc_info=True)
                self.trades = [] # Hata durumunda boş başlat
        else:
            logging.info("Kaydedilmiş işlem geçmişi bulunamadı. Yeni bir geçmiş oluşturulacak.")
            self.trades = []

    def _recalculate_overall_metrics(self):
        """Yüklenen işlemler üzerinden genel performans metriklerini yeniden hesaplar."""
        self.performance_metrics = {
            'total_trades': len(self.trades),
            'profitable_trades': sum(1 for t in self.trades if t['success']),
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'peak_equity': 0.0, # Bu yeniden hesaplanmalı
            'max_drawdown': 0.0, # Bu yeniden hesaplanmalı
            'daily_volume': 0.0 # Bu sadece son 24 saatlik olmalı, o yüzden burada sıfırlanabilir.
                                # Veya get_stats gibi bir fonksiyonda hesaplanmalı.
        }
        # peak_equity ve max_drawdown için bir döngü gerekebilir.
        # Şimdilik basitçe toplamları alalım, daha sonra bu metrikler geliştirilebilir.
        if self.trades:
            cumulative_pnl = np.cumsum([t['pnl'] for t in self.trades])
            if len(cumulative_pnl) > 0:
                 self.performance_metrics['peak_equity'] = np.max(np.maximum.accumulate(cumulative_pnl))
                 if self.performance_metrics['peak_equity'] > 0 : # Sadece karda ise drawdown hesapla
                    self.performance_metrics['max_drawdown'] = np.max(np.maximum.accumulate(cumulative_pnl) - cumulative_pnl)

        logging.info(f"Genel performans metrikleri yeniden hesaplandı: {self.performance_metrics}")   

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
        self.stats = TradeStatistics() # TradeStatistics örneği oluşturuluyor
        self.active_positions = {} 
        self.last_error_notified = {} 
        self.error_notification_cooldown = timedelta(hours=1)
        self.report_task = None
        self.last_model_update = datetime.now(IST_TIMEZONE) - MODEL_UPDATE_INTERVAL
        self.symbol_info = {}
        self.last_fast_market_direction = None
        self.last_fast_market_direction_update_time = datetime.now(IST_TIMEZONE) - timedelta(minutes=10) # Bot başlar başlamaz güncellensin diye
    
    async def get_balance(self, *args, **kwargs):
        if getattr(self, "backtest", False):
            return 10000  # Backtest için sabit yüksek bir bakiye döndür
    # Gerçek API çağrısı buraya
        ...

    async def __aenter__(self):
        """Bot başladığında oturumu başlatır, exchange bilgilerini ve pozisyonları yükler."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        
        # 1. Önce gerekli borsa bilgilerini ve hassasiyet ayarlarını çek
        await self.fetch_exchange_info()

        # 2. Takip edilen tüm semboller için kaldıraçları ayarla
        for symbol in self.symbols_to_trade:
            if symbol in self.symbol_info:
                await self.set_leverage(symbol, SYMBOL_LEVERAGE.get(symbol, 20))
        
        # 3. Gerçek durumu (Binance) yükle ve botun hafızasını senkronize et
        await self.load_initial_positions()
        
        self.stats.load_trades() # <<--- BOT BAŞLARKEN İŞLEM GEÇMİŞİNİ YÜKLE
        
        await self.load_initial_positions() # Bu zaten Binance'den pozisyonları yüklüyordu
        return self

    async def load_initial_positions(self):
        """
        Başlangıçta Binance'deki tüm açık pozisyonları çeker ve self.active_positions'ı
        bu gerçek duruma göre sıfırdan oluşturur. Böylece manuel kapatılan pozisyonlar
        "hayalet" olarak kalmaz.
        """
        logging.info("Binance'deki mevcut açık pozisyonlar yükleniyor ve senkronize ediliyor...")
        
        # Önce botun kendi hafızasını tamamen temizle
        self.active_positions = {}
        
        # Binance'den güncel ve gerçek pozisyon listesini çek
        current_open_positions = await self.get_all_open_positions_with_details()
        
        if not isinstance(current_open_positions, list):
            logging.error(f"Başlangıç pozisyonları yüklenemedi, API'den liste dönmedi: {current_open_positions}")
            return

        # Sadece Binance'de gerçekten açık olan pozisyonları botun hafızasına kaydet
        for pos_details in current_open_positions:
            symbol = pos_details.get('symbol')
            try:
                amt = float(pos_details.get('positionAmt', 0))

                # Sadece pozisyon miktarı sıfırdan büyük olanları ve takip listemizde olanları dikkate al
                if symbol and abs(amt) > 0 and symbol in self.symbols_to_trade:
                    direction = 'LONG' if amt > 0 else 'SHORT'
                    
                    self.active_positions[symbol] = {
                        'entry': float(pos_details.get('entryPrice', 0)),
                        'quantity': abs(amt),
                        'original_total_quantity': abs(amt),
                        'direction': direction,
                        'sl_moved_to_entry': False,
                        'open_timestamp': int(pos_details.get('updateTime', int(time.time() * 1000))),
                        'main_sl_order_id': None,
                        'tp1_order_id': None,
                        'tp2_order_id': None,
                        'tp3_order_id': None,
                        'stop_loss_price': None,
                        'tp1_price': None,
                        'tp2_price': None,
                        'tp3_price': None,
                        'tp1_status': 'UNKNOWN',
                        'tp2_status': 'UNKNOWN',
                        'tp3_status': 'UNKNOWN'
                    }
                    logging.info(f"[{symbol}] Binance'den mevcut pozisyon yüklendi ve senkronize edildi.")

            except (ValueError, TypeError) as e:
                logging.error(f"[{symbol}] Pozisyon bilgisi yüklenirken format hatası: {e} - Veri: {pos_details}")
        
        logging.info(f"Senkronizasyon tamamlandı. Yönetilecek başlangıç pozisyonları: {list(self.active_positions.keys())}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.stats.save_trades() # <<--- BOT DURDURULURKEN İŞLEM GEÇMİŞİNİ KAYDET
        if self.session and not self.session.closed:
            await self.session.close()
        if self.report_task and not self.report_task.done():
            self.report_task.cancel()
            try:
                await self.report_task
            except asyncio.CancelledError:
                logging.info("Rapor görevi çıkışta iptal edildi.")
 
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
            logging.info(f"DEBUG: {symbol} - handle_position_close GİRİLDİ. pos_info: {pos_info}")
            return

        entry = pos_info['entry']
        original_quantity = pos_info['original_total_quantity']
        direction = pos_info['direction']
        open_timestamp = pos_info.get('open_timestamp', 0)

        pnl = 0.0
        exit_price_for_stats = entry
        pnl_reason = "Unknown"

        try: # ANA TRY BLOĞU BAŞLANGICI
            pnl_data = await self.get_realized_pnl_from_trades(symbol, open_timestamp)
            if pnl_data:
                pnl = pnl_data.get('realized_pnl', 0.0)
                exit_price_for_stats = pnl_data.get('avg_exit_price', entry)
                if exit_price_for_stats == 0.0: exit_price_for_stats = entry
            else:
                raise ValueError("get_realized_pnl_from_trades None döndürdü.")

            if not isinstance(pnl, (int, float)):
                logging.error(f"[{symbol}] get_realized_pnl_from_trades'ten dönen PNL sayısal değil: {pnl} (tip: {type(pnl)})")
                pnl = 0.0 
                pnl_reason = "PnL Type Error" # PnL tip hatası için özel bir sebep
            
            # pnl_reason ataması (pnl sayısal olduktan sonra)
            if pnl > 0:
                if pos_info.get('tp3_status') == 'EXECUTED': pnl_reason = "Take Profit (TP3)"
                elif pos_info.get('tp2_status') == 'EXECUTED': pnl_reason = "Take Profit (TP2)"
                elif pos_info.get('tp1_status') == 'EXECUTED': pnl_reason = "Take Profit (TP1)"
                else: pnl_reason = "Take Profit (Unknown/Manual)"
            elif pnl < 0: pnl_reason = "Stop Loss"
            else: pnl_reason = "Breakeven"

        except Exception as e: # get_realized_pnl_from_trades veya pnl_reason atamasında hata olursa
            logging.error(f"[{symbol}] PnL çekilirken/hesaplanırken hata (handle_position_close): {e}. PnL 0 olarak kabul edildi.", exc_info=True)
            pnl = 0.0 
            exit_price_for_stats = entry # Hata durumunda çıkış fiyatı olarak girişi kullan
            pnl_reason = "PnL Calculation Error" 

        # İstatistiklere ekle (DOĞRU ÇAĞRI)
        self.stats.add_trade(symbol, direction, entry, exit_price_for_stats, original_quantity, pnl, pnl_reason)
        self.stats.save_trades() # <<--- HER İŞLEM KAPANDIĞINDA GEÇMİŞİ KAYDET (Daha sık kayıt için)
        # ... (Telegram mesajı gönderme kısmı) ...

        price_precision = self.symbol_info.get(symbol, SymbolInfo(symbol)).price_precision
        message = (
            f"{'🔴' if pnl < 0 else '🟢'} {symbol} {direction} Pozisyon Kapatıldı\n"
            f"• Sebep: {pnl_reason}\n"
            f"• Giriş: {entry:.{price_precision}f}\n"
            f"• Çıkış: {exit_price_for_stats:.{price_precision}f}\n"
            f"• Kar/Zarar: {pnl:.2f} USDT\n"
            f"• Zaman: {datetime.now(IST_TIMEZONE).strftime('%H:%M')}\n"
        )
        await self.send_smart_telegram_message(message, msg_type='TRADE_CLOSED', symbol=symbol) # Bu mesaj gitmeli

        # Cooldown uygula
        if pnl_reason.startswith("Stop Loss") or pnl_reason == "PnL Calculation Error": # Hata durumunda da cooldown uygulanabilir
            self.cooldowns[symbol] = datetime.now(IST_TIMEZONE) + timedelta(minutes=COOLDOWN_MINUTES)
            logging.info(f"[{symbol}] {pnl_reason} nedeniyle {COOLDOWN_MINUTES} dakika cooldown'a alındı.")
        
        # Marjin temizleme çağrısı (bu da try-except içinde olabilir kendi içinde)
        try:
            await self.clean_margin(symbol)
        except Exception as e_clean:
            logging.error(f"[{symbol}] clean_margin çağrısında hata: {e_clean}", exc_info=True)

        # Son olarak bot içinden pozisyonu sil (Bu finally bloğuna alınabilir)
        if symbol in self.active_positions: # Hata olup olmadığına bakılmaksızın silinmesi için finally daha iyi
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

        # === YENİ EKLENECEK FONKSİYONLAR ===

    async def get_fast_market_direction(self, specific_df_for_symbol: dict = None, current_time_for_backtest_ms: int = None):
        """
        Daha hızlı ve hassas piyasa yönü belirleme fonksiyonu.
        EMA, RSI, MACD, VWAP üzerinden skorlayarak yön tespiti yapar.
        Backtest için specific_df_for_symbol ve current_time_for_backtest_ms alabilir.
        Canlıda ise API'den veri çeker.
        """
        interval = '15m' 
        lookback = 50    
        long_score_total = 0
        short_score_total = 0 # short skorunu da sayalım
        neutral_score_total = 0 # nötr skorunu da sayalım
        checked_symbols_count = 0

        for symbol in MARKET_DIRECTION_SYMBOLS: 
            if symbol not in self.symbol_info: 
                logging.warning(f"[{symbol}] Hızlı piyasa yönü için exchange bilgisi eksik. Atlanıyor.")
                continue

            df = None
            if specific_df_for_symbol and symbol in specific_df_for_symbol:
                df = specific_df_for_symbol[symbol]
            else:
                # Canlıda veya backtestte bu sembol için veri yoksa API'den çek
                end_time_param = current_time_for_backtest_ms if current_time_for_backtest_ms else int(time.time() * 1000)
                df = await self.fetch_data_multi(symbol, interval=interval, total_limit=lookback, endTime=end_time_param)


            if df.empty or len(df) < 20: 
                logging.warning(f"[{symbol}] Hızlı piyasa yönü için yeterli veri yok ({len(df)} mum). Atlanıyor.")
                continue

            try:
                close = df['close']
                ema5 = ta.trend.ema_indicator(close, window=5, fillna=False)
                ema20 = ta.trend.ema_indicator(close, window=20, fillna=False)
                rsi = ta.momentum.rsi(close, window=14, fillna=False)
                macd_diff = ta.trend.macd_diff(close, fillna=False)
                vwap = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], fillna=False)

                if pd.isna(ema5.iloc[-1]) or pd.isna(ema20.iloc[-1]) or \
                   pd.isna(rsi.iloc[-1]) or pd.isna(macd_diff.iloc[-1]) or \
                   pd.isna(vwap.iloc[-1]) or pd.isna(close.iloc[-1]):
                    logging.warning(f"[{symbol}] Hızlı yön analizi için bazı indikatör değerleri NaN. Atlanıyor.")
                    continue

                current_close = close.iloc[-1]
                current_ema5 = ema5.iloc[-1]
                current_ema20 = ema20.iloc[-1]
                current_rsi = rsi.iloc[-1]
                current_macd_diff = macd_diff.iloc[-1]
                current_vwap = vwap.iloc[-1]

                symbol_long_score = 0
                symbol_short_score = 0

                if current_ema5 > current_ema20: symbol_long_score += 1
                else: symbol_short_score +=1
                
                if current_rsi > 70: symbol_long_score += 1
                elif current_rsi < 30: symbol_short_score += 1 # Short için de bir eşik

                if current_macd_diff > 0: symbol_long_score += 1
                elif current_macd_diff < 0: symbol_short_score += 1

                if current_close > current_vwap: symbol_long_score += 1
                elif current_close < current_vwap: symbol_short_score += 1
                
                if symbol_long_score >= 3: # 4 indikatörden en az 3'ü LONG ise
                    long_score_total += 1
                elif symbol_short_score >= 3: # 4 indikatörden en az 3'ü SHORT ise
                    short_score_total += 1
                else: # Diğer durumlar (örn: 2 LONG, 2 SHORT)
                    neutral_score_total +=1


                checked_symbols_count += 1
                logging.debug(f"[{symbol}] Hızlı Yön Skorları - Long: {symbol_long_score}, Short: {symbol_short_score}")

            except Exception as e:
                logging.warning(f"[{symbol}] Hızlı piyasa yönü analizi sırasında hata: {e}", exc_info=False)
                continue

        if checked_symbols_count == 0:
            logging.info(f"[FAST_MARKET_DIRECTION] Hiçbir referans sembol analiz edilemedi. Yön: SHORT")
            return 'SHORT'

        if long_score_total > short_score_total and long_score_total >= (checked_symbols_count / 2):
            determined_direction = 'LONG'
        else:
            determined_direction = 'SHORT'
        
        logging.info(f"[FAST_MARKET_DIRECTION] Belirlenen Yön: {determined_direction} (Long Toplam: {long_score_total}, Short Toplam: {short_score_total}, Kontrol Edilen: {checked_symbols_count}/{len(MARKET_DIRECTION_SYMBOLS)})")
        return determined_direction

    async def get_cached_fast_market_direction(self, cache_duration_minutes=5, **kwargs_for_get_fast):
        now = datetime.now(IST_TIMEZONE)
        if not hasattr(self, 'last_fast_market_direction') or \
           not hasattr(self, 'last_fast_market_direction_update_time') or \
           self.last_fast_market_direction is None or \
           (now - self.last_fast_market_direction_update_time) > timedelta(minutes=cache_duration_minutes):
            
            logging.info(f"Önbellek süresi doldu veya ilk çağrı, hızlı piyasa yönü yeniden hesaplanıyor...")
            self.last_fast_market_direction = await self.get_fast_market_direction(**kwargs_for_get_fast)
            self.last_fast_market_direction_update_time = now
            logging.info(f"Hızlı piyasa yönü güncellendi ve önbelleğe alındı: {self.last_fast_market_direction}")
        else:
            logging.debug(f"Önbellekten hızlı piyasa yönü kullanılıyor: {self.last_fast_market_direction}")
            
        return self.last_fast_market_direction
    # --- YENİ PİYASA YÖNÜ FONKSİYONLARI BİTTİ ---

    async def get_cached_fast_market_direction(self, cache_duration_minutes=5):
        """
        Piyasa yönünü belirli aralıklarla önbelleğe alarak API çağrılarını azaltır.
        """
        now = datetime.now(IST_TIMEZONE)
        if not hasattr(self, 'last_fast_market_direction') or \
           not hasattr(self, 'last_fast_market_direction_update_time') or \
           self.last_fast_market_direction is None or \
           (now - self.last_fast_market_direction_update_time) > timedelta(minutes=cache_duration_minutes):
            
            logging.info(f"Önbellek süresi doldu veya ilk çağrı, hızlı piyasa yönü yeniden hesaplanıyor...")
            self.last_fast_market_direction = await self.get_fast_market_direction()
            self.last_fast_market_direction_update_time = now
            logging.info(f"Hızlı piyasa yönü güncellendi ve önbelleğe alındı: {self.last_fast_market_direction}")
        else:
            logging.debug(f"Önbellekten hızlı piyasa yönü kullanılıyor: {self.last_fast_market_direction}")
            
        return self.last_fast_market_direction
    
    async def clean_margin(self, symbol):
        try:
            pos_data = await self.get_open_position(symbol)
            if not pos_data or 'amt' not in pos_data:
                logging.warning(f"[{symbol}] Pozisyon bilgisi alınamadı veya 'amt' anahtarı eksik: {pos_data}. Marjin temizlemeye gerek yok.")
                return

            quantity_to_close = abs(pos_data['amt'])
            if quantity_to_close <= 0:
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

    async def fetch_data_multi(self, symbol, interval='5m', total_limit=1500, endTime=None): # <<-- endTime parametresini ekleyin (varsayılan None)
        limit_per_call = 1500  # Binance genellikle max 1500 veya 1000 mum verir, kontrol edin.
        all_data = []

        # Eğer endTime parametre olarak verilmemişse, şu anki zamanı kullan
        current_end_time_ms = endTime if endTime is not None else int(time.time() * 1000)

        data_fetched_in_first_call = 0 # İlk çağrıda kaç veri geldiğini saymak için

        while total_limit > 0:
            limit_to_fetch = min(limit_per_call, total_limit)
            
            # API'ye gönderilecek parametreleri oluştur
            params_for_api = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit_to_fetch
            }
            # Sadece endTime belirtilmişse URL'ye ekle
            if current_end_time_ms is not None:
                params_for_api['endTime'] = current_end_time_ms

            # URL'yi oluştur (parametreleri urlencode ile eklemek daha güvenli olabilir ama f-string de çalışır)
            query_string = urllib.parse.urlencode(params_for_api)
            url = f"{BASE_URL}/fapi/v1/klines?{query_string}"
            
            # logging.debug(f"fetch_data_multi URL: {url}") # Debug için URL'yi loglayabilirsiniz

            async with self.session.get(url) as response:
                data = await response.json()
                if not data or not isinstance(data, list) or len(data) == 0:
                    if not all_data: # Eğer hiç veri çekilemediyse ve ilk çağrıda da boş geldiyse
                        logging.warning(f"[{symbol}] İçin veri çekilemedi veya boş geldi: {data} (URL: {url})")
                    else: # Daha önce veri çekilmiş ama artık gelmiyorsa
                        logging.info(f"[{symbol}] Daha fazla veri bulunamadı (URL: {url}). Toplam çekilen: {len(all_data)}")
                    break # Veri yoksa döngüden çık

                # Gelen veriyi mevcut listenin başına ekle (en eski veriler sona, en yeni başa)
                all_data = data + all_data
                
                # Bir sonraki istek için endTime'ı bu çağrıdaki ilk mumun açılış zamanından 1ms önceye ayarla
                # data[0][0] en eski mumun zaman damgasıdır (listeyi tersine çevirmediğimiz için)
                # Ancak biz verileri `data + all_data` şeklinde birleştirdiğimiz için
                # ve Binance'den gelen veri en yeniden en eskiye doğru (endTime'a göre) sıralı olduğu için:
                # `data[0]` aslında o anki çağrının en yeni mumu olur.
                # Bizim amacımız ise bir sonraki çağrıda bu gelen `data` listesindeki en eski mumdan daha eski mumları çekmek.
                # Bu yüzden `data[0][0]` (gelen verinin en eski mumu) doğru olmalı.
                
                # Binance'den gelen veriler şu şekilde sıralıdır:
                # [ [en yeni_timestamp, o,h,l,c,...], ..., [en eski_timestamp, o,h,l,c,...] ]
                # Bu yüzden bir sonraki sorgu için endTime = en_eski_timestamp - 1 olmalı.
                current_end_time_ms = int(data[0][0]) - 1 # Gelen listenin ilk elemanı (en eski)

                # Eğer ilk çağrıdaysak ve gelen veri sayısı total_limit'ten azsa ve endTime parametresi verilmemişse,
                # bu, tüm geçmiş verinin bu kadar olduğu anlamına gelebilir.
                if endTime is None and len(all_data) == data_fetched_in_first_call and len(data) < limit_to_fetch :
                    logging.info(f"[{symbol}] İlk çağrıda istenenden ({limit_to_fetch}) az veri ({len(data)}) geldi ve endTime belirtilmedi. Muhtemelen tüm geçmiş bu kadar.")
                    # total_limit'i sıfırlayarak döngüden çıkabiliriz veya zaten çıkacaktır.
                
                if len(all_data) >= total_limit: # İstenen miktarda veya daha fazla veri toplandıysa
                     break


                # total_limit'i azalt (bu mantık biraz kafa karıştırıcı olabilir, alternatif aşağıda)
                # total_limit -= len(data) # Bu yaklaşım yerine doğrudan all_data uzunluğunu kontrol etmek daha iyi olabilir.

                # Önemli: Eğer Binance API'si limit parametresine rağmen daha az veri dönerse
                # (örn. coinin geçmişi o kadar uzun değilse), döngü gereksiz yere devam etmesin.
                if len(data) < limit_to_fetch:
                    logging.info(f"[{symbol}] API'den istenen ({limit_to_fetch}) adetten az ({len(data)}) veri geldi. Veri çekme tamamlandı.")
                    break
        
        if not all_data:
            return pd.DataFrame() # Boş DataFrame döndür

        # Toplamda istenen 'total_limit' kadarını al (eğer daha fazla çekildiyse sondan kırp)
        # Veriler en yeniden en eskiye doğru biriktiği için, sondan (en eskiden) başlayarak almalıyız.
        # all_data = data_chunk_N + ... + data_chunk_1 (data_chunk_1 en yeni, endTime'a en yakın olan)
        # df'e çevirmeden önce ters çevirip sonra ilk total_limit kadarını alabiliriz ya da sondan alırız.
        # Mevcut `data + all_data` mantığıyla `all_data` en yeniden en eskiye doğru gidiyor.
        # Yani `all_data[-total_limit:]` en eski `total_limit` kadarını alır.
        # Bizim istediğimiz ise en yeni `total_limit` kadarı. Bu yüzden `all_data[:total_limit]` olmalı
        # ama bu da verilerin ters sırada olmasına neden olabilir.
        
        # En iyisi: Gelen verileri doğru sıraya (eskiden yeniye) sokup sonra dataframe yapmak.
        # Binance'den gelen her `data` bloğu [en_yeni, ..., en_eski] şeklindedir.
        # `all_data = data + all_data` ile birleştirince:
        # [chunk_N_yeni,...,chunk_N_eski, chunk_N-1_yeni,..., chunk_N-1_eski, ..., chunk_1_yeni,...,chunk_1_eski]
        # Bu listeyi tersine çevirirsek [chunk_1_eski, ..., chunk_N_yeni] olur, yani eskiden yeniye.
        
        all_data.reverse() # Eskiden yeniye sırala

        # Şimdi toplamda istenen 'total_limit' kadarını al (eğer daha fazla çekildiyse sondan kırp)
        # Ters çevirdiğimiz için artık en yeni veriler listenin sonunda.
        # Bu yüzden, eğer all_data'nın boyutu total_limit'ten büyükse, son total_limit kadarını almalıyız.
        if len(all_data) > total_limit:
             all_data = all_data[-total_limit:]


        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        # Timestamp'ı IST'ye çevir ve UTC yap
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Önce datetime yap
        df = df.set_index('timestamp').tz_localize('UTC').tz_convert(IST_TIMEZONE).reset_index() # Sonra localize ve convert

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
        except aiohttp.ClientError as e:
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

        initial_stop_price = self.format_price(symbol, stop_loss)
        sl_params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": qty,
            "stopPrice": initial_stop_price,
            "closePosition": "false",
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }

        for attempt in range(max_retries):
            sl_signed = sign_params(sl_params)
            try:
                async with self.session.post(url, headers=self.headers, data=sl_signed) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if resp.status == 200:
                        logging.info(f"[{symbol}][{label}] STOP-LOSS emri gönderildi. OrderId: {data.get('orderId')}")
                        return data
                    elif resp.status == 400 and data.get('code') == -2021:
                        stop_price = self.format_price(symbol, stop_loss + min_tick_size * (attempt + 1))
                        sl_params["stopPrice"] = stop_price
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
            s_info = self.symbol_info[symbol]
            # ----- HESAPLAMA BLOKLARI -----
            min_tick = s_info.tick_size
            # ... diğer hesaplamalar ...
            position_size = ... # hesapla
            final_quantity_to_send = self.format_quantity(symbol, position_size)
            # -------------------------------
            if getattr(self, "backtest", False):
                logging.info(f"[BACKTEST] {symbol} için trade açıldı (simülasyon): {direction}, fiyat: {current_price}, qty: {final_quantity_to_send}")
                profit = ... # dummy bir simülasyon
                return {"profit": profit}
            
            s_info = self.symbol_info[symbol]

            logging.info(f"[{symbol}] Model Confidence: {confidence:.4f}") # Güven skorunu logla
            logging.info(f"[{symbol}] Trade açma işlemi başlıyor. Yön: {direction}, Güncel Fiyat: {current_price:.{s_info.price_precision}f}, ATR: {atr:.{s_info.price_precision}f}")
            
            if current_price <= 0 or atr <= 0:
                logging.error(f"[{symbol}] Güncel fiyat ({current_price}) veya ATR ({atr}) sıfır veya negatif. Trade açılamadı.")
                return
                
            min_tick = s_info.tick_size
            # ATR hesaplaması ve SL mesafesi
            max_atr_multiplier = 0.03 
            min_atr_multiplier = 0.0005 
            atr_calculated = max(min(current_price * max_atr_multiplier, atr), current_price * min_atr_multiplier)
            
            min_sl_percentage = 0.003
            sl_distance_base = max(atr_calculated * 2.5, current_price * min_sl_percentage)
            sl_distance = max(sl_distance_base, min_tick * 3) # En az 3 tick kadar SL mesafesi
            sl_distance = (round(sl_distance / min_tick)) * min_tick if min_tick > 0 else sl_distance_base # tick_size sıfır değilse yuvarla
            
            if sl_distance == 0:
                logging.error(f"[{symbol}] Hesaplanan SL mesafesi sıfır. Trade açılamadı.")
                return

            entry = current_price # Güncel fiyatı giriş olarak kabul et
            
            if direction == 'LONG':
                stop_loss = entry - sl_distance
                if stop_loss <= s_info.min_price or (entry - stop_loss) < (min_tick * 2): # SL'in min_price altında veya girişe çok yakın olmasını engelle
                    logging.error(f"[{symbol}] LONG için stop_loss ({stop_loss:.{s_info.price_precision}f}) geçersiz veya girişe çok yakın. Trade açılamadı.")
                    return
                tp1 = entry + (sl_distance * 1.0)
                tp2 = entry + (sl_distance * 1.5)
                tp3 = entry + (sl_distance * 2.5)
                order_side = "BUY"
            else: # SHORT
                stop_loss = entry + sl_distance
                if stop_loss >= s_info.max_price or (stop_loss - entry) < (min_tick * 2): # SL'in max_price üzerinde veya girişe çok yakın olmasını engelle
                    logging.error(f"[{symbol}] SHORT için stop_loss ({stop_loss:.{s_info.price_precision}f}) geçersiz veya girişe çok yakın. Trade açılamadı.")
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
            if len(set(prices_to_check)) < len(prices_to_check): # Eğer herhangi iki fiyat eşitse
                logging.warning(f"[{symbol}] Entry/SL/TP fiyatlarından bazıları birbirine eşit. Trade açılamadı. Fiyatlar: {prices_to_check}")
                await self.send_smart_telegram_message(f"⚠️ {symbol} için Entry/SL/TP fiyatları birbirine çok yakın veya eşit, trade açılamadı.", msg_type='WARNING', symbol=symbol)
                return

            rr = abs(tp3 - entry) / abs(entry - stop_loss) if abs(entry - stop_loss) > 0 else 0
            if rr < 1.3: # Minimum Risk/Reward oranı
                logging.info(f"[{symbol}] için RR ({rr:.2f}) çok düşük (<1.3). Trade açılamadı.")
                await self.send_smart_telegram_message(f"⚠️ {symbol} için RR ({rr:.2f}) çok düşük. Trade açılamadı.", msg_type='WARNING', symbol=symbol)
                return

            balance = await self.get_balance()
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
                calculated_notional = entry * position_size
                logging.warning(f"[{symbol}] Hesaplanan notional ({calculated_notional:.2f}) min notional'dan ({s_info.min_notional:.2f}) küçük. Miktar {self.format_quantity(symbol, position_size)} olarak ayarlandı.")
                
            # === final_quantity_to_send TANIMLAMASI BURADA ===
            final_quantity_to_send = self.format_quantity(symbol, position_size) # <<--- BU SATIRI EKLEYİN
    
            logging.info(f"DEBUG [{symbol}]: position_size hesaplandı: {position_size}")
            logging.info(f"DEBUG [{symbol}]: final_quantity_to_send formatlandı: {final_quantity_to_send}")
            
            if final_quantity_to_send < s_info.min_quantity or final_quantity_to_send > s_info.max_quantity:
                logging.error(f"[{symbol}] Nihai miktar ({final_quantity_to_send}) Binance min/max ({s_info.min_quantity}/{s_info.max_quantity}) limitleri dışında. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için miktar limitleri karşılanamadı. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            actual_notional_after_format = entry * final_quantity_to_send
            if actual_notional_after_format < s_info.min_notional:
                logging.error(f"[{symbol}] Formatlandıktan sonra notional ({actual_notional_after_format:.2f}) hala min notional'dan ({s_info.min_notional:.2f}) küçük. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için notional gereksinimleri karşılanamadı ({actual_notional_after_format:.2f}). Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            leverage = SYMBOL_LEVERAGE.get(symbol, 20)
            required_margin = (entry * final_quantity_to_send) / leverage
            
            if required_margin > balance * 0.98:
                logging.error(f"[{symbol}] Yetersiz marjin: Gereken ({required_margin:.2f}) bakiyeden ({balance:.2f}) fazla. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için yetersiz marjin. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            if final_quantity_to_send <= 0:
                logging.error(f"[{symbol}] Hesaplanan pozisyon miktarı sıfır veya negatif: {final_quantity_to_send}. Trade açılamadı.")
                await self.send_smart_telegram_message(f"⛔ {symbol} için pozisyon miktarı sıfır. Trade açılamadı.", msg_type='ERROR', symbol=symbol)
                return

            logging.info(f"DEBUG [{symbol}]: Market emir göndermeden önce final_quantity_to_send: {final_quantity_to_send}") # YENİ LOG
            # MARKET EMİR GÖNDERME
            order_result = await self.send_binance_market_order(
                symbol=symbol,
                side=order_side,
                quantity=final_quantity_to_send, # Hatanın olduğu satır burasıydı
                leverage=leverage
            )

            if not (order_result and order_result.get('orderId')):
                logging.error(f"[{symbol}] Binance MARKET emir gönderilemedi veya OrderId dönmedi. Detay: {order_result}")
                await self.send_smart_telegram_message(f"⚠️ [BINANCE] {symbol} MARKET emri gönderilemedi. Lütfen manuel kontrol edin.", msg_type='ERROR', symbol=symbol)
                return

            order_id = order_result['orderId'] # OrderId var, alalım

            # EMİR DURUMUNU KONTROL ETME (POLL)
            poll_attempts = 0
            max_poll_attempts = 20
            order_filled = False
            final_order_status = None

            while poll_attempts < max_poll_attempts:
                await asyncio.sleep(0.5) # Yarım saniye bekle
                status_check_params = {
                    "symbol": symbol,
                    "orderId": order_id,
                    "timestamp": int(time.time() * 1000)
                }
                # _api_request veya doğrudan session.get kullanılabilir. Hata yönetimi _api_request'te daha iyi.
                # signed_status_check = sign_params(status_check_params)
                # status_url = BASE_URL + "/fapi/v1/order"
                # check_data = await self._api_request('GET', "/fapi/v1/order", params=status_check_params) 
                # Daha basit ve direkt yöntem:
                query_string = urllib.parse.urlencode(sign_params(status_check_params))
                status_url = f"{BASE_URL}/fapi/v1/order?{query_string}"
                async with self.session.get(status_url) as resp: # Headers __init__'te ayarlandı
                    check_data = await resp.json()
                
                if check_data and check_data.get('status'):
                    current_status = check_data['status']
                    executed_qty_str = check_data.get('executedQty', '0.0')
                    if executed_qty_str: # Boş string gelme ihtimaline karşı
                        executed_qty = float(executed_qty_str)
                    else:
                        executed_qty = 0.0
                        
                    if current_status == 'FILLED' or (current_status == 'PARTIALLY_FILLED' and executed_qty > 0):
                        final_order_status = check_data
                        order_filled = True
                        logging.info(f"[{symbol}] MARKET emir başarıyla dolduruldu! Statü: {current_status}, Doldurulan Miktar: {executed_qty}")
                        break
                    elif current_status in ['CANCELED', 'EXPIRED', 'REJECTED', 'NEW', 'PARTIALLY_FILLED']: # NEW ve PARTIALLY_FILLED (qty=0) durumları da bekleyebilir
                        logging.info(f"[{symbol}] MARKET emir hala bekliyor veya sorunlu... Durum: {current_status}, Doldurulan Miktar: {executed_qty}")
                        if current_status in ['CANCELED', 'EXPIRED', 'REJECTED']:
                             logging.error(f"[{symbol}] MARKET emir {current_status} oldu, doldurulamadı. Detay: {check_data}")
                             break # Bu durumlarda döngüden çık
                    else:
                        logging.info(f"[{symbol}] MARKET emir durumu beklenmedik: {current_status}. Detay: {check_data}")

                else:
                    logging.warning(f"[{symbol}] Emir durumu kontrol edilemedi veya 'status' anahtarı yok: {check_data}")
                
                poll_attempts += 1

            if not order_filled or not final_order_status:
                logging.error(f"[{symbol}] MARKET emir zaman aşımına uğradı veya TAMAMEN DOLMADI/status alınamadı. Son durum: {final_order_status if final_order_status else 'Bilinmiyor'}")
                await self.send_smart_telegram_message(f"⚠️ [BINANCE] {symbol} MARKET emri zaman aşımına uğradı veya dolmadı. Lütfen manuel kontrol edin. Emir ID: {order_id}", msg_type='ERROR', symbol=symbol)
                return
            
            actual_entry_price = float(final_order_status.get('avgPrice', current_price))
            actual_filled_quantity = float(final_order_status.get('executedQty', 0))

            if actual_filled_quantity <= 0:
                logging.error(f"[{symbol}] MARKET emir dolduruldu ama miktar sıfır ({actual_filled_quantity}). Pozisyon açılamadı.")
                return

            logging.info(f"[{symbol}] Pozisyon başarıyla açıldı ve kaydedildi: Entry={actual_entry_price:.{s_info.price_precision}f}, Quantity={actual_filled_quantity:.{s_info.quantity_precision}f}")
            
            open_timestamp_ms = int(final_order_status.get('updateTime', int(time.time() * 1000)))
            
            self.active_positions[symbol] = {
                'entry': actual_entry_price,
                'quantity': actual_filled_quantity,
                'original_total_quantity': actual_filled_quantity,
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
                'tp3_status': 'PENDING',
                'open_timestamp': open_timestamp_ms # Sadece burada atama yapın
            }

            # SL/TP emirlerini yerleştir
            close_side = 'SELL' if direction == 'LONG' else 'BUY'
            
            main_sl_order_data = await self.place_only_sl_order(symbol, close_side, actual_filled_quantity, stop_loss, "MAIN_SL")
            if main_sl_order_data and 'orderId' in main_sl_order_data:
                self.active_positions[symbol]['main_sl_order_id'] = str(main_sl_order_data['orderId'])

            tp1_order_data = await self.place_tp_order(symbol, close_side, actual_filled_quantity * 0.3, tp1, "TP1")
            if tp1_order_data and 'orderId' in tp1_order_data:
                self.active_positions[symbol]['tp1_order_id'] = str(tp1_order_data['orderId'])

            tp2_order_data = await self.place_tp_order(symbol, close_side, actual_filled_quantity * 0.3, tp2, "TP2")
            if tp2_order_data and 'orderId' in tp2_order_data:
                self.active_positions[symbol]['tp2_order_id'] = str(tp2_order_data['orderId'])

            tp3_quantity_nominal = actual_filled_quantity - (actual_filled_quantity * 0.3) - (actual_filled_quantity * 0.3)
            tp3_quantity_nominal = self.format_quantity(symbol, tp3_quantity_nominal) # Son kalan miktarı formatla

            tp3_order_data = await self.place_tp_order(symbol, close_side, tp3_quantity_nominal, tp3, "TP3", close_entire_position=True)
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

        except NameError as ne: # Spesifik olarak NameError'ı yakala
            logging.error(f"[{symbol}] Execute Trade içinde NameError oluştu: {ne} - Değişken tanımsız olabilir.", exc_info=True)
        except Exception as e:
            logging.error(f"[{symbol}] Execute Trade Hatası: {e}", exc_info=True)

    async def process_symbol(self, symbol):
        logging.info(f"[{symbol}] --- Yeni analiz başlıyor ---")
        try:
            if symbol not in self.symbol_info:
                logging.warning(f"[{symbol}] Sembol bilgisi bulunamadı, analiz atlanıyor.")
                return

            df = await self.fetch_data_multi(symbol, interval='3m', total_limit=1500)
            min_raw_candles_needed = max(14, 30) + N_FUTURE_CANDLES # Örnek bir değer
            if df.empty or len(df) < min_raw_candles_needed:
                logging.warning(f"[{symbol}] Yeterli ham veri çekilemedi veya DataFrame boş! (Mevcut: {len(df)} mum, Minimum: {min_raw_candles_needed} mum gerekli)")
                return
            
            trend_up, trend_down, df_processed_ta = teknik_analiz(df)
            df = df_processed_ta # teknik_analiz'den dönen işlenmiş df'i kullan

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

            if symbol not in self.scalers: # Scaler daha önce eğitilmemişse
                self.scalers[symbol] = StandardScaler()
                self.scalers[symbol].fit(features.values) # Tüm özellikler üzerinden fit et
            
            X_last = pd.DataFrame([features.iloc[-1]], columns=features.columns) # Son satırı al
            X_scaled_last = self.scalers[symbol].transform(X_last.values)

            # Model yükleme veya ilk eğitim
            if symbol not in self.models['xgb']: # Herhangi bir model tipi kontrol edilebilir
                logging.info(f"[{symbol}] Modeller kontrol ediliyor/yükleniyor...")
                models_loaded = await self.load_models(symbol)
                
                if not models_loaded:
                    logging.info(f"[{symbol}] Kaydedilmiş modeller bulunamadı veya yüklenemedi. Modeller ilk defa eğitiliyor...")
                    self.models['xgb'][symbol] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    self.models['lgbm'][symbol] = LGBMClassifier()
                    self.models['gbt'][symbol] = GradientBoostingClassifier()
                    
                    # LSTM modelini de burada oluştur
                    self.models['lstm'][symbol] = Sequential([
                        Input(shape=(1, X_scaled_last.shape[1])), # Input layer for LSTM
                        LSTM(16, activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    self.models['lstm'][symbol].compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
                    
                    try:
                        # Modelleri eğitirken tüm 'features' ve 'targets' kullanılmalı
                        X_scaled_all_train = self.scalers[symbol].transform(features.values)
                        self.models['xgb'][symbol].fit(X_scaled_all_train, targets.values)
                        self.models['lgbm'][symbol].fit(X_scaled_all_train, targets.values)
                        self.models['gbt'][symbol].fit(X_scaled_all_train, targets.values)
                        
                        X_lstm_all_train = X_scaled_all_train.reshape(-1, 1, features.shape[1])
                        self.models['lstm'][symbol].fit(X_lstm_all_train, targets.values, epochs=20, batch_size=32, verbose=0)
                        logging.info(f"[{symbol}] Modeller başarıyla eğitildi.")
                        await self.save_models(symbol)
                    except Exception as e:
                        logging.error(f"[{symbol}] Modellerin ilk eğitimi sırasında hata: {e}", exc_info=True)
                        return
                else:
                    logging.info(f"[{symbol}] Modeller başarıyla yüklendi.")

            # Tahminler
            X_lstm_predict_tensor = tf.constant(X_scaled_last.reshape(1, 1, -1), dtype=tf.float32)
            predictions = {}
            try:
                predictions['xgb'] = self.models['xgb'][symbol].predict_proba(X_scaled_last)[0][1] # Long olasılığı
                predictions['lgbm'] = self.models['lgbm'][symbol].predict_proba(X_scaled_last)[0][1]
                predictions['gbt'] = self.models['gbt'][symbol].predict_proba(X_scaled_last)[0][1]
                predictions['lstm'] = float(self.models['lstm'][symbol].predict(X_lstm_predict_tensor, verbose=0)[0][0])
            except Exception as e:
                logging.error(f"[{symbol}] Model tahminleri sırasında hata: {e}", exc_info=True)
                return

            avg_prediction = sum(predictions.values()) / len(predictions)
            current_atr_series = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            current_atr = current_atr_series.iloc[-1] if not current_atr_series.empty and not pd.isna(current_atr_series.iloc[-1]) else 0.0
            current_price = df['close'].iloc[-1]

            now = datetime.now(IST_TIMEZONE)
            logging.info(f"[{symbol}] Model prediction: {avg_prediction:.3f}, trend_up: {trend_up}, trend_down: {trend_down}, ATR: {current_atr:.4f}, Price: {current_price:.{self.symbol_info.get(symbol, SymbolInfo(symbol)).price_precision}f}")
            
            if symbol in self.cooldowns and now < self.cooldowns[symbol]:
                logging.info(f"[{symbol}] Cooldown sürüyor ({self.cooldowns[symbol].strftime('%H:%M')}'e kadar), yeni trade açılmayacak.")
                return

            # === POZİSYON KONTROLLERİ VE handle_position_close ÇAĞRISININ DOĞRU YERİ ===
            pos = await self.get_open_position(symbol)
            pozisyon_acik_binance = pos and abs(pos.get('amt', 0)) > 0 # .get() ile güvenli erişim
            pozisyon_acik_bot = symbol in self.active_positions

            logging.info(f"[{symbol}] Binance'de pozisyon açık mı? {pozisyon_acik_binance} (Miktar: {pos.get('amt', 'N/A') if pos else 'N/A'})")
            logging.info(f"[{symbol}] Bot tarafında pozisyon açık mı? {pozisyon_acik_bot}")

            # Aşağıdaki CRITICAL DEBUG logu kaldırıldı çünkü handle_position_close sadece aşağıdaki koşulda çağrılmalı.
            # logging.critical(f"CRITICAL DEBUG: {symbol} için handle_position_close ÇAĞRILMAK ÜZERE!") 

            if pozisyon_acik_bot and not pozisyon_acik_binance:
                logging.warning(f"[{symbol}] Botta açık pozisyon görünüyor ancak Binance'de kapalı. Pozisyon kapatma işlemini tetikliyorum.")
                await self.cancel_open_orders(symbol)
                await self.handle_position_close(symbol) # SADECE BU KOŞULDA ÇAĞRILMALI
                return # Bu sembol için analiz bitti, çünkü pozisyonu kapattık

            elif not pozisyon_acik_bot and pozisyon_acik_binance:
                logging.warning(f"[{symbol}] Binance'de açık pozisyon var ancak bot kaydında yok. Bu pozisyonu bot takip etmiyor. Manuel kapatma veya inceleme gerekebilir.")
                # İsteğe bağlı: Bu durumda da bir tür senkronizasyon yapılabilir veya bot bu pozisyonu devralabilir.
                # Şimdilik sadece uyarı verip atlıyoruz.
                return # Bu sembol için analiz bitti

            if pozisyon_acik_bot: # Eğer pozisyon bot tarafından yönetiliyorsa ve Binance'de de açıksa
                logging.info(f"[{symbol}] Zaten açık pozisyon var (Bot tarafından yönetiliyor), yeni trade açılmayacak.")
                return # Yeni işlem açma

            # === YENİ İŞLEM AÇMA MANTIĞI (EĞER YUKARIDAKİ return'LERDEN GEÇİLDİYSE) ===
            market_direction = await self.get_cached_fast_market_direction() # Önbellekli hızlı piyasa yönünü çek
            logging.info(f"[{symbol}] Hızlı Piyasa Yönü Filtresi (Önbellekli): {market_direction}")

            # Sinyal eşiklerini buradan alabilirsiniz veya konfigürasyondan
            long_signal_threshold = 0.45 
            short_signal_threshold = 0.55

            if avg_prediction > long_signal_threshold:
                await self.execute_trade(symbol, 'LONG', ...)
            elif avg_prediction < short_signal_threshold:
                await self.execute_trade(symbol, 'SHORT', ...)
            else:
                logging.debug(f"[{symbol}] için geçerli bir işlem sinyali yok. Ortalama Tahmin: {avg_prediction:.3f}")
            
            if i < 50:
                print(f"{i}: avg_prediction: {avg_prediction}")

        except Exception as e:
            logging.error(f"[{symbol}] process_symbol içinde genel analiz hatası: {str(e)}", exc_info=True)

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
                open_positions_binance = await self.get_all_open_positions() # Binance'deki gerçek açık pozisyonlar
                current_bot_positions = list(self.active_positions.keys()) # Botun o anki hafızası

                for symbol_in_bot_memory in current_bot_positions:
                    if symbol_in_bot_memory not in open_positions_binance:
                        logging.warning(f"[{symbol_in_bot_memory}] Botta açık pozisyon görünüyor ({self.active_positions.get(symbol_in_bot_memory)}) ancak Binance'de kapalı. Kapatma işlemi tetikleniyor.")
                        await self.cancel_open_orders(symbol_in_bot_memory) # Önce emirleri iptal et
                        await self.handle_position_close(symbol_in_bot_memory) # Sonra bot içinden pozisyonu temizle
            
                # Binance'de olup botta olmayanlar için sadece uyarı ver (bu kısım doğru)
                for symbol_on_binance in open_positions_binance:
                    if symbol_on_binance not in self.active_positions: # Artık self.active_positions güncel olmalı
                        logging.warning(f"[{symbol_on_binance}] Binance'de açık pozisyon var ancak bot kaydında yok. Manuel kontrol gerekebilir.")
            
                current_active_symbols = list(self.active_positions.keys()) # Güncel listeyi tekrar al
                logging.info(f"Bot tarafından yönetilen açık pozisyonlar (senkronizasyon sonrası): {current_active_symbols} (max {MAX_ACTIVE_TRADES})")

                # 5. Mevcut Açık Pozisyonların Durumunu Kontrol Etme (SL/TP Yönetimi)
                analyzed_open_positions_count = 0 # Yeni değişken: Sadece açık pozisyonları saymak için
                for symbol in current_active_symbols:
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
                    if pos_details.get('tp1_price') and not pos_details.get('sl_moved_to_entry') and pos_details.get('tp1_status') == 'PENDING':
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
                            await asyncio.sleep(1) # Durumun güncellenmesi için kısa bir bekleme
                    
                    # SL/TP Emirlerinin Kontrolü ve Yeniden Yerleştirme Mantığı
                    if pos_details.get('stop_loss_price') is not None: # Pozisyon için SL/TP fiyatları tanımlıysa
                        current_open_sl_tp_orders = await self.get_open_sl_tp_orders_for_symbol(symbol)
                        current_tp_order_ids_on_binance = {str(order['orderId']) for order in current_open_sl_tp_orders if order.get('type') == 'TAKE_PROFIT_MARKET'}
                        main_sl_is_open_on_binance = any(
                            pos_details.get('main_sl_order_id') and
                            str(order['orderId']) == str(pos_details.get('main_sl_order_id'))
                            for order in current_open_sl_tp_orders if order.get('type') == 'STOP_MARKET'
                        )

                        missing_sl = pos_details.get('main_sl_order_id') and not main_sl_is_open_on_binance
                        missing_tp1 = pos_details.get('tp1_status') == 'PENDING' and (
                            not pos_details.get('tp1_order_id') or str(pos_details.get('tp1_order_id')) not in current_tp_order_ids_on_binance
                        )
                        missing_tp2 = pos_details.get('tp2_status') == 'PENDING' and (
                            not pos_details.get('tp2_order_id') or str(pos_details.get('tp2_order_id')) not in current_tp_order_ids_on_binance
                        )
                        missing_tp3 = pos_details.get('tp3_status') == 'PENDING' and (
                            not pos_details.get('tp3_order_id') or str(pos_details.get('tp3_order_id')) not in current_tp_order_ids_on_binance
                        )

                        if missing_sl or missing_tp1 or missing_tp2 or missing_tp3:
                            logging.warning(f"[{symbol}] Pozisyonda eksik SL/TP emirleri bulundu. Yeniden gönderilecekler: SL:{missing_sl}, TP1:{missing_tp1}, TP2:{missing_tp2}, TP3:{missing_tp3}")
                            
                            # 1. Önce mevcut tüm SL/TP emirlerini iptal et (temiz bir başlangıç için)
                            await self.cancel_open_orders(symbol)
                            await asyncio.sleep(0.5) # Binance API'sine zaman tanımak için kısa bir bekleme

                            tps_re_sent_labels_log = [] # Yeniden gönderilen TP'leri loglamak için
                            original_qty = pos_details.get('original_total_quantity')
                            if original_qty is None:
                                logging.error(f"[{symbol}] 'original_total_quantity' bulunamadı! TP miktarları yanlış olabilir. Fallback: current_binance_quantity.")
                                original_qty = current_binance_quantity # current_binance_quantity'nin bu kapsamda tanımlı olması gerekir

                            # 2. Şimdi eksik olanları yeniden gönder
                            # Ana SL'i yeniden gönder
                            if missing_sl:
                                sl_price_for_re_placement = pos_details['entry'] if pos_details.get('sl_moved_to_entry', False) else pos_details['stop_loss_price']
                                # current_binance_quantity'nin burada güncel olması önemli
                                new_sl_order = await self.place_only_sl_order(symbol, close_side, current_binance_quantity, sl_price_for_re_placement, "MAIN_SL_RE-SEND")
                                if new_sl_order and 'orderId' in new_sl_order:
                                    pos_details['main_sl_order_id'] = str(new_sl_order['orderId'])
                                else:
                                    logging.error(f"[{symbol}] Ana SL emri yeniden GÖNDERİLEMEDİ.")

                            # TP1'i yeniden gönder
                            if missing_tp1:
                                qty_tp1 = original_qty * 0.3
                                tp_order_data = await self.place_tp_order(symbol, close_side, qty_tp1, pos_details['tp1_price'], "TP1")
                                if tp_order_data and 'orderId' in tp_order_data:
                                    pos_details['tp1_order_id'] = str(tp_order_data['orderId'])
                                    tps_re_sent_labels_log.append("TP1")
                                elif tp_order_data and tp_order_data.get('code') == -2021: # Emir hemen tetiklenirse
                                    logging.warning(f"[{symbol}] TP1 yeniden gönderilirken 'Order would immediately trigger' hatası. Muhtemelen tetiklenmiş.")
                                    pos_details['tp1_status'] = 'EXECUTED'
                                    pos_details['tp1_order_id'] = None

                            # TP2'yi yeniden gönder
                            if missing_tp2:
                                qty_tp2 = original_qty * 0.3
                                tp_order_data = await self.place_tp_order(symbol, close_side, qty_tp2, pos_details['tp2_price'], "TP2")
                                if tp_order_data and 'orderId' in tp_order_data:
                                    pos_details['tp2_order_id'] = str(tp_order_data['orderId'])
                                    tps_re_sent_labels_log.append("TP2")
                                elif tp_order_data and tp_order_data.get('code') == -2021:
                                    logging.warning(f"[{symbol}] TP2 yeniden gönderilirken 'Order would immediately trigger' hatası. Muhtemelen tetiklenmiş.")
                                    pos_details['tp2_status'] = 'EXECUTED'
                                    pos_details['tp2_order_id'] = None
                            
                            # TP3'ü yeniden gönder
                            if missing_tp3:
                                qty_tp3_nominal = original_qty * 0.4 # Bu %40 kalan miktarı temsil etmeli, format_quantity ile ayarlanabilir
                                # Örnek: qty_tp3_nominal = original_qty - (original_qty * 0.3) - (original_qty * 0.3)
                                tp_order_data = await self.place_tp_order(symbol, close_side, qty_tp3_nominal, pos_details['tp3_price'], "TP3", close_entire_position=True)
                                if tp_order_data and 'orderId' in tp_order_data:
                                    pos_details['tp3_order_id'] = str(tp_order_data['orderId'])
                                    tps_re_sent_labels_log.append("TP3")
                                elif tp_order_data and tp_order_data.get('code') == -2021:
                                    logging.warning(f"[{symbol}] TP3 yeniden gönderilirken 'Order would immediately trigger' hatası. Muhtemelen tetiklenmiş.")
                                    pos_details['tp3_status'] = 'EXECUTED'
                                    pos_details['tp3_order_id'] = None
                                
                            if tps_re_sent_labels_log or missing_sl: # Eğer herhangi bir emir yeniden gönderildiyse
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

# === BACKTEST FONKSİYONU (SINIF DIŞINDA GLOBAL OLARAK TANIMLANIYOR) ===
async def backtest_market_direction_accuracy(
    trader: QuantumTrader,
    start_date_str: str,  # <<--- BU PARAMETRE
    end_date_str: str,    # <<--- BU PARAMETRE
    direction_interval: str = '15m',
    future_look_hours: int = 6,
    price_change_threshold_percent: float = 0.5
):
    results = []
    start_dt = pd.Timestamp(start_date_str, tz=IST_TIMEZONE)
    end_dt = pd.Timestamp(end_date_str, tz=IST_TIMEZONE)

    all_symbol_data = {}
    for symbol in MARKET_DIRECTION_SYMBOLS:
        if symbol not in trader.symbol_info:
            logging.warning(f"Backtest: [{symbol}] için exchange bilgisi yok, atlanıyor.")
            continue
        logging.info(f"Backtest: [{symbol}] için {start_date_str} - {end_date_str} arası veri çekiliyor...")
        pass

    current_dt_loop = start_dt
    loop_count = 0

    trader.last_fast_market_direction = None
    trader.last_fast_market_direction_update_time = datetime.now(IST_TIMEZONE) - timedelta(days=1)

    while current_dt_loop < end_dt:
        loop_count += 1
        if loop_count % 24 == 0:
             logging.info(f"Backtest ilerliyor: {current_dt_loop}")

        predicted_direction = await trader.get_fast_market_direction(current_time_for_backtest_ms=int(current_dt_loop.timestamp() * 1000))

        actual_future_direction = "SHORT"
        avg_future_price_change = 0.0

        future_start_dt_for_actual = current_dt_loop
        future_end_dt_for_actual = current_dt_loop + timedelta(hours=future_look_hours)

        df_future_eth = await trader.fetch_data_multi(
            'ETHUSDT',
            interval=direction_interval,
            total_limit=int(future_look_hours * (60 / int(direction_interval[:-1])))+5,
            endTime=int(future_end_dt_for_actual.timestamp()*1000)
        )

        if not df_future_eth.empty:
            df_future_eth = df_future_eth[
                (df_future_eth['timestamp'] >= future_start_dt_for_actual) &
                (df_future_eth['timestamp'] < future_end_dt_for_actual)
            ]

        if not df_future_eth.empty and len(df_future_eth) > 1:
            start_price = df_future_eth['open'].iloc[0]
            end_price = df_future_eth['close'].iloc[-1]
            avg_future_price_change = ((end_price - start_price) / start_price) * 100

            if avg_future_price_change > price_change_threshold_percent:
                actual_future_direction = "LONG"
            elif avg_future_price_change < -price_change_threshold_percent:
                actual_future_direction = "SHORT"
        else:
            logging.warning(f"Gelecek fiyat verisi çekilemedi veya yetersiz: ETHUSDT {future_start_dt_for_actual} - {future_end_dt_for_actual}")

        results.append({
            'timestamp': current_dt_loop,
            'predicted_direction': predicted_direction,
            'actual_future_direction': actual_future_direction,
            'future_price_change_%': round(avg_future_price_change, 2)
        })

        current_dt_loop += timedelta(hours=1)
        if loop_count % 10 == 0:
            await asyncio.sleep(1)

    if not results:
        print("Backtest için hiç sonuç üretilemedi.")
        return

    df_results = pd.DataFrame(results)
    df_results['correct_prediction'] = df_results['predicted_direction'] == df_results['actual_future_direction']

    overall_accuracy = df_results['correct_prediction'].mean() * 100

    print("\n=== Piyasa Yönü Backtest Sonuçları ===")
    print(f"Test Periyodu: {start_date_str} - {end_date_str}")
    print(f"Tahmin Aralığı: Her 1 saatte bir")
    print(f"Gelecek Gözlem Periyodu: {future_look_hours} saat")
    print(f"Fiyat Değişim Eşiği: %{price_change_threshold_percent}")
    print(f"Referans Semboller: {MARKET_DIRECTION_SYMBOLS}")
    print(f"Piyasa Yönü Zaman Dilimi: {direction_interval}")
    print("---")
    print(f"Toplam Tahmin Sayısı: {len(df_results)}")
    print(f"Genel Doğruluk Oranı: {overall_accuracy:.2f}%")
    print("---")
    print("Tahmin Edilen Yöne Göre Dağılım ve Doğruluk:")
    for p_dir in ['LONG', 'SHORT']:
        subset = df_results[df_results['predicted_direction'] == p_dir]
        if not subset.empty:
            accuracy = subset['correct_prediction'].mean() * 100
            print(f"  Tahmin: {p_dir} (Adet: {len(subset)}) -> Doğruluk: {accuracy:.2f}%")
            print(f"    Gerçekleşenler: {subset['actual_future_direction'].value_counts(normalize=True).mul(100).round(1).to_dict()}")
    print("---")
    print("Detaylı Sonuçlar (ilk 5 satır):")
    print(df_results.head())

    results_filename = f"market_direction_backtest_{start_date_str}_to_{end_date_str}_{direction_interval}.csv"
    df_results.to_csv(results_filename, index=False)
    print(f"\nSonuçlar '{results_filename}' dosyasına kaydedildi.")


async def main_backtest_runner():
    if not all([TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
        logging.critical("API anahtarları veya Telegram bilgileri .env dosyasında eksik.")
        return

    async with QuantumTrader(SYMBOLS, TELEGRAM_TOKEN, CHAT_ID) as trader:
        logging.info("Backtest için QuantumTrader örneği oluşturuldu.")
        # Tarihleri ve interval'i istediğiniz gibi ayarlayın
        await backtest_market_direction_accuracy(
            trader,
            start_date_str="2025-05-07",
            end_date_str="2025-05-14",
            direction_interval='15m', # get_fast_market_direction bu intervali kullanacak
            future_look_hours=4, # Tahminden sonraki 4 saatlik performansa bak
            price_change_threshold_percent=0.75 # %0.75'lik hareket trendi teyit eder
        )

async def main():  # Bu satır 2309 olabilir
    # BURAYA GERÇEK main FONKSİYONUNUN İÇERİĞİNİ EKLEYİN
    if not all([TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
        logging.critical("API anahtarları veya Telegram bilgileri .env dosyasında eksik. Bot çalıştırılamıyor.")
        return

    # ÖNEMLİ: API anahtarlarının yüklendiğini kontrol et (QuantumTrader __init__ içinde de var ama burada da iyi olur)
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logging.critical("Binance API anahtarları yüklenemedi. .env dosyasını kontrol edin.")
        return # Anahtarlar yoksa botu başlatma

    max_retries = 3
    retry_delay = 10  # saniye

    for attempt in range(max_retries):
        try:
            logging.info(f"Quantum AI Trader başlatılıyor... (Deneme {attempt + 1}/{max_retries})")
            async with QuantumTrader(SYMBOLS, TELEGRAM_TOKEN, CHAT_ID) as trader:
                await trader.send_smart_telegram_message("🤖 Quantum AI Trader V32 başlatıldı!")
                # Başlangıçta kısa bir bekleme, Binance'in rate limitlerine karşı nazik olmak için
                await asyncio.sleep(5)
                await trader.run_analysis_loop()
            break # Başarılı olursa döngüden çık
        except aiohttp.ClientConnectorError as e: # Ağ bağlantı hataları
            logging.error(f"Ağ bağlantı hatası (ClientConnectorError): {e}. Yeniden denenecek ({retry_delay} saniye sonra)...")
            if attempt + 1 == max_retries:
                logging.critical("Maksimum yeniden deneme sayısına ulaşıldı. Bot durduruluyor.")
                await trader.send_smart_telegram_message("❌ Bot maksimum yeniden deneme sayısına ulaştıktan sonra durduruldu (Ağ Hatası).", msg_type='ERROR')
                break
            await asyncio.sleep(retry_delay)
        except requests.exceptions.ConnectionError as e: # Bu kütüphane kullanılmıyorsa bu bloğu kaldırın
            logging.error(f"Ağ bağlantı hatası (requests.ConnectionError): {e}. Yeniden denenecek ({retry_delay} saniye sonra)...")
            # ... (requests için retry mantığı) ...
        except Exception as e:
            logging.critical(f"QuantumTrader başlatılırken veya çalışırken kritik hata: {e}", exc_info=True)
            # await trader.send_smart_telegram_message(f"❌ Bot kritik bir hata nedeniyle durduruldu: {e}", msg_type='ERROR') # 'trader' tanımlı olmayabilir
            # Genel bir mesaj gönderilebilir
            # await send_telegram_message_static(f"❌ Bot kritik bir hata nedeniyle durduruldu: {e}") # Eğer statik bir gönderici varsa
            # Şimdilik sadece loglayalım, çünkü trader örneği oluşmamış olabilir.
            break # Diğer hatalarda da döngüden çık, yeniden deneme sadece ağ hataları için mantıklı


async def main():
    async with QuantumTrader(SYMBOLS, TELEGRAM_TOKEN, CHAT_ID) as trader:
        df = await trader.fetch_data_multi('ETHUSDT', interval='1h', total_limit=300)
        print(get_classic_market_direction(df, ema_fast=5, ema_slow=20, rsi_period=14, rsi_ob=60, rsi_os=40))
        print(get_classic_market_direction(df, ema_fast=12, ema_slow=30, rsi_period=14, rsi_ob=75, rsi_os=25))

if __name__ == "__main__":
    asyncio.run(main())