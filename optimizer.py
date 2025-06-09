import os
import pandas as pd
import numpy as np
import ta
from binance.client import Client # Senkron veri çekme için
from itertools import product
from datetime import datetime, timezone
import logging
from tqdm import tqdm # İlerleme çubuğu için

# --- TEMEL KONFİGÜRASYON ---
API_KEY = os.getenv('BINANCE_API_KEY_FUTURES') # .env dosyanızdaki Binance API anahtarınız
API_SECRET = os.getenv('BINANCE_API_SECRET_FUTURES') # .env dosyanızdaki Binance API secret'ınız

# API anahtarlarının yüklenip yüklenmediğini kontrol et (isteğe bağlı, public endpoint kullanacağız)
# Eğer private endpoint'ler (örn: hesap bilgisi) kullanılmayacaksa API key'e gerek olmayabilir klines için.
# Ancak python-binance Client() başlatmak için isteyebilir. Şimdilik bırakalım.
if not API_KEY or not API_SECRET:
    print("UYARI: Binance API anahtarları .env dosyasında bulunamadı veya yüklenemedi! Sadece public endpoint'ler çalışacaktır.")
    # exit() # Eğer API anahtarı kesin gerekliyse çıkış yapılabilir. Kline için genelde gerekmez.

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OPTİMİZASYON PARAMETRELERİ ---
SYMBOL_TO_OPTIMIZE = "ETHUSDT"
INTERVAL_TO_OPTIMIZE = Client.KLINE_INTERVAL_15MINUTE # "15m"
KLINE_LIMIT = 1500  # Çekilecek mum sayısı (örn: 2500 mum * 15dk = ~26 gün)

# "Gerçekleşen Gelecek Yönü" Belirleme Parametreleri
FUTURE_LOOK_PERIODS = 16  # 15dk'lık barlar için 4 saat = 16 periyot
PRICE_CHANGE_THRESHOLD = 0.0075  # %0.75

# Test Edilecek Parametre Aralıkları
PARAM_GRID = {
    'ema_short_period': [8, 10, 12],             # Biraz daralttım
    'ema_long_period': [26, 30, 50],              # Biraz daralttım
    'rsi_period': [14],
    'rsi_ob_threshold': [70, 75],                 # Daha az seçenek
    'rsi_os_threshold': [30, 25],                 # Daha az seçenek
    'rsi_interpretation_mode': ['momentum', 'reversal'],
    'macd_fast_period': [12],
    'macd_slow_period': [26],
    'macd_signal_period': [9],
    'use_vwap': [True],                           # VWAP her zaman kullanılıyor

    # YENİ EKLENEN PARAMETRELER
    'atr_period': [14, 20],                       # ATR periyodu
    'use_atr_filter': [True, False],              # ATR filtresi aktif mi? (örn: volatilite çok düşükse işlem yapma)
    'atr_volatility_threshold_factor': [0.5, 1.0], # ATR'nin ortalamasına göre bir eşik faktörü (örnek)

    'bb_period': [20, 25],                        # Bollinger Bantları periyodu
    'bb_std_dev': [2.0, 2.5],                     # Bollinger Bantları standart sapma
    'use_bb_filter': [True, False],               # Bollinger Bantları filtresi aktif mi? (örn: fiyat banda değdi mi?)

    'use_obv_slope': [True, False],               # OBV eğimi pozitif/negatif mi?
    'obv_slope_window': [5, 10],                  # OBV eğimi için bakılacak pencere

    'adx_period': [14, 20],                       # ADX periyodu
    'use_adx_filter': [True, False],              # ADX filtresi aktif mi?
    'adx_trend_threshold': [20, 25],              # ADX trend gücü eşiği (örn: >25 ise trend var)

    # Skorlama eşiği şimdi daha fazla indikatöre göre ayarlanacak
    # Eğer tüm filtreler aktifse 4 (EMA,RSI,MACD,VWAP) + 4 (ATR,BB,OBV,ADX) = 8 potansiyel sinyal
    # Bu yüzden skor eşiğini daha geniş bir aralıkta tutabiliriz.
    'score_threshold': [3, 4, 5]
}

# --- YARDIMCI FONKSİYONLAR ---

def fetch_historical_klines(symbol, interval, limit):
    """Binance'den geçmiş kline verilerini çeker."""
    try:
        # API anahtarları olmadan da public klines çekilebilir.
        # client = Client(API_KEY, API_SECRET)
        client = Client() # Anahtarsız başlatma (public data için)
        logging.info(f"{symbol} için {limit} adet {interval} kline verisi çekiliyor...")
        # Binance Futures için: client.futures_klines
        # Binance Spot için: client.get_klines
        # Varsayılan olarak Futures için yapalım, sembolleriniz ona göre.
        # Eğer spot ise client.get_historical_klines veya client.get_klines kullanın.
        # get_historical_klines daha esnek zaman aralığı sunar.
        # Şimdilik basitlik için client.futures_klines kullanalım.
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        logging.info("Veri çekme tamamlandı.")
    except Exception as e:
        logging.error(f"Veri çekilirken hata oluştu: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.set_index('timestamp', inplace=True)
    df.dropna(inplace=True) # Sayısala çevrilemeyen veya eksik veri varsa at
    return df

def calculate_direction_with_params(df_klines, params):
    """
    Verilen DataFrame ve parametrelerle indikatörleri hesaplar
    ve bir 'predicted_direction' serisi döndürür.
    """
    df = df_klines.copy()

    # 1. TEMEL İNDİKATÖRLER (Mevcutlar)
    df['ema_short'] = ta.trend.ema_indicator(df['close'], window=params['ema_short_period'], fillna=False)
    df['ema_long'] = ta.trend.ema_indicator(df['close'], window=params['ema_long_period'], fillna=False)
    df['rsi'] = ta.momentum.rsi(df['close'], window=params['rsi_period'], fillna=False)
    df['macd_diff'] = ta.trend.macd_diff(
        df['close'],
        window_fast=params['macd_fast_period'],
        window_slow=params['macd_slow_period'],
        window_sign=params['macd_signal_period'],
        fillna=False
    )
    if params['use_vwap']:
        if 'volume' in df.columns and not df['volume'].isnull().all():
            df['vwap'] = ta.volume.volume_weighted_average_price(
                df['high'], df['low'], df['close'], df['volume'], fillna=False
            )
        else:
            df['vwap'] = np.nan
    else:
        df['vwap'] = np.nan

    # 2. YENİ İNDİKATÖRLER
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=params['atr_period'], fillna=False)

    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=params['bb_period'], window_dev=params['bb_std_dev'], fillna=False)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_mavg'] = bb_indicator.bollinger_mavg() # Orta bant (EMA ile aynı periyotta SMA)

    # OBV
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'], fillna=False)
    # OBV Eğimini hesaplamak için basit bir fark veya hareketli ortalama kullanılabilir
    df['obv_slope'] = df['obv'].diff(params['obv_slope_window']) # Son N periyottaki değişim

    # ADX
    adx_indicator = ta.trend.ADXIndicator(
        high=df['high'], low=df['low'], close=df['close'], window=params['adx_period'], fillna=False
    )
    df['adx'] = adx_indicator.adx()
    df['adx_pos'] = adx_indicator.adx_pos() # +DI
    df['adx_neg'] = adx_indicator.adx_neg() # -DI


    # İndikatör hesaplamalarından sonra NaN içeren satırları at
    df.dropna(inplace=True)
    if df.empty:
        return pd.Series(dtype='object')

    # --- SKORLAMA MANTIĞI ---
    long_score = pd.Series(0, index=df.index)
    short_score = pd.Series(0, index=df.index)
    num_active_indicators = 0 # Skorlama eşiğini dinamik yapmak için aktif indikatör sayısını tutabiliriz

    # EMA Skorlaması
    long_score[df['ema_short'] > df['ema_long']] += 1
    short_score[df['ema_short'] < df['ema_long']] += 1
    num_active_indicators +=1

    # RSI Skorlaması
    if params['rsi_interpretation_mode'] == 'momentum':
        long_score[df['rsi'] > params['rsi_ob_threshold']] += 1
        short_score[df['rsi'] < params['rsi_os_threshold']] += 1
    elif params['rsi_interpretation_mode'] == 'reversal':
        long_score[df['rsi'] < params['rsi_os_threshold']] += 1
        short_score[df['rsi'] > params['rsi_ob_threshold']] += 1
    num_active_indicators +=1

    # MACD Skorlaması
    long_score[df['macd_diff'] > 0] += 1
    short_score[df['macd_diff'] < 0] += 1
    num_active_indicators +=1
    
    # VWAP Skorlaması
    if params['use_vwap'] and 'vwap' in df.columns and not df['vwap'].isnull().all():
        long_score[df['close'] > df['vwap']] += 1
        short_score[df['close'] < df['vwap']] += 1
        num_active_indicators +=1

    # ATR Filtresi/Skorlaması (Örnek: Yeterli volatilite varsa puan ekle veya filtrele)
    if params['use_atr_filter'] and 'atr' in df.columns:
        # Örnek: ATR, kapanış fiyatının belirli bir yüzdesinden büyükse trendi destekler
        # Veya ATR'nin kendi hareketli ortalamasının üzerindeyse
        atr_avg = df['atr'].rolling(window=params['atr_period']).mean() # ATR'nin kendi ortalaması
        min_volatility_condition = df['atr'] > (atr_avg * params['atr_volatility_threshold_factor'])
        # Bu koşul direkt LONG/SHORT skoru vermez, ama diğer sinyalleri güçlendirebilir
        # Şimdilik, eğer volatilite varsa genel bir "işleme girilebilir" puanı ekleyelim.
        # Bu, yön belirtmez ama sinyallerin geçerliliğini artırabilir.
        # VEYA: Yönlü sinyallerle birlikte KULLANILABİLİR.
        # Şimdilik basitçe, volatilite varsa skorlara +0.5 ekleyelim (yön belirtmeden)
        # long_score[min_volatility_condition] += 0.5 # Bu yaklaşım karmaşıklaştırır
        # short_score[min_volatility_condition] += 0.5
        # DAHA İYİ YAKLAŞIM: ATR'yi bir filtre olarak kullanmak. Eğer volatilite düşükse,
        # diğer sinyaller ne olursa olsun NEUTRAL kalmak.
        # Ya da yönlü skorları sadece volatilite varsa artırmak.
        # Şimdilik, ATR'yi doğrudan yön skoruna eklemeyelim, bunun yerine bir "koşul" olarak tutalım.
        # num_active_indicators +=1 # Eğer skorlamaya dahil edilecekse
        pass # ATR'yi şimdilik skorlamaya direkt dahil etmiyorum, kullanımı daha karmaşık. Filtre olarak daha iyi.

    # Bollinger Bands Filtresi/Skorlaması
    if params['use_bb_filter'] and 'bb_upper' in df.columns:
        # Örnek: Fiyat alt banda değip yükseliyorsa LONG, üst banda değip düşüyorsa SHORT (Reversal)
        # VEYA: Fiyat üst bandı kırarsa LONG, alt bandı kırarsa SHORT (Breakout)
        # Şimdilik Reversal deneyelim:
        touches_lower_band_then_up = (df['low'].shift(1) < df['bb_lower'].shift(1)) & (df['close'] > df['bb_lower']) & (df['close'] > df['open'])
        touches_upper_band_then_down = (df['high'].shift(1) > df['bb_upper'].shift(1)) & (df['close'] < df['bb_upper']) & (df['close'] < df['open'])
        long_score[touches_lower_band_then_up] += 1
        short_score[touches_upper_band_then_down] += 1
        # num_active_indicators +=1 # Eğer skorlamaya dahil edilecekse
        pass # Bollinger'ı bu şekilde eklemek iyi bir başlangıç olabilir.

    # OBV Eğim Skorlaması
    if params['use_obv_slope'] and 'obv_slope' in df.columns:
        long_score[df['obv_slope'] > 0] += 1  # OBV yükseliyorsa LONG
        short_score[df['obv_slope'] < 0] += 1 # OBV düşüyorsa SHORT
        num_active_indicators +=1

    # ADX Filtresi/Skorlaması
    if params['use_adx_filter'] and 'adx' in df.columns and 'adx_pos' in df.columns and 'adx_neg' in df.columns:
        strong_trend_condition = df['adx'] > params['adx_trend_threshold']
        # Sadece güçlü trend varken EMA, MACD gibi trend takip eden sinyallere güven
        # Veya ADX yönüyle (+DI > -DI ise LONG) birleştir
        adx_long_signal = strong_trend_condition & (df['adx_pos'] > df['adx_neg'])
        adx_short_signal = strong_trend_condition & (df['adx_neg'] > df['adx_pos'])
        long_score[adx_long_signal] += 1
        short_score[adx_short_signal] += 1
        num_active_indicators +=1
        

    # Tahmin Edilen Yön
    predicted_direction = pd.Series('NEUTRAL', index=df.index)
    # score_thresh = params['score_threshold'] # Bu parametre zaten var
    # Dinamik bir eşik kullanabiliriz: örn: num_active_indicators * 0.6 (yani %60'ı aynı yönde olmalı)
    # Ya da PARAM_GRID'den gelen sabit eşiği kullanırız. Şimdilik sabit eşik.
    
    # Kaç tane "filtre" aktifse (use_... == True ise) ona göre aktif indikatör sayısı belirlenmeli.
    # Şimdilik EMA,RSI,MACD,OBV,ADX (eğer use_obv_slope ve use_adx_filter True ise) 5 ana skor kaynağı gibi düşünelim.
    # VWAP da vardı. 6 etti. Bollinger da bir skor kaynağı olabilir. 7 etti.
    # Bu skorlama mantığını basitleştirmek gerekebilir veya `score_threshold`'u buna göre ayarlamak.

    # Basitleştirilmiş skorlama eşiği (PARAM_GRID'den gelen)
    score_thresh = params['score_threshold']

    predicted_direction[long_score >= score_thresh] = 'LONG'
    predicted_direction[short_score >= score_thresh] = 'SHORT'
    predicted_direction[(long_score >= score_thresh) & (short_score >= score_thresh)] = 'NEUTRAL'
    
    return predicted_direction


def calculate_actual_future_direction(df_klines, look_periods, price_thresh):
    """
    Her bir mum için, 'look_periods' sonraki fiyat hareketine göre
    'actual_direction' (LONG, SHORT, NEUTRAL) serisini hesaplar.
    """
    actual_direction = pd.Series(dtype='object', index=df_klines.index)
    # df_klines.shift(-look_periods) ile daha vektörel yapılabilir, ancak iloc ile daha net
    for i in range(len(df_klines) - look_periods):
        current_price = df_klines['close'].iloc[i]
        future_price = df_klines['close'].iloc[i + look_periods]
        price_change_pct = (future_price - current_price) / current_price

        if price_change_pct > price_thresh:
            actual_direction.iloc[i] = 'LONG'
        elif price_change_pct < -price_thresh:
            actual_direction.iloc[i] = 'SHORT'
        else:
            actual_direction.iloc[i] = 'NEUTRAL'
    return actual_direction

# --- ANA OPTİMİZASYON FONKSİYONU ---
def run_param_optimizer():
    logging.info("Parametre Optimizasyonu Başlatılıyor...")

    # 1. Geçmiş Veriyi Çek
    df_hist = fetch_historical_klines(SYMBOL_TO_OPTIMIZE, INTERVAL_TO_OPTIMIZE, KLINE_LIMIT)
    if df_hist.empty or len(df_hist) < FUTURE_LOOK_PERIODS + 50: # İndikatörler ve gelecek için yeterli veri
        logging.error("Optimizasyon için yeterli geçmiş veri çekilemedi veya veri çok kısa.")
        return

    # 2. Gerçekleşen Gelecek Yönlerini Hesapla (bir kere yapılır)
    logging.info("'Actual future direction' hesaplanıyor...")
    df_hist['actual_direction'] = calculate_actual_future_direction(df_hist, FUTURE_LOOK_PERIODS, PRICE_CHANGE_THRESHOLD)
    # 'actual_direction' hesaplanamayan son satırları at
    df_hist.dropna(subset=['actual_direction'], inplace=True)
    if df_hist.empty:
        logging.error("'Actual future direction' hesaplandıktan sonra DataFrame boş kaldı.")
        return
    logging.info(f"Analiz için kullanılacak veri boyutu: {len(df_hist)}")


    all_results = []
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    
    # Tüm parametre kombinasyonlarını oluştur
    # product(*param_values) bir jeneratör döndürür
    # tqdm ile ilerleme çubuğu ekleyelim
    combinations = list(product(*param_values))
    logging.info(f"Toplam {len(combinations)} parametre kombinasyonu test edilecek.")

    for combo in tqdm(combinations, desc="Optimizasyon İlerlemesi"):
        current_params = dict(zip(param_names, combo))

        # EMA uzun periyodunun kısa periyottan büyük olduğundan emin ol
        if current_params['ema_long_period'] <= current_params['ema_short_period']:
            continue # Bu kombinasyonu atla

        # RSI eşiklerinin mantıklı olduğundan emin ol
        if current_params['rsi_ob_threshold'] <= current_params['rsi_os_threshold'] + 10: # Arada en az 10 fark olsun
             continue


        # Tahmin edilen yönleri hesapla
        predicted_directions = calculate_direction_with_params(df_hist.copy(), current_params) # df_hist'in kopyasını gönder
        
        if predicted_directions.empty:
            # logging.warning(f"Parametreler: {current_params} için tahmin üretilemedi (muhtemelen tüm veriler NaN oldu).")
            continue

        # Performansı hesapla (predicted_directions ve df_hist['actual_direction'] aynı indekse sahip olmalı)
        # İndeksleri birleştirerek hizala ve NaN olmayanları al
        comparison_df = pd.DataFrame({
            'predicted': predicted_directions,
            'actual': df_hist['actual_direction']
        }).dropna() # Hem tahmin hem de gerçek yön olan satırları al

        if comparison_df.empty:
            # logging.warning(f"Parametreler: {current_params} için karşılaştırılacak veri yok (dropna sonrası).")
            continue

        total_predictions = len(comparison_df)
        if total_predictions == 0:
            accuracy = 0
        else:
            correct_predictions = (comparison_df['predicted'] == comparison_df['actual']).sum()
            accuracy = (correct_predictions / total_predictions) * 100

        # Detaylı doğruluklar
        num_long_pred = (comparison_df['predicted'] == 'LONG').sum()
        num_short_pred = (comparison_df['predicted'] == 'SHORT').sum()
        num_neutral_pred = (comparison_df['predicted'] == 'NEUTRAL').sum()

        acc_long = ((comparison_df['predicted'] == 'LONG') & (comparison_df['actual'] == 'LONG')).sum() / num_long_pred * 100 if num_long_pred > 0 else 0
        acc_short = ((comparison_df['predicted'] == 'SHORT') & (comparison_df['actual'] == 'SHORT')).sum() / num_short_pred * 100 if num_short_pred > 0 else 0
        acc_neutral = ((comparison_df['predicted'] == 'NEUTRAL') & (comparison_df['actual'] == 'NEUTRAL')).sum() / num_neutral_pred * 100 if num_neutral_pred > 0 else 0
        
        # Sadece anlamlı sayıda tahmin yapıldıysa sonuçları kaydet (örn: en az 10 tahmin)
        if total_predictions < 10:
            continue

        all_results.append({
            'params': str(current_params), # CSV'ye yazmak için string'e çevir
            'accuracy': round(accuracy, 2),
            'num_long_pred': num_long_pred,
            'acc_long': round(acc_long, 2),
            'num_short_pred': num_short_pred,
            'acc_short': round(acc_short, 2),
            'num_neutral_pred': num_neutral_pred,
            'acc_neutral': round(acc_neutral, 2),
            'total_predictions': total_predictions
        })

    # Sonuçları DataFrame'e çevir ve sırala
    if not all_results:
        logging.warning("Optimizasyon sonucunda değerlendirilebilecek hiçbir sonuç üretilemedi.")
        return

    results_df = pd.DataFrame(all_results)
    # En iyi genel doğruluğa göre sırala, sonra LONG doğruluğuna, sonra SHORT doğruluğuna
    results_df = results_df.sort_values(by=['accuracy', 'acc_long', 'acc_short'], ascending=[False, False, False])

    logging.info("\n--- En İyi Optimizasyon Sonuçları (İlk 20) ---")
    print(results_df.head(20).to_string()) # .to_string() ile tüm sütunlar daha iyi görünür

    # Sonuçları CSV'ye kaydet
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{SYMBOL_TO_OPTIMIZE}_{INTERVAL_TO_OPTIMIZE}_{timestamp_str}.csv"
    results_df.to_csv(filename, index=False)
    logging.info(f"Optimizasyon sonuçları '{filename}' dosyasına kaydedildi.")


if __name__ == "__main__":
    # .env dosyasındaki API anahtarlarını yüklemek için (eğer henüz yüklenmediyse)
    from dotenv import load_dotenv
    dotenv_loaded = load_dotenv()
    if not dotenv_loaded:
        print("UYARI: .env dosyası bulunamadı veya yüklenemedi.")

    # API anahtarlarını tekrar kontrol et (load_dotenv sonrası)
    API_KEY = os.getenv('BINANCE_API_KEY_FUTURES')
    API_SECRET = os.getenv('BINANCE_API_SECRET_FUTURES')
    if not API_KEY or not API_SECRET:
         print("UYARI: .env yüklemesinden sonra Binance API anahtarları hala eksik. Lütfen .env dosyanızı kontrol edin (BINANCE_API_KEY_FUTURES, BINANCE_API_SECRET_FUTURES).")
         # Kline verisi çekmek için API anahtarına genelde gerek olmaz, yine de Client() başlatmak için isteyebilir.

    run_param_optimizer()