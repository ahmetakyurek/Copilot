# train_market_predictor.py

import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
import os
from dotenv import load_dotenv
import optuna # Optuna importu

# ml_data_utils.py dosyasındaki fonksiyonları import et
from ml_data_utils import get_klines_iterative, create_market_direction_target, create_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna log seviyesini ayarla

load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY_FUTURES')
API_SECRET = os.getenv('BINANCE_API_SECRET_FUTURES')

if not API_KEY or not API_SECRET:
    logging.warning("UYARI: .env dosyasından Binance API anahtarları yüklenemedi.")

# --- KONFİGÜRASYON ---
SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
START_DATE_UTC = "2022-01-01 00:00:00"
END_DATE_UTC = "2024-05-01 00:00:00"

LOOK_FORWARD_PERIODS = 12
PRICE_CHANGE_THRESHOLD_PCT = 0.0040

# --- OPTUNA OBJECTIVE FONKSİYONU ---
def objective(trial, X_train_scaled, y_train, X_val_scaled, y_val):
    """Optuna için objective fonksiyonu."""
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss', # veya 'multi_error'
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2 regularization
        'random_state': 42,
        'class_weight': 'balanced' # Sınıf dengesizliği için
        # 'n_jobs': -1 # Kullanılabilir tüm işlemcileri kullan
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_scaled, y_train)
    
    y_pred_val = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred_val)
    return accuracy # Optuna bu değeri maksimize etmeye çalışacak

# --- ANA EĞİTİM VE OPTİMİZASYON FONKSİYONU ---
def optimize_and_train_model():
    logging.info(f"{SYMBOL} için veri çekiliyor ({START_DATE_UTC} - {END_DATE_UTC})...")
    client = Client()
    df_raw = get_klines_iterative(client, SYMBOL, INTERVAL, START_DATE_UTC, END_DATE_UTC, limit=1000)

    if df_raw.empty or len(df_raw) < 200:
        logging.error("Eğitim için yeterli geçmiş veri çekilemedi.")
        return

    logging.info("Özellikler oluşturuluyor...")
    # ml_data_utils.py içindeki create_features'ta ADD_ENGULFING_FEATURES = True/False ayarını kontrol edin
    X_features = create_features(df_raw)

    logging.info("Hedef değişken ('target_direction') oluşturuluyor...")
    y_target = create_market_direction_target(df_raw['close'], LOOK_FORWARD_PERIODS, PRICE_CHANGE_THRESHOLD_PCT)
    
    combined_df = X_features.copy()
    combined_df['target'] = y_target
    combined_df.dropna(inplace=True)

    if combined_df.empty or len(combined_df['target'].unique()) < 2 :
        logging.error("NaN'lar temizlendikten sonra veri kalmadı veya hedefte yeterli çeşitlilik yok.")
        return

    X = combined_df.drop('target', axis=1)
    y = combined_df['target'].astype(int)

    logging.info(f"Eğitim/Validasyon/Test için {len(X)} örnek veri hazırlandı.")
    logging.info(f"Sınıf dağılımı:\n{y.value_counts(normalize=True)}")

    # Veri Setini Bölme: Train -> Validation -> Test (Zaman sıralı)
    # Örnek: İlk %60 Train, sonraki %20 Validation, son %20 Test
    n_total = len(X)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    # n_test = n_total - n_train - n_val # Geri kalanı test

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val :], y.iloc[n_train + n_val :]

    if X_train.empty or X_val.empty or X_test.empty:
        logging.error("Train/Validation/Test split sonrası en az bir set boş kaldı.")
        return

    logging.info(f"Eğitim seti: {len(X_train)}, Validasyon seti: {len(X_val)}, Test seti: {len(X_test)}")

    # Özellik Ölçekleme
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_val_scaled_np = scaler.transform(X_val) # Sadece transform (eğitim verisiyle fit edildi)
    X_test_scaled_np = scaler.transform(X_test) # Sadece transform

    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled_np, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=X_test.columns, index=X_test.index)

    # Hiperparametre Optimizasyonu (Optuna)
    logging.info("Optuna ile hiperparametre optimizasyonu başlatılıyor...")
    study = optuna.create_study(direction='maximize') # Doğruluğu maksimize et
    # n_trials: Kaç farklı parametre seti deneneceği. Daha yüksek daha iyi ama daha uzun sürer.
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val), n_trials=50) # Örnek 50 deneme

    best_params = study.best_params
    logging.info(f"En iyi hiperparametreler bulundu: {best_params}")
    logging.info(f"En iyi validasyon doğruluğu: {study.best_value:.4f}")

    # En iyi parametrelerle son modeli tüm eğitim verisi üzerinde (train+val) eğitme (isteğe bağlı ama yaygın)
    # Veya sadece train seti üzerinde en iyi parametrelerle eğitip test edebiliriz.
    # Şimdilik sadece train üzerinde en iyi parametrelerle eğitelim ve test edelim.
    
    logging.info("En iyi parametrelerle final model eğitiliyor (eğitim seti üzerinde)...")
    final_model = LGBMClassifier(**best_params, random_state=42, class_weight='balanced', verbosity=-1)
    # Tüm eğitim verisi (X_train_scaled + X_val_scaled) üzerinde de eğitilebilir:
    # X_train_val_scaled = pd.concat([X_train_scaled, X_val_scaled])
    # y_train_val = pd.concat([y_train, y_val])
    # final_model.fit(X_train_val_scaled, y_train_val)
    final_model.fit(X_train_scaled, y_train) # Sadece orijinal eğitim setiyle

    # Model Değerlendirmesi (Test Seti Üzerinde)
    logging.info("\n--- Test Seti Performansı (Optimize Edilmiş Model) ---")
    y_pred_test = final_model.predict(X_test_scaled)
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    logging.info(f"Genel Doğruluk (Test Seti): {accuracy_test*100:.2f}%")
    
    target_names = ['SHORT', 'LONG', 'NEUTRAL'] 
    try:
        unique_labels_test = np.unique(y_test)
        unique_labels_pred_test = np.unique(y_pred_test)
        combined_labels_numeric = np.unique(np.concatenate((unique_labels_test, unique_labels_pred_test))).astype(int)
        current_target_names = [target_names[i] for i in combined_labels_numeric if i < len(target_names)]
        
        report = classification_report(y_test, y_pred_test, labels=combined_labels_numeric, target_names=current_target_names, zero_division=0)
        logging.info(f"Sınıflandırma Raporu (Test Seti):\n{report}")
    except Exception as e:
        logging.error(f"Classification report (Test Seti) oluşturulurken hata: {e}")

    logging.info("Karışıklık Matrisi (Test Seti):")
    try:
        cm = confusion_matrix(y_test, y_pred_test, labels=combined_labels_numeric)
        cm_df = pd.DataFrame(cm, 
                             index=current_target_names, 
                             columns=[f"Pred_{name}" for name in current_target_names])
        print(cm_df)
    except Exception as e:
        logging.error(f"Confusion matrix (Test Seti) oluşturulurken hata: {e}")

    # Özellik Önem Sıralaması
    if hasattr(final_model, 'feature_importances_'):
        logging.info("\n--- Özellik Önem Sıralaması ---")
        feature_importances = pd.Series(final_model.feature_importances_, index=X_train.columns)
        print(feature_importances.sort_values(ascending=False).to_string())


    # Modeli ve Scaler'ı Kaydet
    logging.info("Optimize edilmiş model ve scaler kaydediliyor...")
    joblib.dump(final_model, "market_direction_predictor_optimized_model.joblib")
    joblib.dump(scaler, "market_direction_predictor_scaler.joblib") # Scaler aynı kalır
    logging.info("Optimize edilmiş model ve scaler başarıyla kaydedildi.")

if __name__ == "__main__":
    optimize_and_train_model()