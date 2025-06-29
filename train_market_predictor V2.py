import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
from dotenv import load_dotenv
import optuna
from ml_data_utils import get_klines_iterative, create_features
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY_FUTURES') # .env dosyanızda olduğundan emin olun
API_SECRET = os.getenv('BINANCE_API_SECRET_FUTURES') # .env dosyanızda olduğundan emin olun

if not API_KEY or not API_SECRET:
    logging.warning("UYARI: .env dosyasından Binance API anahtarları (BINANCE_API_KEY_FUTURES, BINANCE_API_SECRET_FUTURES) yüklenemedi veya eksik.")

# --- KONFİGÜRASYON ---
SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
START_DATE_UTC = "2022-11-22 00:00:00" # Veri başlangıç tarihi
END_DATE_UTC   = "2024-03-11 00:00:00" # Veri bitiş tarihi (Test için bir miktar veri bırakın)

LOOK_FORWARD_PERIODS = 12 # Kaç periyot sonrasına bakılacak (örn: 15dk için 12 periyot = 3 saat)
# PRICE_CHANGE_THRESHOLD_PCT global olarak tanımlamak yerine find_optimal_threshold ile bulunacak

# --- YARDIMCI FONKSİYONLAR (Bu script'e özel) ---

def find_optimal_threshold(close_prices, lookforward_periods, target_positive_ratio=0.15, step=0.0005, min_samples_per_class_ratio=0.05):
    """
    Verilen kapanış fiyatları ve ileriye bakış periyodu için,
    pozitif sınıf oranını hedeflenen bir dengeye yaklaştıran optimal
    fiyat değişim eşiğini bulur.
    """
    logging.info("Optimal fiyat değişim eşiği aranıyor...")
    best_threshold = 0.005 # Varsayılan bir başlangıç
    smallest_diff_to_target = float('inf')
    actual_ratio_at_best_thr = 0

    # Test edilecek eşik aralığı
    for current_threshold_pct in np.arange(0.002, 0.030, step): # %0.2 ile %3.0 arası
        future_price_changes = close_prices.shift(-lookforward_periods) / close_prices - 1
        # İkili hedef: 1 (LONG - pozitif sınıf), 0 (SHORT - negatif sınıf)
        # NEUTRAL durumu bu fonksiyonda yok, binary hedef için.
        binary_target = (future_price_changes > current_threshold_pct).astype(int)
        
        # NaN olmayan hedefleri al (özellikle serinin sonundaki lookforward_periods kadar NaN olur)
        valid_targets = binary_target.dropna()
        if len(valid_targets) == 0:
            continue

        positive_class_ratio = valid_targets.mean() # 1'lerin oranı
        negative_class_ratio = 1 - positive_class_ratio

        # Her iki sınıfın da minimum örnek oranını karşıladığından emin ol
        if positive_class_ratio >= min_samples_per_class_ratio and negative_class_ratio >= min_samples_per_class_ratio:
            current_diff = abs(positive_class_ratio - target_positive_ratio)
            if current_diff < smallest_diff_to_target:
                smallest_diff_to_target = current_diff
                best_threshold = current_threshold_pct
                actual_ratio_at_best_thr = positive_class_ratio
    
    if best_threshold is None: # Eğer uygun bir threshold bulunamazsa
        logging.warning(f"Uygun bir optimal threshold bulunamadı, varsayılan {PRICE_CHANGE_THRESHOLD_PCT_DEFAULT} kullanılıyor.")
        return PRICE_CHANGE_THRESHOLD_PCT_DEFAULT, 0 # Varsayılanı döndür
        
    logging.info(f"Optimal threshold bulundu: {best_threshold:.4f} (Pozitif sınıf oranı: {actual_ratio_at_best_thr:.2%})")
    return best_threshold, actual_ratio_at_best_thr

def create_binary_target(close_prices, lookforward_periods, price_change_threshold_pct):
    """
    İkili hedef değişkeni oluşturur (1: LONG/Yükseliş, 0: SHORT/Düşüş). NEUTRAL yok.
    """
    future_price_changes = close_prices.shift(-lookforward_periods) / close_prices - 1
    # Yükseliş (LONG): 1, Düşüş veya Yeterince Yükselmeyen (SHORT): 0
    target = (future_price_changes > price_change_threshold_pct).astype(int)
    return target

# --- XGBOOST İÇİN OPTUNA OBJECTIVE FONKSİYONU ---
def objective_xgboost(trial, X_train_fs, y_train_bal, X_val_fs, y_val):
    """Optuna için XGBoost objective fonksiyonu."""
    # Sınıf dengesizliği için scale_pos_weight hesapla
    # (negatif_sayısı / pozitif_sayısı)
    # y_train_bal zaten SMOTE ile dengelendiği için scale_pos_weight=1 olmalı veya kullanılmamalı.
    # Eğer orijinal y_train kullanılacaksa hesaplanabilir.
    # count_negative = np.sum(y_train_bal == 0)
    # count_positive = np.sum(y_train_bal == 1)
    # scale_pos_weight_val = count_negative / count_positive if count_positive > 0 else 1
    
    params = {
        'objective': 'binary:logistic', # İkili sınıflandırma
        'eval_metric': 'logloss',       # veya 'auc', 'error'
        'verbosity': 0,                 # XGBoost loglarını kapat
        'use_label_encoder': False,     # Önerilen ayar
        # 'scale_pos_weight': scale_pos_weight_val, # Eğer y_train_bal yerine y_train kullanılacaksa
        
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0), # Minimum loss reduction
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Training instance subsample ratio
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Subsample ratio of columns
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2 regularization
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    print("DEBUG: model.fit =", model.fit)
    print("DEBUG: model.fit module =", model.fit.__module__)
    model.fit(
        X_train_fs,
        y_train_bal,
        eval_set=[(X_val_fs, y_val)],
        # early_stopping_rounds=25,  # Bunu KALDIR
        verbose=False
    )
    
    y_pred_val = model.predict(X_val_fs)
    accuracy = accuracy_score(y_val, y_pred_val)
    return accuracy

# --- ANA EĞİTİM VE OPTİMİZASYON FONKSİYONU (Sadece XGBoost) ---
def optimize_and_train_xgboost_model(n_optuna_trials=30): # Optuna deneme sayısı
    logging.info(f"{SYMBOL} için veri çekiliyor ({START_DATE_UTC} - {END_DATE_UTC})...")
    client = Client(API_KEY, API_SECRET) # API anahtarlarıyla başlatmayı deneyelim
    df_raw = get_klines_iterative(client, SYMBOL, INTERVAL, START_DATE_UTC, END_DATE_UTC, limit=1000)

    if df_raw.empty or len(df_raw) < (LOOK_FORWARD_PERIODS + 100): # Özellikler ve hedef için min veri
        logging.error("Eğitim için yeterli geçmiş veri çekilemedi.")
        return

    # Optimal fiyat değişim eşiğini bul (pozitif sınıf oranını %15 civarında tutmaya çalışır)
    price_change_threshold, actual_ratio = find_optimal_threshold(
        df_raw['close'], 
        LOOK_FORWARD_PERIODS, 
        target_positive_ratio=0.30, # LONG sınıfının oranını biraz daha artırmayı deneyelim
        min_samples_per_class_ratio=0.10 # Her sınıf en az %10 olmalı
    )
    if price_change_threshold is None:
        logging.error("Uygun bir fiyat değişim eşiği bulunamadı.")
        return

    logging.info("Teknik indikatörler ve özellikler oluşturuluyor (`ml_data_utils.create_features` kullanılıyor)...")
    # ml_data_utils.py'deki create_features, engulfing vb. ayarlarını kontrol edin.
    # Örneğin, ADD_ENGULFING_FEATURES = True/False
    X_all_features = create_features(df_raw)

    logging.info("İkili hedef değişken oluşturuluyor (NEUTRAL yok)...")
    y_binary_target = create_binary_target(df_raw['close'], LOOK_FORWARD_PERIODS, price_change_threshold)

    # Özellikler ve hedefi birleştir, NaN'ları ve uyumsuz indeksleri temizle
    combined_df = X_all_features.copy()
    combined_df['target'] = y_binary_target
    
    # Önce hedefte NaN olanları (genellikle serinin sonu) at
    combined_df.dropna(subset=['target'], inplace=True)
    # Sonra özelliklerde NaN olanları at (indikatörlerin başlangıç periyotları nedeniyle)
    combined_df.dropna(subset=X_all_features.columns, inplace=True)
    
    # Hedefin hala 0 ve 1 içerdiğinden emin ol (bazen hepsi 0 veya 1 olabilir)
    if len(combined_df['target'].unique()) < 2:
        logging.error("Hedef değişken (target) tek bir sınıf içeriyor. Eşik veya veri aralığını kontrol edin.")
        return

    X = combined_df.drop('target', axis=1)
    y = combined_df['target'].astype(int)
    
    if X.empty or len(X) < 100: # Minimum örnek sayısı
        logging.error("Özellik (X) DataFrame'i NaN temizleme sonrası boş veya çok az veri içeriyor.")
        return

    logging.info(f"Son özellik setiyle eğitim için {len(X)} örnek veri hazırlandı.")
    logging.info(f"İkili sınıf dağılımı:\n{y.value_counts(normalize=True)}")

    # Veri Setini Bölme: Train -> Validation -> Test (Zaman sıralı)
    n_total = len(X)
    if n_total < 100: # Çok az veri varsa anlamlı bölme yapılamaz
        logging.error("Anlamlı Train/Val/Test bölmesi için yeterli veri yok.")
        return
        
    n_train_val = int(n_total * 0.8) # %80 Train+Val
    n_test = n_total - n_train_val   # %20 Test

    X_train_val, y_train_val = X.iloc[:n_train_val], y.iloc[:n_train_val]
    X_test, y_test = X.iloc[n_train_val:], y.iloc[n_train_val:]

    # Train_val setini de Train ve Validation olarak böl
    n_train_val_total = len(X_train_val)
    n_train = int(n_train_val_total * 0.75) # %75'i train, %25'i validation
    
    X_train, y_train = X_train_val.iloc[:n_train], y_train_val.iloc[:n_train]
    X_val, y_val = X_train_val.iloc[n_train:], y_train_val.iloc[n_train:]


    if X_train.empty or X_val.empty or X_test.empty:
        logging.error("Train/Validation/Test split sonrası en az bir set boş kaldı.")
        return
    logging.info(f"Eğitim seti: {len(X_train)}, Validasyon seti: {len(X_val)}, Test seti: {len(X_test)}")

    # Sınıf Dengesizliği için SMOTE (Sadece Eğitim Setine Uygula)
    logging.info("SMOTE ile eğitim verisi dengeleniyor...")
    smote = SMOTE(random_state=42)
    try:
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logging.info(f"SMOTE sonrası eğitim seti sınıf dağılımı:\n{pd.Series(y_train_smote).value_counts(normalize=True)}")
    except ValueError as e:
        logging.warning(f"SMOTE uygulanamadı (muhtemelen bir sınıf çok az örnek içeriyor): {e}. Orijinal eğitim verisi kullanılacak.")
        X_train_smote, y_train_smote = X_train, y_train # Hata durumunda orijinali kullan

    # Özellik Ölçekleme (SMOTE sonrası eğitim verisiyle fit et)
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train_smote)
    X_val_scaled_np = scaler.transform(X_val)
    X_test_scaled_np = scaler.transform(X_test)

    # Özellik isimlerini korumak için DataFrame'e geri çevir
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train.columns) # index X_train_smote'un indeksi olmalı ama önemli değil
    X_val_scaled = pd.DataFrame(X_val_scaled_np, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=X_test.columns, index=X_test.index)

    # Hiperparametre Optimizasyonu (Optuna ile XGBoost için)
    logging.info("Optuna ile XGBoost hiperparametre optimizasyonu başlatılıyor...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X_train_scaled, y_train_smote, X_val_scaled, y_val), 
                       n_trials=n_optuna_trials)

    best_xgb_params = study_xgb.best_params
    logging.info(f"XGBoost için en iyi hiperparametreler bulundu: {best_xgb_params}")
    logging.info(f"XGBoost için en iyi validasyon doğruluğu: {study_xgb.best_value:.4f}")

    # En iyi parametrelerle final XGBoost modelini eğit (SMOTE'lanmış tüm eğitim verisi üzerinde)
    logging.info("En iyi parametrelerle final XGBoost modeli eğitiliyor...")
    final_xgb_model = xgb.XGBClassifier(**best_xgb_params, use_label_encoder=False, objective='binary:logistic', random_state=42)
    # İsteğe bağlı: Tüm train+val üzerinde eğitme
    # X_train_val_smote_scaled = pd.concat([X_train_scaled, X_val_scaled]) # Dikkat: X_val_scaled SMOTE'lanmadı
    # y_train_val_smote = pd.concat([pd.Series(y_train_smote), y_val]) # Bu da SMOTE'lanmadı, dikkat
    # Bu kısmı basitleştirmek için sadece SMOTE'lanmış train setiyle eğitelim
    final_xgb_model.fit(X_train_scaled, y_train_smote) # SMOTE'lanmış eğitim verisiyle

    # Model Değerlendirmesi (Test Seti Üzerinde)
    logging.info("\n--- XGBoost Test Seti Performansı (Optimize Edilmiş Model) ---")
    y_pred_test_xgb = final_xgb_model.predict(X_test_scaled)
    y_pred_proba_test_xgb = final_xgb_model.predict_proba(X_test_scaled)[:, 1] # Pozitif sınıfın olasılığı

    accuracy_test_xgb = accuracy_score(y_test, y_pred_test_xgb)
    logging.info(f"Genel Doğruluk (Test Seti - XGBoost): {accuracy_test_xgb*100:.2f}%")
    
    class_labels_binary_str = ['SHORT', 'LONG'] # Sadece 0 ve 1 var artık
    report_xgb = classification_report(y_test, y_pred_test_xgb, target_names=class_labels_binary_str, zero_division=0)
    logging.info(f"Sınıflandırma Raporu (Test Seti - XGBoost):\n{report_xgb}")

    logging.info("Karışıklık Matrisi (Test Seti - XGBoost):")
    cm_xgb = confusion_matrix(y_test, y_pred_test_xgb)
    cm_df_xgb = pd.DataFrame(cm_xgb, index=class_labels_binary_str, columns=[f"Pred_{s}" for s in class_labels_binary_str])
    print(cm_df_xgb)

    # ROC AUC
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_test_xgb)
    logging.info(f"ROC AUC Skoru (Test Seti - XGBoost): {roc_auc_xgb:.4f}")
    # plot_roc_auc(y_test, y_pred_proba_test_xgb) # Görselleştirme kaldırıldı

    # Özellik Önem Sıralaması
    if hasattr(final_xgb_model, 'feature_importances_'):
        logging.info("\n--- Özellik Önem Sıralaması (XGBoost) ---")
        feature_importances_xgb = pd.Series(final_xgb_model.feature_importances_, index=X_train.columns) # X_train orijinal özellik isimlerini taşır
        print(feature_importances_xgb.sort_values(ascending=False).to_string())

    # Modeli ve Scaler'ı Kaydet
    logging.info("Optimize edilmiş XGBoost modeli ve scaler kaydediliyor...")
    joblib.dump(final_xgb_model, "market_direction_xgb_model.joblib")
    joblib.dump(scaler, "market_direction_xgb_scaler.joblib")
    logging.info("XGBoost modeli ve scaler başarıyla kaydedildi.")

    logging.info("\n--- XGBoost Tahmin Olasılıkları ile Performans Analizi (Test Seti) ---")
    # y_pred_proba_test_xgb zaten hesaplanmıştı: final_xgb_model.predict_proba(X_test_scaled)
    # Bu (n_samples, n_classes) şeklinde bir arraydir.
    # Bizim durumumuzda n_classes = 2 (0:SHORT, 1:LONG)

    # Modelin sınıflarının sıralamasını alalım (genellikle [0, 1] olur ama teyit edelim)
    model_classes_xgb = final_xgb_model.classes_
    logging.info(f"XGBoost Model Sınıfları (sıralama): {model_classes_xgb} (Beklenen: 0:SHORT, 1:LONG)")
    
    try:
        # Sınıf etiketlerine karşılık gelen indeksleri bul
        # Bu, model.classes_ sıralaması her zaman [0, 1] olmasa bile doğru çalışır.
        idx_short_xgb = np.where(model_classes_xgb == 0)[0][0]
        idx_long_xgb = np.where(model_classes_xgb == 1)[0][0]
    except IndexError:
        logging.error("XGBoost model sınıfları beklenenden farklı veya eksik. Varsayılan sıralama (0:SHORT, 1:LONG) kullanılıyor.")
        idx_short_xgb, idx_long_xgb = 0, 1 # Varsayılan, eğer model.classes_ [0,1] ise

    # Olasılıkları alalım
    # y_pred_proba_test_xgb[:, idx_short_xgb] -> SHORT olasılığı
    # y_pred_proba_test_xgb[:, idx_long_xgb]  -> LONG olasılığı
    # Daha önce y_pred_proba_test_xgb'yi sadece pozitif sınıfın (LONG) olasılığı olarak almıştık.
    # Şimdi tüm olasılık matrisine ihtiyacımız var.
    all_probabilities_xgb = final_xgb_model.predict_proba(X_test_scaled)


    for threshold_pct in [50, 55, 60, 65, 70, 75, 80]: # Yüzde olarak eşikler
        threshold = threshold_pct / 100.0
        logging.info(f"\n--- Olasılık Eşiği (XGBoost): {threshold_pct}% ---")
        
        # Yeni tahminleri eşiğe göre yap
        # Bu ikili bir sınıflandırma olduğu için, bir yöne karar veremezsek
        # "işlem yapma" veya "belirsiz" gibi bir durum olmayacak doğrudan.
        # Ya SHORT ya da LONG diyeceğiz (veya birini varsayılan yapacağız).
        # Ya da, eğer her iki olasılık da eşiğin altındaysa, bir "NEUTRAL_IMPLIED" durumu yaratabiliriz.
        
        # Strateji 1: Sadece bir sınıf eşiği geçerse o sınıfı ata, yoksa bir varsayılan ata (örn: çoğunluk sınıfı)
        # Strateji 2: Yüksek olasılıklı olanı seç, eğer o olasılık da eşiği geçiyorsa.
        # Strateji 3 (Burada uygulayacağımız): Her iki yönde de sinyal yoksa, bunu pratikte "işlem yok" (NEUTRAL gibi) olarak yorumlayabiliriz.
        # Ancak classification_report hala ikili (SHORT/LONG) metrikler verecektir.
        # Karışıklığı önlemek için, eğer hiçbir sınıf eşiği geçmezse, çoğunluk sınıfını (SHORT) tahmin edelim.
        # Ya da daha iyisi, eğer amacımız yüksek kesinlikli sinyallerse, eşiği geçmeyenleri "tahmin yok" (NaN) gibi işaretleyip,
        # sadece eşiği geçenler üzerinden precision/recall hesaplayabiliriz.
        # Şimdilik, "tahmin yok" yerine, en olası olanı veya bir varsayılanı atayalım.

        # Yeni y_pred_thresholded_xgb oluşturalım
        # Eğer LONG olasılığı >= threshold ise LONG (1)
        # Eğer SHORT olasılığı >= threshold ise SHORT (0)
        # Eğer ikisi de değilse (veya ikisi de eşiği geçiyorsa), ne yapmalı?
        # Öncelik: Daha yüksek olasılıklı olan ve eşiği geçen.
        # Eğer ikisi de eşiği geçiyorsa, en yüksek olasılıklı olan.
        # Eğer hiçbiri eşiği geçmiyorsa, modelin normalde tahmin edeceği (en yüksek olasılıklı) sınıfı alalım.
        
        y_pred_thresholded_xgb = np.argmax(all_probabilities_xgb, axis=1) # Önce varsayılan tahminleri al
        
        prob_short_class = all_probabilities_xgb[:, idx_short_xgb]
        prob_long_class = all_probabilities_xgb[:, idx_long_xgb]

        # Koşulları belirle
        is_confident_short = prob_short_class >= threshold
        is_confident_long = prob_long_class >= threshold

        # Güncelleme mantığı:
        # 1. Güvenli LONG varsa ata
        # 2. Güvenli LONG yoksa AMA güvenli SHORT varsa ata
        # 3. İkisi de güvenli değilse, ilk (argmax) tahmini kalsın.

        temp_preds = y_pred_thresholded_xgb.copy() # Orijinali değiştirmemek için kopya

        # Herhangi bir yön için eşik geçilmiyorsa, o tahmini geçersiz kılabiliriz (örn: -1 ata)
        # ve sadece geçerli tahminler üzerinden metrik hesaplayabiliriz. Bu, "işlem yapmama" durumunu simüle eder.
        # Bu, "support" sayılarını azaltır ama precision'ı daha anlamlı kılar.
        
        # Yeni yaklaşım: Eğer bir sinyal eşiği geçmiyorsa, onu "tahmin edilmedi" sayalım.
        # Ve metrikleri sadece tahmin edilenler üzerinden hesaplayalım.
        # Ya da, 3 sınıflı (SHORT, LONG, NO_SIGNAL) bir çıktıya dönüştürelim.
        # Şimdilik, sadece precision'a odaklanmak için, eşiği geçmeyenleri filtreden çıkaralım.

        actual_y_for_eval = []
        predicted_y_for_eval = []

        for i in range(len(y_test)):
            predicted_class = -1 # Varsayılan: sinyal yok
            if is_confident_long[i] and (not is_confident_short[i] or prob_long_class[i] >= prob_short_class[i]):
                predicted_class = idx_long_xgb
            elif is_confident_short[i]: # Zaten ~is_confident_long[i] veya long olasılığı daha düşük
                predicted_class = idx_short_xgb
            
            if predicted_class != -1: # Sadece güvenli bir tahmin varsa değerlendirmeye kat
                actual_y_for_eval.append(y_test.iloc[i])
                predicted_y_for_eval.append(predicted_class)
        
        if not actual_y_for_eval: # Eğer hiçbir güvenli tahmin üretilemediyse
            logging.warning(f"Olasılık eşiği {threshold_pct}% ile hiç güvenli tahmin üretilemedi.")
            continue

        actual_y_for_eval = np.array(actual_y_for_eval)
        predicted_y_for_eval = np.array(predicted_y_for_eval)
        
        accuracy_thresh = accuracy_score(actual_y_for_eval, predicted_y_for_eval)
        logging.info(f"Genel Doğruluk (Eşikli, Sadece Güvenli Tahminler): {accuracy_thresh*100:.2f}%")
        
        # Sınıflandırma raporu ve karışıklık matrisi için etiketleri doğru alalım
        unique_labels_actual_thresh = np.unique(actual_y_for_eval)
        unique_labels_pred_thresh = np.unique(predicted_y_for_eval)
        # Hem gerçekte hem de tahminde olan tüm etiketleri alalım
        combined_labels_numeric_thresh = np.unique(np.concatenate((unique_labels_actual_thresh, unique_labels_pred_thresh))).astype(int)
        
        # Eğer bir sınıf hiç oluşmadıysa (örn: tüm tahminler SHORT ise ve gerçekte LONG yoksa), target_names'te sorun olabilir.
        # target_names her zaman ['SHORT', 'LONG'] olmalı.
        # labels parametresi, raporda hangi sınıfların gösterileceğini belirler.
        # Eğer bir sınıf hiç tahmin edilmediyse veya gerçekte yoksa, raporda 0 support ile görünebilir.

        current_target_names_thresh = [class_labels_binary_str[i] for i in combined_labels_numeric_thresh if i < len(class_labels_binary_str)]
        if not current_target_names_thresh: # Eğer hiç etiket yoksa (çok nadir)
             if combined_labels_numeric_thresh.size > 0: # En az bir etiket varsa ama isim bulunamadıysa
                 current_target_names_thresh = [str(l) for l in combined_labels_numeric_thresh] # Sayısal olarak kullan
             else: # Hiç etiket yoksa (çok çok nadir)
                 logging.warning("Eşikli değerlendirme için hiç etiket bulunamadı.")
                 continue


        report_thresh = classification_report(actual_y_for_eval, predicted_y_for_eval, 
                                              labels=combined_labels_numeric_thresh, 
                                              target_names=current_target_names_thresh, 
                                              zero_division=0)
        logging.info(f"Sınıflandırma Raporu (Eşikli, Sadece Güvenli Tahminler):\n{report_thresh}")

        cm_thresh = confusion_matrix(actual_y_for_eval, predicted_y_for_eval, labels=combined_labels_numeric_thresh)
        # Eğer cm_thresh boşsa veya current_target_names_thresh boşsa hata alabiliriz.
        if cm_thresh.size > 0 and current_target_names_thresh:
            cm_df_thresh = pd.DataFrame(cm_thresh, 
                                         index=current_target_names_thresh, 
                                         columns=[f"Pred_{name}" for name in current_target_names_thresh])
            print("Karışıklık Matrisi (Eşikli, Sadece Güvenli Tahminler):")
            print(cm_df_thresh)
        else:
            logging.warning("Karışıklık matrisi oluşturulamadı (veri yok veya etiket isimleri eksik).")

if __name__ == "__main__":
    # n_optuna_trials sayısını buradan ayarlayabilirsiniz veya script içinde sabit bırakabilirsiniz.
    optimize_and_train_xgboost_model(n_optuna_trials=30) # Örnek 30 deneme
