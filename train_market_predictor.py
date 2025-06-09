import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
import os
from dotenv import load_dotenv
import optuna

from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

import ta  # Teknik indikatörler için
from ml_data_utils import get_klines_iterative, create_market_direction_target, create_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY_FUTURES')
API_SECRET = os.getenv('BINANCE_API_SECRET_FUTURES')

if not API_KEY or not API_SECRET:
    logging.warning("UYARI: .env dosyasından Binance API anahtarları yüklenemedi.")

SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
START_DATE_UTC = "2022-01-01 00:00:00"
END_DATE_UTC = "2024-05-01 00:00:00"
LOOK_FORWARD_PERIODS = 12
PRICE_CHANGE_THRESHOLD_PCT = 0.03  # NEUTRAL'i azaltmak için artırıldı

def create_binary_target(close, lookforward, pct):
    future_change = close.shift(-lookforward) / close - 1
    # Yükseliş: 1, Düşüş: 0, NEUTRAL yok
    return (future_change > pct).astype(int)

def optimize_and_train_binary_model():
    logging.info(f"{SYMBOL} için veri çekiliyor ({START_DATE_UTC} - {END_DATE_UTC})...")
    client = Client()
    df_raw = get_klines_iterative(client, SYMBOL, INTERVAL, START_DATE_UTC, END_DATE_UTC, limit=1000)

    if df_raw.empty or len(df_raw) < 200:
        logging.error("Eğitim için yeterli geçmiş veri çekilemedi.")
        return

    logging.info("Teknik indikatörler ve özellikler oluşturuluyor...")
    df_with_ta = create_advanced_features(df_raw)
    X_features = create_features(df_with_ta)

    logging.info("Binary hedef değişken oluşturuluyor (NEUTRAL yok)...")
    y_target = create_binary_target(df_raw['close'], LOOK_FORWARD_PERIODS, PRICE_CHANGE_THRESHOLD_PCT)
    
    combined_df = X_features.copy()
    combined_df['target'] = y_target
    combined_df.dropna(inplace=True)

    # Sadece 0 ve 1 içeren satırları al
    combined_df = combined_df[combined_df['target'].isin([0, 1])]

    X = combined_df.drop('target', axis=1)
    y = combined_df['target'].astype(int)
    logging.info(f"Yeni sınıf dağılımı:\n{y.value_counts(normalize=True)}")

def create_advanced_features(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_ratio'] = df['ema_9'] / df['ema_21']
    df['price_zscore'] = (df['close'] - df['close'].rolling(100).mean()) / df['close'].rolling(100).std()
    df['momentum'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_3'] = df['close'].pct_change(3)
    return df

def plot_confusion_matrix_heatmap(y_true, y_pred, class_labels_num, class_labels_str, title="Karışıklık Matrisi Heatmap"):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels_num)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels_str, yticklabels=class_labels_str)
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_auc(y_true, y_pred_proba, n_classes, class_labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true==i).astype(int), y_pred_proba[:, i])
        roc_auc[i] = roc_auc_score((y_true==i).astype(int), y_pred_proba[:, i])
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Eğrisi')
    plt.legend()
    plt.tight_layout()
    plt.show()

def optimize_catboost(X_train, y_train, X_val, y_val, n_trials=15):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 3, 10),
            'loss_function': 'MultiClass',
            'verbose': False,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print("CatBoost Best Params:", study.best_params)
    return study.best_params

def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=15):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }
        # ESKİ: model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print("XGBoost Best Params:", study.best_params)
    return study.best_params

def optimize_and_train_model():
    logging.info(f"{SYMBOL} için veri çekiliyor ({START_DATE_UTC} - {END_DATE_UTC})...")
    client = Client()
    df_raw = get_klines_iterative(client, SYMBOL, INTERVAL, START_DATE_UTC, END_DATE_UTC, limit=1000)

    if df_raw.empty or len(df_raw) < 200:
        logging.error("Eğitim için yeterli geçmiş veri çekilemedi.")
        return

    # --- HATALARI ÖNLE: İLK TA, SONRA KENDİ FEATURES ---
    logging.info("Teknik indikatörler ve özellikler oluşturuluyor...")
    df_with_ta = create_advanced_features(df_raw)
    X_features = create_features(df_with_ta)

    logging.info("Hedef değişken ('target_direction') oluşturuluyor...")
    y_target = create_market_direction_target(df_raw['close'], LOOK_FORWARD_PERIODS, PRICE_CHANGE_THRESHOLD_PCT)
    
    combined_df = X_features.copy()
    combined_df['target'] = y_target
    combined_df.dropna(inplace=True)

    if combined_df.empty or len(combined_df['target'].unique()) < 2:
        logging.error("NaN'lar temizlendikten sonra veri kalmadı veya hedefte yeterli çeşitlilik yok.")
        return

    X = combined_df.drop('target', axis=1)
    y = combined_df['target'].astype(int)

    logging.info(f"Eğitim/Validasyon/Test için {len(X)} örnek veri hazırlandı.")
    logging.info(f"Sınıf dağılımı:\n{y.value_counts(normalize=True)}")

    # Train/Val/Test split
    n_total = len(X)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train: n_train + n_val], y.iloc[n_train: n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    if X_train.empty or X_val.empty or X_test.empty:
        logging.error("Train/Validation/Test split sonrası en az bir set boş kaldı.")
        return

    logging.info(f"Eğitim seti: {len(X_train)}, Validasyon seti: {len(X_val)}, Test seti: {len(X_test)}")

    # SMOTE: Sınıf dengesizliği için
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # Feature Selection (RFE ile en iyi 25 özellik)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    n_features = X_train_scaled.shape[1]
    n_select = min(24, n_features)  # 24 özelliğin varsa 24 seç
    selector = RFE(estimator=LogisticRegression(max_iter=1500),
    n_features_to_select=n_select, step=1)
    X_train_fs = pd.DataFrame(selector.fit_transform(X_train_scaled, y_train_bal))
    X_val_fs = pd.DataFrame(selector.transform(X_val_scaled))
    X_test_fs = pd.DataFrame(selector.transform(X_test_scaled))
    feat_indices = selector.get_support(indices=True)
    selected_columns = X_train.columns[feat_indices]
    print("Seçilen özellikler:", list(selected_columns))

    # Hiperparametre optimizasyonu
    xgb_params = optimize_xgboost(X_train_fs, y_train_bal, X_val_fs, y_val, n_trials=10)
    cat_params = optimize_catboost(X_train_fs, y_train_bal, X_val_fs, y_val, n_trials=10)

    # Ana modeller
    lgbm = LGBMClassifier(n_estimators=300, random_state=42, class_weight='balanced', verbosity=-1)
    xgbc = xgb.XGBClassifier(**xgb_params, use_label_encoder=False, num_class=3, objective='multi:softmax')
    catb = CatBoostClassifier(**cat_params, loss_function='MultiClass', verbose=False, random_seed=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')

    # ENSEMBLE: stacking!
    estimators = [
        ('lgbm', lgbm),
        ('xgb', xgbc),
        ('cat', catb),
        ('rf', rf)
    ]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1200), n_jobs=-1)

    # Hepsini testte karşılaştır
    models = {
        "LightGBM": lgbm,
        "XGBoost": xgbc,
        "CatBoost": catb,
        "RandomForest": rf,
        "StackingEnsemble": stack
    }
    class_labels_num = [0, 1, 2]
    class_labels_str = ['SHORT', 'LONG', 'NEUTRAL']

    scores = {}
    for name, model in models.items():
        print(f"\n========== {name} ==========")
        model.fit(X_train_fs, y_train_bal)
        y_pred_test = model.predict(X_test_fs)
        acc = accuracy_score(y_test, y_pred_test)
        scores[name] = acc
        print(f"{name} Doğruluk: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred_test, target_names=class_labels_str, zero_division=0))
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=selected_columns)
            print("Önemli Özellikler:\n", fi.sort_values(ascending=False).head(10))
        plot_confusion_matrix_heatmap(
            y_test, y_pred_test, 
            class_labels_num=class_labels_num, 
            class_labels_str=class_labels_str, 
            title=f"{name} Karışıklık Matrisi"
        )
        # ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_fs)
            plot_roc_auc(y_test, y_proba, n_classes=3, class_labels=class_labels_str)
    best_model = max(scores, key=scores.get)
    print(f"\n*** EN İYİ MODELİN: {best_model} | Doğruluk: {scores[best_model]*100:.2f}% ***")

if __name__ == "__main__":
    optimize_and_train_binary_model()
