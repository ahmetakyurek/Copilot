import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# 1. Veri Hazırlama
df = pd.read_csv("BTCUSDT_1h.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Feature Engineering (ATR, ATR_PCT ve ADX dahil!)
def add_features(df):
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

    # ATR ve ATR_PCT
    df["tr"] = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    # ADX hesaplama
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = low.shift(1) - low
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df["adx"] = dx.rolling(14).mean()

    df["volume_change"] = df["volume"].pct_change(5)
    df["vol_spike"] = (df["volume"] > df["volume"].rolling(20).mean() * 1.5).astype(int)
    return df

df = add_features(df)

# 3. Etiketleme
LOOKFORWARD = 24
THRESHOLD = 0.003  # Optimize edilmiş, cesur ama abartısız

labels = []
for idx in range(len(df) - LOOKFORWARD):
    future_prices = df["close"].iloc[idx+1:idx+LOOKFORWARD+1].values
    future_pct_chg = (future_prices - df["close"].iloc[idx]) / df["close"].iloc[idx]
    max_up = np.max(future_pct_chg)
    min_down = np.min(future_pct_chg)
    if (max_up >= THRESHOLD):
        labels.append(1)
    elif (min_down <= -THRESHOLD):
        labels.append(0)
    else:
        labels.append(np.nan)

df = df.iloc[:len(labels)]
df["target"] = labels

# 4. Temizleme ve Feature Listesi
features = [
    "ema_diff", "ema_cross_up", "ema_cross_down",
    "rsi", "macd", "macd_signal", "macd_cross_up", "macd_cross_down",
    "bb_mavg", "bb_upper", "bb_lower", "bb_breakout",
    "momentum_10", "roc_10", "atr14", "atr_pct", "adx",
    "volume_change", "vol_spike"
]
df = df.dropna(subset=features + ["target"])

# 5. Eğitim/Validasyon/Test
X = df[features]
y = df["target"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Tüm veri setlerinde sonsuz veya NaN değer varsa temizle
def clean_inf_nan(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()

X_train = clean_inf_nan(X_train)
X_valid = clean_inf_nan(X_valid)
X_test  = clean_inf_nan(X_test)
y_train = y_train.loc[X_train.index]
y_valid = y_valid.loc[X_valid.index]
y_test  = y_test.loc[X_test.index]

# Sınıf dağılımı kontrolü
print("Eğitimdeki sınıf dağılımı:\n", y_train.value_counts())

# 6. Model ve Class Weight (optimize)
scale_pos_weight = 0.5  # Daha dengeli, optimize edilmiş değer

xgb = XGBClassifier(
    n_estimators=120,
    max_depth=5,
    learning_rate=0.12,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train, verbose=True)

y_pred = xgb.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

importances = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
print("Önemli Feature'lar:\n", importances)