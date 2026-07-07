"""
Préparation des données pour l'entraînement ML (XGBoost, LSTM).
Split train/test temporel + standardisation.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from features.engineering import load_data, engineer_features

# Colonnes à exclure des features : identifiants, prix bruts non stationnaires, la cible elle-même
EXCLUDE_COLS = ['date', 'ticker', 'Close', 'Volume', 'target_vol_10d']

FEATURE_COLS_BASE = [
    'daily_return', 'realized_vol_20d', 'realized_vol_60d', 'vix', 'yield_10y',
    'vol_lag1', 'vol_lag5', 'vol_lag20', 'vol_trend',
    'returns_std_5d', 'returns_skew', 'returns_kurtosis',
    'vix_change', 'yield_change',
    'volume_ma20', 'volume_ratio',
    'day_of_week', 'month', 'quarter'
]


def prepare_ml_data(df, test_size=0.2):
    """
    Prépare X_train, X_test, y_train, y_test pour l'entraînement ML.

    - Encode le ticker en one-hot (pour que le modèle distingue les actions)
    - Split TEMPOREL (pas aléatoire) : les dates les plus récentes vont dans le test
    - Standardise les features, fit uniquement sur le train (jamais sur le test)
    """
    df = df.copy()

    # 1. One-hot encoding du ticker : "AMD" devient une colonne ticker_AMD = 1, les autres = 0
    ticker_dummies = pd.get_dummies(df['ticker'], prefix='ticker')
    df = pd.concat([df, ticker_dummies], axis=1)

    feature_cols = FEATURE_COLS_BASE + list(ticker_dummies.columns)

    # 2. On ne garde que les lignes complètes (features ET target présents)
    mask = ~(df[feature_cols].isna().any(axis=1) | df['target_vol_10d'].isna())
    df_clean = df[mask].sort_values('date')

    # 3. Split temporel : une date de coupure, pas un split aléatoire
    split_date = df_clean['date'].quantile(1 - test_size)

    train_df = df_clean[df_clean['date'] < split_date]
    test_df = df_clean[df_clean['date'] >= split_date]

    X_train, y_train = train_df[feature_cols], train_df['target_vol_10d']
    X_test, y_test = test_df[feature_cols], test_df['target_vol_10d']

    # 4. Standardisation : fit UNIQUEMENT sur le train, pour ne pas "fuiter"
    #    d'information du test vers l'entraînement
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, split_date


if __name__ == "__main__":
    df = load_data(config.DB_PATH)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test, scaler, split_date = prepare_ml_data(df)

    print(f"Date de coupure train/test : {split_date.date()}")
    print(f"Train : {len(X_train)} lignes")
    print(f"Test  : {len(X_test)} lignes")
    print(f"Nombre de features : {X_train.shape[1]}")
    print(f"\nColonnes : {list(X_train.columns)}")
    print(f"\nAperçu X_train (standardisé) :")
    print(X_train.head())
    print(f"\nStats y_train : min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")
    print(f"Stats y_test  : min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")