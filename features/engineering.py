"""
Feature engineering : transforme les données brutes en indicateurs
exploitables par les modèles (GARCH, XGBoost, LSTM).
"""
import pandas as pd
import numpy as np
import sys
import os
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(db_path):
    """Charge les données depuis SQLite et les trie proprement."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM volatility_data', conn, parse_dates=['date'])
    conn.close()

    # Tri essentiel : par ticker puis par date, pour que tous les calculs
    # "rolling" et "shift" respectent l'ordre chronologique correct
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    return df


def engineer_features(df):
    """
    Crée l'ensemble des features à partir des données brutes.
    Tous les calculs se font ticker par ticker (groupby) pour ne jamais
    mélanger les séries temporelles de deux actions différentes.
    """
    df = df.copy()
    g = df.groupby('ticker')  # on réutilise ce groupby pour toutes les features

    # --- 1. Features de volatilité ---
    df['vol_lag1'] = g['realized_vol_20d'].transform(lambda x: x.shift(1))
    df['vol_lag5'] = g['realized_vol_20d'].transform(lambda x: x.shift(5))
    df['vol_lag20'] = g['realized_vol_20d'].transform(lambda x: x.shift(20))
    df['vol_trend'] = g['realized_vol_20d'].transform(lambda x: x.pct_change())

    # --- 2. Features basées sur les rendements ---
    df['returns_std_5d'] = g['daily_return'].transform(lambda x: x.rolling(5).std())
    df['returns_skew'] = g['daily_return'].transform(lambda x: x.rolling(20).skew())
    df['returns_kurtosis'] = g['daily_return'].transform(lambda x: x.rolling(20).kurt())

    # --- 3. Features de régime de marché ---
    # VIX et yield_10y sont identiques pour tous les tickers à une date donnée.
    # On calcule donc la variation sur les dates UNIQUES, triées chronologiquement,
    # pour ne jamais mélanger les séries de deux tickers différents.
    macro_unique = df[['date', 'vix', 'yield_10y']].drop_duplicates(subset='date').sort_values('date')
    macro_unique['vix_change'] = macro_unique['vix'].pct_change()
    macro_unique['yield_change'] = macro_unique['yield_10y'].pct_change()

    df = df.merge(
        macro_unique[['date', 'vix_change', 'yield_change']],
        on='date',
        how='left'
    )

    # --- 4. Features de volume ---
    df['volume_ma20'] = g['Volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_ratio'] = df['Volume'] / df['volume_ma20']

    # --- 5. Features calendaires ---
    df['day_of_week'] = df['date'].dt.dayofweek  # 0 = lundi, 4 = vendredi
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    return df


if __name__ == "__main__":
    # 1. Charger les données brutes
    df = load_data(config.DB_PATH)
    print(f"Données chargées : {len(df)} lignes")

    # 2. Créer les features
    df_features = engineer_features(df)

    # Liste des nouvelles colonnes créées (pour vérifier qu'on a bien tout)
    new_cols = [
        'vol_lag1', 'vol_lag5', 'vol_lag20', 'vol_trend',
        'returns_std_5d', 'returns_skew', 'returns_kurtosis',
        'vix_change', 'yield_change',
        'volume_ma20', 'volume_ratio',
        'day_of_week', 'month', 'quarter'
    ]
    print(f"\nNombre de nouvelles features créées : {len(new_cols)}")

    # 3. Aperçu pour un ticker
    print(f"\nAperçu (AMD, lignes 60-65, pour avoir tout amorcé) :")
    amd = df_features[df_features['ticker'] == 'AMD']
    print(amd[['date'] + new_cols].iloc[60:65])

    # 4. Valeurs manquantes par colonne
    print(f"\nValeurs manquantes par nouvelle feature :")
    print(df_features[new_cols].isna().sum())