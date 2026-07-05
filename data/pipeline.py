"""
Pipeline de récupération des données de prix (OHLCV).
Étape 1 du projet : télécharger l'historique de prix pour chaque ticker,
calculer les rendements et la volatilité réalisée.
"""
import yfinance as yf
import pandas as pd
import sys
import os

# Permet d'importer config.py qui est à la racine du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def fetch_ohlcv(ticker, start_date, end_date):
    """
    Télécharge les données OHLCV (Open, High, Low, Close, Volume)
    pour un ticker donné, sur une période donnée.

    Retourne un DataFrame avec : ticker, close, volume, daily_return
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # yfinance renvoie des colonnes "MultiIndex" (Price, Ticker).
    # On aplatit pour ne garder que le premier niveau (Close, High, Low, Open, Volume).
    df.columns = df.columns.get_level_values(0)

    # On ajoute le nom du ticker comme colonne (utile plus tard quand on combine plusieurs tickers)
    df['ticker'] = ticker

    # Rendement journalier = variation en % du prix de clôture d'un jour à l'autre
    df['daily_return'] = df['Close'].pct_change()

    # On ne garde que les colonnes qui nous intéressent pour la suite
    result = df[['ticker', 'Close', 'Volume', 'daily_return']].copy()
    result.index.name = 'date'

    return result


def fetch_all_tickers(tickers, start_date, end_date):
    """
    Télécharge les données OHLCV pour une liste de tickers,
    et les combine en un seul DataFrame (empilés les uns sous les autres).
    """
    all_data = []

    for ticker in tickers:
        print(f"Téléchargement de {ticker}...")
        df = fetch_ohlcv(ticker, start_date, end_date)
        all_data.append(df)

    # pd.concat empile tous les DataFrames les uns sous les autres
    combined = pd.concat(all_data)
    return combined


def calculate_realized_volatility(df, window_short=20, window_long=60):
    """
    Calcule la volatilité réalisée (rolling std des rendements) pour un DataFrame
    contenant potentiellement plusieurs tickers.

    Important : le calcul se fait ticker par ticker, pour ne jamais mélanger
    les rendements d'AMD avec ceux de NVDA par exemple.
    """
    df = df.copy()

    # groupby('ticker') : on calcule séparément pour chaque ticker
    # transform garde le même nombre de lignes que le DataFrame d'origine
    df['realized_vol_20d'] = df.groupby('ticker')['daily_return'].transform(
        lambda x: x.rolling(window=window_short).std() * 100
    )
    df['realized_vol_60d'] = df.groupby('ticker')['daily_return'].transform(
        lambda x: x.rolling(window=window_long).std() * 100
    )

    return df


if __name__ == "__main__":
    # 1. Récupération des prix pour tous les tickers
    all_data = fetch_all_tickers(config.TICKERS, config.START_DATE, config.END_DATE)

    # 2. Calcul de la volatilité réalisée (20j et 60j)
    all_data = calculate_realized_volatility(
        all_data,
        window_short=config.REALIZED_VOL_WINDOW_SHORT,
        window_long=config.REALIZED_VOL_WINDOW_LONG
    )

    print(f"\nNombre total de lignes : {len(all_data)}")
    print(f"Tickers présents : {sorted(all_data['ticker'].unique())}")

    print(f"\nAperçu pour AMD (à partir du jour 60, pour voir vol_60d rempli) :")
    amd_data = all_data[all_data['ticker'] == 'AMD']
    print(amd_data.iloc[58:65])

    print(f"\nValeurs manquantes par colonne :")
    print(all_data.isna().sum())