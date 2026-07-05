"""
Pipeline de récupération des données de prix (OHLCV).
Étape 1 du projet : télécharger l'historique de prix pour chaque ticker.
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


if __name__ == "__main__":
    # Test avec les vrais paramètres du projet (définis dans config.py)
    all_data = fetch_all_tickers(config.TICKERS, config.START_DATE, config.END_DATE)

    print(f"\nNombre total de lignes : {len(all_data)}")
    print(f"Tickers présents : {sorted(all_data['ticker'].unique())}")
    print(f"\nAperçu :")
    print(all_data.head())
    print(f"\nValeurs manquantes par colonne :")
    print(all_data.isna().sum())
