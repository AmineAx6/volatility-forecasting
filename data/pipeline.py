"""
Pipeline de récupération des données de prix (OHLCV).
Étape 1 du projet : télécharger l'historique de prix pour chaque ticker,
calculer les rendements et la volatilité réalisée.
"""
import yfinance as yf
import pandas as pd
import sys
import os

from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

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

def fetch_macro_data(start_date, end_date):
    """
    Récupère les données macro depuis FRED :
    - VIX (indice de volatilité du marché)
    - Taux du Trésor américain à 10 ans

    Retourne un DataFrame avec une ligne par jour, indexé par date.
    """
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError(
            "Clé FRED_API_KEY introuvable. Vérifie que le fichier .env existe "
            "et contient bien FRED_API_KEY=ta_cle"
        )

    fred = Fred(api_key=api_key)

    # VIXCLS = code FRED pour le VIX (CBOE Volatility Index)
    vix = fred.get_series('VIXCLS', observation_start=start_date, observation_end=end_date)

    # DGS10 = code FRED pour le taux du Trésor US à 10 ans
    yield_10y = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)

    macro_df = pd.DataFrame({
        'vix': vix,
        'yield_10y': yield_10y
    })
    macro_df.index.name = 'date'

    return macro_df

if __name__ == "__main__":
    # 1. Récupération des prix pour tous les tickers
    all_data = fetch_all_tickers(config.TICKERS, config.START_DATE, config.END_DATE)

    # 2. Calcul de la volatilité réalisée (20j et 60j)
    all_data = calculate_realized_volatility(
        all_data,
        window_short=config.REALIZED_VOL_WINDOW_SHORT,
        window_long=config.REALIZED_VOL_WINDOW_LONG
    )

    print(f"\nNombre total de lignes (prix) : {len(all_data)}")

    # 3. Récupération des données macro (VIX, taux 10 ans)
    macro_data = fetch_macro_data(config.START_DATE, config.END_DATE)

    print(f"\nAperçu des données macro :")
    print(macro_data.head())
    print(f"\nNombre de lignes (macro) : {len(macro_data)}")
    print(f"Valeurs manquantes :")
    print(macro_data.isna().sum())