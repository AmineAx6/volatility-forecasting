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

def merge_data(price_df, macro_df):
    """
    Fusionne les données de prix/volatilité (par ticker) avec les données
    macro (VIX, taux 10 ans), qui sont identiques pour tous les tickers
    un jour donné.
    """
    # join fusionne sur l'index (la date), en gardant toutes les lignes de price_df
    # et en associant la bonne ligne macro à chaque date
    merged = price_df.join(macro_df, how='left')

    # Les jours fériés bancaires US peuvent créer des trous dans le VIX/taux.
    # On "remplit vers l'avant" : si une valeur manque, on reprend la dernière connue.
    merged['vix'] = merged['vix'].ffill()
    merged['yield_10y'] = merged['yield_10y'].ffill()

    return merged

def create_target(df, horizon=10):
    """
    Crée la variable cible : volatilité réalisée dans `horizon` jours.
    Le calcul se fait ticker par ticker pour ne jamais mélanger les séries.
    """
    df = df.copy()

    # shift(-horizon) décale les valeurs vers le HAUT de `horizon` lignes :
    # la ligne du jour T récupère la valeur de realized_vol_20d du jour T+10
    df['target_vol_10d'] = df.groupby('ticker')['realized_vol_20d'].transform(
        lambda x: x.shift(-horizon)
    )

    return df


import sqlite3


def save_to_sqlite(df, db_path):
    """
    Sauvegarde le DataFrame final dans une base SQLite.
    """
    # On s'assure que le dossier de destination existe
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    df.to_sql('volatility_data', conn, if_exists='replace', index=True)
    conn.close()

    print(f"\n✅ {len(df)} lignes sauvegardées dans {db_path}")

if __name__ == "__main__":
    # 1. Récupération des prix pour tous les tickers
    all_data = fetch_all_tickers(config.TICKERS, config.START_DATE, config.END_DATE)

    # 2. Calcul de la volatilité réalisée (20j et 60j)
    all_data = calculate_realized_volatility(
        all_data,
        window_short=config.REALIZED_VOL_WINDOW_SHORT,
        window_long=config.REALIZED_VOL_WINDOW_LONG
    )

    # 3. Récupération des données macro
    macro_data = fetch_macro_data(config.START_DATE, config.END_DATE)

    # 4. Fusion
    merged_data = merge_data(all_data, macro_data)

    # 5. Création de la target
    final_data = create_target(merged_data, horizon=config.FORECAST_HORIZON)

    # 6. Sauvegarde dans SQLite
    save_to_sqlite(final_data, config.DB_PATH)

    # --- Vérification finale : on relit depuis la base pour confirmer que tout est bien stocké ---
    print("\n--- Vérification depuis SQLite ---")
    conn = sqlite3.connect(config.DB_PATH)
    check = pd.read_sql('SELECT * FROM volatility_data', conn)
    conn.close()

    print(f"Lignes dans la base : {len(check)}")
    print(f"Tickers dans la base : {sorted(check['ticker'].unique())}")
    print(f"Colonnes : {list(check.columns)}")
    print(f"\nAperçu :")
    print(check.head())