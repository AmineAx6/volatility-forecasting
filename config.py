"""
Configuration centrale du projet.
Modifie ce fichier pour ajuster tickers, dates, ou paramètres des modèles.
"""
import datetime

# --- Données ---
TICKERS = ['AMD', 'NVDA', 'GOOG', 'RDDT', 'SPY', 'QQQ']
START_DATE = '2022-01-01'
END_DATE = '2026-06-30'

# --- Chemins ---
DB_PATH = 'data/volatility_data.db'

# --- Paramètres de volatilité ---
REALIZED_VOL_WINDOW_SHORT = 20   # fenêtre courte (jours)
REALIZED_VOL_WINDOW_LONG = 60    # fenêtre longue (jours)
FORECAST_HORIZON = 10            # jours dans le futur à prédire
