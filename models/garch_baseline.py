"""
Modèle GARCH(1,1) : baseline traditionnelle de la finance quantitative
pour la modélisation de la volatilité.
"""
import sys
import os
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from features.engineering import load_data


def fit_garch_baseline(returns_series):
    """
    Fit un modèle GARCH(1,1) sur les rendements journaliers d'UN ticker.
    Retourne les résultats du fit et la volatilité conditionnelle estimée.
    """
    returns = returns_series.dropna()

    # On multiplie par 100 : GARCH converge mieux numériquement sur des rendements
    # exprimés en % (ex: 1.5) plutôt qu'en décimales très petites (ex: 0.015)
    model = arch_model(returns * 100, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')

    # conditional_volatility est déjà en %, cohérent avec l'échelle de realized_vol_20d
    predicted_vol = results.conditional_volatility

    return results, predicted_vol


def evaluate_garch(predicted_vol, actual_vol):
    """
    Compare la volatilité GARCH à la volatilité réalisée observée,
    en alignant les deux séries sur leurs dates communes.
    """
    aligned = pd.DataFrame({'predicted': predicted_vol, 'actual': actual_vol}).dropna()

    mae = mean_absolute_error(aligned['actual'], aligned['predicted'])
    rmse = np.sqrt(mean_squared_error(aligned['actual'], aligned['predicted']))
    corr = aligned['predicted'].corr(aligned['actual'])

    return {'mae': mae, 'rmse': rmse, 'correlation': corr, 'n_obs': len(aligned)}


if __name__ == "__main__":
    df = load_data(config.DB_PATH)
    all_results = {}

    for ticker in config.TICKERS:
        ticker_df = df[df['ticker'] == ticker].set_index('date')

        print(f"\n--- {ticker} ---")
        results, predicted_vol = fit_garch_baseline(ticker_df['daily_return'])

        # On réaligne les dates : le modèle GARCH renvoie un index numérique,
        # on lui remet les vraies dates pour pouvoir comparer avec realized_vol_20d
        predicted_vol.index = ticker_df['daily_return'].dropna().index

        metrics = evaluate_garch(predicted_vol, ticker_df['realized_vol_20d'])
        all_results[ticker] = metrics

        print(f"  Alpha (choc):        {results.params['alpha[1]']:.4f}")
        print(f"  Beta (persistance):  {results.params['beta[1]']:.4f}")
        print(f"  Alpha + Beta:        {results.params['alpha[1]'] + results.params['beta[1]']:.4f}")
        print(f"  MAE:                 {metrics['mae']:.4f}")
        print(f"  RMSE:                {metrics['rmse']:.4f}")
        print(f"  Corrélation:         {metrics['correlation']:.4f}")

    print("\n=== RÉSUMÉ GARCH (tous tickers) ===")
    summary = pd.DataFrame(all_results).T
    print(summary)