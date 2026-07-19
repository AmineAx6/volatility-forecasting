"""
Ensemble : combine GARCH (décalé de 10j), XGBoost, et LSTM
avec des poids pour produire une prédiction finale.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from features.engineering import load_data, engineer_features
from models.data_prep import prepare_ml_data
from models.garch_baseline import fit_garch_baseline
from models.xgboost_model import train_xgboost
from models.lstm_model import create_sequences, build_lstm_model, train_lstm

# Poids de l'ensemble, décidés dans CLAUDE.md : XGBoost est le plus fiable
# (meilleur R² individuel une fois LSTM et XGBoost comparés), GARCH sert de
# stabilisateur "baseline financière", LSTM apporte la diversité de modèle.
WEIGHTS = {'garch': 0.3, 'xgboost': 0.5, 'lstm': 0.2}


def get_garch_shifted_predictions(df, horizon=10):
    """
    Fit un GARCH par ticker, et décale sa volatilité conditionnelle de `horizon`
    jours en arrière pour l'aligner sur le même horizon de prédiction que
    XGBoost/LSTM (approximation assumée : la vol GARCH d'aujourd'hui sert
    d'estimation pour dans `horizon` jours).
    """
    results_list = []

    for ticker in config.TICKERS:
        ticker_df = df[df['ticker'] == ticker].sort_values('date').set_index('date')
        _, predicted_vol = fit_garch_baseline(ticker_df['daily_return'])
        predicted_vol.index = ticker_df['daily_return'].dropna().index

        # On décale la prédiction GARCH de `horizon` jours vers l'avant :
        # la valeur du jour T devient la "prédiction pour T+horizon"
        shifted = predicted_vol.shift(horizon)
        shifted.name = 'garch_pred'

        ticker_result = pd.DataFrame({'garch_pred': shifted})
        ticker_result['ticker'] = ticker
        ticker_result['date'] = ticker_result.index

        results_list.append(ticker_result)

    return pd.concat(results_list).reset_index(drop=True)


def get_xgboost_predictions(df):
    """
    Entraîne XGBoost via prepare_ml_data()/train_xgboost() (déjà existants,
    split temporel géré par prepare_ml_data) et retourne les prédictions sur
    le test set, avec (date, ticker, actual) pour pouvoir les réaligner avec
    GARCH et LSTM.
    """
    X_train, X_test, y_train, y_test, scaler, split_date = prepare_ml_data(df)
    model, y_pred_train, y_pred_test, metrics = train_xgboost(X_train, X_test, y_train, y_test)

    # X_test conserve l'index original de `df` (prepare_ml_data ne le reset
    # jamais) : on peut donc récupérer date/ticker directement via df.loc
    meta = df.loc[X_test.index, ['date', 'ticker']].reset_index(drop=True)
    result = meta.copy()
    result['xgb_pred'] = y_pred_test
    result['actual'] = y_test.values

    return result, metrics


def get_lstm_predictions(df):
    """
    Entraîne le LSTM via train_lstm() (boucle d'entraînement manuelle, voir
    CLAUDE.md pour le bug TensorFlow résolu) et retourne les prédictions sur
    le test set, avec (date, ticker) pour le réalignement.
    """
    model, scaler, metrics, dates_test, tickers_test, y_pred_test = train_lstm(
        df, seq_length=config.LSTM_SEQ_LENGTH
    )

    result = pd.DataFrame({
        'date': dates_test,
        'ticker': tickers_test,
        'lstm_pred': y_pred_test
    })

    return result, metrics


def align_predictions(garch_preds, xgb_preds, lstm_preds):
    """
    Fusionne les 3 séries de prédictions sur (date, ticker), en ne gardant
    que les lignes où les 3 modèles ET la target sont valides (pas de NaN).
    C'est indispensable : GARCH décalé perd ses `horizon` premières valeurs
    par ticker (NaN), XGBoost et LSTM n'ont pas exactement les mêmes lignes
    de test (LSTM perd `seq_length` jours de warmup en plus par ticker).
    """
    # On normalise le type de la colonne date pour être sûr que le merge
    # matche bien (datetime64[ns] des deux côtés)
    garch_preds = garch_preds.copy()
    xgb_preds = xgb_preds.copy()
    lstm_preds = lstm_preds.copy()
    garch_preds['date'] = pd.to_datetime(garch_preds['date'])
    xgb_preds['date'] = pd.to_datetime(xgb_preds['date'])
    lstm_preds['date'] = pd.to_datetime(lstm_preds['date'])

    merged = xgb_preds.merge(garch_preds, on=['date', 'ticker'], how='inner')
    merged = merged.merge(lstm_preds, on=['date', 'ticker'], how='inner')

    print(f"  XGBoost (test) : {len(xgb_preds)} lignes")
    print(f"  LSTM (test)    : {len(lstm_preds)} lignes")
    print(f"  GARCH (décalé) : {len(garch_preds)} lignes (sur tout l'historique, pas juste le test)")
    print(f"  Après merge (avant dropna) : {len(merged)} lignes")

    before = len(merged)
    merged = merged.dropna(subset=['garch_pred', 'xgb_pred', 'lstm_pred', 'actual'])
    print(f"  Après dropna (les 3 prédictions valides) : {len(merged)} lignes (supprimées : {before - len(merged)})")

    return merged.sort_values('date').reset_index(drop=True)


def compute_metrics(y_pred, y_true):
    """Calcule MAE, RMSE, R², corrélation — les mêmes métriques que GARCH/XGBoost/LSTM individuels."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'corr': np.corrcoef(y_pred, y_true)[0, 1]
    }


if __name__ == "__main__":
    df = load_data(config.DB_PATH)
    df = engineer_features(df)

    print("=== 1. GARCH décalé (10 jours) ===")
    garch_preds = get_garch_shifted_predictions(df, horizon=config.FORECAST_HORIZON)
    print(f"Lignes GARCH : {len(garch_preds)}")
    print(garch_preds.head())
    print(f"NaN garch_pred : {garch_preds['garch_pred'].isna().sum()} / {len(garch_preds)}")

    print("\n=== 2. XGBoost (test set) ===")
    xgb_preds, xgb_metrics = get_xgboost_predictions(df)
    print(f"Lignes XGBoost (test) : {len(xgb_preds)}")
    print(xgb_preds.head())
    print(f"NaN xgb_pred : {xgb_preds['xgb_pred'].isna().sum()} / {len(xgb_preds)}")

    print("\n=== 3. LSTM (test set) ===")
    lstm_preds, lstm_metrics = get_lstm_predictions(df)
    print(f"Lignes LSTM (test) : {len(lstm_preds)}")
    print(lstm_preds.head())
    print(f"NaN lstm_pred : {lstm_preds['lstm_pred'].isna().sum()} / {len(lstm_preds)}")

    print("\n=== 4. Alignement des 3 séries sur (date, ticker) ===")
    aligned = align_predictions(garch_preds, xgb_preds, lstm_preds)
    print(f"\nAperçu de l'ensemble aligné :")
    print(aligned.head())
    print(f"\nRépartition par ticker :")
    print(aligned['ticker'].value_counts())

    print("\n=== 5. Calcul de l'ensemble pondéré ===")
    print(f"Poids : GARCH={WEIGHTS['garch']}, XGBoost={WEIGHTS['xgboost']}, LSTM={WEIGHTS['lstm']}")
    aligned['ensemble_pred'] = (
        WEIGHTS['garch'] * aligned['garch_pred'] +
        WEIGHTS['xgboost'] * aligned['xgb_pred'] +
        WEIGHTS['lstm'] * aligned['lstm_pred']
    )
    print(aligned[['date', 'ticker', 'garch_pred', 'xgb_pred', 'lstm_pred', 'ensemble_pred', 'actual']].head())

    print("\n=== 6. Évaluation comparative (même sous-ensemble aligné pour les 4 approches) ===")
    approaches = {
        'GARCH (décalé)': 'garch_pred',
        'XGBoost': 'xgb_pred',
        'LSTM': 'lstm_pred',
        'Ensemble': 'ensemble_pred'
    }

    summary_rows = []
    for name, col in approaches.items():
        m = compute_metrics(aligned[col], aligned['actual'])
        summary_rows.append({
            'Modèle': name,
            'MAE': round(m['mae'], 4),
            'RMSE': round(m['rmse'], 4),
            'R²': round(m['r2'], 4),
            'Corrélation': round(m['corr'], 4)
        })

    summary = pd.DataFrame(summary_rows).set_index('Modèle')

    print(f"\n=== TABLEAU RÉCAPITULATIF (n={len(aligned)} observations, test set aligné) ===")
    print(summary)

    best_mae = summary['MAE'].idxmin()
    print(f"\nMeilleur MAE : {best_mae} ({summary.loc[best_mae, 'MAE']})")
