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

# Le LSTM n'est pas parfaitement déterministe sur cette machine, même à seed
# fixe (testé : poids initiaux + ordre des batches donnent un MAE test qui
# varie d'un run à l'autre, ex. 0.42 à 0.51 — voir CLAUDE.md). Plutôt que de
# forcer un déterminisme complet (coûteux et pas garanti à 100%), on répète
# l'entraînement N_LSTM_RUNS fois avec des seeds différents et on rapporte
# moyenne ± écart-type : c'est la variance réelle du modèle, pas cachée.
N_LSTM_RUNS = 5
LSTM_SEED_BASE = 42


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
    GARCH et LSTM. XGBoost a un random_state fixe : ce résultat est stable
    d'un run à l'autre, pas besoin de le répéter.
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


def get_lstm_predictions(df, seed=42):
    """
    Entraîne le LSTM via train_lstm() (boucle d'entraînement manuelle, voir
    CLAUDE.md pour le bug TensorFlow résolu) et retourne les prédictions sur
    le test set, avec (date, ticker) pour le réalignement.
    """
    model, scaler, metrics, dates_test, tickers_test, y_pred_test = train_lstm(
        df, seq_length=config.LSTM_SEQ_LENGTH, seed=seed
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

    merged = merged.dropna(subset=['garch_pred', 'xgb_pred', 'lstm_pred', 'actual'])

    return merged.sort_values('date').reset_index(drop=True)


def compute_metrics(y_pred, y_true):
    """Calcule MAE, RMSE, R², corrélation — les mêmes métriques que GARCH/XGBoost/LSTM individuels."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'corr': np.corrcoef(y_pred, y_true)[0, 1]
    }


def aggregate_metrics(metrics_runs):
    """
    Moyenne ± écart-type (échantillon, ddof=1) de chaque métrique sur
    plusieurs runs indépendants. C'est ce qu'on rapporte pour LSTM/Ensemble
    au lieu d'un seul run, puisque le LSTM n'est pas déterministe ici.
    """
    keys = metrics_runs[0].keys()
    agg = {}
    for k in keys:
        values = np.array([m[k] for m in metrics_runs])
        agg[k] = {'mean': values.mean(), 'std': values.std(ddof=1)}
    return agg


def format_mean_std(agg, key, decimals=4):
    return f"{agg[key]['mean']:.{decimals}f} ± {agg[key]['std']:.{decimals}f}"


def format_single(metrics, key, decimals=4):
    return f"{metrics[key]:.{decimals}f}"


if __name__ == "__main__":
    df = load_data(config.DB_PATH)
    df = engineer_features(df)

    print("=== 1. GARCH décalé (10 jours) ===")
    garch_preds = get_garch_shifted_predictions(df, horizon=config.FORECAST_HORIZON)
    print(f"Lignes GARCH : {len(garch_preds)}")
    print(garch_preds.head())
    print(f"NaN garch_pred : {garch_preds['garch_pred'].isna().sum()} / {len(garch_preds)}")

    print("\n=== 2. XGBoost (test set, déterministe — un seul run) ===")
    xgb_preds, xgb_metrics_raw = get_xgboost_predictions(df)
    print(f"Lignes XGBoost (test) : {len(xgb_preds)}")
    print(xgb_preds.head())
    print(f"NaN xgb_pred : {xgb_preds['xgb_pred'].isna().sum()} / {len(xgb_preds)}")

    print(f"\n=== 3. LSTM : {N_LSTM_RUNS} runs indépendants (seeds {LSTM_SEED_BASE} à {LSTM_SEED_BASE + N_LSTM_RUNS - 1}) ===")
    print("Le LSTM n'est pas parfaitement reproductible sur cette machine même à seed fixe (cf. CLAUDE.md) :")
    print("on entraîne plusieurs fois et on mesure la variance plutôt que de se fier à un seul run.\n")

    lstm_metrics_runs = []
    ensemble_metrics_runs = []
    aligned_preview = None
    garch_metrics_aligned = None
    xgb_metrics_aligned = None

    for i in range(N_LSTM_RUNS):
        seed = LSTM_SEED_BASE + i
        print(f"--- Run LSTM {i + 1}/{N_LSTM_RUNS} (seed={seed}) ---")
        lstm_preds, _ = get_lstm_predictions(df, seed=seed)
        print(f"Lignes LSTM (test) : {len(lstm_preds)}, NaN : {lstm_preds['lstm_pred'].isna().sum()}")

        aligned = align_predictions(garch_preds, xgb_preds, lstm_preds)
        aligned['ensemble_pred'] = (
            WEIGHTS['garch'] * aligned['garch_pred'] +
            WEIGHTS['xgboost'] * aligned['xgb_pred'] +
            WEIGHTS['lstm'] * aligned['lstm_pred']
        )

        run_lstm_metrics = compute_metrics(aligned['lstm_pred'], aligned['actual'])
        run_ensemble_metrics = compute_metrics(aligned['ensemble_pred'], aligned['actual'])

        print(f"  Lignes alignées : {len(aligned)}")
        print(f"  LSTM     -> MAE={run_lstm_metrics['mae']:.4f}  RMSE={run_lstm_metrics['rmse']:.4f}  "
              f"R²={run_lstm_metrics['r2']:.4f}  Corr={run_lstm_metrics['corr']:.4f}")
        print(f"  Ensemble -> MAE={run_ensemble_metrics['mae']:.4f}  RMSE={run_ensemble_metrics['rmse']:.4f}  "
              f"R²={run_ensemble_metrics['r2']:.4f}  Corr={run_ensemble_metrics['corr']:.4f}\n")

        lstm_metrics_runs.append(run_lstm_metrics)
        ensemble_metrics_runs.append(run_ensemble_metrics)

        # GARCH et XGBoost sont déterministes : les lignes alignées et leurs
        # métriques sont identiques à chaque run, donc on ne les calcule
        # qu'une seule fois (au premier passage)
        if aligned_preview is None:
            aligned_preview = aligned
            garch_metrics_aligned = compute_metrics(aligned['garch_pred'], aligned['actual'])
            xgb_metrics_aligned = compute_metrics(aligned['xgb_pred'], aligned['actual'])

    print("=== 4. Aperçu de l'ensemble aligné (run 1, mêmes lignes pour tous les runs) ===")
    print(aligned_preview[['date', 'ticker', 'garch_pred', 'xgb_pred', 'lstm_pred', 'ensemble_pred', 'actual']].head())
    print(f"\nRépartition par ticker :")
    print(aligned_preview['ticker'].value_counts())

    lstm_agg = aggregate_metrics(lstm_metrics_runs)
    ensemble_agg = aggregate_metrics(ensemble_metrics_runs)

    print(f"\n=== 5. TABLEAU RÉCAPITULATIF (n={len(aligned_preview)} observations, test set aligné) ===")
    print(f"GARCH et XGBoost : valeur unique (déterministes). LSTM et Ensemble : moyenne ± écart-type sur {N_LSTM_RUNS} runs.\n")

    summary_rows = [
        {
            'Modèle': 'GARCH (décalé)',
            'MAE': format_single(garch_metrics_aligned, 'mae'),
            'RMSE': format_single(garch_metrics_aligned, 'rmse'),
            'R²': format_single(garch_metrics_aligned, 'r2'),
            'Corrélation': format_single(garch_metrics_aligned, 'corr'),
        },
        {
            'Modèle': 'XGBoost',
            'MAE': format_single(xgb_metrics_aligned, 'mae'),
            'RMSE': format_single(xgb_metrics_aligned, 'rmse'),
            'R²': format_single(xgb_metrics_aligned, 'r2'),
            'Corrélation': format_single(xgb_metrics_aligned, 'corr'),
        },
        {
            'Modèle': f'LSTM (moyenne {N_LSTM_RUNS} runs)',
            'MAE': format_mean_std(lstm_agg, 'mae'),
            'RMSE': format_mean_std(lstm_agg, 'rmse'),
            'R²': format_mean_std(lstm_agg, 'r2'),
            'Corrélation': format_mean_std(lstm_agg, 'corr'),
        },
        {
            'Modèle': f'Ensemble (moyenne {N_LSTM_RUNS} runs)',
            'MAE': format_mean_std(ensemble_agg, 'mae'),
            'RMSE': format_mean_std(ensemble_agg, 'rmse'),
            'R²': format_mean_std(ensemble_agg, 'r2'),
            'Corrélation': format_mean_std(ensemble_agg, 'corr'),
        },
    ]

    summary = pd.DataFrame(summary_rows).set_index('Modèle')
    print(summary.to_string())
