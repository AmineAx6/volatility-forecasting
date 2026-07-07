"""
Modèle XGBoost : apprend à prédire target_vol_10d à partir des 25 features
engineered (volatilité, rendements, macro, volume, calendrier, ticker).
"""
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from features.engineering import load_data, engineer_features
from models.data_prep import prepare_ml_data


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle XGBoost et retourne le modèle + métriques d'évaluation.
    """
    model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_corr': np.corrcoef(y_pred_test, y_test)[0, 1]
    }

    return model, y_pred_train, y_pred_test, metrics


if __name__ == "__main__":
    df = load_data(config.DB_PATH)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test, scaler, split_date = prepare_ml_data(df)

    print("Entraînement XGBoost en cours...")
    model, y_pred_train, y_pred_test, metrics = train_xgboost(X_train, X_test, y_train, y_test)

    print(f"\n=== RÉSULTATS XGBOOST ===")
    print(f"Train MAE : {metrics['train_mae']:.4f}")
    print(f"Test MAE  : {metrics['test_mae']:.4f}")
    print(f"Test RMSE : {metrics['test_rmse']:.4f}")
    print(f"Test R²   : {metrics['test_r2']:.4f}")
    print(f"Corrélation (pred vs actual) : {metrics['test_corr']:.4f}")

    # Feature importance : quelles features le modèle utilise-t-il le plus ?
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    print(f"\n=== TOP 10 FEATURES LES PLUS IMPORTANTES ===")
    print(importances.head(10))

    # Sauvegarde du modèle pour réutilisation future (dashboard, etc.)
    os.makedirs('models/trained_models', exist_ok=True)
    model.save_model('models/trained_models/xgb_model.json')
    print(f"\n✅ Modèle sauvegardé dans models/trained_models/xgb_model.json")