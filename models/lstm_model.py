"""
Modèle LSTM : apprend à partir de séquences de rendements journaliers
pour prédire target_vol_10d.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from features.engineering import load_data, engineer_features


def create_sequences(df, seq_length=20):
    """
    Construit des séquences de rendements journaliers, ticker par ticker,
    pour ne jamais mélanger deux séries temporelles différentes.

    Retourne : X (séquences), y (target associée), dates (date de fin de chaque
    séquence), tickers (ticker associé à chaque séquence — utile pour réaligner
    les prédictions LSTM avec GARCH/XGBoost par (date, ticker) dans l'ensemble)
    """
    X_list, y_list, date_list, ticker_list = [], [], [], []

    for ticker, g in df.groupby('ticker'):
        g = g.sort_values('date').reset_index(drop=True)
        returns = g['daily_return'].values
        targets = g['target_vol_10d'].values
        dates = g['date'].values

        for i in range(seq_length, len(g)):
            window = returns[i - seq_length:i]
            target = targets[i]

            # On ignore les séquences contenant un NaN (dans les rendements ou la target)
            if np.isnan(window).any() or np.isnan(target):
                continue

            X_list.append(window)
            y_list.append(target)
            date_list.append(dates[i])
            ticker_list.append(ticker)

    X = np.array(X_list).reshape(-1, seq_length, 1)
    y = np.array(y_list)
    dates = np.array(date_list)
    tickers = np.array(ticker_list)

    return X, y, dates, tickers


def build_lstm_model(seq_length):
    """Construit l'architecture du réseau de neurones LSTM."""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    # run_eagerly=True : sans ça, Keras compile le train_step en tf.function et
    # l'exécute via le FunctionLibraryRuntime de TF, qui peut rester bloqué
    # indéfiniment (deadlock reproduit dans ProcessFunctionLibraryRuntime::RunSync,
    # en attente d'une absl::Notification jamais levée) sur cette machine.
    # L'exécution eager contourne ce chemin de code.
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', run_eagerly=True)
    return model


def train_lstm(df, seq_length=20, test_size=0.2, epochs=20, batch_size=32):
    """
    Pipeline complet : création des séquences, split temporel, standardisation,
    entraînement, évaluation.
    """
    X, y, dates, tickers = create_sequences(df, seq_length)

    # Split TEMPOREL, comme pour XGBoost : on trie par date, puis on coupe
    sort_idx = np.argsort(dates)
    X, y, dates, tickers = X[sort_idx], y[sort_idx], dates[sort_idx], tickers[sort_idx]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test, tickers_test = dates[split_idx:], tickers[split_idx:]

    # Standardisation des rendements : fit sur le train uniquement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # Split validation manuel (derniers 10% du train, comme validation_split=0.1)
    n_val = int(len(X_train_scaled) * 0.1)
    X_fit, y_fit = X_train_scaled[:-n_val], y_train[:-n_val]
    X_val, y_val = X_train_scaled[-n_val:], y_train[-n_val:]

    model = build_lstm_model(seq_length)

    print("Entraînement LSTM en cours (peut prendre 1-2 minutes)...")
    # Boucle d'entraînement manuelle (train_on_batch/test_on_batch/predict_on_batch)
    # plutôt que model.fit()/model.predict() : ces derniers font transiter les
    # données par le pipeline interne tf.data de Keras (PrefetchDataset), qui reste
    # bloqué indéfiniment sur cette machine (deadlock reproduit dans
    # PrefetchDatasetOp::Iterator::GetNextInternal, dû à une collision de symboles
    # Abseil entre libtensorflow_framework et libarrow au sein du wheel TensorFlow
    # macOS). train_on_batch/predict_on_batch opèrent directement sur les tenseurs,
    # sans passer par ce pipeline, et évitent donc le blocage.
    n = len(X_fit)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_losses.append(model.train_on_batch(X_fit[idx], y_fit[idx]))
        val_loss = model.test_on_batch(X_val, y_val)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {np.mean(batch_losses):.4f} - val_loss: {val_loss:.4f}")

    y_pred_train = model.predict_on_batch(X_train_scaled).flatten()
    y_pred_test = model.predict_on_batch(X_test_scaled).flatten()

    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_corr': np.corrcoef(y_pred_test, y_test)[0, 1]
    }

    return model, scaler, metrics, dates_test, tickers_test, y_pred_test


if __name__ == "__main__":
    df = load_data(config.DB_PATH)
    df = engineer_features(df)

    model, scaler, metrics, dates_test, tickers_test, y_pred_test = train_lstm(df, seq_length=config.LSTM_SEQ_LENGTH)

    print(f"\n=== RÉSULTATS LSTM ===")
    print(f"Train MAE : {metrics['train_mae']:.4f}")
    print(f"Test MAE  : {metrics['test_mae']:.4f}")
    print(f"Test RMSE : {metrics['test_rmse']:.4f}")
    print(f"Test R²   : {metrics['test_r2']:.4f}")
    print(f"Corrélation (pred vs actual) : {metrics['test_corr']:.4f}")

    os.makedirs('models/trained_models', exist_ok=True)
    model.save('models/trained_models/lstm_model.keras')
    print(f"\n✅ Modèle sauvegardé dans models/trained_models/lstm_model.keras")