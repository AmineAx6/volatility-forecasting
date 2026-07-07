# Contexte du projet : Volatility Forecasting Engine

## Objectif général

Projet portfolio (quant finance) : système de prédiction de volatilité à 10 jours combinant GARCH, XGBoost et LSTM en ensemble pondéré. Repo GitHub public : `AmineAx6/volatility-forecasting`.

## Décisions de scope (importantes, déjà tranchées)

- **Tickers** : AMD, NVDA, GOOG, RDDT, SPY, QQQ (RDDT n'a d'historique que depuis mars 2024, IPO — c'est normal, pas un bug)
- **Période** : ~4.5 ans, janvier 2022 → juin 2026 (étendu depuis le spec original qui prévoyait 3 ans)
- **Volatilité implicite (IV) historique** : volontairement **exclue**. Pas de source gratuite fiable sur un historique aussi long. On utilise le VIX comme proxy de volatilité de marché à la place. L'IV en temps réel pourra être ajoutée plus tard dans le dashboard (Stage 5), mais pas dans l'entraînement des modèles.
- **Cible de prédiction** : `target_vol_10d`, la volatilité réalisée 20 jours (rolling std des rendements) décalée de 10 jours dans le futur

## Stack technique

- Python 3.13, environnement virtuel (`venv`)
- pandas, numpy, yfinance (prix), fredapi + dotenv (macro : VIX, taux 10 ans, clé API dans `.env`, jamais commit)
- arch (GARCH), xgboost, scikit-learn, tensorflow/keras (LSTM)
- SQLite (`data/volatility_data.db`, exclu de Git via `.gitignore`)
- Git/GitHub avec commits incrémentaux à chaque brique fonctionnelle

## Structure du projet

```
volatility-forecasting/
├── config.py                    # tickers, dates, paramètres centraux
├── data/
│   ├── pipeline.py               # fetch prix, calcul vol réalisée, macro, merge, target, save SQLite
│   └── volatility_data.db        # 6194 lignes, 6 tickers, 10 colonnes (non versionné)
├── features/
│   └── engineering.py            # load_data() + engineer_features() : 14 features dérivées
├── models/
│   ├── garch_baseline.py         # GARCH(1,1) par ticker, baseline
│   ├── data_prep.py              # prepare_ml_data() : split temporel 80/20, one-hot ticker, StandardScaler
│   ├── xgboost_model.py          # XGBoost entraîné, terminé et fonctionnel
│   ├── lstm_model.py             # LSTM — débuggé et fonctionnel (voir section bug résolu ci-dessous)
│   └── trained_models/           # modèles sauvegardés (.json xgb, .keras lstm)
├── backtest/, dashboard/, notebooks/, results/  # pas encore attaqués
└── requirements.txt
```

## État d'avancement

- ✅ **Stage 1 — Data Pipeline** : terminé. 6194 lignes, 6 tickers, prix + rendements + vol réalisée (20j/60j) + VIX + taux 10 ans + target, tout en SQLite.
- ✅ **Stage 2 — Feature Engineering & GARCH** : terminé. 14 features (lags de vol, skew, kurtosis, vix_change, volume_ratio, features calendaires...). GARCH(1,1) fitté par ticker avec alpha/beta/MAE/RMSE/corrélation par ticker.
- 🔄 **Stage 3 — ML Ensemble** : en cours.
  - XGBoost : terminé. Test MAE 0.4858, R² 0.759, corrélation 0.872. Feature la plus importante : `realized_vol_20d` (41%).
  - **LSTM : débuggé et fonctionnel** (voir section bug résolu ci-dessous). Test MAE ≈ 0.42-0.49, Test R² ≈ 0.72-0.79, corrélation ≈ 0.88-0.89 selon les runs (poids initiaux non seedés). Modèle sauvegardé dans `models/trained_models/lstm_model.keras`.
  - Ensemble (blend des 3 modèles) : pas encore fait — **prochaine étape**
- ⬜ Stage 4 — Backtest & signaux : pas commencé
- ⬜ Stage 5 — Dashboard Streamlit & déploiement : pas commencé

## Conventions de code établies

- Tous les calculs par ticker utilisent `groupby('ticker')` pour ne jamais mélanger les séries temporelles de deux tickers différents (piège identifié et corrigé plusieurs fois dans ce projet, notamment sur `vix_change`)
- Split train/test **toujours temporel** (jamais aléatoire) — data leakage sinon
- `StandardScaler` : fit uniquement sur train, jamais sur test
- Commentaires en français, pédagogiques (le propriétaire du projet apprend en marchant)
- Un commit Git par brique fonctionnelle, avec message descriptif

## Bug résolu : LSTM bloqué (model.fit se figeait, 0% CPU)

**Symptôme observé** : `model.fit()` dans `train_lstm()` (models/lstm_model.py) se bloquait indéfiniment (aucune progression, 0% CPU) avec les vraies données du projet, mais fonctionnait avec des données factices de même taille.

**Cause réelle (pas scikit-learn)** : ce n'était pas un conflit `StandardScaler`/`tensorflow`. Diagnostic confirmé en capturant la pile d'exécution du process bloqué (macOS `sample`, équivalent de py-spy) : le thread principal était figé dans le pipeline interne `tf.data` de Keras (`PrefetchDatasetOp::Iterator::GetNextInternal`), en attente d'une notification jamais levée. L'appel de synchronisation passait par `AbslInternalPerThreadSemWait_lts_20250814` situé dans `libarrow.2400.dylib` au lieu de sa propre implémentation dans `libtensorflow_framework.2.dylib` — une collision de symboles Abseil dupliqués entre deux bibliothèques internes du wheel TensorFlow 2.21.0 pour macOS ARM64. Un vrai bug de la distribution TensorFlow sur cette machine, indépendant du contenu des données (le blocage est une race condition dont le déclenchement dépend du timing exact avant l'appel à `fit()` — d'où la différence de comportement entre un script de test minimal et le pipeline réel, plus long à charger).

**Pistes explorées avant de trouver la vraie cause** (pour référence) :
- Limiter les threads TensorFlow (`tf.config.threading.set_*_op_parallelism_threads`, puis variables d'env `TF_NUM_INTRAOP_THREADS`/`TF_NUM_INTEROP_THREADS`) : insuffisant seul, déplaçait parfois le deadlock ailleurs dans le même sous-système `tf.data`.
- `run_eagerly=True` seul : évite le deadlock dans `ProcessFunctionLibraryRuntime::RunSync` mais pas celui dans `PrefetchDatasetOp` (sous-système différent, `model.fit()` construit toujours un `tf.data.Dataset` en interne même en mode eager).

**Fix appliqué** (models/lstm_model.py) :
- `run_eagerly=True` dans `model.compile()`
- `model.fit()`/`model.predict()` remplacés par une boucle d'entraînement manuelle (`train_on_batch`/`test_on_batch`/`predict_on_batch`), qui n'utilise jamais `tf.data.Dataset` et contourne donc complètement le code défaillant

**Vérifié** : 3 exécutions complètes et séquentielles du script réel (20 epochs, entraînement + métriques + sauvegarde) — 3/3 réussies sans blocage.

**Prochaine étape** : passer à l'ensemble (blend GARCH + XGBoost + LSTM).
