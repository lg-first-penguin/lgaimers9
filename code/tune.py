# code/tune.py
import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool

from code.train import process_trackman_features_safe, add_engineered_features

N_TRIALS = int(os.environ.get("TUNE_TRIALS", 40))
DATA_DIR = "./open/data"
TARGET_COL = "control_success"
OUT_PATH = "./open/temp/best_hparams.json"


def compute_bss(preds, y_val):
    r = y_val.mean()
    brier = ((preds - y_val) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def build_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [col for col in train_df.columns if col not in drop_cols]
    categorical_features = [
        c for c in tr_final.select_dtypes(include="object").columns
        if c != "row_id" and c in features
    ]

    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna("missing")

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET_COL].values
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET_COL].values

    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=categorical_features)
    return train_pool, val_pool, y_val


def make_objective(train_pool, val_pool, y_val):
    def objective(trial):
        params = dict(
            iterations=1500,
            loss_function="Logloss",
            eval_metric="BrierScore",
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            border_count=trial.suggest_int("border_count", 32, 254),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 200, log=True),
            bootstrap_type="Bayesian",
        )
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        preds = model.predict_proba(val_pool)[:, 1]
        bss = compute_bss(preds, y_val)
        trial.set_user_attr("best_iteration", model.best_iteration_)
        return bss
    return objective


def main():
    print("[Tune] 데이터 로드 및 피처 엔지니어링 (1회만 수행, 트라이얼 간 재사용)...")
    train_pool, val_pool, y_val = build_data()
    print(f"[Tune] train_pool: {train_pool.num_row()}행 | val_pool: {val_pool.num_row()}행")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    def log_callback(study, trial):
        print(f"[Trial {trial.number:03d}] BSS={trial.value:.5f} "
              f"(best_iter={trial.user_attrs.get('best_iteration')}) params={trial.params}")

    study.optimize(make_objective(train_pool, val_pool, y_val), n_trials=N_TRIALS, callbacks=[log_callback])

    print("\n" + "=" * 60)
    print(f"[Tune 완료] Best BSS: {study.best_value:.5f} (score: {max(0, 100000*study.best_value):.2f})")
    print(f"Best params: {study.best_params}")
    print(f"Best iteration at that trial: {study.best_trial.user_attrs.get('best_iteration')}")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "best_bss": study.best_value,
            "best_params": study.best_params,
            "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ 최적 하이퍼파라미터 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
