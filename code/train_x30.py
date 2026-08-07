# code/train.py
import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL_FEATURES = ['top_bottom', 'game_type', 'base_state']

def build_pitcher_consistency_features(trackman_path):
    print("[Feature Engineering] 과거 트랙맨 로그 기반 투구 메커니즘 일관성 지표 산출 중...")
    tm = pd.read_csv(trackman_path, encoding="utf-8-sig")
    tm = tm.dropna(subset=['pitcher_trackman_id', 'pitch_type_group']).copy()
    
    consistency = tm.groupby(['pitcher_trackman_id', 'pitch_type_group']).agg({
        'horz_break': 'std', 'induced_vert_break': 'std', 'extension': 'std', 'rel_speed': 'std'
    }).reset_index()
    
    consistency.columns = [
        'pitcher_id', 'pitch_type_group',
        'tm_std_horz_break', 'tm_std_vert_break', 'tm_std_extension', 'tm_std_rel_speed'
    ]
    
    pivot_consistency = consistency.pivot(
        index='pitcher_id', columns='pitch_type_group',
        values=['tm_std_horz_break', 'tm_std_vert_break', 'tm_std_extension', 'tm_std_rel_speed']
    )
    pivot_consistency.columns = [f"tm_{c[1]}_{c[0].replace('tm_', '')}" for c in pivot_consistency.columns]
    return pivot_consistency.reset_index()

def main():
    DATA_DIR = "./open/data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    TRACKMAN_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
    TEMP_MODEL_PATH = "./open/temp/latest_model.pkl"
    os.makedirs("./open/temp", exist_ok=True)

    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns
    base_features = [c for c in test_cols if c != ID_COL]
    
    train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=base_features + [TARGET_COL])
    train_df = train_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if os.path.exists(TRACKMAN_PATH):
        pitcher_features = build_pitcher_consistency_features(TRACKMAN_PATH)
        train_df = pd.merge(train_df, pitcher_features, on='pitcher_id', how='left')
    
    full_features = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
    for col in CATEGORICAL_FEATURES:
        train_df[col] = train_df[col].astype(str).fillna('missing')

    # ==========================================
    # [전략 변경] 2024년 시즌의 마지막 경기 시점 기준 후반 30% 검증 분할
    # ==========================================
    is_2024 = train_df["season"] == 2024
    df_2024 = train_df[is_2024].copy()
    
    # 시간 순서 유지를 위해 원래 인덱스를 기반으로 후반 30% 커팅 포인트 연산
    split_idx = int(len(df_2024) * 0.7)
    
    val_split = df_2024.iloc[split_idx:].reset_index(drop=True)
    
    # 학습셋은 2024 이전 시즌 전체 + 2024년의 전반 70% 결합
    train_before_2024 = train_df[train_df["season"] < 2024]
    train_2024_front = df_2024.iloc[:split_idx]
    train_split = pd.concat([train_before_2024, train_2024_front], axis=0).reset_index(drop=True)

    X_train, y_train = train_split[full_features], train_split[TARGET_COL].values
    X_val, y_val = val_split[full_features], val_split[TARGET_COL].values

    print(f"📊 데이터 분할 완료 :: Train={len(X_train)} (2024년 70% 포함) | Val={len(X_val)} (2024년 후반 30%)")

    train_pool = Pool(data=X_train, label=y_train, cat_features=CATEGORICAL_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CATEGORICAL_FEATURES)

    eval_model = CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=6,
        loss_function='Logloss', eval_metric='BrierScore',
        random_seed=42, early_stopping_rounds=40, verbose=100, l2_leaf_reg=10
    )
    eval_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    eval_model.best_iteration_ = eval_model.get_best_iteration()

    with open(TEMP_MODEL_PATH, 'wb') as f:
        pickle.dump(eval_model, f)
    print(f"✅ [Train 완료] 임시 버퍼 저장 성공: {TEMP_MODEL_PATH}")

if __name__ == "__main__":
    main()
