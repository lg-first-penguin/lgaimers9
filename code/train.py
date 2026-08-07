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
        'horz_break': 'std',
        'induced_vert_break': 'std',
        'extension': 'std',
        'rel_speed': 'std'
    }).reset_index()
    
    consistency.columns = [
        'pitcher_id', 'pitch_type_group',
        'tm_std_horz_break', 'tm_std_vert_break', 
        'tm_std_extension', 'tm_std_rel_speed'
    ]
    
    pivot_consistency = consistency.pivot(
        index='pitcher_id',
        columns='pitch_type_group',
        values=['tm_std_horz_break', 'tm_std_vert_break', 'tm_std_extension', 'tm_std_rel_speed']
    )
    pivot_consistency.columns = [f"tm_{c[1]}_{c[0].replace('tm_', '')}" for c in pivot_consistency.columns]
    pivot_consistency = pivot_consistency.reset_index()
    return pivot_consistency

def main():
    DATA_DIR = "./open/data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    TRACKMAN_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
    
    # 훈련 단계에서는 임시폴더(temp)에 먼저 보관합니다.
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

    # 2024년 검증 분할
    is_val = train_df["season"] == 2024
    train_split = train_df[~is_val].reset_index(drop=True)
    test_split = train_df[is_val].reset_index(drop=True)

    X_train, y_train = train_split[full_features], train_split[TARGET_COL].values
    X_val, y_val = test_split[full_features], test_split[TARGET_COL].values

    train_pool = Pool(data=X_train, label=y_train, cat_features=CATEGORICAL_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CATEGORICAL_FEATURES)

    # 다른 모델 구조 확장 시 이 부분을 인터페이스화하거나 Sklearn API 모델로 변경 가능합니다.
    eval_model = CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=6,
        loss_function='Logloss', eval_metric='BrierScore',
        random_seed=42, early_stopping_rounds=40, verbose=100, l2_leaf_reg=10
    )
    eval_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    # 사후 활용을 위해 베스트 이터레이션 수 등을 모델 객체에 동적 바인딩
    eval_model.best_iteration_ = eval_model.get_best_iteration()

    with open(TEMP_MODEL_PATH, 'wb') as f:
        pickle.dump(eval_model, f)
    print(f"✅ [Train 완료] 임시 버퍼 저장 성공: {TEMP_MODEL_PATH}")

if __name__ == "__main__":
    main()
