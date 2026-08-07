# code/test.py
import os
import pickle
import pandas as pd
from catboost import Pool
from sklearn.model_selection import train_test_split

ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL_FEATURES = ['top_bottom', 'game_type', 'base_state']

def calculate_bss(model, X_val, y_val):
    if hasattr(model, 'predict_proba'):
        if type(model).__name__ == 'CatBoostClassifier':
            val_pool = Pool(data=X_val, label=y_val, cat_features=CATEGORICAL_FEATURES)
            preds = model.predict_proba(val_pool)[:, 1]
        else:
            preds = model.predict_proba(X_val)[:, 1]
    else:
        preds = model.predict(X_val)
        
    brier = ((preds - y_val) ** 2).mean()
    r = y_val.mean()
    baseline_brier = r * (1 - r)
    bss = 1.0 - (brier / baseline_brier)
    score = max(0.0, 100000.0 * bss)
    return bss, score

def main():
    DATA_DIR = "./open/data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    TRACKMAN_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
    
    TEMP_MODEL_PATH = "./open/temp/latest_model.pkl"
    REF_MODEL_PATH = "./open/reference/best_model.pkl"
    
    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns
    base_features = [c for c in test_cols if c != ID_COL]
    train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=base_features + [TARGET_COL])
    train_df = train_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    
    if os.path.exists(TRACKMAN_PATH):
        from train import build_pitcher_consistency_features
        pitcher_features = build_pitcher_consistency_features(TRACKMAN_PATH)
        train_df = pd.merge(train_df, pitcher_features, on='pitcher_id', how='left')
        
    full_features = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
    for col in CATEGORICAL_FEATURES:
        train_df[col] = train_df[col].astype(str).fillna('missing')
        
    # train.py와 완벽히 동일한 구조의 랜덤 30% 테스트셋 분리 고정
    _, val_split = train_test_split(
        train_df, 
        test_size=0.3, 
        random_state=42, 
        stratify=train_df[TARGET_COL]
    )
    val_split = val_split.reset_index(drop=True)
    X_val, y_val = val_split[full_features], val_split[TARGET_COL].values

    # 1. 최신 Temp 모델 평가
    with open(TEMP_MODEL_PATH, 'rb') as f:
        latest_model = pickle.load(f)
    latest_bss, latest_score = calculate_bss(latest_model, X_val, y_val)
    
    print("\n" + "="*60)
    print(f"{'[랜덤 30% 검증 모델 성능 비교 리포트]':^50}")
    print(f"Brier Skill Score (BSS): {latest_bss:.5f}")
    print(f"대회 환산 점수: {latest_score:.2f}")
    print("="*60)

    # 2. Reference 모델과 비교
    ref_bss = -float('inf')
    if os.path.exists(REF_MODEL_PATH):
        with open(REF_MODEL_PATH, 'rb') as f:
            ref_model = pickle.load(f)
        ref_bss, ref_score = calculate_bss(ref_model, X_val, y_val)
        print(f"-> 기존 Reference 모델 BSS: {ref_bss:.5f} (점수: {ref_score:.2f})")

    with open("./open/temp/compare_result.txt", "w") as f:
        if latest_bss > ref_bss:
            f.write("NEW_BEST")
        else:
            f.write("KEEP_REF")

if __name__ == "__main__":
    main()
