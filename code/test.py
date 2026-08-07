# code/test.py
import os
import pickle
import pandas as pd
from catboost import Pool

ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL_FEATURES = ['top_bottom', 'game_type', 'base_state']

def calculate_bss(model, X_val, y_val):
    """모델 종류와 상관없이 predict_proba가 지원되는 형태면 BSS 계산 가능"""
    if hasattr(model, 'predict_proba'):
        # CatBoost Pool 대처 및 일반 DataFrame 대처 구조
        if type(model).__name__ == 'CatBoostClassifier':
            val_pool = Pool(data=X_val, label=y_val, cat_features=CATEGORICAL_FEATURES)
            preds = model.predict_proba(val_pool)[:, 1]
        else:
            preds = model.predict_proba(X_val)[:, 1]
    else:
        preds = model.predict(X_val) # 규격화 대비
        
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
    
    # 검증셋 구성 (train.py와 동일 로직)
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
        
    is_val = train_df["season"] == 2024
    test_split = train_df[is_val].reset_index(drop=True)
    X_val, y_val = test_split[full_features], test_split[TARGET_COL].values

    # 1. 최신 Temp 모델 평가
    with open(TEMP_MODEL_PATH, 'rb') as f:
        latest_model = pickle.load(f)
    latest_bss, latest_score = calculate_bss(latest_model, X_val, y_val)
    
    print("\n" + "="*60)
    print(f"{'[신규 훈련 모델 성능 리포트]':^50}")
    print(f"Brier Skill Score (BSS): {latest_bss:.5f}")
    print(f"대회 환산 점수: {latest_score:.2f}")
    print("="*60)

    # 2. Reference 모델 존재 시 비교 진행
    ref_bss = -float('inf')
    if os.path.exists(REF_MODEL_PATH):
        with open(REF_MODEL_PATH, 'rb') as f:
            ref_model = pickle.load(f)
        ref_bss, ref_score = calculate_bss(ref_model, X_val, y_val)
        print(f"-> 기존 Reference 모델 BSS: {ref_bss:.5f} (점수: {ref_score:.2f})")
    else:
        print("-> 비교할 기존 Reference 모델이 존재하지 않습니다. 첫 루프 실행으로 간주합니다.")

    # 결과를 상위 파이프라인(dopip.py)이 감지할 수 있도록 표준 출력 혹은 파일 신호 처리 구조 마련
    # 여기서는 간단히 비교 결과를 텍스트 파일로 임시 저장하여 통신합니다.
    with open("./open/temp/compare_result.txt", "w") as f:
        if latest_bss > ref_bss:
            f.write("NEW_BEST")
        else:
            f.write("KEEP_REF")

if __name__ == "__main__":
    main()
