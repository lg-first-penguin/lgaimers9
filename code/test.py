# code/test.py
import sys
import os

# [조립 핵심 지점] 실행 디렉토리 위치 독립 무결성 보정식 주입
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
import pandas as pd
from catboost import Pool

ID_COL = "row_id"
TARGET_COL = "control_success"

# 이제 상위 루트 디렉토리가 시스템 패스에 잡혀있으므로 완벽하게 import 성공합니다.
from code.train import process_trackman_features_safe, add_engineered_features

def calculate_bss(model, X_val, y_val, categorical_features):
    """안전장치가 적용된 BSS 평가 루틴"""
    val_pool = Pool(data=X_val, label=y_val, cat_features=categorical_features)
    preds = model.predict_proba(val_pool)[:, 1]
    
    brier = ((preds - y_val) ** 2).mean()
    r = y_val.mean()
    baseline_brier = r * (1 - r)
    bss = 1.0 - (brier / baseline_brier)
    score = max(0.0, 100000.0 * bss)
    return bss, score

def main():
    DATA_DIR = "./open/data"
    TEMP_MODEL_PATH = "./open/temp/latest_model.pkl"
    REF_MODEL_PATH = "./open/reference/best_model.pkl"
    
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    
    # 10-Key 타임 필터 기반 가동 동기화 (is_train_split=True)
    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    
    train_df = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # train.py와 동일하게 train-split(season<2024) 기준 리그 평균으로 파생 피처 산출
    train_mask = train_df['season'] < 2024
    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [col for col in train_df.columns if col not in [ID_COL, TARGET_COL]]
    categorical_features = [col_cat for col_cat in tr_final.select_dtypes(include='object').columns if col_cat != ID_COL and col_cat in features]

    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna('missing')

    # 2024년 데이터는 학습에서 완전히 제외하고 검증셋으로만 사용
    val_split = train_df[train_df['season'] == 2024].reset_index(drop=True)
    
    X_val, y_val = val_split[features], val_split[TARGET_COL].values
    
    if 'top_bottom' in X_val.columns:
        X_val['top_bottom'] = X_val['top_bottom'].astype(np.int64)
        
    # 1. 신규 최신 임시 버퍼 모델 검증
    with open(TEMP_MODEL_PATH, 'rb') as f:
        latest_model = pickle.load(f)
    latest_bss, latest_score = calculate_bss(latest_model, X_val, y_val, categorical_features)
    
    print("\n" + "="*60)
    print(f"{'[10-Key 상황 동기화 검증 모델 성능 리포트]':^50}")
    print(f" Brier Skill Score (BSS): {latest_bss:.5f}")
    print(f" 대회 환산 예측 점수   : {latest_score:.2f}")
    print("="*60)

    # 2. 기존 최고 Reference 모델과 성능 대조
    ref_bss = -float('inf')
    if os.path.exists(REF_MODEL_PATH):
        with open(REF_MODEL_PATH, 'rb') as f:
            ref_model = pickle.load(f)
        ref_bss, ref_score = calculate_bss(ref_model, X_val, y_val, categorical_features)
        print(f" ➔ 기존 최고 Reference 모델 BSS: {ref_bss:.5f} (점수: {ref_score:.2f})")

    with open("./open/temp/compare_result.txt", "w") as f:
        if latest_bss > ref_bss:
            f.write("NEW_BEST")
        else:
            f.write("KEEP_REF")
    print("✅ 검증 세트 스코어 비교 대조록 갱신 성공.")

if __name__ == "__main__":
    main()