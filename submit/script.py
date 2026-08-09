# script.py
import os
import pickle
import numpy as np
import pandas as pd
from catboost import Pool

ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL_FEATURES = ['top_bottom', 'game_type', 'base_state']

def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다. (code/train.py와 수동 동기화 유지)"""
    df = df.copy()

    df['pitcher_recent1_gap'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent3_gap'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent5_gap'] = df['asof_pitcher_prev5_game_success_rate'] - df['asof_pitcher_success_rate']

    df['pitcher_relative_success'] = df['asof_pitcher_success_rate'] - league_success_mean

    df['count_diff'] = df['strikes_before'] - df['balls_before']
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(np.int64)

    df['pitcher_count_advantage_raw'] = df['asof_pitcher_success_rate'] * df['count_diff']
    df['pitcher_count_advantage_rel'] = df['pitcher_relative_success'] * df['count_diff']

    df['pitcher_trend'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_consistency'] = df[[
        'asof_pitcher_prev1_game_success_rate',
        'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate',
    ]].std(axis=1)

    df['matchup'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']

    pressure_signal = df['li'] * ((df['strikes_before'] >= 2) | (df['balls_before'] >= 3)).astype(np.int64)
    df['count_pressure'] = df['pitcher_relative_success'] * pressure_signal

    return df

def main():
    # 대회 서빙 환경 표준 경로 정의
    DATA_DIR = "./data"
    MODEL_PATH = "./model/final_retained_model.pkl"
    OUTPUT_DIR = "./output"

    # 1. 필수 입력 데이터 로드
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
    trackman_path = os.path.join(DATA_DIR, "trackman_history.csv")
    train_path = os.path.join(DATA_DIR, "train.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ 필수 입력 파일이 없습니다: {test_path}")

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    df_sub = pd.read_csv(sample_sub_path, encoding="utf-8-sig")
    df_trm = pd.read_csv(trackman_path, encoding="utf-8-sig")

    # final_retained_model.pkl과 동일하게, 전체 train.csv 기준 리그 평균 성공률 계산
    df_train_raw = pd.read_csv(train_path, encoding="utf-8-sig")
    league_success_mean = df_train_raw[TARGET_COL].mean()
    
    # 2. 저장된 최종 통합 완습 모델 로드
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 제출 구조 내 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        
    with open(MODEL_PATH, 'rb') as f:
        final_model = pickle.load(f)
        
    # 3. 파이프라인 무결성 유지를 위한 동적 모듈 주입 및 전처리 수행
    # train.py의 전처리 함수 구조를 직접 가져와 활용
    df_main_copy = df_test.copy()
    df_trm_copy = df_trm.copy()
    
    # 전처리 결합 기준 컬럼 추출
    match_cols = [dfc for dfc in df_main_copy.columns if (dfc in df_trm_copy.columns) and dfc != 'row_id']
    
    # 과거 트랙맨 로그 기반 통계량 산출 (추론 시점이므로 과거 전체 데이터 활용)
    grouped = df_trm_copy.groupby(match_cols + ['pitch_type_group', 'auto_pitch_type'])
    grouped_phase1 = grouped[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']].agg(['mean', 'std'])
    grouped_phase2 = grouped_phase1.reset_index()
    
    std_cols = [col for col in grouped_phase2.columns if 'std' in col]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ['_'.join(col).strip('_') for col in grouped_phase2.columns]
    
    grouped_phase3 = grouped_phase2.drop(columns='auto_pitch_type')
    grouped_phase3 = grouped_phase3.groupby(match_cols + ['pitch_type_group']).agg(['mean'])
    
    pivoted = grouped_phase3.unstack(level='pitch_type_group')
    pivoted.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)
    
    # 데이터 타입 정밀 매칭 및 인코딩
    df_main_copy['top_bottom'] = df_main_copy['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    tm_final['top_bottom'] = tm_final['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)
    
    for col in ['batter_hand', 'pitcher_hand']:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({'Left': 1, 'Right': 2}).astype(np.int64)
            
    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how='left')
    new_feature_cols = [col for col in tm_final.columns if col not in match_cols]

    # 훈련 시점 피처 통계 기반 결측치 보정 (추론 시점의 결측값은 0으로 예외 처리 방어 조치)
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(0)

    # 3.5 asof_* 및 카운트 정보를 조합한 파생 피처 추가 (code/train.py와 동일 정의)
    tr_final = add_engineered_features(tr_final, league_success_mean)

    # 4. 모델 입력 데이터 정렬 및 풀 구성
    drop_cols = [ID_COL, TARGET_COL]
    features = [col for col in tr_final.columns if col not in drop_cols]
    
    for col in CATEGORICAL_FEATURES:
        if col in tr_final.columns:
            tr_final[col] = tr_final[col].astype(str).fillna('missing')
            
    if 'top_bottom' in tr_final.columns:
        tr_final['top_bottom'] = tr_final['top_bottom'].astype(np.int64)
        
    X_test = tr_final[features]
    
    # 5. 확률 추론 수행
    test_pool = Pool(data=X_test, cat_features=CATEGORICAL_FEATURES)
    preds = final_model.predict_proba(test_pool)[:, 1]
    
    # 6. 제출 서식 동기화 및 저장
    df_sub[TARGET_COL] = preds
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 추론 및 제출용 파일 저장 완료: {out_path}")

if __name__ == "__main__":
    main()