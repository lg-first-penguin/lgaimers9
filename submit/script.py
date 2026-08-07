# submit/script.py
import os
import pickle
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

def main():
    # 대회 규칙에 지정된 서버 인터페이스 경로 매핑
    TEST_PATH = "./data/test.csv"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    MODEL_PATH = "./model/final_retained_model.pkl"
    OUTPUT_PATH = "./output/submission.csv"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"제출용 가중치 파일을 찾을 수 없습니다: {MODEL_PATH}")

    # 모델 복원
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # 데이터 로딩
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sub_df = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")

    # 트랙맨 보조 데이터가 평가서버의 ./data/ 폴더에 주어질 경우 자동 연동 설계
    TRACKMAN_PATH = "./data/trackman_history.csv"
    if os.path.exists(TRACKMAN_PATH):
        # 파이프라인 전처리 피처 빌더 탑재 (서버 오프라인 실행 목적)
        # 훈련용 코드의 로직과 동일하게 작동하되 행 간 독립성을 해치지 않는 매핑 가동
        tm = pd.read_csv(TRACKMAN_PATH, encoding="utf-8-sig")
        tm = tm.dropna(subset=['pitcher_trackman_id', 'pitch_type_group']).copy()
        consistency = tm.groupby(['pitcher_trackman_id', 'pitch_type_group']).agg({
            'horz_break': 'std', 'induced_vert_break': 'std', 'extension': 'std', 'rel_speed': 'std'
        }).reset_index()
        consistency.columns = ['pitcher_id', 'pitch_type_group', 'tm_std_horz_break', 'tm_std_vert_break', 'tm_std_extension', 'tm_std_rel_speed']
        pivot_consistency = consistency.pivot(index='pitcher_id', columns='pitch_type_group', values=['tm_std_horz_break', 'tm_std_vert_break', 'tm_std_extension', 'tm_std_rel_speed'])
        pivot_consistency.columns = [f"tm_{c[1]}_{c[0].replace('tm_', '')}" for c in pivot_consistency.columns]
        pivot_consistency = pivot_consistency.reset_index()
        
        test_df = pd.merge(test_df, pivot_consistency, on='pitcher_id', how='left')

    # 피처 컬럼 규격 정의
    full_features = [c for c in test_df.columns if c != ID_COL]
    
    CATEGORICAL_FEATURES = ['top_bottom', 'game_type', 'base_state']
    for col in CATEGORICAL_FEATURES:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str).fillna('missing')

    X_test = test_df[full_features]

    # 추론 확률값 매핑
    print("Executing server inference...")
    if type(model).__name__ == 'CatBoostClassifier':
        preds = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict_proba(X_test)[:, 1]

    # 샘플 서브미션 레이아웃 순서 보장 매핑
    sub_df[TARGET_COL] = preds

    # 결과물 출력 보관
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sub_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✅ 제출 포맷팅 파일 저장 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
