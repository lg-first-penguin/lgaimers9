# dopip.py
import os
import shutil
import subprocess
import pickle
import pandas as pd
from catboost import CatBoostClassifier, Pool

def run_script(script_path):
    process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"❌ {script_path} 실행 중 오류가 발생했습니다.")

def main():
    print("[Pipeline] 통합 드라이버 파이프라인(dopip.py) 실행중...")
    
    # 폴더 구조 원자성 확립을 위한 사전 세팅
    os.makedirs("./open/temp", exist_ok=True)
    os.makedirs("./open/reference", exist_ok=True)
    os.makedirs("./open/former_model", exist_ok=True)
    os.makedirs("./submit/model", exist_ok=True)
    
    # Step 1. 모델 훈련 가동 (train.py 실행)
    print("\n--- [Step 1] 모델 학습 프로세스 가동 (code/train.py) ---")
    run_script("./code/train.py")
    
    # Step 2. 모델 검증 및 비교 평가 (test.py 실행)
    print("\n--- [Step 2] 성능 검증 및 Reference 비교 프로세스 가동 (code/test.py) ---")
    run_script("./code/test.py")
    
    # 파일 상태 이동 분석
    result_flag_path = "./open/temp/compare_result.txt"
    if not os.path.exists(result_flag_path):
        print("❌ 비교 결과 플래그를 찾을 수 없습니다.")
        return
        
    with open(result_flag_path, "r") as f:
        result = f.read().strip()
        
    latest_model_src = "./open/temp/latest_model.pkl"
    former_model_dest = "./open/former_model/former_latest_model.pkl"
    ref_model_path = "./open/reference/best_model.pkl"
    former_ref_dest = "./open/former_model/former_best_model.pkl"
    
    # 규칙 요건 분기 처리 
    # [조건A] 방금 막 훈련시켰던 모델은 무조건 former_model 폴더로 복사 이동 처리
    if os.path.exists(latest_model_src):
        shutil.copy2(latest_model_src, former_model_dest)
        print(f"최신 훈련 모델 백업 완료 -> {former_model_dest}")
        
    if result == "NEW_BEST":
        print("\n[Result] 신규 모델이 기존 Reference보다 높은 점수. ref model 교체를 진행...")
        # 기존 reference에 있던 모델이 있다면 former_model로 백업 이동 후 삭제 효과
        if os.path.exists(ref_model_path):
            shutil.move(ref_model_path, former_ref_dest)
            print(f"기존 Reference 모델을 백업함 -> {former_ref_dest}")
            
        # 신규 모델을 reference로 등극
        shutil.move(latest_model_src, ref_model_path)
        print(f"신규 모델을 Reference로 등록 완료 -> {ref_model_path}")
        
    else:
        print("\n[Result] 기존 Reference 모델의 성능이 더 우수하거나 동일합니다. 기준 모델을 유지합니다.")
        # temp에 생성된 최신 모델 삭제 (기존 temp 모델 삭제 조건 반영)
        if os.path.exists(latest_model_src):
            os.remove(latest_model_src)
            print("🗑️시 버퍼(temp) 내 최신 모델을 비웠습니다.")

    # Step 3. 최종 확정된 Reference 모델 기반 전체 데이터(2019~2024) 통합 완습 (Retrain) 후 submit 구조 구축
    print("\n--- [Step 3] 최종 검증 완료본 기반 전체 시즌 데이터 통합 완습 (Full Retrain) ---")
    
    if not os.path.exists(ref_model_path):
        print("⚠ 학습된 Reference 모델이 없어 전체 재학습을 진행할 수 없습니다.")
        return
        
    # Reference 모델 사양 로드
    with open(ref_model_path, 'rb') as f:
        best_eval_model = pickle.load(f)
        
    # 최적 이터레이션 수 추출 (일반화 시 파라미터 딕셔너리 추출 구조로 다변화 가능)
    best_iter = getattr(best_eval_model, 'best_iteration_', 300)
    
    # 전체 데이터 로드
    DATA_DIR = "./open/data"
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    trackman_path = os.path.join(DATA_DIR, "trackman_history.csv")
    
    base_features = [c for c in test_df.columns if c != ID_COL]
    train_df = train_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    
    if os.path.exists(trackman_path):
        from code.train import build_pitcher_consistency_features
        pitcher_features = build_pitcher_consistency_features(trackman_path)
        train_df = pd.merge(train_df, pitcher_features, on='pitcher_id', how='left')
        
    full_features = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
    for col in CATEGORICAL_FEATURES:
        train_df[col] = train_df[col].astype(str).fillna('missing')
        
    print(f"[Full Retrain] 총 {len(train_df)}행 전체 데이터에 대해 {best_iter + 10}회 반복 학습을 진행합니다.")
    
    X_full = train_df[full_features]
    y_full = train_df[TARGET_COL].values
    
    # 확장성 가이드: 만약 CatBoost가 아니라 타 프레임워크인 경우 객체 타입 판별 분기 가능
    if type(best_eval_model).__name__ == 'CatBoostClassifier':
        full_pool = Pool(data=X_full, label=y_full, cat_features=CATEGORICAL_FEATURES)
        final_model = CatBoostClassifier(
            iterations=best_iter + 10, learning_rate=0.03, depth=6,
            loss_function='Logloss', random_seed=42, verbose=50, l2_leaf_reg=10
        )
        final_model.fit(full_pool)
    else:
        # 일반 Scikit-learn 이나 타 프레임워크 대응 범용 clone & fit 구조
        from sklearn.base import clone
        final_model = clone(best_eval_model)
        final_model.fit(X_full, y_full)

    # 최종 제출 경로로 가중치 파일 내보내기
    FINAL_SUBMIT_MODEL_PATH = "./submit/model/final_retained_model.pkl"
    with open(FINAL_SUBMIT_MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"[Pipeline 완료] 대형 파이프라인 무결성 통과. 제출용 모델 생성 성공: {FINAL_SUBMIT_MODEL_PATH}")

if __name__ == "__main__":
    main()
