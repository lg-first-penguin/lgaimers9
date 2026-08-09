# dopip.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shutil
import subprocess
import pickle
import pandas as pd
from catboost import CatBoostClassifier, Pool
import numpy as np

ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL_FEATURES = ['top_bottom', 'game_type', 'base_state']

def get_numbered_path(base_dest_path):
    if not os.path.exists(base_dest_path):
        return base_dest_path
    base, ext = os.path.splitext(base_dest_path)
    counter = 1
    while True:
        new_path = f"{base}_v{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def run_script(script_path):
    process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"❌ {script_path} 실행 중 오류가 발생했습니다.")

def main():
    print("[Pipeline] 통합 드라이버 파이프라인(dopip.py) 실행중...")
    
    os.makedirs("./open/temp", exist_ok=True)
    os.makedirs("./open/reference", exist_ok=True)
    os.makedirs("./open/former_model", exist_ok=True)
    os.makedirs("./submit/model", exist_ok=True)
    
    print("\n--- [Step 1] 모델 학습 프로세스 가동 (code/train.py) ---")
    run_script("./code/train.py")
    
    print("\n--- [Step 2] 성능 검증 및 Reference 비교 프로세스 가동 (code/test.py) ---")
    run_script("./code/test.py")
    
    result_flag_path = "./open/temp/compare_result.txt"
    if not os.path.exists(result_flag_path):
        print("❌ 비교 결과 플래그를 찾을 수 없습니다.")
        return
        
    with open(result_flag_path, "r") as f:
        result = f.read().strip()
        
    latest_model_src = "./open/temp/latest_model.pkl"
    former_model_dir = "./open/former_model"
    ref_model_path = "./open/reference/best_model.pkl"
    
    base_former_dest = os.path.join(former_model_dir, "former_latest_model.pkl")
    former_model_dest = get_numbered_path(base_former_dest)
    base_ref_dest = os.path.join(former_model_dir, "former_best_model.pkl")
    former_ref_dest = get_numbered_path(base_ref_dest)

    if os.path.exists(latest_model_src):
        shutil.copy2(latest_model_src, former_model_dest)
        print(f"최신 훈련 모델 백업 완료 -> {former_model_dest}")
        
    if result == "NEW_BEST":
        print("\n[Result] 신규 모델이 기존 Reference보다 높은 점수. ref model 교체를 진행...")
        if os.path.exists(ref_model_path):
            shutil.move(ref_model_path, former_ref_dest)
            print(f"기존 Reference 모델을 백업함 -> {former_ref_dest}")
        shutil.move(latest_model_src, ref_model_path)
        print(f"신규 모델을 Reference로 등록 완료 -> {ref_model_path}")
    else:
        print("\n[Result] 기존 Reference 모델의 성능이 더 우수하거나 동일합니다. 기준 모델을 유지합니다.")
        if os.path.exists(latest_model_src):
            os.remove(latest_model_src)
            print("임시 버퍼(temp) 내 최신 모델을 비웠습니다.")

    # =========================================================================
    # Step 3. 최종 확정된 Reference 모델 기반 전체 데이터(2019~2024) 통합 완습 (Retrain) 후 submit 구조 구축
    # =========================================================================
    print("\n--- [Step 3] 최종 검증 완료본 기반 전체 시즌 데이터 통합 완습 (Full Retrain) ---")
    if not os.path.exists(ref_model_path):
        print("⚠ 학습된 Reference 모델이 없어 전체 재학습을 진행할 수 없습니다.")
        return
        
    with open(ref_model_path, 'rb') as f:
        best_eval_model = pickle.load(f)
        
    best_iter = getattr(best_eval_model, 'best_iteration_', 300)
    if best_iter is None or best_iter < 0:
        best_iter = 400

    DATA_DIR = "./open/data"
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm_raw = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    
    # [수정 완료 지점] train.py에서 선언한 무결성 타임필터 전처리 함수를 그대로 수입하여 1줄로 결합 완수
    from code.train import process_trackman_features_safe, add_engineered_features
    tr_final, match_cols = process_trackman_features_safe(train_df_raw, df_trm_raw, is_train_split=False)

    train_df = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    league_success_mean = train_df[TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = [ID_COL, TARGET_COL]
    full_features = [col for col in train_df.columns if col not in drop_cols]
    
    for col in CATEGORICAL_FEATURES:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str).fillna('missing')
            
    if 'top_bottom' in train_df.columns:
        train_df['top_bottom'] = train_df['top_bottom'].astype(np.int64)
            
    print(f"[Full Retrain] 총 {len(train_df)}행 전체 데이터에 대해 {best_iter + 10}회 반복 학습을 진행합니다.")
    X_full = train_df[full_features]
    y_full = train_df[TARGET_COL].values
    
    if type(best_eval_model).__name__ == 'CatBoostClassifier':
        full_pool = Pool(data=X_full, label=y_full, cat_features=CATEGORICAL_FEATURES)
        final_model = CatBoostClassifier(
            iterations=best_iter + 10,
            learning_rate=0.05040411253232039,
            depth=7,
            l2_leaf_reg=3.14659036827521,
            random_strength=3.715568024268865,
            bagging_temperature=0.4609270457436248,
            border_count=106,
            min_data_in_leaf=81,
            bootstrap_type='Bayesian',
            loss_function='Logloss',
            random_seed=42,
            verbose=50,
        )
        final_model.fit(full_pool)
    else:
        from sklearn.base import clone
        final_model = clone(best_eval_model)
        final_model.fit(X_full, y_full)
        
    FINAL_SUBMIT_MODEL_PATH = "./submit/model/final_retained_model.pkl"
    os.makedirs(os.path.dirname(FINAL_SUBMIT_MODEL_PATH), exist_ok=True)
    with open(FINAL_SUBMIT_MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"[Pipeline 완료] 대형 파이프라인 무결성 통과. 제출용 모델 생성 성공: {FINAL_SUBMIT_MODEL_PATH}")

if __name__ == "__main__":
    main()