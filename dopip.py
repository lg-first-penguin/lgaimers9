# dopip.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shutil
import subprocess
import pickle
import pandas as pd
import numpy as np

ID_COL = "row_id"
TARGET_COL = "control_success"


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
    # "python"이 PATH에 없는 환경(예: venv를 activate하지 않고 venv의 python 바이너리를
    # 직접 호출한 경우)에서도 항상 지금 dopip.py를 실행 중인 것과 동일한 인터프리터로
    # 서브프로세스를 띄우도록 sys.executable을 사용합니다.
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    # Step 3. 최종 확정된 Reference 모델 기반 전체 데이터(2019~2024) 통합 재학습 후 submit 구조 구축
    # =========================================================================
    print("\n--- [Step 3] 최종 검증 완료본 기반 전체 시즌 데이터 통합 재학습 (Full Retrain) ---")
    if not os.path.exists(ref_model_path):
        print("⚠ 학습된 Reference 모델이 없어 전체 재학습을 진행할 수 없습니다.")
        return

    with open(ref_model_path, 'rb') as f:
        best_eval_model = pickle.load(f)

    if not isinstance(best_eval_model, dict) or best_eval_model.get("model_kind") != "mlp_ensemble":
        print("❌ Reference 모델이 MLP 앙상블 형식이 아닙니다. dopip.py를 다시 실행해 "
              "MLP 앙상블 기준 Reference를 먼저 만드세요.")
        return

    # MLP는 그래디언트 학습 루프가 있어 TabPFN과 달리 '몇 에폭을 돌릴지'가 필요합니다.
    # season==2024 홀드아웃으로 조기종료해서 찾은 시드별 best_epoch을 그대로 재사용해,
    # 홀드아웃 없이 전체 데이터로 시드마다 정확히 그 횟수만큼 고정 학습합니다.
    full_retrain_epochs_per_seed = best_eval_model["best_epochs"]
    print(f"[Full Retrain] Reference 모델의 시드별 best_epochs={full_retrain_epochs_per_seed}를 그대로 사용합니다.")

    DATA_DIR = "./open/data"
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")

    from code.train import (
        map_top_bottom,
        apply_f1_filter,
        build_season_end_lookup,
        apply_season_progression_features,
        add_engineered_features,
        add_missing_indicators,
        fit_preprocessing,
        transform_features,
        train_mlp_ensemble,
        build_feature_lists,
        get_device,
        MLP_SEEDS,
    )

    train_df_raw = map_top_bottom(train_df_raw)
    train_df_raw = apply_f1_filter(train_df_raw)
    train_df = train_df_raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    league_success_mean = train_df[TARGET_COL].mean()

    # 최종 제출 모델은 홀드아웃이 없으므로 lookup도 전체 데이터로 만들고, 그대로 모델
    # 딕셔너리에 담아 submit/script.py가 test.csv(2025)에 재계산 없이 적용하게 합니다.
    season_end_lookup = build_season_end_lookup(train_df)
    train_df = apply_season_progression_features(train_df, season_end_lookup)

    train_df = add_engineered_features(train_df, league_success_mean)

    full_mask = pd.Series(True, index=train_df.index)
    categorical_features, numeric_features, na_cols = build_feature_lists(train_df, TARGET_COL, full_mask)
    train_df = add_missing_indicators(train_df, na_cols)
    numeric_features = numeric_features + [f'{c}_isna' for c in na_cols]

    cat_encoders, cat_cardinalities, numeric_medians, scaler = fit_preprocessing(
        train_df, categorical_features, numeric_features)

    X_num_full, X_cat_full = transform_features(
        train_df, categorical_features, numeric_features, cat_encoders, numeric_medians, scaler)
    y_full = train_df[TARGET_COL].values.astype(np.float32)

    device = get_device()
    embed_dims = [best_eval_model["embed_dims"][best_eval_model["categorical_features"].index(c)]
                  for c in categorical_features]

    print(f"[Full Retrain] 총 {len(train_df)}행 전체 데이터(2019~2024)로 {len(MLP_SEEDS)}개 시드를 "
          f"각각 시드별 고정 에폭만큼 학습합니다.")
    state_dicts, _, _ = train_mlp_ensemble(
        X_num_full, X_cat_full, y_full,
        cat_cardinalities=cat_cardinalities, embed_dims=embed_dims, device=device,
        seeds=MLP_SEEDS, fixed_epochs_per_seed=full_retrain_epochs_per_seed,
    )

    final_model = {
        "model_kind": "mlp_ensemble",
        "state_dicts": state_dicts,
        "seeds": MLP_SEEDS,
        "num_numeric": len(numeric_features),
        "cat_cardinalities": cat_cardinalities,
        "embed_dims": embed_dims,
        "hidden_dims": best_eval_model["hidden_dims"],
        "dropout": best_eval_model["dropout"],
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "na_indicator_source_cols": na_cols,
        "cat_encoders": cat_encoders,
        "numeric_medians": numeric_medians,
        "scaler": scaler,
        "league_success_mean": league_success_mean,
        "season_end_lookup": season_end_lookup,
        "best_epochs": full_retrain_epochs_per_seed,
        "val_bss": None,
    }

    FINAL_SUBMIT_MODEL_PATH = "./submit/model/final_retained_model.pkl"
    os.makedirs(os.path.dirname(FINAL_SUBMIT_MODEL_PATH), exist_ok=True)
    with open(FINAL_SUBMIT_MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)

    print(f"[Pipeline 완료] 대형 파이프라인 무결성 통과. 제출용 모델 생성 성공: {FINAL_SUBMIT_MODEL_PATH}")


if __name__ == "__main__":
    main()
