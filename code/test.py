# code/test.py
import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
import pandas as pd
import torch

ID_COL = "row_id"
TARGET_COL = "control_success"

from code.train import (
    map_top_bottom,
    apply_f1_filter,
    build_season_end_lookup,
    apply_season_progression_features,
    add_engineered_features,
    add_missing_indicators,
    transform_features,
    predict_proba_mlp_ensemble,
    bss_score,
    get_device,
)


def is_mlp_dict(model):
    return isinstance(model, dict) and model.get("model_kind") == "mlp_ensemble"


def evaluate(model_dict, base_val_df, device):
    """model_dict 자신의 결측 플래그/인코더/스케일러 기준으로 검증셋을 변환해 앙상블 BSS를 계산합니다."""
    val_df = add_missing_indicators(base_val_df, model_dict["na_indicator_source_cols"])
    X_num, X_cat = transform_features(
        val_df,
        model_dict["categorical_features"],
        model_dict["numeric_features"],
        model_dict["cat_encoders"],
        model_dict["numeric_medians"],
        model_dict["scaler"],
    )
    preds = predict_proba_mlp_ensemble(model_dict, X_num, X_cat, device)
    y_val = val_df[TARGET_COL].values
    return bss_score(preds, y_val)


def main():
    DATA_DIR = "./open/data"
    TEMP_MODEL_PATH = "./open/temp/latest_model.pkl"
    REF_MODEL_PATH = "./open/reference/best_model.pkl"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df = map_top_bottom(df)
    df = apply_f1_filter(df)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # train.py와 동일하게 train-split(season<2024) 기준으로 리그 평균/season-progression lookup 산출
    train_mask = train_df['season'] < 2024
    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()

    season_end_lookup = build_season_end_lookup(train_df.loc[train_mask])
    train_df = apply_season_progression_features(train_df, season_end_lookup)

    train_df = add_engineered_features(train_df, league_success_mean)

    # 2024년 데이터는 학습에서 완전히 제외하고 검증셋으로만 사용
    base_val_df = train_df[train_df['season'] == 2024].reset_index(drop=True)

    device = get_device()

    # 1. 신규 최신 임시 버퍼 모델 검증
    with open(TEMP_MODEL_PATH, 'rb') as f:
        latest_model = pickle.load(f)
    if not is_mlp_dict(latest_model):
        raise TypeError(
            f"❌ {TEMP_MODEL_PATH}가 MLP 딕셔너리가 아닙니다. code/train.py를 먼저 실행하세요."
        )
    latest_bss, latest_score = evaluate(latest_model, base_val_df, device)

    print("\n" + "=" * 60)
    print(f"{'[MLP 검증 모델 성능 리포트]':^50}")
    print(f" Brier Skill Score (BSS): {latest_bss:.5f}")
    print(f" 대회 환산 예측 점수   : {latest_score:.2f}")
    print("=" * 60)

    # 2. 기존 최고 Reference 모델과 성능 대조
    # (레퍼런스가 옛 TabPFN/CatBoost 아티팩트 등 MLP 형식이 아니면 비교 없이 -inf로 취급하여
    #  신규 MLP 모델이 자동으로 새 기준선이 되도록 합니다.)
    ref_bss = -float('inf')
    if os.path.exists(REF_MODEL_PATH):
        with open(REF_MODEL_PATH, 'rb') as f:
            ref_model = pickle.load(f)
        if is_mlp_dict(ref_model):
            ref_bss, ref_score = evaluate(ref_model, base_val_df, device)
            print(f" ➔ 기존 최고 Reference 모델 BSS: {ref_bss:.5f} (점수: {ref_score:.2f})")
        else:
            print(" ➔ 기존 Reference 모델이 MLP 형식이 아니므로 비교를 건너뜁니다 "
                  "(신규 모델이 자동으로 새 기준선이 됩니다).")

    with open("./open/temp/compare_result.txt", "w") as f:
        if latest_bss > ref_bss:
            f.write("NEW_BEST")
        else:
            f.write("KEEP_REF")
    print("✅ 검증 세트 스코어 비교 대조록 갱신 성공.")


if __name__ == "__main__":
    main()
