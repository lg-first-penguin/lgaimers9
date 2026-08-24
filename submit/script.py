# script.py
import os

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ID_COL = "row_id"
TARGET_COL = "control_success"
PREDICT_BATCH_SIZE = 20000


# ── 아래 함수/클래스들은 code/train.py와 동일 로직입니다. 이 스크립트는 대회 서버에서
#    submit.zip만으로 독립 실행되어야 하므로 code/를 import하지 않고 그대로 복사해 둡니다.
#    code/train.py의 피처 엔지니어링이나 모델 구조를 바꾸면 반드시 여기도 수동으로 동기화하세요.

def map_top_bottom(df):
    df = df.copy()
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    return df


SEASON_PROGRESSION_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def apply_season_progression_features(df, lookup):
    df = df.copy()
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
        lut = lookup.loc[lookup['role'] == role, ['id', 'season', 'end_n', 'end_rate', 'end_success']]
        merged = df[[id_col, 'season']].merge(
            lut, left_on=[id_col, 'season'], right_on=['id', 'season'], how='left')

        pre_n = (merged['end_n'] + 1).fillna(0).values
        pre_success = ((merged['end_n'] * merged['end_rate']).round().fillna(0)
                        + merged['end_success'].fillna(0)).values

        cum_n = df[n_col].values
        cum_success = np.round(df[n_col].values * df[rate_col].values)

        season_n = np.maximum(cum_n - pre_n, 0)
        season_success = np.maximum(cum_success - pre_success, 0)
        season_rate = np.divide(season_success, season_n,
                                 out=np.full_like(season_success, np.nan, dtype=np.float64),
                                 where=season_n > 0)

        df[f'{role}_season_n'] = season_n
        df[f'{role}_season_success_count'] = season_success
        df[f'{role}_season_success_rate'] = season_rate
        df[f'{role}_season_rate_gap'] = season_rate - df[rate_col].values

    return df


def add_engineered_features(df, league_success_mean):
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


def add_missing_indicators(df, na_cols):
    df = df.copy()
    for col in na_cols:
        df[f'{col}_isna'] = df[col].isna().astype(np.float32)
    return df


class TabularMLP(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, embed_dims, hidden_dims, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim) for card, dim in zip(cat_cardinalities, embed_dims)
        ])
        input_dim = num_numeric + sum(embed_dims)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        if self.embeddings:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + embs, dim=1)
        else:
            x = x_num
        return self.mlp(x).squeeze(-1)


def transform_features(df, categorical_features, numeric_features, cat_encoders, numeric_medians, scaler):
    n = len(df)
    X_cat = np.zeros((n, len(categorical_features)), dtype=np.int64)
    for i, col in enumerate(categorical_features):
        enc = cat_encoders[col]
        codes = enc.transform(df[[col]].astype(str)).astype(np.int64).ravel()
        unknown_index = len(enc.categories_[0])
        codes[codes == -1] = unknown_index
        X_cat[:, i] = codes

    X_num = df[numeric_features].fillna(numeric_medians)
    X_num = scaler.transform(X_num).astype(np.float32)

    return X_num, X_cat


def predict_proba_mlp(model, X_num, X_cat, device, batch_size=PREDICT_BATCH_SIZE):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_num), batch_size):
            end = start + batch_size
            xb_num = torch.from_numpy(X_num[start:end]).to(device)
            xb_cat = torch.from_numpy(X_cat[start:end]).to(device)
            logits = model(xb_num, xb_cat)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def main():
    # 대회 서빙 환경 표준 경로 정의
    DATA_DIR = "./data"
    MODEL_PATH = "./model/final_retained_model.pkl"
    OUTPUT_DIR = "./output"

    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ 필수 입력 파일이 없습니다: {test_path}")

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    df_sub = pd.read_csv(sample_sub_path, encoding="utf-8-sig")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 제출 구조 내 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)

    if not isinstance(model_dict, dict) or model_dict.get("model_kind") != "mlp_ensemble":
        raise TypeError(f"❌ {MODEL_PATH}가 예상한 MLP 앙상블 딕셔너리 형식이 아닙니다.")

    tr_final = map_top_bottom(df_test)

    # league_success_mean과 season_end_lookup은 학습 시점(train.csv 전체)에 이미 계산해
    # 모델 딕셔너리에 저장해 두었으므로, 추론 시점에는 train.csv를 다시 읽을 필요가 없고
    # (대회 규칙상 평가 데이터의 다른 행으로 피처를 만드는 것도 금지되어 있어 test.csv
    # 자신으로부터는 애초에 계산할 수도 없습니다), trackman_history.csv도 더 이상 쓰지 않습니다.
    tr_final = apply_season_progression_features(tr_final, model_dict["season_end_lookup"])
    tr_final = add_engineered_features(tr_final, model_dict["league_success_mean"])
    tr_final = add_missing_indicators(tr_final, model_dict["na_indicator_source_cols"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_num, X_cat = transform_features(
        tr_final,
        model_dict["categorical_features"],
        model_dict["numeric_features"],
        model_dict["cat_encoders"],
        model_dict["numeric_medians"],
        model_dict["scaler"],
    )

    # 시드별로 독립 학습된 모델들을 하나씩 재구성해 예측 확률을 평균합니다(시드 앙상블).
    probs_sum = np.zeros(len(X_num), dtype=np.float64)
    for state_dict in model_dict["state_dicts"]:
        model = TabularMLP(
            model_dict["num_numeric"],
            model_dict["cat_cardinalities"],
            model_dict["embed_dims"],
            model_dict["hidden_dims"],
            model_dict["dropout"],
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        probs_sum += predict_proba_mlp(model, X_num, X_cat, device)
        del model
    preds = probs_sum / len(model_dict["state_dicts"])

    df_sub[TARGET_COL] = preds

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 추론 및 제출용 파일 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
