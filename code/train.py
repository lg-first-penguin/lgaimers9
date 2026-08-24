# code/train.py
import sys
import os
import math
import random

# CUDA에서 결정적 연산(예: cuBLAS 행렬곱)을 쓰려면 CUDA 컨텍스트가 생성되기 전에 이
# 환경변수가 설정돼 있어야 합니다. torch를 import하기 전에 미리 지정합니다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

# 순수 MLP 단독 모델(임베딩 + 수치 피처 타워)로 전환. CatBoost/TabPFN 등 다른 모델과
# 블렌딩하지 않는다는 뜻의 "단독"이며, 같은 MLP 아키텍처를 시드만 바꿔 여러 개 학습해
# 예측 확률을 평균하는 시드 앙상블은 허용됩니다(MLP_SEEDS 참고) — 이 모델 하나(계열)로
# 대회 리더보드 1,000점(BSS 환산 점수) 돌파가 목표입니다.
SEED = 42
# 서로 다른 시드로 학습한 MLP 20개의 예측 확률을 평균합니다. 시드별로 초기화·미니배치 셔플
# 순서가 달라 각 모델이 조금씩 다른 국소해에 수렴하는데, 평균을 내면 그 분산이 상쇄되어
# 단일 모델보다 안정적으로 높은 val BSS를 보였습니다(실측 근거는 EXPERIMENTS.md 참고).
# 7->15 확장은 EXPERIMENTS.md "후속 검증 Part C"에서 실측(786.81->800.17, +13.36) 후 채택.
# 그 실측 당시 추가한 8개 신규 시드 [1,2,3,4,5,6,7,8] 중 7이 기존 목록과 우연히 중복돼
# 실질 14개뿐이었던 문제를, 마지막 값만 9로 바꿔 15개 전부 서로 다른 시드가 되도록 수정.
# 15->20 확장은 2026-08-24 스크래치패드 측정(EXPERIMENTS.md "시드 수 확장 15 -> 30" 절)에서
# 795.71->798.51(+2.80) 채택. 20개를 넘겨 25/30까지 늘려봐도 798~803 구간에서 오르내릴 뿐
# 뚜렷한 추가 상승이 없어(정체/plateau 시작점), 20개가 학습 비용 대비 합리적인 지점으로 선택
# — 시드를 더 늘리기 전에 그 절의 plateau 데이터부터 확인할 것.
MLP_SEEDS = [42, 123, 7, 2024, 99, 555, 31337, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]

# 원래는 game_month/balls_before 같은 이산 컬럼들도 전부 임베딩으로 넣어봤지만(트리가 split을
# 잘 찾는 것과 같은 이유를 기대), season==2024 홀드아웃 기준 val BSS가 오히려 0.0055 -> 0.0038
# 로 떨어져서 되돌렸습니다. `season`은 그 중에서도 특히 위험한데, 검증 split이 season==2024를
# 통째로 홀드아웃하는 구조라(코드/모델의 "Train/eval split convention" 참고) season을 범주형
# 으로 넣으면 검증 시점에 2024는 100% 학습 때 못 본 값이 되어 전부 unknown 슬롯(=미학습 랜덤
# 벡터)으로 빠지고, 실측상 val BSS가 0.0017까지 폭락했습니다. season/game_month/dayofweek/
# inning/balls_before/strikes_before/outs_before는 모두 수치형(스케일된 정수)으로 남겨두세요.
# pitcher_id/batter_id/team_id는 asof_* 집계 통계로만 쓰지 않고 임베딩으로 직접 학습시키면
# 트리 계열이 잡기 어려운 선수별 잠재 특성을 MLP가 스스로 학습할 여지가 생겨 유지합니다.
EMBED_CATEGORICAL_FEATURES = [
    'top_bottom', 'game_type', 'base_state',
    'pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
]
EMBED_DIMS = {
    'top_bottom': 2,
    'game_type': 2,
    'base_state': 4,
    'pitcher_id': 8,
    'batter_id': 8,
    'pitcher_team_id': 4,
    'batter_team_id': 4,
}

# 아래 학습 하이퍼파라미터는 EXPERIMENTS.md에 기록된 결정적(재현 가능한) 스윕 결과 기준
# 채택값입니다. hidden=[128,64] + dropout=0.2 조합이 season==2024 홀드아웃 val BSS 기준
# 반복 재현되는 최적점이었습니다(0.00702, 환산 점수 701.71) — hidden을 256/512로 넓히거나
# dropout을 0.15~0.3 범위에서 바꿔봐도 이보다 낫지 않았습니다. 2026-08-24 재확인: F1필터+
# season-progression+12개 파생피처+15-seed 앙상블이 전부 반영된 현재 파이프라인 위에서
# 3-seed[42,123,7]로 hidden([256,128]/[64,32]/[128,64,32])×dropout(0.1/0.15/0.3) 7종을
# 다시 스윕해도 [128,64]+0.2(782.19)가 여전히 최고였고 나머지는 전부 −27~−41점 — 초기
# 스윕이 stale하지 않음을 재확인. EXPERIMENTS.md "용량/드롭아웃 재스윕" 절 참고.
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.2
BATCH_SIZE = 16384
MAX_EPOCHS = 40
PATIENCE = 15
LR = 2e-3
WEIGHT_DECAY = 1e-6
# 이 문제의 수치형 피처(asof_* 비율, win_expectancy 등)는 분포가 많이 치우쳐 있어,
# StandardScaler(z-score)보다 QuantileTransformer(정규분포로 랭크 변환)가 실측상 더 안정적인
# val BSS를 보였습니다 — MLP는 트리와 달리 입력 스케일/분포 형태에 민감하기 때문으로 추정.
QUANTILE_N_QUANTILES = 2000
QUANTILE_SUBSAMPLE = 200000


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # nn.Embedding의 CUDA backward는 기본적으로 atomic scatter-add로 그래디언트를 누적하는데,
    # 부동소수점 덧셈은 결합법칙이 성립하지 않아 스레드 완료 순서에 따라 결과가 미세하게
    # 달라지는(공식 문서화된) 비결정성이 있습니다. 실측으로도 동일 하이퍼파라미터를 재실행했을 때
    # val BSS가 크게 흔들리는 걸 확인했습니다 — 아래 설정으로 결정적 구현이 있는 연산은 전부
    # 결정적 경로를 쓰도록 강제합니다(warn_only=True라 결정적 구현이 없는 연산은 에러 대신
    # 경고 후 기존 방식으로 폴백).
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def map_top_bottom(df):
    """top_bottom의 T/B 문자열을 0/1 정수로 바꿉니다."""
    df = df.copy()
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    return df


def apply_f1_filter(df):
    """`game_type=='F'`(퓨처스/2군) 데이터는 2022년까지는 R(1군)보다 유리했다가 2023년부터
    불리하게 역전되는 오염이 있어(원인 불명, 정착된 패턴으로 보임), 그 구간만 학습 데이터에서
    제거합니다. `game_type`이 피처 중요도 최상위권이라 이 오염이 학습에 큰 영향을 줍니다.
    검증/추론 데이터에는 절대 적용하지 마세요 — 실제 서빙 분포를 그대로 반영해야 합니다.
    """
    return df[~((df['game_type'] == 'F') & (df['season'] <= 2022))].reset_index(drop=True)


SEASON_PROGRESSION_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def build_season_end_lookup(df):
    """선수별/시즌별 "그 시즌 마지막 행"의 커리어 누적치를 다음 시즌(season+1)의 시작
    기준값으로 저장해 둡니다. `df`는 항상 라벨(control_success)이 있는 학습용 데이터여야
    하며(테스트 데이터로 호출 금지 — 대회 규칙상 평가 데이터의 다른 행을 이용한 피처 생성에
    해당할 수 있음), 이 lookup은 `apply_season_progression_features`로 어떤 데이터프레임에도
    안전하게(자기 자신을 참조하지 않고) 적용할 수 있습니다.
    """
    tables = []
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
        idx = df.groupby([id_col, 'season'])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, 'season', n_col, rate_col, 'control_success']].copy()
        season_end.columns = ['id', 'season', 'end_n', 'end_rate', 'end_success']
        season_end['season'] = season_end['season'] + 1  # 다음 시즌의 시작 기준값으로 이동
        season_end.insert(0, 'role', role)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_season_progression_features(df, lookup):
    """`asof_*` 커리어 누적 성공률에서 "직전 시즌까지의 누적분"을 빼서, 그 시즌에 들어와서만
    쌓인 성공률(=최근 컨디션)을 분리해 냅니다. df 자신의 다른 행에 의존하지 않으므로
    test.csv에도 안전하게 적용할 수 있습니다.
    """
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
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다."""
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
    """결측이었다는 사실 자체가 신호(cold-start)이므로, 중앙값 대체 전에 결측 플래그를 남깁니다."""
    df = df.copy()
    for col in na_cols:
        df[f'{col}_isna'] = df[col].isna().astype(np.float32)
    return df


class TabularMLP(nn.Module):
    """저카디널리티/ID 범주형은 임베딩으로, 나머지는 표준화된 수치 타워로 합쳐 학습하는 단일 MLP."""

    def __init__(self, num_numeric, cat_cardinalities, embed_dims, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim)  # +1: 학습 시 못 본 범주(예: 2025시즌 신인)를 위한 unknown 슬롯
            for card, dim in zip(cat_cardinalities, embed_dims)
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


def fit_preprocessing(fit_df, categorical_features, numeric_features):
    cat_encoders = {}
    cat_cardinalities = []
    for col in categorical_features:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(fit_df[[col]].astype(str))
        cat_encoders[col] = enc
        cat_cardinalities.append(len(enc.categories_[0]))

    numeric_medians = fit_df[numeric_features].median()
    scaler = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(QUANTILE_N_QUANTILES, len(fit_df)),
        subsample=QUANTILE_SUBSAMPLE,
        random_state=SEED,
    )
    scaler.fit(fit_df[numeric_features].fillna(numeric_medians))

    return cat_encoders, cat_cardinalities, numeric_medians, scaler


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


def bss_score(preds, y):
    preds = np.clip(preds, 0.0, 1.0)
    y = np.asarray(y, dtype=np.float64)
    brier = ((preds - y) ** 2).mean()
    r = y.mean()
    baseline_brier = r * (1 - r)
    bss = 1.0 - (brier / baseline_brier)
    score = max(0.0, 100000.0 * bss)
    return bss, score


def predict_proba_mlp(model, X_num, X_cat, device, batch_size=20000):
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


def train_mlp(X_num_train, X_cat_train, y_train,
              X_num_val=None, X_cat_val=None, y_val=None,
              cat_cardinalities=None, embed_dims=None, device="cpu",
              max_epochs=MAX_EPOCHS, patience=PATIENCE, verbose=True,
              hidden_dims=None, dropout=None, seed=SEED):
    """X_num_val/y_val이 주어지면 검증 BSS 기준 조기종료(+베스트 에폭 상태 복원).
    None이면(=최종 제출용 전체 데이터 재학습) 검증 없이 max_epochs 그대로 고정 학습합니다.

    hidden_dims/dropout을 명시적 인자로 받는 이유: `TabularMLP.__init__`의
    `hidden_dims=HIDDEN_DIMS`같은 기본값은 함수 "정의 시점"에 한 번 바인딩되므로, 모듈을
    import한 뒤 `code.train.HIDDEN_DIMS = [...]`로 나중에 바꿔도 반영되지 않습니다
    (LR/WEIGHT_DECAY/BATCH_SIZE는 함수 본문에서 매번 전역 이름을 그대로 참조하므로 이 문제가
    없음). None이면 모듈 전역 HIDDEN_DIMS/DROPOUT을 그때그때 조회해 하위호환을 유지합니다.

    seed: 시드 앙상블(`train_mlp_ensemble`)의 각 멤버가 서로 다른 초기화/미니배치 셔플
    순서를 갖도록 명시적으로 받습니다.
    """
    set_seed(seed)
    hidden_dims = HIDDEN_DIMS if hidden_dims is None else hidden_dims
    dropout = DROPOUT if dropout is None else dropout
    model = TabularMLP(X_num_train.shape[1], cat_cardinalities, embed_dims, hidden_dims, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    has_val = X_num_val is not None

    train_ds = TensorDataset(
        torch.from_numpy(X_num_train),
        torch.from_numpy(X_cat_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 이 문제(신호가 매우 약한 이진분류)에서 ReduceLROnPlateau는 val BSS가 에폭마다 크게
    # 흔들려 반응이 늦고, 초반 몇 스텝의 큰 gradient step에 학습이 통째로 휘둘리는 걸
    # 실측으로 확인했습니다(동일 설정 재실행에도 best_epoch/BSS가 크게 튀는 현상).
    # 스텝 단위 선형 워밍업 + 코사인 감쇠 + gradient clipping으로 초반 불안정을 줄입니다.
    steps_per_epoch = max(1, -(-len(train_ds) // BATCH_SIZE))
    total_steps = max(1, max_epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_bss = -float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(yb)
        train_loss = total_loss / len(train_ds)

        if not has_val:
            if verbose:
                print(f"[Epoch {epoch}/{max_epochs}] train_loss={train_loss:.5f} (검증 없음 - 고정 에폭 학습)")
            continue

        val_preds = predict_proba_mlp(model, X_num_val, X_cat_val, device)
        val_bss, val_score = bss_score(val_preds, y_val)

        if verbose:
            print(f"[Epoch {epoch}] train_loss={train_loss:.5f} val_bss={val_bss:.5f} val_score={val_score:.2f}")

        if val_bss > best_val_bss:
            best_val_bss = val_bss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EarlyStop] {patience}에폭 연속 개선 없어 조기 종료 "
                      f"(best epoch={best_epoch}, best val_bss={best_val_bss:.5f})")
                break

    if has_val:
        model.load_state_dict(best_state)
        return model, best_epoch, best_val_bss
    else:
        return model, max_epochs, None


def build_ensemble_model(cat_cardinalities, embed_dims, num_numeric, hidden_dims, dropout, state_dict, device):
    model = TabularMLP(num_numeric, cat_cardinalities, embed_dims, hidden_dims, dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_proba_mlp_ensemble(model_dict, X_num, X_cat, device, batch_size=20000):
    """model_dict["state_dicts"] 각각으로 모델을 새로 구성해 예측한 뒤 확률을 평균합니다."""
    probs_sum = np.zeros(len(X_num), dtype=np.float64)
    for state_dict in model_dict["state_dicts"]:
        model = build_ensemble_model(
            model_dict["cat_cardinalities"], model_dict["embed_dims"], model_dict["num_numeric"],
            model_dict["hidden_dims"], model_dict["dropout"], state_dict, device,
        )
        probs_sum += predict_proba_mlp(model, X_num, X_cat, device, batch_size=batch_size)
        del model
    return probs_sum / len(model_dict["state_dicts"])


def train_mlp_ensemble(X_num_train, X_cat_train, y_train,
                        X_num_val=None, X_cat_val=None, y_val=None,
                        cat_cardinalities=None, embed_dims=None, device="cpu",
                        seeds=MLP_SEEDS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                        fixed_epochs_per_seed=None, verbose=True):
    """`seeds`개의 독립적으로 학습된 MLP를 만들고, 검증셋이 있으면 그 예측 확률 평균으로
    앙상블 val BSS까지 계산합니다. `fixed_epochs_per_seed`가 주어지면(=최종 제출용 전체 데이터
    재학습) 검증 없이 시드별로 그 값만큼 고정 학습합니다(seed 순서와 1:1 대응하는 리스트).
    """
    state_dicts = []
    best_epochs = []
    has_val = X_num_val is not None
    val_probs_sum = np.zeros(len(y_val), dtype=np.float64) if has_val else None

    for i, seed in enumerate(seeds):
        seed_max_epochs = fixed_epochs_per_seed[i] if fixed_epochs_per_seed is not None else max_epochs
        model, best_epoch, best_val_bss = train_mlp(
            X_num_train, X_cat_train, y_train,
            X_num_val if has_val else None, X_cat_val if has_val else None, y_val if has_val else None,
            cat_cardinalities=cat_cardinalities, embed_dims=embed_dims, device=device,
            max_epochs=seed_max_epochs, patience=patience, verbose=False, seed=seed,
        )
        state_dicts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        best_epochs.append(best_epoch)

        if has_val:
            val_probs_sum += predict_proba_mlp(model, X_num_val, X_cat_val, device)
            running_bss, running_score = bss_score(val_probs_sum / (i + 1), y_val)
            if verbose:
                print(f"[Seed {i + 1}/{len(seeds)}={seed}] solo_best_epoch={best_epoch} "
                      f"solo_val_bss={best_val_bss:.5f} | 누적 앙상블 val_bss={running_bss:.5f} "
                      f"(점수 {running_score:.2f})")
        elif verbose:
            print(f"[Seed {i + 1}/{len(seeds)}={seed}] {seed_max_epochs}에폭 고정 학습 완료 (검증 없음)")
        del model

    if has_val:
        ensemble_bss, _ = bss_score(val_probs_sum / len(seeds), y_val)
        return state_dicts, best_epochs, ensemble_bss
    else:
        return state_dicts, best_epochs, None


def build_feature_lists(train_df, target_col, train_mask):
    """범주형(임베딩)/수치형 피처 목록과 결측 플래그 대상을 확정합니다."""
    categorical_features = [c for c in EMBED_CATEGORICAL_FEATURES if c in train_df.columns]
    drop_cols = ['row_id', target_col]
    all_features = [c for c in train_df.columns if c not in drop_cols]
    numeric_features = [c for c in all_features if c not in categorical_features]

    na_cols = [c for c in numeric_features if train_df.loc[train_mask, c].isna().any()]
    return categorical_features, numeric_features, na_cols


def main():
    DATA_DIR = "./open/data"
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df = map_top_bottom(df)
    df = apply_f1_filter(df)

    target_col = 'control_success'
    train_df = df.dropna(subset=[target_col]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024

    league_success_mean = train_df.loc[train_mask, target_col].mean()

    season_end_lookup = build_season_end_lookup(train_df.loc[train_mask])
    train_df = apply_season_progression_features(train_df, season_end_lookup)

    train_df = add_engineered_features(train_df, league_success_mean)

    categorical_features, numeric_features, na_cols = build_feature_lists(train_df, target_col, train_mask)
    train_df = add_missing_indicators(train_df, na_cols)
    numeric_features = numeric_features + [f'{c}_isna' for c in na_cols]

    tr_split = train_df.loc[train_mask].reset_index(drop=True)
    val_split = train_df.loc[train_df['season'] == 2024].reset_index(drop=True)

    cat_encoders, cat_cardinalities, numeric_medians, scaler = fit_preprocessing(
        tr_split, categorical_features, numeric_features)

    X_num_train, X_cat_train = transform_features(
        tr_split, categorical_features, numeric_features, cat_encoders, numeric_medians, scaler)
    y_train = tr_split[target_col].values.astype(np.float32)

    X_num_val, X_cat_val = transform_features(
        val_split, categorical_features, numeric_features, cat_encoders, numeric_medians, scaler)
    y_val = val_split[target_col].values.astype(np.float32)

    device = get_device()
    embed_dims = [EMBED_DIMS[c] for c in categorical_features]
    print(f"[Train] device={device} | train={len(tr_split)} val={len(val_split)} | "
          f"수치형 {len(numeric_features)}개 + 임베딩 {len(categorical_features)}개(합계 dim={sum(embed_dims)})")

    state_dicts, best_epochs, ensemble_val_bss = train_mlp_ensemble(
        X_num_train, X_cat_train, y_train,
        X_num_val, X_cat_val, y_val,
        cat_cardinalities=cat_cardinalities, embed_dims=embed_dims, device=device,
    )

    model_dict = {
        "model_kind": "mlp_ensemble",
        "state_dicts": state_dicts,
        "seeds": MLP_SEEDS,
        "num_numeric": len(numeric_features),
        "cat_cardinalities": cat_cardinalities,
        "embed_dims": embed_dims,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "na_indicator_source_cols": na_cols,
        "cat_encoders": cat_encoders,
        "numeric_medians": numeric_medians,
        "scaler": scaler,
        "league_success_mean": league_success_mean,
        "best_epochs": best_epochs,
        "val_bss": ensemble_val_bss,
    }

    os.makedirs("./open/temp", exist_ok=True)
    with open("./open/temp/latest_model.pkl", 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"✅ MLP {len(MLP_SEEDS)}-seed 앙상블 저장 완료 -> ./open/temp/latest_model.pkl "
          f"(best_epochs={best_epochs}, ensemble val_bss={ensemble_val_bss:.5f}, "
          f"점수={max(0.0, 100000.0 * ensemble_val_bss):.2f})")


if __name__ == "__main__":
    main()
