# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A private team pipeline for the LG Aimers / DACON competition (야구 투구 제어 성공 예측, `control_success` binary target, evaluated via Brier Skill Score). Built for Ubuntu 24.04 (WSL2), Python 3.11.

**Current goal: a single standalone MLP (PyTorch, embeddings + numeric tower — no ensembling, no TabPFN, no CatBoost) that clears a leaderboard score of 1,000+ (BSS-converted) on its own.** This replaced an earlier TabPFN v2 bagging-ensemble approach — see git history / `EXPERIMENTS.md` if you need the old CatBoost/TabPFN context. `torch`, `pandas`, `numpy`, and `scikit-learn` are all preinstalled on the eval server at pinned versions, so the MLP needs no extra weight files or offline-checkpoint handling the way TabPFN did — `submit/requirements.txt` is effectively empty and `submit/model/` holds nothing but our own trained `state_dict`.

**Status as of the latest seed-count expansion**: reproducibly measured `season == 2024` holdout score is **798.51** (BSS 0.00799, 20-seed ensemble, promoted into `open/reference/best_model.pkl` and `submit/model/final_retained_model.pkl` on 2026-08-24 — see the seed-count-expansion paragraph further down for how this superseded the prior 795.71/15-seed reference) — clears the Phase 2 completion cutoff (549.51) but not yet the 1,000 target. A round of six new hand-engineered candidate features (platoon same-hand flag, batter-relative-success, RISP pressure, pitcher command-score composite, Bayesian-shrunk pitcher rate, pitcher-team win expectancy) was screened the same day and **none were adopted** — see `EXPERIMENTS.md`'s "신규 피처 후보 스크리닝" section for the full negative result and why (they're linear/multiplicative combinations of columns already in the numeric tower, so the MLP can already approximate them). A follow-up round the same day tried three MLP-architecture-specific candidates instead (cyclical sin/cos encoding for month/dayofweek, a `pitcher_id`×`batter_id` matchup embedding, and an explicit bilinear/FM-style dot-product interaction layer between paired entity embeddings) — all three also **failed at 3-seed screening** (−39 to −58 points each, worse combined), so none were adopted either; see `EXPERIMENTS.md`'s "신규 피처 후보 스크리닝 3차" section, including a likely cause for each (the matchup embedding in particular hits the same "high-unknown-fraction at eval time" failure mode already documented below for a `season` embedding). Between the two rounds, 12 feature candidates have now been screened and rejected — treat further hand-engineered/architectural feature additions as a low-probability lever until a different axis (capacity/regularization, seed count, cross-regime stability) is tried instead. A fourth round (2026-08-24, run in parallel with a sibling session sharing the same GPU) tried two more: `season_interaction` (`pitcher_season_rate_gap × batter_season_rate_gap`) and `team_matchup_embed` (a low-cardinality `pitcher_team_id`×`batter_team_id` embedding, cardinality 92, deliberately much smaller than the failed 3rd-round `pitcher_id`×`batter_id` matchup embedding to sidestep its high-unseen-fraction failure mode). Both looked positive at 3-seed (+2.14 noise-level, +10.12 borderline) but **`team_matchup_embed` reversed to −13.73 at 7-seed** — confirmed on the same 7-seed baseline (786.81) already documented below, so the reversal is real, not a baseline drift artifact. A sibling session independently saw the same 3-seed→7-seed reversal pattern for its own version of `season_interaction` in a different codebase at the same time. **Methodology update: 3-seed is no longer sufficient to adopt a feature for this problem — confirm at 7-seed before adopting, and treat 3-seed as a pre-filter only.** A fifth round the same day (2026-08-24) targeted four raw columns that had never been touched by any prior engineered feature (pitch-mix rates via an entropy transform, `asof_pitcher_reverse_rate`, `asof_batter_middle_rate`, and `inning` for late-game leverage) — all four came back negative at 3-seed (two clearly so, −26/−24; two within the noise band, −4/−5) with no positive or borderline result, so all were rejected outright without needing 7-seed confirmation (per the updated rule, only positive/borderline results need that step). 18 feature candidates total now screened and rejected across five rounds; nearly all 47 raw input columns have now been used as material for at least one attempt, so the "new combination of raw columns" axis looks close to exhausted — see `EXPERIMENTS.md`'s "신규 피처 후보 스크리닝 4차"/"5차" sections for the full detail.

With the feature axis exhausted, the user redirected effort to the other two axes: capacity/regularization and seed count (2026-08-24). A hidden_dims/dropout re-sweep (3-seed, GPU, on top of the full current pipeline — the original sweep predates season-progression/F1-filter/12-engineered-features/seed-ensembling) tried `[256,128]`/`[64,32]`/`[128,64,32]` and dropout `0.1`/`0.15`/`0.3` against the current `[128,64]+0.2` — **all 7 alternatives underperformed the current setting by −28 to −41 points**, reconfirming it's still optimal, not stale. No code change. Seed count was then pushed 15→30 (reusing the promoted 15 state_dicts via pure inference + training 15 new distinct seeds), reaching **800.86 at 30 seeds (+5.15 vs. the 15-seed reference)** — but the gain **plateaus around 20 seeds**: scores oscillate in a 798–803 band from 20 through 30 with no further trend (23 seeds hit a one-run-noise peak of 803.41, not a true optimum). Because this is pure ensemble variance reduction (not new information), it's lower-risk than the feature candidates that kept reversing. The user chose 20 seeds (the plateau's cost-effective entry point) and it was formally adopted the same day: `MLP_SEEDS` in `code/train.py` extended to the original 15 plus `[10, 11, 12, 13, 14]`, then the full `dopip.py` pipeline was run for real (not a scratchpad shortcut) — Step 1 reproduced **798.51** (BSS 0.00799) exactly matching the scratchpad measurement, Step 2 promoted it as `NEW_BEST` over the 15-seed reference (795.71), and Step 3's full-data retrain updated `submit/model/final_retained_model.pkl`. **Current production reference: 798.51**, old 15-seed models backed up under `open/former_model/former_*_v7.pkl`. See `EXPERIMENTS.md`'s "용량/드롭아웃 재스윕" and "시드 수 확장 15 → 30" sections for the full detail. See "MLP design" and "Data engineering" under Architecture and `EXPERIMENTS.md`'s "MLP 전환" section for the full tuning/score history, a CUDA non-determinism bug and fix, a hyperparameter-sweep footgun that invalidated several early sweep results, and what was ported from a sibling project (F1 filter, dropping trackman, season-progression features) — read those before starting a new round of tuning so you don't repeat a mistake or re-litigate an already-settled question.

Competition data link: https://dacon.io/competitions/official/236743/data

## Competition's rule

[배경] 
최근 스포츠 현장에서는 선수의 경기력 분석과 전략 수립에 데이터 기반 의사결정의 중요성이 빠르게 커지고 있습니다. 특히 야구에서 투구의 제구력은 실점 억제, 볼카운트 운영, 타자 대응 전략에 직접적인 영향을 주는 핵심 요소입니다.

기존에는 투수의 제구력을 평균자책점, 볼넷 수, 스트라이크 비율 등 경기 후 집계 지표로 평가하는 경우가 많았습니다. 그러나 실제 경기에서는 매 투구 직전의 볼카운트, 주자 상황, 타자·투수 특성, 과거 투구 이력 등 다양한 정보가 복합적으로 작용합니다.

따라서 단순한 결과 통계가 아니라, 투구가 이루어지기 전까지 확인 가능한 정보만을 바탕으로 해당 투구가 원하는 제구 범위에 성공할 가능성을 예측하는 AI 모델링이 중요해지고 있습니다.

이번 해커톤은 이러한 문제의식을 바탕으로, 야구 경기 데이터를 중심으로 투구 직전의 상황과 과거 이력을 활용하여 제구 성공 확률을 예측하는 실전형 AI 문제를 수행하게 됩니다. 

트랙맨(Trackman) 데이터는 2019~2024년의 과거 투구 특성을 참고할 수 있는 보조 데이터로 제공됩니다.



### [주제]
투구 단위의 제구 성공 확률 예측 AI 모델 개발



### [설명]
이번 온라인 해커톤(Phase 2)은 투구 직전까지 확인 가능한 경기 상황, 선수 정보, 주자 상황, 과거 이력을 바탕으로 각 투구의 제구 성공 확률을 예측하는 AI 모델을 개발하는 것을 목표로 합니다.

참가자는 제공된 데이터를 활용하여 테스트 데이터의 각 투구에 대해 control_success 의 확률값을 예측할 수 있어야 합니다. 

학습 데이터의 control_success는 제구 성공을 1, 제구 실패를 0으로 정의한 학습용 Target이며, 예측해야하는 control_success는 제구 성공 가능성을 나타내는 확률입니다.

또한 참가자는 투구 이전 시점에서 활용 가능한 정보만을 바탕으로 예측 모델을 설계할 수 있어야합니다.



온라인 해커톤(Phase 2)에서의 제구 성공은 각 투구의 공 위치를 기준으로 정의합니다.

아래의 3가지 경우는 제구 실패에 해당하며, 그 외 유효한 투구는 제구 성공에 해당합니다.

1) 스트라이크존 가운데 부근으로 들어간 공

2) 스트라이크존에서 크게 벗어난 공

3) 포수의 요구 방향과 반대로 들어간 공



온라인 해커톤(Phase2)에서 교육생들의 문제 해결 능력을 검증하여 오프라인 해커톤(Phase3)에 진출자(약 100명)를 선발하기 위한 과정입니다.

오프라인 해커톤(Phase 3)은 1박 2일간 오프라인으로 진행되며, 세부 과제는 추후 안내될 예정이며, 온라인 해커톤(Phase 2)과 동일하게 야구 데이터를 기반으로 한 AI 문제로 진행될 예정입니다.



[코드 제출 대회]

본 대회는 submit.zip 업로드 방식의 코드 제출 형식 대회로 진행됩니다.

전체 추론 실행 시간 ≤ 10분 (245,789개 샘플 추론)
패키지(라이브러리) 설치 시간 ≤ 10분
제출 파일 용량 ≤ 10GB (*압축해제 후 최대 32GB)
오프라인 환경 실행 (패키지 설치 외 인터넷 연결 불가능)
6 vCPU, 28GB RAM, L4 GPU 22.4GiB VRAM 환경에서 실행
자세한 사항은 평가 탭과 코드 제출 가이드를 반드시 참고하여 진행하시길 바랍니다.

1. 리더 보드

평가 산식 : Brier Skill Score
본 대회는 각 투구의 control_success = 1일 확률을 예측하는 확률 예측 과제입니다. 추론 확률이 실제 정답에 가까울수록 높은 점수를 받습니다.
Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))
Brier Score = mean((p_i - y_i)^2)
r = mean(y_i)
평균 제구율 Brier Score = r × (1 - r)
p_i : i번째 샘플의 제구 성공 예측 확률
y_i : i번째 샘플의 실제 정답 (0, 1)
r : 전체 평가 데이터의 평균 제구 성공률 (비공개 수치)


Public Score : 전체 테스트 데이터 100%
Private Score : 대회 종료 시점의 Public Score


2. 평가 방식

LG Aimers 수료 조건
Phase1을 이수하고 Phase2의 Public Score (LB: 549.51) 이상
기준 점수는 운영진이 제공한 베이스라인 추론 코드를 운영진 평가 환경에서 실행했을 때의 점수를 기준으로 측정
1차 평가 : 리더보드 Private Score 100%
동점자의 경우, 기존 리더보드 순위 산정 방식을 따름 [링크]의 '리더보드 점수' 부분을 참고
2차 평가 : 오프라인 해커톤(Phase3) 진출을 희망하는 팀은 코드 제출 후 코드 검증
Private 리더보드 상위팀(약 100명)은 코드 및 PPT 필수 제출 대상
코드 및 PPT 제출과 검증를 모두 통과한 Private 리더보드 상위팀(약 100명)이 오프라인 해커톤(Phase3) 진출


3. 코드 제출 대회 가이드

본 대회는 submit.zip 파일을 제출하는 방식의 '코드 제출 대회'로 진행됩니다. (기본 가이드 문서)

참가자는 아래와 같은 구조로 submit.zip을 구성하여 제출해야 합니다.

아래의 구조와 동일하고 디렉토리 명과 파일 명을 모두 일치 시켜야합니다.

제출 파일 구조 (submit.zip)

submit.zip
├── model/        # 모델 가중치 파일을 저장하는 디렉토리
│      └── (예: model.pt 등)
├── script.py       # 실제 추론이 수행되는 실행 코드
└── requirements.txt   # 필요한 패키지 및 버전 명시
script.py는 submit.zip을 제출 시 평가 서버에서 자동으로 실행됩니다.
requirements.txt는 pip install -r requirements.txt 명령어로 설치 가능한 형태여야 하며, 추론 시 필요한 모든 패키지를 포함해야 합니다.
submit.zip 내 구조는 반드시 일치해야하며, 추가 최상위 폴더가 zip 구조 내 존재하는 경우 등 구조가 불일치하는 경우 설치 오류가 발생합니다.


평가 서버에서 추가되는 항목

제출 시, 평가 서버에서 참가자가 제출한 submit.zip 파일에는 아래 항목이 자동으로 추가됩니다.

submit.zip
├── model/        # 참가자 구성
├── script.py       # 참가자 구성
├── requirements.txt   # 참가자 구성
├── data/         # 평가에 사용될 테스트 데이터 (디렉토리 자동 생성)
└── output/submission.csv        # 참가자 추론 결과가 저장되는 경로 (디렉토리 자동 생성)
data/ 디렉토리는 실제 평가 데이터를 포함한 경진대회 데이터가 포함되며, 읽기전용으로 쓰기 및 수정이 불가능한 디렉토리입니다.
output/ 디렉토리는 참가자의 script.py 실행 결과로 생성된 예측 결과 파일이 저장되는 디렉토리이며, 해당 디렉토리 내에 반드시 submission.csv으로 생성될 수 있어야합니다.


 제출 파일 용량 제한

제출 파일(zip) 용량 제한: 최대 10GB 이내 (*압축해제 후 최대 32GB)


⏱ 실행 시간 제한
패키지 설치 시간: 최대 10분 이내 (시간 초과 시 설치 오류)
추론 코드 실행 시간: 최대 10분 이내 (시간 초과 시 제출 오류)

 평가 서버 사양
OS : Ubuntu 22.04.5 LTS
GPU : NVIDIA L4 (VRAM 22.4GiB)
CPU: 6 vCPU
CPU RAM: 28GB
Python : 3.11.15
인터넷 접속:  비활성화 (패키지 설치 외 외부 서버 연결 및 다운로드 불가)
CUDA : 12.8


 평가 서버 기본 설치 패키지(라이브러리) 목록

아래의 패키지(라이브러리)는 평가 서버에 기본적으로 설치되어 있으며, 버전이 명시된 아래의 패키지(라이브러리)에 한해서는 다른 버전을 사용할 때 설치 에러가 발생할 수 있으므로 가급적 평가 서버에 기본 설치된 패키지(라이브러리)를 활용하고 제출하는 requirements.txt에는 포함하지 않는 것을 권장드립니다. 
라이브러리 설치 에러가 발생하면 설치 오류에 해당하며, 일일 제출 횟수에는 반영되지 않습니다.


1) 주요 설치 패키지(라이브러리)

torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1


2) 주요 설치 시스템 패키지﻿

git
build-essential
python3.11
python3.11-dev
python3.11-venv
python3-pip
libffi-dev
libblas3
liblapack3
libomp-dev
tzdata
unzip
p7zip-full
gfortran
libatlas-base-dev
default-jre-headless
cmake
pkg-config
ninja-build
libgl1
libglib2.0-0


유의사항
제출 시 발생하는 오류의 종류는 두 가지로 정의되며, 일일 제출 횟수 반영에 대한 기준이 다르므로 반드시 숙지하여 진행해야 합니다.
1) 설치 오류 : 제출하는 submit.zip 내부 구조가 불일치한 경우, 패키지 설치 오류 -> 일일 제출 횟수 반영되지 않음
2) 제출 오류 : script.py 코드 실행 후 발생하는 모든 오류 -> 일일 제출 횟수 반영됨
script.py 내에서 open/ 디렉토리의 데이터를 로드하고, output/ 디렉토리에 예측 결과를 반드시 submission.csv의 파일명으로 저장되어야 합니다.
평가 서버 환경은 인터넷 접속이 불가능하므로, 패키지 설치 이후 외부 다운로드가 필요한 코드나 모델은 작동하지 않습니다.

2. 대회 규칙

1) 사전학습모델 및 가중치 사용 가능 범위

공식적으로 누구에게나 가중치가 공개되었으며, 최소한 비상업적 이용이 허용된 라이선스(MIT, Apache 2.0 등) 하에 배포된 모델 및 가중치만 사용 가능합니다.
해당 조건을 만족하지 않는 모델 및 가중치는 사용할 수 없습니다.
2) 외부 API 사용 제한

원격 서버 기반의 API 형태로만 접근 가능한 모델(OpenAI API, Gemini API 등)은 사용이 불가합니다.
모든 작업은 로컬 환경에서 직접 코드로 실행 및 재현 가능해야 하며, 외부 서버에 의존하는 방식은 제한됩니다.
3) 외부 데이터 사용 금지

온라인 해커톤(Phase 2)에서 제공하는 공식 데이터 외의 외부 데이터는 사용할 수 없습니다.
		   4) 추론 코드의 평가 데이터 예측 원칙

평가 데이터(test.csv)의 각 행은 하나의 독립적인 예측 대상입니다.
참가자는 각 행에 포함된 입력 변수와 주최 측이 제공한 공식 학습 데이터만을 이용하여 해당 행의 예측값(제구 성공 확률)을 추론해야 합니다. 
평가 데이터의 다른 행이나 전체 평가 데이터의 분포를 이용해 특정 행의 예측값을 보정하거나 생성하는 방식은 정상적인 추론 절차로 인정되지 않습니다.



## 제출 모델 만들기 설명서

[Baseline] RandomForest — 학습
KBO 투구 하나가 제구 성공 투구일 확률을 예측하는 베이스라인입니다.

입력: test.csv 의 47개 컬럼 (경기 상황, 투수·타자의 직전까지 누적 기록 등)
출력: 제구 성공 확률 (0 이상 1 이하의 실수)
평가지표: Brier Skill Score
trackman_history.csv 는 이 베이스라인에서 사용하지 않습니다. 2019~2024 과거 로그 179만 행이 그대로 남아 있으니 직접 활용해 보세요.

이 노트북은 모델을 학습하여 ./model/rf.pkl 로 저장합니다. 저장한 모델은 추론용 script.py 와 함께 baseline_submit.zip 으로 묶어 제출합니다.

1. 라이브러리 불러오기
데이터 처리(pandas)와 모델 학습(scikit-learn)에 필요한 라이브러리를 불러옵니다. joblib 은 학습한 모델을 파일로 저장할 때 사용합니다.

```
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
```

2. 데이터 불러오기
train.csv 는 2019~2024 시즌이고 평가 데이터는 2025 시즌입니다.

사용할 피처 목록은 test.csv 가 정합니다. train.csv 에만 있는 컬럼을 학습에 넣으면 평가 시점에 그 컬럼이 없어 추론이 실패하기 때문입니다.

```
test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns
FEATURES = [c for c in test_cols if c != ID]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                    encoding="utf-8-sig", usecols=FEATURES + [TARGET])

print("train:", train.shape, "| 피처:", len(FEATURES),
      f"(범주형 {len(CAT_COLS)}, 수치형 {len(NUM_COLS)})")
print("시즌:", train["season"].min(), "~", train["season"].max())
print(f"제구 성공률: {train[TARGET].mean():.4f}")
```

3. 전처리 정의
범주형 3개(top_bottom, game_type, base_state)는 정수로 바꾸고, 수치형 44개의 결측값은 중앙값으로 채웁니다.

두 변환을 ColumnTransformer 로 묶어 모델 파이프라인 안에 넣습니다. 이렇게 하면 추론할 때도 같은 변환이 자동으로 따라가므로, 전처리를 빠뜨려 생기는 실수를 막을 수 있습니다. handle_unknown="use_encoded_value" 는 학습 때 보지 못한 범주가 평가 데이터에 나타나면 -1 로 처리하라는 뜻입니다.

```
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1), CAT_COLS),
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
])
```

4. 모델 정의와 학습
트리 깊이를 10, 잎 노드의 최소 샘플 수를 200으로 제한해 얕게 두었습니다. 학습이 1분 안에 끝나고 모델 파일도 4MB 정도로 가볍습니다. random_state 를 고정했고 GPU 를 사용하지 않으므로, 같은 데이터와 같은 패키지 버전이면 어느 환경에서 실행해도 결과가 같습니다.

```
model = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=200,
        n_jobs=-1,
        random_state=42,
    )),
])
```

2024 시즌을 검증용으로 떼어 두고 2019~2023 으로 학습합니다.

```
is_val = train["season"] == 2024
X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
print("train:", len(X_train), "| val:", len(X_val))

t = time.time()
model.fit(X_train, y_train)
print(f"학습 완료 :: {time.time() - t:.1f}s")
```

5. 검증 — Brier Skill Score
학습 데이터에서 떼어 둔 2024 시즌으로 검증 점수를 계산합니다.

Brier 는 예측 확률과 실제값(0/1) 차이의 제곱 평균이고, 이를 상수 예측의 Brier 인 r(1-r) 로 나누어 Brier Skill Score 를 구합니다. 검증 분할 방식은 참가자가 자유롭게 바꿀 수 있습니다.

```
val_pred = model.predict_proba(X_val)[:, 1]

r = y_val.mean()
brier = ((val_pred - y_val) ** 2).mean()
baseline_brier = r * (1 - r)
score = max(0, 100000 * (1 - brier / baseline_brier))

print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
print(f"Validation Score: {score:.2f}")
```

6. 전체 데이터로 재학습 & 모델 저장
검증으로 성능을 확인했으니 이제 전체 학습 데이터로 다시 학습합니다.

학습한 파이프라인을 ./model/rf.pkl 로 저장합니다. 이 파일을 추론용 script.py, requirements.txt 와 함께 baseline_submit.zip 으로 묶으면 제출 준비가 끝납니다.

```
t = time.time()
model.fit(train[FEATURES], train[TARGET])
print(f"재학습 완료 :: {time.time() - t:.1f}s")

os.makedirs("./model", exist_ok=True)
joblib.dump(model, "./model/rf.pkl", compress=3)
print("저장 완료: ./model/rf.pkl")
```

## 데이터 설명서

### [배포용 데이터 구조]

```text
open.zip

baseline_submit.zip : 베이스라인 코드 기반 리더보드 제출 파일(zip) 예시 (참고용)
data/
  - train.csv : 학습 입력 및 정답 데이터 (1,475,092행 x 49컬럼)

  - test.csv : 평가 입력 데이터 (형식 확인용 5건 샘플, 48컬럼)

  - sample_submission.csv : 제출 양식 파일 (형식 확인용 5행 x 2컬럼)

  - trackman_history.csv : 2019~2024년 Trackman 과거 로그 (1,793,078행 x 30컬럼)

data_description.md : 데이터 설명서
```

※ `test.csv`는 배포본에 형식 확인용 5건만 포함됩니다. 실제 평가 데이터는 비공개이며, 제출용 파일(zip)을 리더보드 평가에 제출하면 평가 서버에서 동일한 경로와 동일한 컬럼 구조의 실제 평가 데이터로 교체되어 처리됩니다.

※ `sample_submission.csv` 역시 형식 확인용 5건 샘플입니다. 실제 평가 시에는 평가 서버의 `test.csv` 행 수와 동일한 제출 양식을 생성해 제출해야 합니다.

※ 모든 데이터 파일은 CSV 형식입니다. `trackman_history.csv`는 메인 학습/평가 데이터와 1:1로 결합하는 정답 테이블이 아니라, 참가자가 과거 이력 기반 피처를 만들 때 참고할 수 있는 로그 데이터입니다.

### [세부 설명]

#### 1) 학습/추론 데이터: `train.csv` · `test.csv`

각 행은 한 개의 투구 시점 상태를 나타냅니다. 참가자는 투구 직전까지 알 수 있는 경기 상황, 선수/팀 정보, 과거 이력 피처를 바탕으로 해당 투구의 제구 성공 확률을 예측합니다.

`train.csv`와 `test.csv`의 입력 피처 구조는 동일합니다. 단, `train.csv`에는 학습 정답인 `control_success`가 포함되고, `test.csv`에는 정답 컬럼이 포함되지 않습니다.

##### 기본 식별자 및 경기 정보

| 컬럼 | 설명 |
| --- | --- |
| `row_id` | 샘플 고유 식별자입니다. 제출 파일과 매칭하는 데 사용합니다. |
| `season` | 시즌 연도입니다. |
| `game_month` | 경기 월입니다. |
| `game_dayofweek` | 경기 요일입니다. 월요일은 0, 일요일은 6입니다. |
| `inning` | 투구 직전 이닝입니다. |
| `top_bottom` | 초/말 구분입니다. `T`는 초, `B`는 말을 의미합니다. |
| `game_type` | 경기 유형 코드입니다. |

##### 투구 직전 카운트 및 점수 상황

| 컬럼 | 설명 |
| --- | --- |
| `balls_before` | 투구 직전 볼 카운트입니다. |
| `strikes_before` | 투구 직전 스트라이크 카운트입니다. |
| `outs_before` | 투구 직전 아웃 카운트입니다. |
| `run_top_before` | 투구 직전 초 공격 팀의 점수입니다. |
| `run_bot_before` | 투구 직전 말 공격 팀의 점수입니다. |
| `run_total_before` | 투구 직전 양 팀 합산 점수입니다. |
| `score_diff_home` | 투구 직전 홈 팀 기준 점수 차입니다. |
| `score_diff_pitcher_team` | 투구 직전 투수 소속 팀 기준 점수 차입니다. |

##### 주자 및 상황 중요도

| 컬럼 | 설명 |
| --- | --- |
| `runner_on_1b` | 투구 직전 1루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `runner_on_2b` | 투구 직전 2루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `runner_on_3b` | 투구 직전 3루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `num_runners_on` | 투구 직전 출루 주자 수입니다. |
| `base_state` | 투구 직전 주자 상황입니다. `___`=주자 없음, `1__`=1루, `_2_`=2루, `__3`=3루, `12_`=1/2루, `1_3`=1/3루, `_23`=2/3루, `123`=만루입니다. |
| `home_win_expectancy` | 투구 직전 경기 상황에서 홈 팀의 기대 승률입니다. 0~100 범위의 값입니다. |
| `away_win_expectancy` | 투구 직전 경기 상황에서 원정 팀의 기대 승률입니다. 0~100 범위의 값입니다. |
| `li` | 투구 직전 상황 중요도 지표입니다. 값이 클수록 경기 흐름에 미치는 영향이 큰 상황을 의미합니다. |

##### 선수 및 팀 정보

| 컬럼 | 설명 |
| --- | --- |
| `pitcher_id` | 투수 익명 ID입니다. |
| `batter_id` | 타자 익명 ID입니다. |
| `pitcher_hand` | 투수의 좌우 유형 코드입니다. |
| `batter_hand` | 타자의 좌우 유형 코드입니다. |
| `pitcher_team_id` | 투수 소속 팀 ID입니다. |
| `batter_team_id` | 타자 소속 팀 ID입니다. |

##### 투구 직전 기준 과거 이력 피처

`asof_*` 컬럼은 해당 행의 투구 직전까지 확인 가능한 과거 기록으로 사전 계산된 피처입니다. 현재 투구 이후에 확정되는 정보는 사용하지 않았습니다.

| 컬럼 | 설명 |
| --- | --- |
| `asof_pitcher_n` | 해당 투구 직전까지 해당 투수의 누적 투구 수입니다. |
| `asof_pitcher_success_rate` | 해당 투구 직전까지 해당 투수의 제구 성공률입니다. |
| `asof_pitcher_reverse_rate` | 해당 투구 직전까지 해당 투수의 의도 반대성 투구 비율입니다. |
| `asof_pitcher_middle_rate` | 해당 투구 직전까지 해당 투수의 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_ball_rate` | 해당 투구 직전까지 해당 투수의 볼성 결과 비율입니다. |
| `asof_pitcher_strike_rate` | 해당 투구 직전까지 해당 투수의 스트라이크성 결과 비율입니다. |
| `asof_pitcher_prev1_game_success_rate` | 해당 투수의 직전 1경기 제구 성공률입니다. |
| `asof_pitcher_prev3_game_success_rate` | 해당 투수의 직전 3경기 제구 성공률입니다. |
| `asof_pitcher_prev5_game_success_rate` | 해당 투수의 직전 5경기 제구 성공률입니다. |
| `asof_pitcher_prev1_game_middle_rate` | 해당 투수의 직전 1경기 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_prev3_game_middle_rate` | 해당 투수의 직전 3경기 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_prev5_game_middle_rate` | 해당 투수의 직전 5경기 가운데 또는 위험 코스 비율입니다. |
| `asof_batter_n` | 해당 투구 직전까지 해당 타자가 상대한 누적 투구 수입니다. |
| `asof_batter_success_rate` | 해당 투구 직전까지 해당 타자가 상대한 투구의 제구 성공률입니다. |
| `asof_batter_middle_rate` | 해당 투구 직전까지 해당 타자가 상대한 투구의 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_pitchmix_n` | 해당 투구 직전까지 해당 투수의 구종 사용 이력 표본 수입니다. |
| `asof_pitcher_fastball_rate` | 해당 투구 직전까지 해당 투수의 fastball 계열 사용 비율입니다. |
| `asof_pitcher_breaking_rate` | 해당 투구 직전까지 해당 투수의 breaking 계열 사용 비율입니다. |
| `asof_pitcher_offspeed_rate` | 해당 투구 직전까지 해당 투수의 offspeed 계열 사용 비율입니다. |

※ 표본 수가 0인 경우 일부 rate 컬럼은 결측값일 수 있습니다. 이런 cold-start 상황의 결측 처리, smoothing, fallback 전략은 참가자가 자유롭게 설계할 수 있습니다.

#### 2) 학습 정답 데이터: `train.csv`의 `control_success`

`train.csv`에는 아래 정답 컬럼이 포함됩니다.

| 컬럼 | 설명 |
| --- | --- |
| `control_success` | 예측 대상입니다. `1`은 제구 성공, `0`은 제구 실패를 의미합니다. |

`control_success`는 운영 기준에 따라 산출된 제구 성공 여부입니다. Target 산출에 사용되는 현재 투구의 사후 정보는 입력 피처로 제공되지 않습니다.

#### 3) 과거 Trackman 로그: `trackman_history.csv`

`trackman_history.csv`는 2019~2024년 Trackman 과거 로그입니다. 2025년 Trackman 데이터는 제공되지 않습니다.

참가자는 이 파일을 이용해 과거 투구 특성, 구종 특성, 투수 단위 요약값 등 추가 피처를 만들 수 있습니다. 단, 이 파일은 `train.csv` 또는 `test.csv`와 1:1로 직접 결합되는 테이블이 아니며, 평가 시점 이후 정보를 포함하는 방식으로 사용할 수 없습니다.

| 컬럼 | 설명 |
| --- | --- |
| `trackman_id` | Trackman 과거 로그의 행 식별자입니다. |
| `season` | Trackman 투구가 속한 시즌입니다. 2019~2024만 포함됩니다. |
| `game_date` | Trackman 투구의 경기 날짜입니다. |
| `game_month` | Trackman 투구의 경기 월입니다. |
| `game_dayofweek` | Trackman 투구의 경기 요일입니다. 월요일은 0, 일요일은 6입니다. |
| `trackman_game_id` | Trackman 기준 경기 ID입니다. 메인 데이터의 `row_id`와 직접 대응하지 않습니다. |
| `pitch_no` | Trackman 기준 경기 내 투구 번호입니다. |
| `inning` | Trackman 로그의 이닝입니다. |
| `top_bottom` | Trackman 로그의 초/말 표기입니다. |
| `balls_before` | 투구 직전 볼 카운트입니다. |
| `strikes_before` | 투구 직전 스트라이크 카운트입니다. |
| `outs_before` | 투구 직전 아웃 카운트입니다. |
| `pitch_of_pa` | 해당 타석에서의 투구 순번입니다. |
| `pitcher_trackman_id` | Trackman 기준 투수 ID입니다. |
| `batter_trackman_id` | Trackman 기준 타자 ID입니다. |
| `pitcher_hand` | 투수의 좌우 유형 코드입니다. |
| `batter_hand` | 타자의 좌우 유형 코드입니다. |
| `pitcher_team` | 투수 소속 팀입니다. |
| `batter_team` | 타자 소속 팀입니다. |
| `tagged_pitch_type` | 수동 또는 태깅 기반 구종명입니다. |
| `auto_pitch_type` | 자동 분류 기반 구종명입니다. |
| `pitch_type_group` | 구종을 `fastball`, `breaking`, `offspeed`, `other`로 단순화한 구종군입니다. |
| `rel_speed` | 릴리스 시점의 구속입니다. |
| `spin_rate` | 투구 회전수입니다. |
| `induced_vert_break` | 유도 수직 무브먼트입니다. |
| `horz_break` | 수평 무브먼트입니다. |
| `extension` | 릴리스 확장 거리입니다. |
| `rel_height` | 릴리스 높이입니다. |
| `rel_side` | 릴리스 좌우 위치입니다. |
| `zone_speed` | 홈플레이트 근처 구속입니다. |

#### 4) 모델 추론 결과 양식 파일: `sample_submission.csv`

| 컬럼 | 설명 |
| --- | --- |
| `row_id` | 평가 데이터 `test.csv`의 샘플 식별자입니다. |
| `control_success` | 예측값입니다. 해당 투구가 제구에 성공할 확률을 0 이상 1 이하의 실수로 입력합니다. |

※ 제출 시 `row_id` 값은 평가 서버에서 제공되는 `test.csv`와 정확히 일치해야 합니다.

#### 5) 평가 데이터 예측 원칙

평가 데이터의 각 행은 독립적으로 예측해야 합니다. 평가 서버에서 실제 `test.csv` 전체가 주어지더라도, 참가자는 `test.csv`의 다른 행을 이용해 현재 행의 피처를 만들 수 없습니다.

금지되는 예시는 다음과 같습니다.

- `test.csv` 내부 행들을 이용한 선수별, 팀별, 월별 누적 통계
- `test.csv` 내부 빈도값 또는 분포 통계
- `test.csv` 내부 target encoding
- `test.csv` 행 순서 기반 rolling 또는 expanding feature
- 평가 데이터 전체를 보고 만든 사후 보정값

운영 측에서 제공한 `asof_*` 컬럼은 각 행의 투구 직전 시점까지의 과거 기록만으로 계산된 공식 입력 피처이므로 사용할 수 있습니다.

#### 6) 사용 금지 정보

공정한 평가를 위해 다음 정보는 입력으로 사용할 수 없습니다.

- 현재 투구 이후에 확정되는 모든 정보
- 현재 투구의 실제 위치 또는 코스 정보
- 현재 투구의 실제 판정, 결과, 제구 성공 여부
- 현재 투구의 실제 구종
- 현재 투구의 Trackman 측정값
- 2025년 Trackman 데이터
- 평가 데이터 내부의 다른 행을 이용해 만든 누적, 빈도, 분포, rolling, target encoding 피처

제공된 `train.csv`, 평가 환경의 `test.csv`, 2019~2024년 `trackman_history.csv`, 그리고 대회 규칙상 허용되는 외부 데이터만 사용할 수 있습니다.



## Commands

There is no build/lint/test tooling — this is a data science pipeline run as plain scripts.

```bash
# Run the full pipeline: train -> validate against reference -> full retrain -> submit artifact
python dopip.py

# Run pieces individually
python code/train.py   # trains, writes ./open/temp/latest_model.pkl
python code/test.py    # evaluates latest_model.pkl vs open/reference/best_model.pkl, writes open/temp/compare_result.txt
```

Data must be manually downloaded into `open/data/` (`train.csv`, `test.csv`, `trackman_history.csv`, `sample_submission.csv`) — this directory is gitignored.

## Architecture

### Pipeline flow (`dopip.py`)

`dopip.py` is the orchestrator, run end-to-end:
1. Runs `code/train.py` as a subprocess → produces `open/temp/latest_model.pkl`: a **dict** (`model_kind: "mlp_ensemble"`) holding a **list** of `MLP_SEEDS` (currently 15) independently-trained `TabularMLP` `state_dict`s, plus every preprocessing artifact needed to reproduce its inputs (fitted `OrdinalEncoder`s, numeric medians, `QuantileTransformer`, feature-name lists, the `season_end_lookup` table — see "Data engineering" below). Built from `season < 2024` only, with `season == 2024` used as a validation split for each seed's early stopping (see "Train/eval split convention").
2. Runs `code/test.py` as a subprocess → rebuilds the same val split (`season == 2024`), rebuilds+scores `latest_model.pkl`'s **ensemble** (average predicted probability across all seeds) against `open/reference/best_model.pkl`'s ensemble (the current best), writes `NEW_BEST` or `KEEP_REF` to `open/temp/compare_result.txt`. If the reference file isn't an `mlp_ensemble` dict (e.g. a leftover pre-migration single-`mlp`/TabPFN/CatBoost artifact), it's treated as `-inf` and skipped so the new model always wins.
3. Based on that flag: backs up the previous latest/reference model into `open/former_model/` (auto-numbered `_v1`, `_v2`, ... via `get_numbered_path`), and if `NEW_BEST`, promotes `latest_model.pkl` to `open/reference/best_model.pkl`.
4. **Full retrain**: rebuilds features from the **entire** dataset (all seasons 2019–2024, no held-out split, F1-filtered — see "Data engineering"), refits preprocessing on all of it, and trains **each of the 7 seeds** for exactly *that seed's own* `best_epoch` — the per-seed epoch counts the *promoted* reference model's early-stopping runs settled on (`open/reference/best_model.pkl["best_epochs"]`, a list, one entry per seed). There's no held-out set at this stage to early-stop against, so the epoch counts found during the season-split validation run are the only signal available for "how long to train" each seed — this mirrors the pre-TabPFN CatBoost convention of carrying over `best_iteration_`. This becomes `submit/model/final_retained_model.pkl`. There's no separate checkpoint file to resolve/download/copy — all 7 models' weights are entirely our own and already inside the pickled dict (the whole dict is still only a few MB, since dropping trackman cut the numeric feature count roughly in half).

`submit/script.py` is the standalone inference entrypoint used by the competition server — it does **not** import from `code/`; it reimplements `map_top_bottom`, `apply_season_progression_features`, `add_engineered_features`, `add_missing_indicators`, `TabularMLP`, and `predict_proba_mlp` as plain copies, then loops over `model_dict["state_dicts"]` rebuilding+predicting+averaging. Keep both in sync manually if `code/train.py`'s feature set or model architecture changes. `league_success_mean` and `season_end_lookup` are computed once at training time and stored in the model dict, so `submit/script.py` never needs to read `train.csv` at inference time (and structurally *can't* derive `season_end_lookup` from `test.csv` itself — see "Data engineering" below for why that would violate the competition's anti-leakage rule) — it only needs `test.csv` and `sample_submission.csv`. `trackman_history.csv` is no longer read anywhere in the pipeline.

### MLP design

`TabularMLP` (defined in `code/train.py`, duplicated in `submit/script.py`) is a single feed-forward network architecture. "MLP 단독" (MLP-only) means no blending with other model families (CatBoost, TabPFN, etc.) — averaging several independently-seeded fits of *this same* architecture is an explicitly allowed and adopted exception (see "Seed ensembling" below), not a violation of that constraint. Two input paths are concatenated before the hidden tower:

- **Embeddings** for `EMBED_CATEGORICAL_FEATURES` — `top_bottom`, `game_type`, `base_state` (the original low-cardinality categoricals) *plus* `pitcher_id`, `batter_id`, `pitcher_team_id`, `batter_team_id`. The four ID columns have small cardinalities in this dataset (≈790/830/13/13 unique values respectively, checked directly against `train.csv`), so instead of feeding them in as raw ints (meaningless ordinal magnitude) or leaving them out, each gets its own `nn.Embedding` sized `cardinality + 1` — the `+1` slot is a reserved "unknown" bucket for any id/category `OrdinalEncoder` didn't see during fit (e.g. a rookie pitcher who only appears in 2025). `EMBED_DIMS` in `code/train.py` fixes each column's embedding width by name (currently small — 8 for pitcher/batter id, 4 for team id/base_state, 2 for the rest; wider embeddings measurably hurt val BSS, see `EXPERIMENTS.md`). **Do not embed `season`** — the validation split holds `season == 2024` out entirely, so a `season` embedding would be 100% unknown-bucket at eval time; keep it in the numeric tower. A sweep found embedding the other small-cardinality situational columns (`game_month`, `balls_before`, etc.) doesn't help either — see `EXPERIMENTS.md` "MLP 전환" section for the measured numbers before changing this list.
- **Numeric tower** for everything else: the base situational/count/score/win-expectancy columns, the `asof_*` history rates, the 8 season-progression columns from `apply_season_progression_features`, and the 12 engineered features from `add_engineered_features` (see "Data engineering" below for both). Missing values (several `asof_*`/season-progression rate columns are `NaN` for pitchers/batters with zero prior history/season sample — a documented cold-start case) are median-imputed, but **before** imputing, `add_missing_indicators` adds a same-named `_isna` binary flag column for every numeric column that had any `NaN` in the fit split — otherwise imputation silently erases the "this is a cold start" signal, which is informative on its own.

Numeric features are `QuantileTransformer(output_distribution='normal')`-normalized rather than `StandardScaler` (fit on the same split used to fit the `OrdinalEncoder`s — train-only for the validation run, full data for the final retrain) — measurably more stable than z-scoring for this dataset's skewed `asof_*` rate columns. Embeddings and the scaled numeric block are concatenated and passed through `HIDDEN_DIMS` (`[128, 64]`) with `BatchNorm1d` + `ReLU` + `Dropout(0.2)` per layer, ending in a single logit. Loss is `BCEWithLogitsLoss`; optimizer is `AdamW` (`lr=2e-3`, `weight_decay=1e-6`) with a **step-based linear-warmup + cosine-decay `LambdaLR`** schedule (5% warmup) plus gradient clipping (`max_norm=1.0`) — this replaced an earlier `ReduceLROnPlateau` schedule, which reacted too slowly to this problem's noisy epoch-to-epoch val BSS. The schedule's period is keyed off `max_epochs`, so `max_epochs` isn't just an early-stopping ceiling here — raising or lowering it changes the LR decay rate mid-training, which changes the actual result. Don't change `MAX_EPOCHS` without re-sweeping.

**Seed ensembling** (`train_mlp_ensemble`/`predict_proba_mlp_ensemble` in `code/train.py`): `MLP_SEEDS = [42, 123, 7, 2024, 99, 555, 31337, 1, 2, 3, 4, 5, 6, 8, 9]` — 15 fully independent `TabularMLP` fits (own initialization, own minibatch shuffle order, own early-stopped epoch count), whose predicted probabilities are simply averaged. This measurably beats any single seed (solo scores in the ~660–735 range vs. ensemble ~796 on the current feature set, see `EXPERIMENTS.md`) by cancelling out per-seed noise rather than adding new signal — consistent with how weak/noisy this target is. `model_dict["state_dicts"]` is a list of 15 state dicts (not one); `model_dict["best_epochs"]` is a list of 15 epoch counts, one per seed, each independently reused by the full-retrain step. There is currently no clean evidence for how many seeds is optimal beyond this (a related project's exploration found diminishing returns past ~7–15) — if you change `MLP_SEEDS` again, re-measure rather than assuming more is strictly better. History: a follow-up probe (2026-08-23, see `EXPERIMENTS.md`'s "후속 검증" section, Part C) extended 7 seeds to 15 (holdout=2024) and got a noisy, non-monotonic but net-positive gain (786.81 → 800.17, +13.36), but the 8 extra seed values used in that probe accidentally duplicated one of the original 7 (`seed=7`), so only 14 were actually distinct. The de-duplicated 15-seed list above (last value swapped `7`→`9`) was adopted into production the same day, ran through the full `dopip.py` pipeline, and reproducibly measured **795.71** (BSS 0.00796) — the exact number differs slightly from 800.17 since one seed differs, but the gain over the 7-seed baseline (786.81) held up.

**Reproducibility**: `nn.Embedding`'s CUDA backward accumulates gradients via atomic scatter-add, which is a documented non-deterministic operation (thread-completion order affects floating-point summation order). This was confirmed to cause real result variance — the same hyperparameters producing val scores anywhere from ~380 to ~554 across runs before the fix. `set_seed()` now also sets `torch.use_deterministic_algorithms(True, warn_only=True)`, `torch.backends.cudnn.deterministic = True`/`benchmark = False`, and the module sets `CUBLAS_WORKSPACE_CONFIG` before `torch` is imported. Two full `code/train.py` runs with this in place produced byte-identical logs. **If you add any new op to the training loop, verify determinism still holds (run twice, diff the logs) before trusting a sweep result** — see `EXPERIMENTS.md` for how this was caught and fixed.

**Known footgun when sweeping hyperparameters programmatically**: `TabularMLP.__init__(self, ..., hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)` binds those defaults at function-*definition* time. Setting `code.train.HIDDEN_DIMS = [...]` after import and then calling `train_mlp()` **silently has no effect** — several early sweeps were invalidated by this (dropout 0/0.1/0.2 all producing the identical score before the fix). `train_mlp()` now takes explicit `hidden_dims=`/`dropout=` keyword arguments that are resolved as globals inside the function body (same pattern `LR`/`WEIGHT_DECAY`/`BATCH_SIZE` already used correctly) — always pass architecture overrides through these, never by mutating the module attribute and relying on `TabularMLP`'s own defaults.

**Current measured result**: `season == 2024` holdout ensemble val BSS = 0.00796 (score 795.71) — reproducible, see `EXPERIMENTS.md` for the full score progression (463 → 701.71 solo → 710.30 after data-engineering changes below → 786.81 after 7-seed ensembling → 795.71 after de-duplicated 15-seed ensembling). This clears the LG Aimers Phase 2 completion cutoff (549.51) but is still short of the 1,000-point goal. Full inference-time benchmark on 245,789 rows (sampled-with-replacement from `train.csv` to simulate the real eval-set size) was only measured with 7 seeds (~5 seconds end-to-end on a local RTX 4060); with 15 seeds it should scale roughly linearly (~10s) — still nowhere near the competition's 10-minute budget, but re-benchmark before assuming this if seed count changes again.

**Why store `state_dict`s + preprocessing artifacts instead of a full `sklearn`-style pipeline object**: the model needs to be loadable regardless of what absolute paths or exact library internals differ between dev machine and the offline eval server. A list of `state_dict`s plus an architecture-config dict rebuilt via `TabularMLP(**config)` per seed is the most version-portable way to persist several plain `nn.Module`s.

**Training budget note**: an MLP forward pass over the full 245,789-row eval set is on the order of seconds even ×7 seeds — there is no realistic risk of blowing the competition's 10-minute inference budget with this architecture, so `PREDICT_BATCH_SIZE` (20,000 in `code/train.py`, `submit/script.py`) is a memory-friendliness knob, not a hard requirement.

### Data engineering: F1 filter, no trackman, season-progression features

These three changes were ported from a separate, far more mature sibling project (`lgaimers9-ensemble-v1`, a different clone doing a CatBoost+MLP blend with a 2000+-line, 55+-section experiment log) after independently re-confirming the underlying data patterns against *this* repo's `train.csv` — don't re-derive these from first principles, but do sanity-check any further port the same way (re-verify the claim against this repo's own data before trusting it).

**F1 filter** (`apply_f1_filter` in `code/train.py`): `game_type` has two values, `R` (정규, regular-season) and `F` (퓨처스, developmental-league). `F` rows' success rate was substantially *higher* than `R` through 2022 (e.g. 2022: F=70.9% vs R=50.4%) and then *flips lower* from 2023 onward (2023: F=47.3% vs R=50.3%) — confirmed directly against this repo's `train.csv`, cause unknown but the pattern looks settled, not noise. Since `game_type` is a high-importance feature, this contamination measurably hurts training. The filter drops `season <= 2022` `F` rows — **training data only; never applied to the validation split or to `test.csv`/inference**, since eval must reflect the real serving distribution. Because the val split is always `season == 2024` (never touched by a `season <= 2022` condition), it's safe to call `apply_f1_filter` unconditionally right after loading `train.csv`, before the train/val split happens.

**No trackman**: `trackman_history.csv` is no longer read or joined anywhere (`process_trackman_features_safe` was deleted). The sibling project ablated this repeatedly across several matching strategies and found the net effect on holdout score was ~0 and inconsistent in sign, while identifying a structural risk that doesn't show up in a same-distribution holdout: at train time trackman always matches the row's own season+situation (sharp distribution), but at serving time (`test.csv` is always a future, unseen season) it always misses in the same way — a systematic train-serve skew that "partial matching" fallbacks made *worse*, not better, on their real leaderboard. `map_top_bottom` (mapping `top_bottom`'s `T`/`B` to `0`/`1`) is the one piece of what `process_trackman_features_safe` used to do that's still needed and is now its own tiny function.

**Season-progression features** (`build_season_end_lookup` + `apply_season_progression_features` in `code/train.py`, 8 new columns: `{pitcher,batter}_season_{n,success_count,success_rate,rate_gap}`): `asof_pitcher_success_rate`/`asof_batter_success_rate` are *career-cumulative*, so a player's current-season form gets diluted by years of history. This decomposes the career-cumulative rate into "career rate as of the end of last season" (looked up) vs. "rate accumulated so far *this* season only" (subtracted out), isolating recent-form signal the cumulative rate buries. This was the single highest-value port — consistent, same-direction improvement across every validation regime the sibling project tried (unlike trackman), and this repo's own measurement went from 701.71 → 710.30 in the same run that also added the F1 filter and dropped trackman (a smaller net gain than the sibling project saw in isolation, plausibly because effects from three simultaneous changes aren't separable in one run — if isolating this feature's individual contribution matters later, A/B it alone).

**Leakage trap — read this before touching `build_season_end_lookup`**: lookup *construction* and *application* are deliberately separate functions. `build_season_end_lookup(df)` must only ever be called on labeled training data (it reads `control_success`) — never on `test.csv`, which has no label and whose rows can't legally be used to build features for each other per the competition rules ("평가 데이터 예측 원칙" in this file). `apply_season_progression_features(df, lookup)` is safe on any dataframe, including `test.csv`, because it only *joins* a precomputed `lookup` by `(id, season)` — it never derives anything from `df`'s own other rows. Concretely: `code/train.py`/`code/test.py` build `lookup` from the `season < 2024` train split each time (deterministic, matches the "test.py reconstructs everything from scratch" convention below); `dopip.py`'s full-retrain step builds `lookup` from the *entire* `train.csv` (post F1-filter) and bakes it into `submit/model/final_retained_model.pkl` as `model_dict["season_end_lookup"]`, so `submit/script.py` only ever *applies* it to `test.csv`, never (re)builds it from `test.csv`.

### Train/eval split convention

`code/train.py` and `code/test.py` share the same split logic and must be kept consistent with each other — `test.py` reconstructs the identical validation set from scratch to score against the reference model, it does not receive it from `train.py`.

Current convention: **season == 2024 is held out entirely as validation; seasons < 2024 are training-only** (`train_df['season'] < 2024` / `== 2024`). For the MLP this validation split also drives early stopping (`PATIENCE = 15` epochs with no BSS improvement) inside `train_mlp` — each seed's epoch with the best validation BSS is what gets saved, and its epoch number is later reused by `dopip.py`'s full retrain for that same seed (see "Pipeline flow" above). If touching the split again, verify actual row-count/season composition with a quick pandas check rather than trusting the code's comments — the "time filter" logic elsewhere in this codebase has a documented gap (see below).

**Known limitation, probed but still unresolved**: a sibling project (see "Data engineering" above) found that a *single*-season holdout's BSS is sensitive to that season's own league success rate `r`, because `baseline_brier = r(1-r)` is maximized at `r = 0.5` — a season where `r` happens to land near 0.5 makes BSS look structurally worse for reasons unrelated to model quality. `season == 2024`'s `r` (0.4861) is not actually an outlier among 2019–2024 (all fall in 0.486–0.550; 2024's `baseline_brier` is the *second-highest* of the six seasons, not the lowest), so it isn't a red flag on its own. A follow-up experiment (2026-08-23, see `EXPERIMENTS.md`'s "후속 검증" section, Part B) tried holding out `season == 2019` instead and got a much higher score (1099.81 vs 782.19 for the same 3-seed config) — but that swing runs in the *opposite* direction `r`-scaling alone would predict (2019's `baseline_brier` is lower than 2024's), so it's dominated by a confound: holding out 2019 leaves 2020–2024 in the training set, turning the task into interpolation rather than the real forward-only extrapolation `season == 2024` represents. **r-sensitivity itself is still not cleanly isolated** — a clean test would keep the `season < X` / `season == X` convention (always train on the past only) and vary `X` instead. Still out of scope for the changes made so far.

Alternate split strategies were prototyped (pre-MLP, CatBoost-era) and are kept as untracked/parallel scripts rather than merged in — `code/train_x30.py`/`test_x30.py` (train on <2024 + first 70% of 2024, validate on the last 30%) and `code/train_rd30.py`/`test_rd30.py` (fully random 30% holdout, ignoring time ordering). These predate the MLP migration and have **not** been updated to match the current feature set or model — treat them as reference-only unless you're deliberately reviving one of these split strategies for the MLP.

### Model storage convention

- `open/temp/latest_model.pkl` — most recent `train.py` output, transient. An `mlp_ensemble` dict (list of state_dicts + preprocessing artifacts), not a bare `nn.Module`.
- `open/reference/best_model.pkl` — current best-scoring MLP ensemble dict, gates whether a new run's architecture/training is an improvement, and supplies `best_epochs` (list, one per seed) to the full-retrain step.
- `open/former_model/` — auto-numbered backups of superseded latest/reference models, written by `dopip.py` every run. Pre-ensemble-migration backups here are single-`mlp`/TabPFN-bagging/CatBoost artifacts and may need extra libraries installed to even unpickle.
- `submit/model/final_retained_model.pkl` — the actual artifact submitted to the competition server, always a full-data (all seasons) MLP ensemble dict, self-contained including `season_end_lookup`. This is the *only* file `submit/model/` needs.

