# 실험 기록 (Feature Engineering & Hyperparameter Tuning)

`control_success` 예측 CatBoost 모델에 적용한 피처 엔지니어링과 하이퍼파라미터 튜닝 내역을 정리합니다. 기준은 이번 작업 시작 시점의 `open/reference/best_model.pkl` (BSS 0.00737 / 점수 736.61)입니다.

> **2026-08-22 이후**: 프로젝트 목표가 "MLP 단독 모델로 1,000점 돌파"로 바뀌면서 모델을 TabPFN v2 배깅 앙상블(중간 단계)을 거쳐 순수 PyTorch MLP로 전면 교체했습니다. 아래는 CatBoost 시절 기록이고, MLP 전환 이후 기록은 맨 아래 "MLP 전환 (2026-08-22)" 절을 참고하세요.

## MLP 전환 (2026-08-22)

`code/train.py`를 TabPFN 배깅에서 순수 PyTorch MLP(`TabularMLP`, 임베딩 + 수치 타워)로 전면 재작성. `season==2024` 홀드아웃 기준 실측 점수 추이:

| 단계 | val BSS | 환산 점수 | 비고 |
| --- | --- | --- | --- |
| 1차 기본 설정 (dropout 0.3, hidden [256,128,64], embed pitcher/batter=24) | 0.00463 | 463 | 최초 MLP 구현, 튜닝 전 |
| dropout↓(0~0.1), hidden [128,64], embed 축소(8/4/4), batch↑ | 0.00555 | 554 | **비결정적 실행 — 재현 안 됨(아래 참고)** |
| `season`을 범주형 임베딩에 추가 시도 | 0.00173 | 173 | **실패**. `season==2024`가 검증셋 전체라, 임베딩으로 넣으면 검증 시점에 100% unknown 슬롯(미학습 랜덤 벡터)으로 빠짐. `season`은 반드시 수치형으로 유지. |
| `game_month`/`balls_before` 등 나머지 이산 컬럼도 임베딩에 추가 시도 | 0.00380 | 380 | 개선 없음, 되돌림 |
| CUDA 비결정성 수정 후 동일 설정 재실행 | 0.00487 | 487 | **재현 확인**(두 번 실행 결과 완전히 동일) — 아래 "CUDA 재현성 문제" 참고 |
| **하이퍼파라미터 몬키패치 버그 발견 및 수정** (아래 참고) | — | — | 그동안 dropout/hidden_dims 스윕이 전부 무효였음을 확인 |
| 버그 수정 후 재스윕: dropout=0.2, hidden=[128,64] | **0.00702** | **701.71** | 결정적 재현 확인(여러 번 재현), 현재 `code/train.py` 채택값 |

**참고**: LG Aimers Phase2 수료 커트라인(549.51)은 넘었지만, 목표인 1,000점에는 아직 못 미칩니다. CatBoost 시절 튜닝 최고 기록(BSS 0.00819, 818.54점, 위 "1. 추가된 피처" 절)에도 아직 못 미치는 수준 — 이 문제(BSS가 전반적으로 0.005~0.008 수준에 머무는, 신호가 매우 약한 이진분류) 자체가 트리 계열보다 MLP에 불리한 출발점일 가능성이 있습니다.

### CUDA 재현성 문제

`torch.manual_seed(seed)`로 시드를 고정해도 완전히 결정적으로 동작하지 않았습니다. `TabularMLP`가 `pitcher_id`/`batter_id` 등 범주형 컬럼을 `nn.Embedding`으로 처리하는데, `nn.Embedding`의 CUDA backward는 배치 내 여러 행이 같은 임베딩 행을 인덱싱할 때 그래디언트를 atomic scatter-add로 누적합니다. 이 누적은 GPU 스레드 완료 순서에 따라 달라지고 부동소수점 덧셈은 결합법칙이 성립하지 않아, 동일한 시드로 동일한 설정을 재실행해도 결과가 흔들리는 현상(예: 554점 vs 380점)이 실측으로 확인됐습니다.

`code/train.py`의 `set_seed()`에 아래를 추가해 해결했습니다:
- 모듈 최상단에서 `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` (CUDA 컨텍스트 생성 전에 필요)
- `torch.use_deterministic_algorithms(True, warn_only=True)` — 결정적 구현이 없는 연산은 경고 후 폴백
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`

동일 설정으로 `code/train.py`를 두 번 실행해 로그가 한 글자도 다르지 않게 완전히 일치함을 확인했습니다(`diff` 결과 동일).

### 하이퍼파라미터 몬키패치 버그

스윕 스크립트에서 `code.train.HIDDEN_DIMS = [...]`, `code.train.DROPOUT = ...`처럼 모듈 속성을 실행 중에 바꿔가며 `train_mlp()`를 호출했는데, `TabularMLP.__init__(self, ..., hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)`의 기본값은 **함수 정의 시점**에 한 번 바인딩되는 파이썬 특성상 이후의 모듈 속성 변경이 반영되지 않았습니다(반면 `LR`/`WEIGHT_DECAY`/`BATCH_SIZE`는 함수 본문에서 매번 전역 이름을 참조하므로 문제 없었음). 즉 여러 차례의 dropout/hidden_dims 스윕이 실제로는 전부 같은 구조로 돌아간 것이었습니다 — 스윕 로그에서 dropout 0/0.1/0.2와 hidden [128,64]/[256,128,64] 조합이 전부 점수 486.65로 정확히 똑같이 나온 게 단서였습니다.

`train_mlp()`에 `hidden_dims=None, dropout=None` 명시적 인자를 추가하고 함수 본문에서 `HIDDEN_DIMS`/`DROPOUT` 전역을 그때그때 조회하도록 고쳐 해결. 버그 수정 후 재스윕한 결과가 위 표의 마지막 행(701.71)입니다.

## 데이터 엔지니어링 + 시드 앙상블 (2026-08-23)

사용자가 별도 세션(`/home/user/contest/mlp/lgaimers9`, `lgaimers9-ensemble-v1`, CatBoost+MLP 블렌드, EXPERIMENTS.md 2293줄 §1~§55+)에서 나온 수치를 공유해줘서, 그 저장소를 read-only로 조사(fork 에이전트)한 뒤 이 프로젝트에 이식 가능한 부분만 실제로 반영했습니다. 이식 전 이 저장소의 `train.csv`로 직접 재확인 후 적용(아래 각 항목 참고).

### 이식한 것

1. **F1 필터** (`apply_f1_filter`) — `game_type=='F'`(퓨처스)가 2022년까지는 R(1군)보다 유리했다가(예: 2022 F=70.9% vs R=50.4%) 2023년부터 불리하게 역전(2023 F=47.3% vs R=50.3%)되는 오염을 이 저장소 `train.csv`로 직접 재확인. `season<=2022`인 F 행을 **학습 데이터에서만** 제거(검증/추론 데이터는 원본 유지).
2. **트랙맨 완전 드롭** — `process_trackman_features_safe` 삭제, `trackman_history.csv` 로드 전부 제거. 근거: 트랙맨은 학습 시점엔 항상 같은 시즌으로 정확 매칭되지만 서빙 시점(test.csv, 미래 시즌)엔 항상 매칭 실패라는 구조적 train-serve skew가 있고, 그쪽 프로젝트에서 여러 매칭 방식으로 반복 검증해도 순효과가 사실상 0으로 수렴했습니다. `top_bottom` T/B→0/1 매핑만 `map_top_bottom`으로 남김.
3. **season-progression 피처 8개** (`build_season_end_lookup` + `apply_season_progression_features`) — `asof_{pitcher,batter}_success_rate`는 커리어 누적이라 최근 컨디션이 희석되는 문제를, "직전 시즌 마지막 행의 누적치"를 시즌 시작 기준값으로 빼서 그 시즌만의 성공률(`{role}_season_success_rate`)과 커리어 대비 격차(`{role}_season_rate_gap`)로 분리. **lookup 구성과 적용을 분리**하는 게 핵심 — `build_season_end_lookup`은 라벨 있는 학습 데이터에서만 호출(대회 규칙상 test.csv 자신으로는 계산 불가), `apply_season_progression_features`는 어떤 df에도(`test.csv` 포함) 안전하게 적용 가능. `dopip.py` full retrain은 lookup을 `final_retained_model.pkl`에 통째로 담아 `submit/script.py`가 재계산 없이 병합만 하도록 함.
4. **7-seed 앙상블** (`train_mlp_ensemble`/`predict_proba_mlp_ensemble`, `MLP_SEEDS=[42,123,7,2024,99,555,31337]`) — 사용자가 "단독"은 다른 모델 계열과 블렌딩만 안 하면 되고 같은 아키텍처 시드 앙상블은 허용된다고 확인해줘서 반영. `model_kind`를 `"mlp"`→`"mlp_ensemble"`로 바꾸고, `state_dict` 단수 필드를 `state_dicts` 리스트로, `best_epoch`을 `best_epochs`(시드별) 리스트로 변경.

### 이식하지 않은 것 (근거 포함)

- **TE-residual**: 그쪽 프로젝트에서 CatBoost엔 도움, **MLP엔 손해**(cutoff7 기준 MLP 단독 -24.51)로 확정됐다는 조사 결과라 이식 안 함.
- **`add_engineered_features` 12개 제거**: 그쪽에서 3-seed 스크리닝으론 유리해 보였지만 7-seed+2023 홀드아웃 재검증에서 뒤집힘(-8.93) → "유지"가 결론. 이 프로젝트도 그대로 유지(원래도 제거한 적 없음, 조치 불필요).
- **R-only + rolling-origin(cutoff7 등) 다중 레짐 검증**, **season==2024 단일 홀드아웃의 r-민감성 대응**: 조사로 메커니즘은 확인(baseline_brier=r(1-r)가 r=0.5에서 최댓값)했지만, 검증 인프라를 다시 짜야 하는 큰 작업이라 이번 세션에서는 보류. CLAUDE.md에 "확인은 했지만 아직 미반영"으로 기록해 둠.
- **CAT_COLS에서 pitcher_id/batter_id 제외**: 그쪽은 고카디널리티 raw ID를 임베딩에도 안 넣는 쪽으로 갔지만(CatBoost 위주 설계), 이 프로젝트는 이미 pitcher_id/batter_id를 작은 embed_dim(8)으로 넣어 좋은 결과를 냈으므로 그대로 유지.

### 실측 점수 추이 (`season==2024` 홀드아웃, 결정적 재현 확인됨)

| 단계 | val BSS | 점수 | 비고 |
| --- | --- | --- | --- |
| 위 "MLP 전환" 절 최종값 (버그 수정 후 하이퍼파라미터) | 0.00702 | 701.71 | 트랙맨 포함, F1 필터·season-progression 없음, 단일 시드 |
| F1 필터 + 트랙맨 드롭 + season-progression 동시 적용 | 0.00710 | 710.30 | 단일 시드(42). 수치형 피처 142개→92개(트랙맨 드롭으로 감소) |
| + 7-seed 앙상블 | **0.00787** | **786.81** | 시드별 solo 점수는 660~735 범위, 평균이 아니라 확률 평균 앙상블로 분산 상쇄 |

`dopip.py` 전체 파이프라인(학습 7-seed → 검증 → reference 승격 → 전체 데이터 7-seed 재학습)을 실제로 끝까지 실행해 통과 확인. `submit/script.py`도 대회 서버와 동일한 디렉토리 구조로 스모크 테스트 — 24만 5,789행 추론에 7개 모델 포함 약 5초(로컬 RTX 4060), 트랙맨 로드가 없어져서 오히려 이전 단일 모델(약 11초)보다 빨라짐.

### 남은 후보 (착수 → 아래 "후속 검증" 절 참고)

- 1,000점까지 아직 부족(786.81). season-progression 개별 A/B, r-민감성 교차검증, 시드 수(7 vs 15) 재검토를 모두 실측 완료(아래 "후속 검증" 절). r-민감성은 여전히 미해결로 남음.

## 후속 검증: season-progression 기여도 분리 / r-민감성 / 시드 수 재검토 (2026-08-23)

위 "남은 후보"에 적어둔 세 가지를 모두 실측했습니다. 스크립트는 `code/train.py`를 그대로 import해 재사용했고(스크래치패드 1회성 스크립트, 저장소에는 남기지 않음), 모두 F1 필터+트랙맨 드롭이 적용된 현재 파이프라인 전제 위에서 돌렸습니다.

### A. season-progression 개별 기여도 (holdout=2024, 3-seed[42,123,7])

| 구성 | val BSS | 점수 |
| --- | --- | --- |
| season-progression 포함 (기존 dopip 로그) | 0.00782 | 782.19 |
| season-progression 제거 | 0.00740 | 740.18 |
| **차이 (순수 기여도)** | **+0.00042** | **+42.01** |

F1 필터·트랙맨 드롭과 분리해서 측정하니 season-progression 8개 피처 단독으로 +42점 기여 — 세 변경을 한 번에 묶어 쟀을 때(701.71→710.30, +8.59)보다 훨씬 큰 순수 효과입니다. 세 변경을 같이 넣었을 때 상호작용으로 순효과가 상쇄됐던 것으로 보이며(F1 필터/트랙맨 드롭 쪽이 오히려 소폭 마이너스였을 가능성), 이 두 개는 아직 개별로 분리 측정하지 않았습니다.

### B. r-민감성 확인 (season별 r, 2019 홀드아웃 vs 2024 홀드아웃)

`season==2024`(r=0.4861)의 baseline_brier(0.249807)가 다른 시즌들보다 특별히 낮은 이상치는 아니었습니다(2019~2023 baseline_brier 범위 0.2476~0.2500, 2024는 그중 r=0.5에 두 번째로 가까운 값):

| season | r | baseline_brier |
| --- | --- | --- |
| 2019 | 0.5495 | 0.247551 |
| 2020 | 0.5269 | 0.249275 |
| 2021 | 0.5128 | 0.249837 |
| 2022 | 0.5037 | 0.249986 |
| 2023 | 0.5000 | **0.250000 (최댓값)** |
| 2024 | 0.4861 | 0.249807 |

같은 구성(F1필터+트랙맨드롭+season-progression, 3-seed[42,123,7])에서 holdout만 2019로 바꿔 재학습하니 val BSS=0.01100(점수 1099.81) — 2024 holdout(782.19) 대비 **+317.62** 폭증했습니다.

**주의 — 이 결과를 "2024가 불리한 시즌이라 저평가됐다"로 곧이곧대로 해석하면 안 됩니다.** baseline_brier만 보면 2019(0.2476)가 2024(0.2498)보다 오히려 낮아서, r-스케일링 효과만으로는 2019 holdout 점수가 더 낮게 나와야 정상입니다(분모가 작을수록 같은 절대 오차에서 BSS가 나빠짐). 실제로는 정반대로 나왔다는 건 r-민감성보다 **홀드아웃 위치 효과**가 훨씬 크다는 뜻입니다 — 2019를 홀드아웃하면 학습 데이터에 2020~2024(미래)까지 포함되어 사실상 "과거를 보간(interpolate)"하는 쉬운 문제가 되는 반면, 2024 홀드아웃은 진짜 "미래 추정(extrapolate)"이라 구조적으로 더 어렵습니다. 즉 이번 실험은 r-민감성 자체를 깨끗이 분리하지 못했고(교란요인: 시간적 보간 vs 외삽), r-민감성이 실제로 얼마나 되는지는 **여전히 미해결**입니다. 순수하게 보려면 `season < X` / `season == X` 형태(항상 과거만 학습)를 유지한 채 X를 2022·2023 등으로 바꿔가며 비교해야 하는데, 이번엔 하지 않았습니다.

### C. 시드 수 7 → 15 (holdout=2024, 프로덕션과 동일한 전체 피처셋)

reference 모델(`best_model.pkl`)의 기존 7-seed는 추론만 재사용하고, 8개 신규 시드를 이어 붙여 누적 앙상블 점수를 관찰:

| 누적 시드 수 | 점수 | 누적 시드 수 | 점수 |
| --- | --- | --- | --- |
| 7 (기존) | 786.81 | 12 | 791.40 |
| 8 | 783.85 | 13 | 795.19 |
| 9 | 785.87 | 14 | 797.19 |
| 10 | 794.12 | 15 | **800.17** |
| 11 | 795.82 | | |

7→15로 늘리면 786.81→800.17로 **+13.36** 개선되지만 8~13 구간에서 오히려 떨어졌다 오르는 등 단조 증가가 아니라 노이즈 낀 완만한 우상향입니다. 사이드 프로젝트가 말한 "7~15 이후 수확체감"과 방향은 일치합니다. **주의**: 신규 시드값 `[1,2,3,4,5,6,7,8]` 중 `7`이 기존 `MLP_SEEDS`에 이미 있던 값과 우연히 겹쳐서(`set_seed(7)`이라 완전 동일한 모델), 15개 중 실질적으로 서로 다른 시드는 14개뿐입니다. 결과 방향에 큰 영향은 없어 보이지만, 프로덕션에 15시드를 채택한다면 중복 없는 시드 리스트로 다시 잡아야 합니다.

### 결론 / 다음 액션

- season-progression은 확실히 유효(+42점 단독 기여) — 유지.
- 시드 수는 7→15가 소폭(+13점) 개선, 추론 비용도 거의 공짜(10분 제한에 한참 못 미침) — 채택 검토 가치 있음, 단 중복 시드부터 수정 필요.
- r-민감성은 이번 실험으로 깨끗이 결론 내지 못함 — 시간순서를 지키는 형태(`season<X`/`==X`)의 다중 시즌 검증이 여전히 필요.

## 신규 피처 후보 스크리닝 (2026-08-23, 1,000점 목표 재도전)

1,000점 목표를 향해 raw 컬럼 조합 기반 파생 피처 6종을 새로 시도했으나, **전부 채택 기준(3-seed 앙상블 기준 seed noise를 넘는 안정적 개선)을 통과하지 못했습니다.** 스크립트는 `code/train.py`를 그대로 import하는 스크래치패드 1회성 스크립트로 작성(저장소에 남기지 않음), 현재 프로덕션 파이프라인(F1필터+트랙맨드롭+season-progression) 위에 얹어 측정.

### 시도한 피처

| 피처 | 정의 | 의도 |
| --- | --- | --- |
| `same_hand` | `pitcher_hand == batter_hand` | 좌우 상성(platoon) 신호 |
| `batter_relative_success` / `matchup_rel` | `asof_batter_success_rate - league_success_mean` / `pitcher_relative_success - batter_relative_success` | 타자도 투수처럼 리그 대비 상대값으로 (기존 `pitcher_relative_success`는 있었지만 타자 쪽은 없었음) |
| `risp_pressure` | `pitcher_relative_success × 1[runner_on_2b or runner_on_3b]` | 득점권 주자 상황 압박 |
| `pitcher_command_score` | `strike_rate - ball_rate - middle_rate - reverse_rate` | 기존에 있던 `asof_pitcher_{strike,ball,middle,reverse}_rate` 4개를 하나의 종합 제구 지표로 압축 |
| `pitcher_success_rate_shrunk` | `(n·rate + K·league_mean)/(n+K)`, K=50 | 표본 적은 투수의 rate를 리그 평균 쪽으로 축소(베이지안 shrinkage) |
| `pitcher_team_win_expectancy` | `top_bottom`에 따라 `home_win_expectancy`/`away_win_expectancy` 중 투수 팀 쪽 선택 | 투수 소속팀 관점 승리 기대값 |

### 1차 스크리닝 (단일 시드=42, season==2024 홀드아웃)

| config | val score | Δ (baseline 710.30 대비) |
| --- | --- | --- |
| baseline | 710.30 | — |
| same_hand | 712.49 | +2.19 |
| batter_relative | 724.96 | +14.66 |
| risp_pressure | 693.75 | **−16.55** |
| command_score | **729.65** | **+19.35** |
| shrunk_rate | 707.84 | −2.46 |
| team_win_exp | 702.04 | −8.26 |
| all_combined(6개 전부) | 717.67 | +7.37 (마이너스 피처들이 섞여 희석) |

여기까지만 보면 `batter_relative`/`command_score`가 유망해 보였습니다.

### 2차 확인 (3-seed[42,123,7] 앙상블, sibling project 컨벤션과 동일)

| config | ensemble BSS | ensemble score | Δ (baseline 782.19 대비) |
| --- | --- | --- | --- |
| baseline | 0.00782 | **782.19** | — |
| batter_relative + command_score | 0.00772 | 771.98 | **−10.21** |
| batter_relative + command_score + same_hand | 0.00784 | 784.44 | +2.25 |

**1차(단일 시드) 결과가 통째로 뒤집혔습니다** — `command_score`가 단독으로는 +19점처럼 보였지만 3-seed 앙상블로 재확인하니 (다른 후보와 조합했을 때) −10점에서 +2점 사이로 노이즈 범위 안에 머뭅니다. `code/train.py`의 "CUDA 재현성" 절, `EXPERIMENTS.md` 위쪽 "후속 검증 Part C"에서 이미 경고한 그대로 — **단일 시드 스크리닝은 방향이 뒤집힐 수 있으므로 반드시 다중 시드로 재확인해야 한다**는 교훈이 이번에도 그대로 재현됐습니다.

### 결론 — 채택하지 않음

여섯 피처 모두 `code/train.py`에 반영하지 않았습니다(코드 변경 없음). 근거 추정: 이 여섯 피처가 전부 이미 수치형 타워에 raw로 들어가 있는 컬럼(`asof_pitcher_{strike,ball,middle,reverse}_rate`, `asof_batter_success_rate`, `pitcher_hand`/`batter_hand`, `home/away_win_expectancy`, `runner_on_{2b,3b}`)의 선형/곱셈 조합이라, MLP가 은닉층에서 이미 근사 가능한 정보이고 새 신호를 주지 못한 채 파라미터/노이즈만 늘린 것으로 보입니다. 기존에 유효했던 `add_engineered_features`의 12개(특히 `pitcher_relative_success`, `count_diff`와의 상호작용)와 `season-progression` 8개는 raw 컬럼에 없던 정보(리그 평균 대비 편차, 시즌 내 컨디션 분리)를 실제로 새로 만들어냈다는 점이 차이로 보임 — **다음에 피처를 추가한다면 "이미 있는 컬럼의 산술 조합"보다 "raw 데이터에 없는 새 정보"(예: 시즌 진행 상황처럼 원본에 없는 걸 lookup으로 구성)를 우선해야 함**.

### 남은 다음 후보 (미착수)

- ~~7→15 시드 확장 채택~~ → **완료(아래 절 참고)**.
- **투수 단위 정적 Trackman 프로필**: 시도 전 조사로 기각. `train.csv`의 `pitcher_id`(5자리, 예: `20700`)와 `trackman_history.csv`의 `pitcher_trackman_id`(6~8자리, 예: `13515155`)가 **직접 겹치는 값이 하나도 없음**(792개 vs 906개 고유값, 교집합 0)을 확인 — 두 ID가 서로 다른 익명화 공간이라 클린한 ID join 자체가 불가능. 기존에 버린 "상황 기반 매칭" 방식과 다를 바 없이 결국 간접 매칭을 다시 설계해야 하므로, 이미 결론 난 이슈를 재론하는 것으로 판단해 착수하지 않음.

## 7→15 시드(중복 제거) 프로덕션 채택 (2026-08-23)

새 피처 스크리닝이 전부 실패해서, 위 "남은 다음 후보"의 저위험 카드였던 15-seed 확장을 대신 채택했습니다. `code/train.py`의 `MLP_SEEDS`를 `[42, 123, 7, 2024, 99, 555, 31337, 1, 2, 3, 4, 5, 6, 8, 9]`(기존 7개 + 신규 8개 중 중복이던 마지막 값만 `7`→`9`로 교체, 15개 전부 distinct)로 바꾸고 `dopip.py` 전체 파이프라인을 끝까지 실행했습니다.

| 단계 | val BSS | 점수 | 비고 |
| --- | --- | --- | --- |
| 기존 reference (7-seed) | 0.00787 | 786.81 | 교체 전 |
| **신규 15-seed (중복 제거)** | **0.00796** | **795.71** | `dopip.py` 실행 완료, reference 승격 + 전체 데이터 재학습까지 반영됨 |

시드별 solo 점수: `[710.30, 748.41→누적, ...]` 상세 로그는 생략(각 시드 solo val_bss 0.0066~0.0074 범위, 앙상블이 항상 최고 solo보다 높음). `best_epochs=[11, 4, 15, 10, 14, 7, 9, 5, 11, 16, 15, 6, 14, 11, 7]`(시드 순서 대응)가 `open/reference/best_model.pkl`과 `submit/model/final_retained_model.pkl` 양쪽에 반영됨 — 전체 재학습(1,369,784행, F1필터 적용된 2019~2024 전체)도 이 epoch 수 그대로 사용해 정상 완료.

**주의**: 이전 프로브에서 측정한 800.17은 15개 중 시드 하나가 우연히 중복(14-distinct)된 상태였고, 이번 795.71은 진짜 15개 distinct 시드 기준입니다 — 두 숫자가 정확히 같지 않은 건 이 차이 때문이며, 7-seed 대비 개선(+8.90) 자체는 노이즈 범위를 벗어난 것으로 판단됩니다.

**한계**: 이 채택도 `season==2024` 단일 레짐 기준입니다. r-민감성/dual-regime 검증은 여전히 미해결로 남아있습니다(위 "후속 검증 Part B" 참고).

## 신규 피처 후보 스크리닝 2차: MLP 표현력 한계를 겨냥한 피처 (2026-08-23)

옆 프로젝트(`lgaimers9-5e`)와 겹치지 않는 방향으로, "raw 컬럼의 단순 산술 조합"(1차 스크리닝, 전부 실패)이 아니라 **shallow MLP(hidden=[128,64], ReLU+BN 2층)가 원본 스칼라 몇 개만으로 근사하기 구조적으로 어려운 연산**을 겨냥한 피처 3종을 새로 시도했습니다.

### 시도한 피처

| 피처 | 정의 | 의도 |
| --- | --- | --- |
| `cyclical` (`month_sin/cos`, `dow_sin/cos`) | `game_month`/`game_dayofweek`를 주기 12/7 기준 sin/cos 쌍으로 인코딩 | 정수 스케일 값은 "12월≈1월" 같은 wraparound 근접성을 MLP가 학습 못 함(트리는 애초에 무관) |
| `ratios` (`matchup_ratio`, `pitcher_recent{1,3}_ratio`) | 기존 차이(gap) 기반 피처의 나눗셈 버전, `(a+eps)/(b+eps)` | 나눗셈은 2층 ReLU+BN으로 두 원본 스칼라만으로 정확히 근사하기 어려운 연산 |
| `zscore` (`{role}_season_rate_zscore`) | `season_rate_gap × sqrt(season_n + 1)` | season-progression의 gap을 표본 크기로 가중해 "표본이 적어 우연일 수 있는 큰 격차"를 명시적으로 축소 |

### 1차 스크리닝 (단일 시드=42)

| config | val score | Δ (baseline 710.30) |
| --- | --- | --- |
| baseline | 710.30 | — |
| cyclical | 710.08 | -0.22 |
| ratios | 718.58 | +8.28 |
| **zscore** | **738.05** | **+27.75** |
| all_combined | 700.55 | −9.75 |

### 2차 확인 (3-seed[42,123,7] 앙상블)

| config | ensemble score | Δ (baseline 782.19) |
| --- | --- | --- |
| baseline | 782.19 | — |
| zscore | 753.90 | **−28.29** |
| zscore+ratios | 750.67 | **−31.52** |

**1차 결과가 또 완전히 뒤집혔습니다.** 단일시드에서 가장 유망해 보였던 `zscore`(+27.75)가 3-seed로는 오히려 baseline보다 −28점 낮게 나왔습니다. 1차 스크리닝(batter_relative/command_score)에 이어 이번이 두 번째로, 이 문제(BSS ~0.007~0.008 수준의 매우 약한 신호)에서는 **단일 시드 스크리닝이 사실상 신뢰할 수 없다**는 게 반복 확인됐습니다 — 시드별 조기종료 시점(3~16에폭 범위로 들쭉날쭉)에 따른 노이즈가 실제 피처 효과보다 훨씬 큰 것으로 보임.

### 결론 — 채택하지 않음

`cyclical`/`ratios`/`zscore` 모두 `code/train.py`에 반영하지 않았습니다. 이번 세션에서 시도한 피처는 총 9개(1차 6개 + 2차 3개)이고 **전부 3-seed 이상 검증에서 탈락**했습니다. 이 정도로 일관되게 실패하는 걸 보면, 현재 아키텍처(hidden=[128,64], dropout=0.2)와 피처셋(F1필터+season-progression+기존 12개 파생피처+15-seed 앙상블) 조합이 이 데이터로 만들 수 있는 손쉬운 파생 피처의 한계에 이미 근접했을 가능성이 있습니다. **앞으로 이 방향(기존 컬럼의 산술/삼각함수/비율 재조합)으로 더 파는 것은 비용 대비 낮은 우선순위로 판단됩니다.**

### 방법론 교훈 (다음 스크리닝에 반드시 적용)

1,000점 목표에 다시 도전할 때는 **1-seed 스크리닝 결과를 절대 그 자체로 신뢰하지 말 것** — 이번 세션에서만 두 번(batter_relative/command_score, zscore) 방향이 완전히 뒤집혔습니다. 가능하면 처음부터 3-seed(혹은 그 이상)로 스크리닝하거나, 1-seed는 "완전히 마이너스인 것만 걸러내는 예비 필터" 용도로만 쓰고 최종 채택 판단은 반드시 3-seed 이상에서 내려야 합니다.

## 신규 피처 후보 스크리닝 3차: MLP 아키텍처 특화 피처 (2026-08-23)

"MLP만의 특색을 살리는 피처"를 만들어보자는 요청으로, 손으로 만든 산술 조합이 아니라 **MLP 아키텍처 자체(임베딩 테이블 확장, 명시적 내적 레이어)로만 표현 가능한** 세 가지를 시도했습니다. 위 2차 스크리닝의 `cyclical`(주기 인코딩)도 단일시드 결과(−0.22, 사실상 0)만 있고 3-seed 재확인이 없었던 채 남아 있었기 때문에, 이번에 다른 두 후보와 함께 정식으로 3-seed 재확인했습니다.

### 시도한 피처

| 피처 | 정의 | 의도 |
| --- | --- | --- |
| `cyclical` | `game_month`/`game_dayofweek`를 주기 12/7 기준 sin/cos 쌍으로 인코딩 (2차 스크리닝과 동일 구현) | 순환 구조를 MLP에 명시적으로 제공 |
| `matchup_embed` | `pitcher_id`+`batter_id` 조합을 새 범주(`pitcher_batter_matchup_id`)로 만들어 별도 임베딩(dim=6) | 투수·타자 조합 고유의 궁합을 히든레이어가 아닌 임베딩 자체로 직접 학습 |
| `bilinear` | `pitcher_id`↔`batter_id`, `pitcher_team_id`↔`batter_team_id` 임베딩 쌍의 내적(dot product)을 스칼라 피처로 추가 (FM 계열 아이디어) | concat+ReLU MLP가 근사하기 어려운 두 임베딩 벡터 간 곱셈적 상호작용을 아키텍처로 직접 제공 |

구현은 `code/train.py`(+ `code/test.py`, `submit/script.py`, `dopip.py` 동일 반영)에 실제로 적용해 `add_matchup_id`/`add_cyclical_features`/`TabularMLP.bilinear_pairs`로 커밋 직전까지 만들었으나, 아래 결과로 전부 되돌렸습니다(코드 변경 없음, 순수 실험).

### 15-seed 프로덕션 전체 측정 (세 피처 전부 적용)

`code/train.py`를 그대로(15-seed, season==2024 홀드아웃) 실행한 결과:

| config | ensemble val BSS | 점수 | Δ (reference 795.71 대비) |
| --- | --- | --- | --- |
| reference (기존, 피처 없음) | 0.00796 | 795.71 | — |
| **+matchup_embed +cyclical +bilinear (전부)** | 0.00744 | 743.99 | **−51.72** |

명확한 하락이라, 세 피처 중 무엇이 원인인지 3-seed(`[42,123,7]`)로 개별 분리했습니다.

### 3-seed 개별 분리 (스크래치패드 1회성 스크립트, 저장소에 남기지 않음)

| config | ensemble score | Δ (baseline 782.19 대비) |
| --- | --- | --- |
| baseline (피처 없음, 기존 프로덕션과 동일 레시피) | 782.19 | — |
| +cyclical only | 742.83 | **−39.36** |
| +matchup_embed only | 724.06 | **−58.13** |
| +bilinear only | 767.85 | **−14.34** |
| +전부 조합 | 733.14 | **−49.05** |

**셋 다 단독으로도 명확히 마이너스**입니다(가장 덜 나쁜 `bilinear`도 −14.34로, 이전 세션에서 노이즈 범위로 간주한 폭 ±10 내외를 벗어남). 조합해도 개선되지 않고 오히려 서로 겹쳐 더 나빠집니다.

### 원인 추정

- **`cyclical`**: 2차 스크리닝의 1-seed 결과(−0.22, 거의 0)와 이번 3-seed 결과(−39.36)가 또다시 정반대 방향 — "1-seed 스크리닝은 신뢰 불가"라는 기존 방법론 교훈이 세 번째로 재현됨.
- **`matchup_embed`**: 사전에 측정한 `season<2024` 기준 unique 조합 82,297개 중앙값 9회 등장 자체는 극단적으로 희소하진 않았지만, `season==2024` 검증 행의 **48.8%가 학습 때 못 본 조합**이라 절반 가까이가 unknown 슬롯(미학습 랜덤 벡터)으로 빠짐 — `code/train.py` 상단 주석에 이미 경고돼 있는 "`season` 임베딩이 unknown 슬롯 폭증으로 val BSS 0.0017까지 폭락했다"는 실패 패턴과 구조적으로 동일. 세 후보 중 가장 크게 하락(−58.13)한 것도 이 설명과 일치.
- **`bilinear`**: 셋 중 유일하게 새 파라미터(임베딩 테이블)를 늘리지 않고 기존 임베딩의 내적만 추가하는 저위험 변경이었는데도 −14.34로 마이너스 — 이 문제(신호가 매우 약한 이진분류, BSS ~0.007~0.008)에서는 입력 차원을 늘리는 시도 자체가 대체로 득보다 실이 크다는 2차 스크리닝의 결론과 일치.

### 결론 — 채택하지 않음

세 피처 모두 되돌렸고 프로덕션 레시피(F1필터+season-progression+기존 12개 파생피처+15-seed 앙상블, reference 795.71)는 변경 없이 유지됩니다. 1차(6개)·2차(3개)·3차(3개) 세 라운드, 총 12개 피처 후보가 전부 3-seed 이상 검증에서 탈락했습니다 — 현재 아키텍처+피처셋이 이 데이터에서 손쉬운 파생 피처로는 더 이상 개선되지 않는 지점에 근접했다는 정황이 세 번째로 반복 확인된 것으로 판단됩니다. **다음에 1,000점 목표를 다시 시도한다면 피처 재조합보다 다른 축(아키텍처 용량/정규화, 시드 수, 다른 검증 레짐에서의 안정성)을 우선하는 게 나아 보입니다.**

## 신규 피처 후보 스크리닝 4차: `season_interaction` / `team_matchup_embed` (2026-08-24)

옆 저장소(`/home/user/contest/lgaimers9`, CatBoost+MLP 블렌드, 별도 세션)와 GPU를 공유하며 병행 진행. 스크립트는 `code/train.py`를 그대로 import하는 스크래치패드 1회성 스크립트(저장소에 남기지 않음), 프로덕션 레시피(F1필터+season-progression+기존 12개 파생피처) 위에 얹어 측정.

### 시도한 피처

| 피처 | 정의 | 의도 |
| --- | --- | --- |
| `season_interaction` | `pitcher_season_rate_gap × batter_season_rate_gap` | season-progression 8개 컬럼(3차까지는 손대지 않았던 축) 중 투수/타자 "이번 시즌 컨디션 gap" 두 개의 곱셈 교차. 옆 저장소가 자기 피처셋(TE-residual/pitchmix 포함, 블렌드)에서 이 아이디어로 cutoff7/season2023 양쪽에서 뚜렷한 플러스를 3-seed에서 관측해 공유했고, 이 저장소(순수 MLP, season==2024 단일 홀드아웃)에도 같은 방향이 재현되는지 독립적으로 이식해 테스트 |
| `team_matchup_embed` | `pitcher_team_id`+`batter_team_id` 조합을 새 범주(`team_matchup_id`, 카디널리티 92)로 만들어 저차원(dim=4) 임베딩 | 3차에서 실패한 `pitcher_id×batter_id` matchup 임베딩(카디널리티 수십만, val 시점 unseen 48.8%)과 동일 아이디어를 훨씬 낮은 카디널리티(팀은 13개뿐이라 조합 최대 169, 실측 92)로 재시도 — unseen 비율이 낮으면 3차의 실패 원인(unknown 슬롯 폭증)을 피할 수 있는지 확인 |

### 1차 확인 (3-seed[42,123,7], **CPU 실행** — 옆 세션이 GPU를 쓰는 동안 충돌 피하려고 의도적으로 CPU만 사용. baseline 절대값(746.03)이 GPU 기준 782.19와 다른 건 CPU/GPU 부동소수점 경로 차이 때문이며, 같은 실행 내 상대비교만 유효)

| config | ensemble score | Δ (CPU baseline 746.03 대비) |
| --- | --- | --- |
| baseline | 746.03 | — |
| +season_interaction | 748.17 | +2.14 (노이즈 범위) |
| +team_matchup_embed | 756.15 | +10.12 (경계선, `team_matchup_id` unseen_frac=0.23%로 예상대로 낮음 확인) |

`season_interaction`은 1차에서 이미 노이즈 범위라 이 시점에 기각. `team_matchup_embed`는 경계선이라 7-seed로 추가 확인.

### 2차 확인 (7-seed, `MLP_SEEDS`의 앞 7개 — GPU, 옆 세션 GPU 작업 종료 확인 후 전환)

| config | ensemble val_bss | ensemble score | Δ |
| --- | --- | --- | --- |
| baseline(7seed) | 0.00787 | **786.81** | — (기존 문서화된 7-seed 기준값과 정확히 일치, 재현성 재확인) |
| +team_matchup_embed(7seed) | 0.00773 | 773.08 | **−13.73** |

**3-seed에서 경계선(+10.12)이던 `team_matchup_embed`가 7-seed에서 뚜렷한 마이너스(−13.73)로 뒤집혔습니다.** 카디널리티를 낮춰 3차의 실패 원인(unseen 슬롯 폭증)은 실제로 피했지만(unseen_frac 0.23%), 그와 무관하게 여전히 마이너스 — 저카디널리티 임베딩 자체가 아니라 "이 신호 강도(BSS 0.007~0.008)에서 입력 차원을 늘리는 시도는 대체로 득보다 실이 크다"는 2차 스크리닝의 결론이 여기서도 재확인된 것으로 보입니다.

**독립 교차검증**: 같은 시간대에 옆 세션도 자기 저장소에서 `season_interaction`(및 다른 후보 2개)을 3-seed→7-seed로 재검증했는데, 3-seed에서 뚜렷하게 플러스였던 것이 7-seed에서 전부 마이너스(−4.94~−11.60)로 반전됐습니다. 서로 다른 저장소/피처셋/모델(블렌드 vs 순수 MLP)에서 독립적으로 "3-seed 낙관 → 7-seed 반전" 패턴이 동시에 재현된 셈이라, 이 문제(약한 이진분류 신호)에서는 **3-seed조차 최종 판단 기준으로 부족할 수 있다**는 경고로 받아들이는 게 맞아 보입니다.

### 결론 — 둘 다 채택하지 않음

`season_interaction`, `team_matchup_embed` 모두 `code/train.py`에 반영하지 않았습니다(코드 변경 없음). 지금까지 4개 라운드에 걸쳐 시도한 피처 후보는 총 14개(1차 6 + 2차 3 + 3차 3 + 4차 2)이며 전부 탈락했습니다. **방법론 교훈 갱신**: 기존엔 "1-seed는 신뢰 불가, 3-seed 이상에서 판단"이었는데, 이번엔 두 저장소에서 독립적으로 3-seed도 뒤집힐 수 있음이 확인됐습니다 — 앞으로 피처 채택을 최종 결정할 때는 가능하면 7-seed(혹은 그 이상)까지 확인하고, 3-seed 결과는 "완전한 마이너스만 걸러내는 예비 필터"로만 쓰는 게 안전합니다. `CLAUDE.md`의 "MLP 설계" 절에 이 라운드도 반영.

## 신규 피처 후보 스크리닝 5차: 지금까지 손대지 않은 원본 컬럼 4종 (2026-08-24)

사용자가 "새로운 피처들 마구 시도해봐"라고 요청해, 1~4차에서 한 번도 엔지니어링에 쓰이지 않았던 원본 컬럼(구종 비율 3개, `asof_pitcher_reverse_rate`, `asof_batter_middle_rate`, `inning`)을 겨냥한 후보 4개를 시도했습니다. GPU가 완전히 비어 있어(옆 세션 작업 종료) 처음부터 GPU 3-seed[42,123,7]로 스크리닝. 스크립트는 스크래치패드 1회성(저장소에 남기지 않음), 프로덕션 레시피 위에 얹어 측정.

### 시도한 피처

| 피처 | 정의 | 의도 |
| --- | --- | --- |
| `pitch_mix_entropy` | `-Σ p·log(p)` (fastball/breaking/offspeed rate) | 구종 비율 3개는 지금까지 수치형 타워에만 raw로 들어가 있었고 엔지니어링에 쓰인 적이 없었음. 엔트로피는 2층 ReLU+BN이 근사하기 어려운 비선형 연산이라는 점에서 2차 스크리닝의 "ratio/zscore" 논리와 같은 맥락 |
| `late_inning_pressure` | `li × 1[inning≥7]` | `inning`을 상황 피처로 조합한 적이 이 저장소에서 한 번도 없었음(count_pressure/risp_pressure는 카운트/주자 기반) |
| `pitcher_reverse_pressure` | `asof_pitcher_reverse_rate × 1[2스트라이크 또는 3볼]` | `reverse_rate`(포수 요구 반대 방향 투구 비율)를 엔지니어링에 쓴 적이 없었음 |
| `batter_middle_relative` | `asof_batter_middle_rate − league_middle_mean` | `batter_middle_rate`를 엔지니어링에 쓴 적이 없었음(`batter_relative_success`는 1차에서 시도했지만 그건 success_rate 기준) |

### 3-seed 결과 (GPU)

| config | ensemble score | Δ (baseline 782.19 대비) |
| --- | --- | --- |
| baseline | 782.19 | — (기존 문서값과 정확히 일치) |
| `pitch_mix_entropy` | 756.13 | **−26.06** |
| `late_inning_pressure` | 758.18 | **−24.01** |
| `pitcher_reverse_pressure` | 778.46 | −3.73 |
| `batter_middle_relative` | 777.67 | −4.52 |
| 전부 조합 | 761.01 | **−21.18** |

### 결론 — 전부 채택하지 않음, 7-seed 재확인 불필요

4개 후보 전부 마이너스(플러스/경계선 없음)라, 갱신된 방법론("확실한 마이너스는 3-seed에서 바로 기각, 플러스/경계선만 7-seed 재확인")에 따라 추가 검증 없이 전부 기각했습니다(코드 변경 없음). `pitch_mix_entropy`/`late_inning_pressure`는 2차 스크리닝에서 확인된 "새로운 비선형 변환(엔트로피/삼각함수/비율)도 이 신호 강도에서는 대체로 손해"라는 패턴과 일치합니다. `pitcher_reverse_pressure`/`batter_middle_relative`는 노이즈 범위 내이긴 하지만 플러스가 아니므로 채택 근거가 없습니다.

지금까지 5개 라운드, 총 18개 피처 후보(1차 6 + 2차 3 + 3차 3 + 4차 2 + 5차 4)가 전부 탈락했습니다. 이 저장소의 47개 원본 컬럼 거의 전부가 최소 한 번씩은 엔지니어링 시도의 재료로 쓰인 상태라, **"raw 컬럼의 새로운 조합"이라는 축 자체가 사실상 소진된 것으로 판단됩니다.** 다음에 1,000점 목표를 다시 시도한다면 피처보다 아키텍처 용량/정규화, 시드 수, 검증 레짐 안정성 쪽이 훨씬 유망해 보입니다(3차/4차 결론과 동일).

## 용량/드롭아웃 재스윕 (2026-08-24)

사용자가 "MLP가 좋은 선택이면 다른 축(용량/정규화, 시드 수)으로 옮겨서 튜닝해봐"라고 지시. `code/train.py`의 `HIDDEN_DIMS=[128,64]`/`DROPOUT=0.2`는 season-progression/F1필터/12개 파생피처/시드앙상블이 전부 반영되기 **이전**(초기 solo-MLP 701.71점 시점)에 스윕된 값이라, 지금의 완성된 파이프라인 위에서도 여전히 최적인지 재확인. GPU 3-seed[42,123,7], 스크래치패드 1회성 스크립트(저장소에 남기지 않음), 프로덕션 레시피 그대로 위에 hidden_dims/dropout만 바꿔 측정 — `train_mlp()`가 이 값들을 반드시 명시적 kwarg로 받아야 한다는 기존 문서화된 풋건(모듈 속성 재할당은 무시됨)에 유의해 구현.

### Phase 1: hidden_dims 스윕 (dropout=0.2 고정)

| hidden_dims | score | Δ (기존 782.19 대비) |
| --- | --- | --- |
| `[128, 64]` (기존) | **782.19** | — |
| `[256, 128]` (넓게) | 749.89 | −32.30 |
| `[64, 32]` (좁게) | 748.80 | −33.39 |
| `[128, 64, 32]` (깊게) | 741.44 | −40.75 |

### Phase 2: dropout 스윕 (`[128, 64]` 고정, Phase 1 최고값)

| dropout | score | Δ |
| --- | --- | --- |
| 0.2 (기존) | **782.19** | — |
| 0.3 | 754.06 | −28.13 |
| 0.1 | 752.88 | −29.31 |
| 0.15 | 743.73 | −38.46 |

### 결론 — 변경 없음, 기존 설정 재확인됨

7개 config 전부 기존 `hidden=[128,64], dropout=0.2`보다 낮았습니다(최선의 대안조차 −28.13). 초기 스윕이 stale하지 않다는 걸 확인했을 뿐 개선은 없었으므로 `code/train.py` 코드 변경 없음(주석만 이 재확인 내용으로 갱신). 이 결과는 3차/4차/5차의 반복된 결론("이 문제는 용량 부족이 병목이 아니라 신호 자체가 약함")과 다시 한번 일치합니다 — 용량을 늘리거나 줄이거나 모두 손해라는 게 이번에도 확인됐습니다.

## 시드 수 확장 15 → 30 (2026-08-24)

용량/드롭아웃 축이 개선 없이 끝나서, 사용자 지시대로 유일하게 지금까지 안정적으로 이득이었던 축(시드 앙상블 확장)을 계속 밀어붙였습니다. 효율을 위해 `open/reference/best_model.pkl`의 기존 15개 state_dict를 재사용(순수 추론)하고, 새 시드만 추가 학습하는 방식으로 진행 — 재사용한 15-seed 예측이 저장된 reference val_bss(0.00796)와 정확히 일치함을 먼저 확인해 이 방식이 안전함을 검증했습니다. GPU(옆 세션 `lgaimers9-1f`와 공유), 스크래치패드 1회성 스크립트(저장소에 남기지 않음).

### 1차: 15 → 25 (새 시드 10개: 10~19)

첫 시도는 예측만 하고 학습된 가중치를 디스크에 저장하지 않아(순수 스크리닝용) 이후 30까지 확장할 때 이 10개를 다시 학습해야 했습니다 — 결정적 학습이라 점수는 그대로 재현됐지만 시간이 중복 소요된 점은 비효율이었습니다(교훈: 이후 스크립트는 state_dict를 체크포인트 pickle로 저장하도록 수정).

| 시드 수 | score |
| --- | --- |
| 15 | 795.71 |
| 20 | 798.51 |
| 25 | **801.60** (Δ+5.89) |

### 2차: 25 → 30 (새 시드 5개 추가: 20~24, 이번엔 체크포인트 저장)

진행 중 옆 세션의 다른 작업이 GPU를 100% 포화(VRAM 7.8/8GB)시켜 시드 하나당 소요 시간이 200~800초에서 2200초 이상으로 늘어지는 구간이 있었습니다(seed=13, 2233.8s) — 메모리(available 3~4GB, swap 1.1~2.1GB)는 안정적으로 유지되어 위험하진 않았고, 옆 세션이 후순위 작업을 종료해 GPU를 비워준 뒤 나머지 시드들은 다시 200~400초대로 정상화됐습니다.

| 시드 수 | score |
| --- | --- |
| 25 | 801.60 |
| 27 | 802.02 |
| 28 | 802.17 |
| **23** | **803.41** (전체 구간 중 peak) |
| **30** | **800.86** (Δ+5.15 vs 15-seed) |

### 결론 — 20개 근처부터 정체(plateau), 15-seed 대비 +5점대는 안정적 이득

20~30 구간은 798~803 사이에서 오르내릴 뿐 뚜렷한 추가 상승이 없습니다 — 23-seed의 803.41은 단일 실행의 우연한 peak이지 진짜 수렴점이 아니고, 20~30 전체를 볼 때 진짜 정체값은 800~801 근처로 판단됩니다. 7→15 확장(+8.90) 때와 마찬가지로 이 이득은 "새 정보 주입"이 아니라 순수 분산 감소(여러 국소해 평균)라서, 지금까지 반복적으로 반전됐던 피처 후보들과 달리 리스크가 낮은 개선입니다. 다만 15→30이 GPU 학습 시간을 2배로 늘리면서 얻는 게 +5점 안팎이라 비용 대비 효율은 7→15 때보다 낮습니다 — **20~25 정도가 비용 대비 합리적인 지점**으로 보입니다.

**반영 완료 (2026-08-24)**: 사용자가 20-seed로 확정 승인. `code/train.py`의 `MLP_SEEDS`를 기존 15개 + `[10, 11, 12, 13, 14]`(5개 추가, 20-seed 체크포인트에서 798.51을 만들었던 조합과 동일)로 갱신하고 `dopip.py` 정식 파이프라인을 처음부터 전체 실행했습니다.

- Step 1(`code/train.py`, season<2024 학습/season==2024 검증): val_bss=0.00799, **score=798.51** — 스크래치패드 측정값과 정확히 일치, 재현성 확인.
- Step 2(`code/test.py`): 기존 reference(15-seed, 795.71) 대비 `NEW_BEST` 판정 → `open/reference/best_model.pkl` 승격(4,438,052 bytes), 이전 모델은 `open/former_model/former_best_model_v7.pkl`(15-seed, 3,705,262 bytes)로 백업.
- Step 3(전체 데이터 2019~2024 재학습, `best_epochs=[11,4,15,10,14,7,9,5,11,16,15,6,14,11,7,6,11,10,11,13]` 시드별 고정 에폭 사용): `submit/model/final_retained_model.pkl` 갱신(4,750,470 bytes).

**현재 프로덕션 reference: 798.51점 (15-seed 795.71 대비 +2.80).**

## 점수 변화 요약

| 단계 | BSS | 환산 점수 | 비고 |
| --- | --- | --- | --- |
| 시작 시점 reference | 0.00737 | 736.61 | 이번 작업 이전 모델 |
| + 파생 피처 12개 추가 (기본 하이퍼파라미터) | 0.00766 | 765.99 | +29.4 |
| + 하이퍼파라미터 튜닝 (Optuna 40 trials) | **0.00819** | **818.54** | +52.5 (누적 +81.9) |

검증은 `code/test.py`의 기존 방식 그대로 — `season == 2024` 전체를 홀드아웃, `season < 2024`로 학습.

## 1. 추가된 피처 (`code/train.py::add_engineered_features`)

새 함수 `add_engineered_features(df, league_success_mean)`으로 12개 피처를 추가했습니다. `code/train.py`, `code/test.py`, `dopip.py`(full retrain), `submit/script.py`(코드 중복, 수동 동기화) 네 곳에 모두 반영했습니다.

| 피처명 | 계산식 | 의도 |
| --- | --- | --- |
| `pitcher_recent1_gap` | `asof_pitcher_prev1_game_success_rate − asof_pitcher_success_rate` | 최근 1경기 컨디션 변화 |
| `pitcher_recent3_gap` | `asof_pitcher_prev3_game_success_rate − asof_pitcher_success_rate` | 최근 3경기 흐름 |
| `pitcher_recent5_gap` | `asof_pitcher_prev5_game_success_rate − asof_pitcher_success_rate` | 최근 5경기 추세 |
| `pitcher_relative_success` | `asof_pitcher_success_rate − league_success_mean` | 리그 평균 대비 상대 실력 |
| `count_diff` | `strikes_before − balls_before` | 현재 카운트가 투수에게 유리한지 |
| `is_full_count` | `(balls_before==3) & (strikes_before==2)` → 0/1 | 풀카운트 여부 |
| `pitcher_count_advantage_raw` | `asof_pitcher_success_rate × count_diff` | 투수 실력 × 현재 카운트 |
| `pitcher_count_advantage_rel` | `pitcher_relative_success × count_diff` | 리그 대비 실력 × 현재 카운트 |
| `pitcher_trend` | `prev1_game_success_rate − prev5_game_success_rate` | 단기(1경기) vs 중기(5경기) 흐름 차이 |
| `pitcher_consistency` | `std([prev1, prev3, prev5]_game_success_rate)` | 최근 제구 안정성 |
| `matchup` | `asof_pitcher_success_rate − asof_batter_success_rate` | 투수·타자 상대 우위 |
| `count_pressure` | `pitcher_relative_success × li × 1[strikes_before≥2 or balls_before≥3]` | 압박 카운트(2스트라이크/3볼)에서의 상대실력 × 상황 중요도 |

**`league_success_mean`은 리크 방지를 위해 학습 시점마다 다르게 계산합니다:**
- `code/train.py`, `code/test.py` (평가용): `season < 2024` 학습 split의 `control_success` 평균만 사용. 검증에 쓰는 2024 시즌 데이터는 이 평균 계산에서 제외 — 홀드아웃을 분리한 의미를 지키기 위함.
- `dopip.py` full retrain, `submit/script.py`: 홀드아웃이 없는 단계이므로 `train.csv` 전체(2019~2024) 평균 사용.

**논의 중 제외/단순화된 부분:**
- 원래 아이디어는 "직전 시즌 리그 평균"(`prev_season_league_success`, 시즌별 lookup)이었으나, 구현 복잡도 대비 이득이 크지 않다고 판단해 **단일 스칼라 상수(전체 평균)**로 단순화함.
- `pitcher_relative_success`를 유지하기로 하면서 여기 의존하는 `pitcher_count_advantage_rel`, `count_pressure`도 함께 유지.

### Feature Importance (튜닝 전 reference 모델 기준, PredictionValuesChange)

| 피처 | Importance |
| --- | --- |
| `pitcher_relative_success` | 3.98 (전체 4위) |
| `pitcher_recent5_gap` | 0.96 |
| `pitcher_count_advantage_raw` | 0.82 |
| `count_diff` | 0.72 |
| `pitcher_recent1_gap` | 0.42 |
| `pitcher_consistency` | 0.36 |
| `count_pressure` | 0.35 |
| `pitcher_recent3_gap` | 0.30 |
| `pitcher_trend` | 0.27 |
| `pitcher_count_advantage_rel` | 0.23 |
| `matchup` | 0.18 |
| `is_full_count` | 0.04 |

새 피처 12개 합산 importance ≈ 8.6% (전체 대비). `pitcher_relative_success`가 압도적으로 기여도가 높고, `is_full_count`는 거의 쓰이지 않음 (제거 후보로 남겨둠, 아직 제거하지 않음). 참고로 `game_type`(25.4%), `season`(16.3%)이 여전히 가장 큰 비중을 차지함.

## 2. 하이퍼파라미터 튜닝

새 스크립트 `code/tune.py`로 Optuna(TPE sampler, seed=42) 기반 탐색을 40 trials 수행. 데이터 로딩·피처 엔지니어링은 1회만 수행하고 트라이얼 간 재사용(속도 최적화). 목적함수는 `season==2024` 홀드아웃 BSS 최대화, `early_stopping_rounds=50`으로 트라이얼별 반복수는 가변.

### 탐색 공간

| 파라미터 | 범위 |
| --- | --- |
| `depth` | 4 ~ 8 |
| `learning_rate` | 0.01 ~ 0.2 (log) |
| `l2_leaf_reg` | 1 ~ 30 (log) |
| `random_strength` | 0 ~ 10 |
| `bagging_temperature` | 0 ~ 5 |
| `border_count` | 32 ~ 254 |
| `min_data_in_leaf` | 1 ~ 200 (log) |
| `bootstrap_type` | `Bayesian` (고정) |

### 적용된 최적값 (Trial 28, BSS 0.00819)

| 파라미터 | 튜닝 전 (train.py 원래값) | 튜닝 후 (적용값) |
| --- | --- | --- |
| `depth` | 6 | **7** |
| `learning_rate` | 0.05 | **0.05040411253232039** |
| `l2_leaf_reg` | (미지정, 기본값 3) | **3.14659036827521** |
| `random_strength` | (미지정, 기본값 1) | **3.715568024268865** |
| `bagging_temperature` | (미지정, 기본값 1) | **0.4609270457436248** |
| `border_count` | (미지정, 기본값 254) | **106** |
| `min_data_in_leaf` | (미지정, 기본값 1) | **81** |
| `bootstrap_type` | (미지정, 기본값) | **Bayesian** (명시) |
| best_iteration | 337 (기존 reference) | **519** |

`code/train.py`의 `CatBoostClassifier`와 `dopip.py` full retrain 단계의 `CatBoostClassifier`에 동일하게 반영. 이전에는 두 곳의 하이퍼파라미터가 서로 달랐던 것(예: `learning_rate` 0.05 vs 0.03, `l2_leaf_reg` 3 vs 10)도 이번에 통일함.

## 3. 변경된 파일

- `code/train.py` — `add_engineered_features()` 추가, 파생 피처 반영, 튜닝된 하이퍼파라미터 적용
- `code/test.py` — 동일 파생 피처 로직 반영 (학습 모델과 동일 입력 셋 유지 목적)
- `dopip.py` — full retrain 단계에 동일 파생 피처 + 튜닝된 하이퍼파라미터 반영
- `submit/script.py` — 코드 중복으로 동일 파생 피처 로직 반영 (train.csv를 추가로 로드해 `league_success_mean` 계산)
- `code/tune.py` — 신규. Optuna 하이퍼파라미터 탐색 스크립트 (제출용 코드에는 포함되지 않음, 로컬 실험 전용)

## 4. 남은 이슈 / 다음 후보

- `is_full_count`는 importance가 거의 0에 가까워 제거해도 무방해 보이지만 아직 유지 중.
- `code/test.py`의 `X_val['top_bottom'] = ...` 라인에서 나는 `SettingWithCopyWarning`은 이번 작업과 무관한 기존 이슈, 미수정.
- 다음 방향으로는 앙상블보다 피처 엔지니어링을 우선하기로 함 — 특히 `asof_batter_*`(현재 3개뿐)가 상대적으로 덜 활용되어 있고, trackman 쪽 pitch-mix/상황 교차 피처도 아직 없음. 앙상블은 대회가 10분 추론 제한이 있는 코드 제출 방식이라 리스크 대비 이득이 낮다고 판단해 후순위로 미룸.
