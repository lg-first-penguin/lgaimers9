우리팀용 비공개 파이프라인입니다. (v1)

```

lgaimers9 (루트 폴더)
├── submit/
│   ├── model/
│   │   └── final_retained_model.pkl (dopip.py 실행 시 전체 데이터 재학습 후 자동 생성)
│   ├── script.py (대회 서버 전용 추론 파일)
│   └── requirements.txt (필수 의존성 명시)
├── open/
│   ├── data/ (참가자 데이터 폴더 - train.csv, test.csv, trackman_history.csv) 
│   ├── former_model/ (이전 temp 및 reference 후보 저장 버퍼)
│   ├── reference/ (현재 최고 스코어를 기록한 기준 모델 저장)
│   └── temp/ (방금 학습을 마친 따끈따끈한 최신 모델 저장)
├── code/
│   ├── train.py (모델 학습 및 피처 엔지니어링 수행)
│   └── test.py (2024 검증 데이터를 통한 BSS 점수 산출 및 최고 모델 비교)
└── dopip.py (전체 파이프라인 제어 오케스트레이터 및 리트레인 수행 드라이버)

```

## <주의사항>

- ai 시킨거라 catboost로 했을 때 한 번 돌아가는 것만 확인해봤습니다.

- open/data/ 아래 직접 데이터 넣어주셔야 합니다.

- 본 파이프라인은 CatBoost 기준으로 만들었습니다만, 확장성을 고려하긴 했습니다. 
> 검증은 안해봤습니다.


## <코드 설명>

`dopip.py`로 실행하면 알아서 훈련, 테스트(reference model과의 1:1비교), 전체 데이터로 재훈련 후 

submit 제출파일 안 model에 .pkl 모델 파일을 넣어줍니다.


`code/` 폴더 아래 `train.py`와 `test.py`를 직접 작성하시면 되겠습니다.
- 각각 훈련 코드와 테스트 코드 입니다.
- 그 외 코드 파일들
	- train/test_x30 : 24년도의 제일 마지막 경기부터 30%(24년도의)를 잘라 테스트 데이터로 쓴 파일들입니다.
	- train/test_rd30 : 19-24년 전체 데이터 중 30%를 랜덤하게 잘라 테스트 데이터로 쓴 파일들입니다.
