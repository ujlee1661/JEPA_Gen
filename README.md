# JETS

**JETS**는 시간과 tabular feature를 가진 개체별 이벤트 시퀀스를 생성하는 joint-embedding 프레임워크입니다. 미래 시간과 feature 표현을 예측해, 이벤트 수·시간·feature를 생성하는 공동 diffusion의 조건으로 사용합니다.

모델의 정보 흐름, target encoder의 학습 목적, ON/OFF 대조군과 인과성 관련 한계는 [프레임워크 설명](docs/JETS_FRAMEWORK.md)을 참고하세요.

## 설치

Python 3.10 이상과 PyTorch 실행 환경이 필요합니다. GPU 사용 시 해당 환경에 맞는 PyTorch를 설치한 다음 저장소를 설치합니다.

```bash
pip install -e .
```

실험 코드가 직접 사용하는 NumPy, pandas, SciPy, scikit-learn은 패키지 의존성에 명시되어 있습니다. 테스트는 `pip install -e '.[dev]'` 후 실행할 수 있습니다.

## 프로젝트 구조

```text
jets/                    모델·target encoder·시간 표현·전처리
  models/                생성기·context encoder·predictor
  encoders/              시간·feature target encoder
  layers/                시간 간격·이벤트 수·diffusion 연산
  data/                  전처리·window·역변환
experiments/             데이터 준비·학습·생성·평가
configs/jets_retail.json Retail 실험의 기본 설정
tests/                   독립 실행 가능한 모델 테스트
docs/JETS_FRAMEWORK.md   프레임워크와 정보 경계 설명
```

## 실행 전 준비

원본 데이터, 가중치, 생성 결과와 외부 원본 평가기는 저장소에 포함하지 않습니다. 아래 파이프라인으로 입력 테이블에서 학습용 window bank, target encoder와 suffix cache를 새로 만들 수 있습니다. 신뢰할 수 있는 PyTorch 파일만 입력하세요. `torch.load`는 파일을 불러올 때 코드를 실행할 수 있습니다.

각 경로는 명령행에서 명시합니다. 기존 실험 checkpoint의 해시나 데이터 경로는 수정하지 않으며, 새 파이프라인 실행은 별도 output 디렉터리와 자체 출처 기록을 사용합니다.

```bash
python -m experiments.retail --help
python -m experiments.pipeline --help
python -m experiments.retail train --help
python -m experiments.retail generate --help
python -m experiments.retail evaluate --help
```

### 데이터 전처리부터 target encoder·cache 생성까지

기본 입력은 `account.csv`와 `trans.csv`가 들어 있는 디렉터리입니다. 이 파일들은 개체·이벤트 단위로 이미 집계된 테이블이며, 기본 Retail 스키마에서 `log_amount`, `log_cumamt`는 입력값을 그대로 사용합니다.

```bash
python -m experiments.pipeline \
  --raw-dir /path/to/retail_tables --config configs/jets_retail.json \
  --output /path/to/new_preparation --device cuda
```

한 명령으로 학습용 전처리기·window bank를 만든 뒤 **시간/feature target encoder 각각을 5,000회 사전학습하고**, 학습 구간 기준 교정과 고정, 각 window의 suffix representation cache 생성, 새 생성기 초기 가중치 준비까지 수행합니다. 완료된 파일의 경로·SHA-256과 다음 생성기 학습 명령은 `complete.json`에 기록됩니다. 생성기 학습은 아래 명령으로 별도 시작합니다.

```text
new_preparation/
  input/prepared/          전처리기, 고객 ID와 준비 상태
  input/bank/              학습용 window bank
  target_encoders/         시간·feature target encoder checkpoint
  target_caches/           각 window의 suffix 표현 cache
  initial_generator.pt     새 생성기 초기 가중치
  config.snapshot.json     실행에 사용한 설정 사본(해시 기록)
  pipeline.json            입력·설정·코드 출처
  complete.json            완성된 산출물 해시와 다음 학습 명령
```

Target encoder의 기본 학습량과 구조는 `configs/jets_retail.json`의 `target_encoder` 항목에서 확인할 수 있습니다. 다음 학습 명령은 외부 설정 파일이 아니라 검증된 `config.snapshot.json`을 사용합니다. 초기 생성기는 **새 데이터와 seed로 만든 초기화**이며, 기존 실험의 가중치를 복사하거나 그 실험과 동일하게 학습됐다고 주장하지 않습니다.

원본 **Online Retail II XLSX**를 입력하려면 `--raw-dir` 대신 `--retail-ii-xlsx /path/to/online_retail_II.xlsx`를 명시적으로 선택하고 `pip install -e '.[retail]'`로 XLSX 의존성을 설치하세요. 이 옵션은 기존 로컬 benchmark 입력 계약을 재현하며, 고객·인보이스 집계, 21–800 인보이스 고객 선택, 최대 500명·seed 9 선택과 음수 인보이스 합계의 0 처리를 포함합니다. 이것은 **입력 데이터 구성 규칙**이지 모델이나 생성 결과에 대한 사후 보정이 아닙니다. 다른 데이터에는 자동 적용하지 않습니다.

```bash
python -m experiments.pipeline \
  --retail-ii-xlsx /path/to/online_retail_II.xlsx \
  --config configs/jets_retail.json --output /path/to/new_preparation --device cuda
```

`--schema`로 개체·이벤트 컬럼 목록을 정의할 수 있고, 집계 테이블 입력에서는 `--train-ids`로 학습 개체를 명시할 수 있습니다. JETS의 전처리·window·target encoder는 선택된 개체에만 맞춰지지만, 입력 테이블이 사전에 전체 모집단을 사용해 구성됐을 가능성까지 배제하지는 못합니다. XLSX benchmark 변환은 전체 선택 모집단으로 Country를 인코딩하므로 `--train-ids`와 동시에 사용할 수 없습니다. 생략하면 입력 전체를 학습에 쓰므로 held-out 일반화 검증이 아닙니다. `--windows`와 `--target-steps`는 명시적인 실험 예산 변경이며, 기본값을 줄인 실행은 원래 전체 학습 결과로 표시하지 않아야 합니다. 이미 완료된 산출물은 해시를 확인해 재사용하며, 다른 입력·설정과 충돌하면 덮어쓰지 않습니다.

중단된 target encoder 학습은 동일한 입력·코드·설정에서 checkpoint부터 재개합니다. 데이터 준비 단계는 완료된 staging 기록이 있으면 해시를 검증해 재개합니다. 기록 이전의 미완성 staging은 자동 삭제하지 않고 새 staging으로 다시 준비합니다. XLSX 변환 도중 출력 표만 남고 영수증이 만들어지지 않았다면 덮어쓰지 않으므로 새 output 경로를 사용해야 합니다.
Target encoder checkpoint와 cache를 안전하게 게시하려면 출력 파일시스템이 하드 링크를 지원해야 합니다. 게시 journal 기록 전 중단돼 검증되지 않은 임시 산출물만 남은 경우에는 자동으로 덮어쓰지 않습니다.

예를 들어 기본 스키마는 `{"numeric":["log_amount","log_cumamt"], "categorical":[], "parent_numeric":[], "parent_categorical":["Country"]}`입니다. 이벤트 표에는 추가로 `trans_id`, `account_id`, `t_offset`, 개체 표에는 `account_id`가 필요합니다. 관측 종료점이 알려져 있다면 `--observation-end`로 지정하세요. 지정하지 않으면 선택된 학습 이벤트의 최대 offset을 사용하며, 이는 데이터 특성에 따라 관측 기간의 추정에 불과할 수 있습니다.
식별자 `001`, `1`, `NA`처럼 문자열을 원형 그대로 구분해야 한다면 schema에 `"id_dtype":"string"`을 추가하고 `--train-ids`에도 같은 문자열을 사용하세요. 같은 시각의 `trans_id`는 문자열 순서로 정렬됩니다. 생략하면 CSV의 기본 타입 추론을 사용합니다.

### 생성기 학습

```bash
python -m experiments.retail train \
  --bank /path/to/data.pt --config configs/jets_retail.json \
  --time-target-encoder /path/to/time_target_encoder.pt --time-cache /path/to/time_codes.pt \
  --feature-target-encoder /path/to/feature_target_encoder.pt --feature-cache /path/to/feature_codes.pt \
  --initial /path/to/generator_initial.pt --output /path/to/new_run --device cuda
```

기본 설정은 feature 표현 예측 손실 ON이며, 생성 조건에는 예측한 평균과 표준편차를 결정적으로 사용합니다. `--feature-weight 0`은 같은 구조에서 feature 예측 손실만 제거하는 대조군입니다. `--steps`는 명시적으로 학습량을 바꾸는 기능이며, 줄인 예산의 실행을 원본 전체 학습 결과로 표시해서는 안 됩니다. 입력 자료 해시와 새 run의 정보는 출력 경로에 기록됩니다.

### 자유 생성

```bash
python -m experiments.retail generate \
  --bank /path/to/data.pt --checkpoint /path/to/final.pt \
  --preprocessor /path/to/preprocessor.json --parents /path/to/parents.json \
  --output /path/to/new_generation --device cuda --seed 1
```

실제 미래의 이벤트 수나 시간·target 코드를 전달하지 않습니다. 출력 경로에 이미 결과 파일이 있으면 덮어쓰지 않습니다.
준비 manifest와 bank manifest에 해시가 있으면 전처리기·고객 ID·학습 bank의 연결을 확인합니다. 체크포인트에 기록된 입력 해시가 있는 경우에도 현재 bank가 일치해야 합니다. 이전 산출물처럼 해당 기록이 없으면 완전히 확인할 수 없으며, `generation.json`의 `artifact_checks`에 검증 여부가 남습니다. 이런 경우 입력 자료의 출처를 별도로 확인해야 합니다.

### 평가

```bash
python -m experiments.retail evaluate \
  --real /path/to/real_trans.csv --synthetic /path/to/child.csv \
  --scorer /path/to/trusted_score.py --dataset online_retail
```

평가기는 저장소에 포함되지 않으며 지정한 Python 파일을 직접 실행합니다. **신뢰할 수 있는 평가기만 사용하세요.** 평가 결과에는 평가기 해시가 포함됩니다.
전처리할 때 `"id_dtype":"string"`을 사용했다면 평가 명령에도 `--id-dtype string`을 지정해 앞자리 0과 `NA` 같은 ID를 보존하세요.

## 검증과 결과

```bash
python -m pytest -q tests/test_jets_model.py
```

기존 실험 checkpoint와 준비된 자료를 별도로 제공한 환경에서 호환 모드의 JETS 생성기·target encoder가 기존 결과와 일치하는지 검증했습니다. 현재 기본값은 결정적 feature 조건이며 과거 확률적 실험의 결과를 그대로 재현한다고 주장하지 않습니다. 기존 Retail 결과는 학습에 포함된 고객에 대한 평가이며, 학습 seed 1개와 생성 seed 3개의 결과입니다. 새로운 고객에 대한 일반화나 다중 학습 seed의 안정성을 의미하지 않습니다.

네 개의 논문 입력 테이블을 학습 seed 1·2·3과 학습별 생성 seed 1·2·3으로 실행할 때는 다음 명령을 사용합니다. 원본 평가기를 별도로 제공해야 하며, 이 명령은 JETS만 실행합니다.

```bash
python -m experiments.multidata --raw-root /path/to/reference/data \
  --scorer /path/to/reference/score.py --config configs/jets_retail.json \
  --output /path/to/new_multidata_run --device cuda
```

## 로컬 산출물

`data/`, `outputs/`, checkpoint 및 실험 로그는 Git에서 제외합니다. 이전 코드와 실험 파일은 로컬 재현을 위해 남아 있을 수 있지만 JETS 패키지에 포함되지 않습니다.
