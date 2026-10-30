# JETS

**JETS**는 시간과 tabular feature를 가진 개체별 이벤트 시퀀스를 생성하는 joint-embedding 프레임워크입니다. 미래 시간과 feature 표현을 예측해, 이벤트 수·시간·feature를 생성하는 공동 diffusion의 조건으로 사용합니다.

모델의 정보 흐름, teacher의 학습 목적, ON/OFF 대조군과 인과성 관련 한계는 [프레임워크 설명](docs/JETS_FRAMEWORK.md)을 참고하세요.

## 설치

Python 3.10 이상과 PyTorch 실행 환경이 필요합니다. GPU 사용 시 해당 환경에 맞는 PyTorch를 설치한 다음 저장소를 설치합니다.

```bash
pip install -e .
```

실험 코드가 직접 사용하는 NumPy, pandas, SciPy, scikit-learn은 패키지 의존성에 명시되어 있습니다. 테스트는 `pip install -e '.[dev]'` 후 실행할 수 있습니다.

## 프로젝트 구조

```text
jets/                    모델·teacher·시간 표현·전처리
experiments/             준비된 데이터로 학습·생성·평가
configs/jets_retail.json Retail 실험의 기본 설정
tests/                   독립 실행 가능한 모델 테스트
docs/JETS_FRAMEWORK.md   프레임워크와 정보 경계 설명
```

## 실행 전 준비

**이 저장소만 복제해서 원본 실험을 바로 재현할 수는 없습니다.** 실행에는 별도로 준비된 이벤트 window bank, 고객 ID·전처리 상태, teacher checkpoint와 suffix-code cache, 생성기 초기 가중치 또는 학습된 checkpoint가 필요합니다. 원본 데이터, 가중치, 생성 결과와 외부 원본 평가기는 저장소에 포함하지 않습니다. 신뢰할 수 있는 PyTorch 파일만 입력하세요. `torch.load`는 파일을 불러올 때 코드를 실행할 수 있습니다.

각 경로는 명령행에서 명시합니다. 현재 진입점은 **이미 준비된 window bank와 고정 teacher cache를 사용한 생성기 학습**, 학습된 checkpoint의 생성, 외부 평가기 실행을 지원합니다. 원천 데이터에서 bank를 만드는 과정과 teacher 사전학습·cache 구축은 이 명령에 포함되지 않습니다.

```bash
python -m experiments.retail --help
python -m experiments.retail train --help
python -m experiments.retail generate --help
python -m experiments.retail evaluate --help
```

### 생성기 학습

```bash
python -m experiments.retail train \
  --bank /path/to/data.pt --config configs/jets_retail.json \
  --time-teacher /path/to/time_teacher.pt --time-cache /path/to/time_codes.pt \
  --feature-teacher /path/to/feature_teacher.pt --feature-cache /path/to/feature_codes.pt \
  --initial /path/to/generator_initial.pt --output /path/to/new_run --device cuda
```

기본 설정은 feature 표현 예측 손실 ON입니다. `--feature-weight 0`은 같은 구조에서 그 손실만 제거하는 대조군입니다. `--steps`는 명시적으로 학습량을 바꾸는 기능이며, 줄인 예산의 실행을 원본 전체 학습 결과로 표시해서는 안 됩니다. 입력 자료 해시와 새 run의 정보는 출력 경로에 기록됩니다.

### 자유 생성

```bash
python -m experiments.retail generate \
  --bank /path/to/data.pt --checkpoint /path/to/final.pt \
  --preprocessor /path/to/preprocessor.json --parents /path/to/parents.json \
  --output /path/to/new_generation --device cuda --seed 1
```

실제 미래의 이벤트 수나 시간·teacher 코드를 전달하지 않습니다. 출력 경로에 이미 결과 파일이 있으면 덮어쓰지 않습니다.
전처리 파일 옆에 준비 manifest가 있으면 전처리기와 고객 ID의 해시를 확인합니다. manifest가 없다면 데이터와 전처리 상태가 같은 원본에서 만들어졌는지 자동으로 완전히 확인할 수 없으므로 입력 자료의 출처를 별도로 확인해야 합니다.

### 평가

```bash
python -m experiments.retail evaluate \
  --real /path/to/real_trans.csv --synthetic /path/to/child.csv \
  --scorer /path/to/trusted_score.py --dataset online_retail
```

평가기는 저장소에 포함되지 않으며 지정한 Python 파일을 직접 실행합니다. **신뢰할 수 있는 평가기만 사용하세요.** 평가 결과에는 평가기 해시가 포함됩니다.

## 검증과 결과

```bash
python -m pytest -q tests/test_jets_model.py
```

기존 실험 checkpoint와 준비된 자료를 별도로 제공한 환경에서 JETS 생성기·teacher의 결과와 기존 결과가 일치하는지 검증했습니다. 원본 Retail 결과는 학습에 포함된 고객에 대한 평가이며, 학습 seed 1개와 생성 seed 3개의 결과입니다. 새로운 고객에 대한 일반화나 다중 학습 seed의 안정성을 의미하지 않습니다.

## 로컬 산출물

`data/`, `outputs/`, checkpoint 및 실험 로그는 Git에서 제외합니다. 이전 코드와 실험 파일은 로컬 재현을 위해 남아 있을 수 있지만 JETS 패키지에 포함되지 않습니다.
