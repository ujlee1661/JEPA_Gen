# JETS: 미래 표현 예측을 이용한 시간형 Tabular 시퀀스 생성

## 개요

JETS는 고객·계정 등 개체별 이벤트 시퀀스를 생성하는 프레임워크다. 각 시퀀스는 길이가 다를 수 있으며, 이벤트 시간과 여러 tabular feature를 가진다. 목표는 이벤트 수, 시간 배치와 feature가 맺는 관계를 함께 생성하는 것이다.

JETS는 학습 중 실제 미래를 사용하여 **미래 시간 패턴 표현**과 **시간 조건부 feature 표현**을 만든다. Student는 현재 조건, 관측 이력, 노이즈가 섞인 prefix로부터 이 표현을 예측하며, 공동 diffusion 생성기는 이 예측을 condition으로 사용한다. 실제 생성 시에는 teacher에 미래를 제공하지 않는다.

```text
학습할 때만 실제 미래 사용

미래 시간·간격 패턴 ─────────→ 시간 teacher ─────→ z_time
미래 시간·feature ────────────→ feature teacher ──→ z_feat
                                                       │
                                       각각 예측 손실의 목표

현재 조건·과거 이력·noisy prefix ─→ student ─→ 미래 표현 예측
                                                      │
                                  조건으로 변환해 더하기 ↓
noisy 시간·feature ───────────────────────→ 공동 diffusion
                                                      ↓
                                          생성 시간·feature 시퀀스
```

두 표현의 학습 역할은 다르지만 서로 완전히 독립이라는 가정은 없다. 시간과 feature가 실제로 관련될 수 있으므로 생성기는 두 조건과 noisy 시퀀스를 함께 사용한다.

## 이벤트 수와 시간 표현

개체의 생성 구간은 정적 속성, 관측 시작 시각, 길이 `Δ`, 선택적으로 주어진 과거 이력을 조건으로 한다. N개 이벤트의 시간은 N+1개 간격으로 표현한다.

```text
구간 시작 → 첫 이벤트 | 이벤트 간 간격 | 마지막 이벤트 → 구간 끝
```

마지막 간격은 종료까지 남은 시간이다. N은 모델이 고객별로 생성하며 고정하지 않는다. 다음으로 N+1개 간격이 0인지 양수인지 나타내는 **face**를 생성한다. 이 face는 금액의 0 여부가 아니라 시간 간격의 zero/positive 패턴이다. 정확한 0 간격을 통해 동일 시각 이벤트도 표현할 수 있다.

양수 간격은 활성 위치의 centered log-ratio(CLR) 좌표로 변환한다. 역변환은 활성 CLR에 softmax를 적용해 양수 간격의 합이 구간 길이 `Δ`가 되도록 한다. 양의 길이 구간의 face는 최소 한 개의 양수 간격을 가진다. 시간 noise와 예측값 역시 활성 CLR의 zero-sum 부분공간으로 투영된다.

수치형 feature는 학습 데이터에 맞춘 Gaussian 분위수 변환으로 표현하고, 생성 후 역변환한다. 이는 분위수 변환만 사용하는 것이며 특정 VAE 구조를 도입한다는 뜻이 아니다. 역변환은 관측 학습 범위의 경계에서 포화될 수 있다. 범주형과 결측 정보는 해당 토큰 표현에 포함된다.

## 미래 표현을 만드는 두 teacher

### 시간 teacher

시간 teacher encoder는 실제 미래 구간의 CLR 시간 좌표, 유효·face 플래그와 위치를 읽어 64차원 표현을 만든다. 미래 금액 등 feature 값은 읽지 않는다. Encoder는 미래 구간 전체를 볼 수 있으며 causal encoder가 아니다.

Decoder는 이 코드와 실제 미래 N, 정적 속성, 관측 시작·기간 및 위치를 사용해 간격 face를 예측하고, 실제 활성 face를 기준으로 CLR 좌표를 복원한다. 사전학습 목적은 face의 음의 로그 우도와 활성 CLR 좌표의 단위분산 Gaussian 복원 오차다. 이 실제 미래 조건은 teacher 학습에만 쓰인다.

Encoder에 N 숫자를 직접 넣지는 않지만 시퀀스 길이와 마스크로 N을 유추할 수 있다. 따라서 시간 표현이 N과 독립이라고 주장하지 않는다.

### Feature teacher

Feature teacher encoder는 실제 미래의 **시간 좌표와 feature를 함께** 읽어 64차원 표현을 만든다. 이름과 달리 encoder가 feature만 보는 구조는 아니다. 길이와 플래그를 통해 N 관련 정보도 드러날 수 있다.

Decoder는 코드와 실제 미래 시간·N·face, 정적 속성·기간 및 독립적인 전역/지역 Gaussian noise를 조건으로 받아 feature 시퀀스를 확률적으로 복원한다. 정답 feature는 decoder에 직접 제공하지 않는다. 두 복원 표본과 실제 feature 사이의 energy score로 학습한다. 범주형과 마스크의 연속 완화 좌표도 이 점수에 포함되므로 정확한 범주형 likelihood라고 해석하지 않는다.

두 teacher는 각각 사전학습하고 학습 구간에서 코드 평균·표준편차를 교정한 뒤 고정한다. 각 prefix 위치의 목표는 그 이후 **남은 suffix**의 teacher 표현이다. Teacher 복원 검사는 자유 생성의 성능 평가가 아니다.

## Student와 diffusion 조건

과거 이력은 history GRU가, 현재 denoising 단계의 noisy prefix는 별도의 prefix GRU가 처리한다. 시간 및 feature 예측 head는 각각 미래 표현의 평균과 표준편차를 낸다. 학습 시 이 분포와 고정된 teacher 표현 사이에 Gaussian NLL을 적용한다.

현재 결합 설정에서 feature 조건은 아래와 같이 만든다.

```text
z_feature = μ_feature + σ_feature ⊙ ε,  ε ~ N(0, I)
```

하나의 생성 경로마다 독립적인 `ε`를 뽑고 해당 경로의 모든 prefix와 역확산 단계에서 공유한다. 이는 선택한 확률적 조건화 방식이다. 공유 잡음이 개체 고유 표현이나 장기 일관성을 보장하지 않으며, Gaussian 예측과 표본화만으로 불확실성의 보정이나 정확한 posterior를 주장할 수 없다.

Student의 시간 표현 평균·표준편차는 시간 조건 MLP로 전달된다. 시간 표현과 feature 표본·표준편차는 별도의 잔차 MLP에도 들어가고, 이 두 조건은 noisy 토큰 임베딩 및 정적 조건과 더해져 Transformer 입력을 이룬다. 잔차 MLP의 마지막 층은 0으로 초기화된다. Transformer는 시간과 feature를 함께 다루며 각 noise 단계에서 v-prediction을 출력한다.

Teacher 목표가 직접 diffusion 조건으로 들어가는 것은 아니다. Teacher 코드는 생성기 학습의 예측 목표로만 사용하며, 생성기 조건에는 student 예측이 들어간다.

## 학습 목표

생성기 학습 목적은 다음과 같다.

```text
L = L_diffusion_v + L_count_NLL + L_face_NLL
  + (L_time_prefix_NLL + L_time_BOS_NLL) / 2
  + L_feature_prefix_NLL
```

- **Diffusion:** 유효 시간·feature 좌표의 v-prediction 제곱오차다.
- **Count와 face:** 이벤트 수의 음의 로그 우도와 양의 기간에서 간격 패턴의 음의 로그 우도다.
- **시간 예측:** 고정된 시간 teacher 코드에 대한 prefix 및 시작 상태(BOS)의 Gaussian NLL이다.
- **Feature 예측:** 고정된 feature teacher 코드에 대한 prefix Gaussian NLL이다.

기준 설정에서 각 바깥 가중치는 1이고, 시간의 prefix/BOS 항은 평균한다. Teacher들의 사전학습 손실은 위 생성기 손실과 별개다. 금액 0 비율이나 꼬리를 직접 맞추는 손실, 특정 값 전용 분기 또는 생성 후 할당량 보정은 사용하지 않는다.

Feature 예측 손실을 끈 대조군도 feature 분기와 생성 손실은 유지한다. 이는 전체 feature 경로를 제거한 모델이 아니며, 시간 표현 예측 손실의 독립 기여를 검증하는 대조군도 아니다.

## 자유 생성

1. 정적 속성, 시작·기간 및 선택적 과거 이력을 입력한다. 이력이 없는 개체는 빈 이력으로 시작한다.
2. 모델이 이벤트 수 N을 분포에서 샘플링한다.
3. 생성된 N으로 N+1개 간격의 face를 샘플링한다.
4. 시간·feature 좌표를 noise에서 시작해 반복 denoising한다. 각 단계의 noisy prefix를 이용해 student 표현을 다시 예측하고 condition으로 주입한다.
5. 시간 좌표를 간격과 timestamp로, feature 좌표를 원래 값으로 변환한다.

자유 생성에는 실제 미래 N, face, 시간, feature 또는 teacher 표현을 입력하지 않는다. 생성 N의 안전 상한을 넘으면 오류 처리하며 자르거나 재샘플링하지 않는다.

## 해석과 한계

- 위치 `i`의 prefix GRU는 `i`보다 앞선 noisy 토큰을 직접 읽지만, CLR은 전체 구간 간격의 중심을 사용한다. 최종 Transformer도 전체 noisy 시퀀스를 공동 처리하므로 전체 모델을 엄격한 causal 생성기로 설명하지 않는다.
- 시간과 feature를 한 행씩 확정해 피드백하는 순차 rollout이 아니라 전체 시퀀스를 반복 정제한다. 따라서 JEPA가 장기 동역학 또는 인과 효과를 보장한다는 주장은 성립하지 않는다.
- 두 teacher는 실제 미래를 보고 만든 표현의 목표를 제공한다. 학습된 예측 표현이 각 정보에 실제로 얼마나 기여하는지는 맞춘 대조군과 개입 진단이 필요하다.
- 학습에서는 실제 N과 face를 조건으로 사용하지만, 자유 생성에서는 모델이 생성한 N과 face를 사용한다. 두 조건의 차이가 후속 생성 성능에 영향을 줄 수 있다.
- Teacher는 복원 목표로 별도 사전학습 후 고정된다. 특정 EMA teacher 학습법과 동일하다고 가정하지 않는다.

## 실험 범위

현재 기준 실험은 하나의 학습 seed, 고정된 최종 checkpoint, 세 생성 seed로 수행됐다. 수치형 변환과 teacher 교정은 학습 구간에서 적합했다. 평가에 사용된 개체가 학습 모집단에 포함된 경우 해당 결과는 같은 모집단의 생성 적합도이며, 새로운 고객에 대한 일반화로 해석하지 않는다.

코드와 실험 실행의 정확한 경로 및 준비물은 저장소의 README를 참고한다.
