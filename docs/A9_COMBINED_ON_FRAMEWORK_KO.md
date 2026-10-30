# A9 결합 ON 프레임워크

> 대상: temporal tabular 이벤트 시퀀스의 합성·생성을 연구하는 동료
>
> 기준: 실제 구현된 `A9 + conditional-time teacher + stochastic feature condition + feature JEPA ON`
>
> 문서 작성일: 2026-10-30

## 1. 한 문장으로

A9 결합 ON은 미래 구간의 **시간 패턴 표현**과 **시간 조건부 feature 표현**을 각각 teacher가 만들고, student가 현재 조건과 과거·노이즈가 섞인 prefix에서 이 표현들을 예측해, 고객별 이벤트 수·시간·feature 시퀀스 생성에 활용하는 프레임워크다.

여기서 시간과 feature는 별도의 결과 모델로 독립 생성되지 않는다. 이벤트 수와 간격 패턴을 먼저 생성한 다음 **하나의 Transformer diffusion이 시간과 feature를 공동 denoising**한다.

## 2. 해결하려는 문제

고객 한 명의 데이터는 속성 한 행이 아니라 시간순으로 발생하는 가변 길이의 이벤트 행들이다. 예를 들어 고객마다 구매 횟수가 다르고, 거래 사이의 간격과 구매 금액이 서로 관련될 수 있다. 목표는 이러한 고객별 이벤트 시퀀스의 시간·feature와 관계를 함께 생성하는 것이다.

핵심 연구 질문은 **생성 이전에 미래 구간에 대한 압축된 표현을 예측하도록 학습시키면 시퀀스의 공동 분포를 더 잘 생성할 수 있는가**다. 모델이 실제 미래를 생성 시에 볼 수 있다는 가정은 하지 않는다.

## 3. 전체 구조

```text
                         [학습: 실제 미래 사용]

실제 미래 시간·간격 패턴 ─────→ 시간 teacher ─────→ z_time (64차원)
실제 미래 시간·feature ───────→ feature teacher ──→ z_feat (64차원)
                                                   │
                                    고정된 예측 목표로만 사용

                         [생성기 학습 및 생성]

고객 속성, 관측 시작·기간, 과거 이력
               │
               ├─→ N 생성 ─→ N+1 간격의 zero/positive 패턴 생성
               │
               └─→ noisy 시퀀스와 함께 prefix GRU
                                    │
                     ┌──────────────┴──────────────┐
                     ↓                             ↓
             시간 표현의 평균·표준편차      feature 표현의 평균·표준편차
                     │                    z_sample = μ_f + σ_f ⊙ ε
                     └──────────────┬──────────────┘
                                    ↓
                         결합 조건 + Transformer diffusion
                                    ↓
                       시간 좌표와 feature 좌표를 공동 복원
```

실제 미래를 인코딩한 teacher 코드는 **학습 손실의 목표**다. 자유 생성에서는 실제 미래의 N·간격 패턴·시간·feature·teacher 코드를 공급하지 않는다.

## 4. 데이터와 시간 표현

한 생성 구간은 고객 속성 `a`, 시작 시각 `s`, 길이 `Δ`, 관측된 과거 이력 `H`와 미래 이벤트 `N`개로 표현한다. 원본 timestamp를 순서대로 `t₁,…,t_N`이라 하면 구간에는 다음 **N+1개 간격**이 있다.

```text
시작 → 첫 이벤트 | 이벤트 사이 간격들 | 마지막 이벤트 → 구간 끝
```

마지막 간격은 종료까지의 여유 구간이다. 이 길이들을 고정한다는 뜻이 아니다. N을 모델이 생성한 뒤 N+1 간격의 양수 여부를 모델이 생성한다. 양수 간격만으로 positive-duration 구간의 전체 길이 `Δ`를 나누고, 양수 부분에 중심화된 로그 비율(CLR)을 사용한다. 예측된 CLR은 active 간격에 대한 softmax를 거쳐 `Δ`에 합이 맞도록 역변환한다. 이를 통해 간격은 음수가 아니고 정렬된 이벤트 시간이 얻어진다. 양수 여부 패턴을 **face**라고 부른다.

*Face는 금액의 0 여부가 아니라 시간 간격의 zero/positive 패턴이다.* 시간 간격이 정확히 0인 경우를 표현하므로 동시 이벤트도 가능하다. 양의 길이 구간에서 face는 적어도 하나의 양수 간격을 갖도록 샘플링된다. N=0인 경우에도 마지막 여유 간격 한 개가 있다.

수치형 feature에는 학습 데이터로 적합한 TabSyn식 Gaussian 분위수 변환을 사용한다. 이는 **TabSyn의 변환을 사용한다는 뜻이며 TabSyn VAE 구조를 도입한 것은 아니다.** 범주형·결측 정보도 해당 토큰 표현에 포함된다. 생성 후에는 기존 역분위수 변환으로 원래 값으로 되돌린다. 학습 범위 밖에서 역변환이 관측 지지 범위의 경계에 포화될 수 있다.

## 5. 두 teacher

### 5.1 시간 teacher

시간 teacher의 encoder는 실제 미래 구간의 **시간 CLR, face·유효 마스크, 위치**를 읽고 CLS 기반 64차원 `z_time`을 만든다. 미래 금액 등 feature 값은 encoder에 넣지 않는다. Transformer encoder는 미래 구간 전체를 볼 수 있으며 causal teacher가 아니다.

Teacher decoder는 이 코드와 실제 미래 N, 고객 속성, 시작·기간, 위치를 조건으로 받는다. 간격의 face를 먼저 예측하는 학습 목적과, 실제 active face를 기준으로 CLR 시간 좌표를 복원하는 목적을 사용한다. 구현에서 시간 teacher 사전학습 손실은 **face NLL + active CLR의 단위분산 Gaussian 오차**다. Decoder의 실제 미래 N과 face는 *teacher 학습용 조건*이지 자유 생성 입력이 아니다.

Encoder에 N 숫자를 직접 넣지는 않지만 입력 길이·마스크를 통해 N 정보를 유추할 수 있다. 따라서 `z_time`이 N과 독립이라고 주장하지 않는다.

### 5.2 Feature teacher

Feature teacher encoder는 실제 미래의 **시간 CLR과 feature를 함께** 읽어 64차원 `z_feat`을 만든다. 이름과 달리 encoder가 feature만 보는 것은 아니다. 직접적인 고객 속성·N 스칼라 주입은 없지만 길이·마스크 등을 통해 N 관련 정보가 들어갈 수 있다.

Feature teacher decoder는 코드와 실제 미래 CLR 시간·N·face, 고객 속성과 관측 기간을 조건으로 받고, 독립적인 전역·지역 Gaussian 잡음을 사용해 **feature 시퀀스만** 확률적으로 복원한다. 정답 feature를 decoder 조건으로 직접 전달하지 않는다. 두 독립 복원과 정답 사이의 energy score를 학습 목표로 사용한다. 범주형·마스크는 연속 완화 좌표에서 점수화하므로 이 손실을 원래 범주형 확률의 정확한 likelihood라고 부르지 않는다.

두 teacher는 별도로 학습한 후 학습 구간으로 코드의 평균·표준편차를 교정하고 고정한다. 각 prefix 위치의 목표는 **그 위치 이후 남은 suffix에 teacher를 적용한 코드**다. Feature teacher 복원과 시간 teacher 복원은 자유 생성의 성능 평가가 아니다.

## 6. Student: 미래 표현을 예측하는 방법

Student는 실제 teacher 코드를 생성 입력으로 받지 않는다. 관측된 과거 이력은 history GRU로, 현재 diffusion 단계의 noisy prefix는 별도의 prefix GRU로 처리한다. 각 prefix 위치에서 두 개의 예측 head가 각각 다음을 출력한다.

```text
시간 표현:     μ_time, σ_time
feature 표현:  μ_feat, σ_feat
```

각 예측의 표준편차는 `softplus`와 수치적 하한으로 양수가 되며, 학습 시 고정된 해당 suffix teacher 코드에 대해 Gaussian NLL을 계산한다. 시간 표현에는 prefix 위치뿐 아니라 생성 시작 상태(BOS)의 예측 손실도 있다.

결합 ON에서는 feature 예측을 생성기에 전달할 때 다음과 같이 표본화한다.

```text
z_sample = μ_feat + σ_feat ⊙ ε,  ε ~ N(0, I)
```

고객·생성 경로별로 ε 하나를 뽑아 해당 경로의 모든 prefix와 역확산 단계에 공유한다. 서로 다른 경로는 독립 잡음을 사용한다. 학습과 생성 양쪽에 이 조건화가 적용된다. **이는 탐색적 조건화 선택**이며, ε 공유가 고객의 고유 상태나 장기 일관성을 보장하지 않는다. Gaussian NLL과 이 표본화만으로 정확한 posterior·VAE·calibrated uncertainty를 주장하지 않는다.

### Prefix와 causal이라는 말의 경계

위치 `i`의 prefix GRU 예측은 **noisy 토큰 `i`보다 앞선 토큰**을 사용한다. 그렇지만 teacher는 미래 전체를 보며, CLR 좌표의 기준점은 구간 전체 간격으로 정해진다. 또한 최종 Transformer는 전체 noisy 시퀀스를 공동 처리한다. 따라서 전체 프레임워크를 엄밀한 인과 생성기나 인과 효과 식별 모델이라고 부르지 않는다. 생성된 행을 한 행씩 최종 확정하고 다시 입력하는 순차 rollout 구조도 아니다.

## 7. 결합과 공동 diffusion

Student의 시간 표현 평균·표준편차는 기존 predictive condition 경로로 들어간다. 시간 표현 평균·표준편차, feature 표본과 feature 표준편차를 연결해 학습 가능한 잔차 MLP에도 넣는다. 이 잔차를 기존 조건에 더해 Transformer의 토큰별 hidden state를 조건화한다. 잔차의 최종 층은 0으로 초기화해 초기 상태에서는 추가 잔차가 0이다.

Transformer는 각 timestep에서 시간 CLR과 feature 좌표에 대한 **v-prediction**을 내며, 정해진 100단계 노이즈 스케줄로 학습한다. Retail 평가의 native 생성에는 50단계 역확산을 사용했다. 이 timestep은 예측 지평 4·8·16을 고정한다는 뜻이 아니다.

## 8. 손실 구성

결합 ON의 **생성기 학습**은 다음 목적을 사용한다.

```text
L = L_diffusion_v + L_count_NLL + L_face_NLL
  + (L_time_prefix_NLL + L_time_BOS_NLL) / 2
  + L_feature_prefix_NLL
```

* `L_diffusion_v`: 유효 시간·feature 좌표의 v-prediction 제곱오차. 시간과 feature가 하나의 손실로 합쳐진다.
* `L_count_NLL`: N의 혼합 분포에 대한 음의 로그 우도.
* `L_face_NLL`: 양의 기간에서 N+1 간격의 유효 zero/positive 패턴에 대한 음의 로그 우도.
* `L_time_prefix_NLL`, `L_time_BOS_NLL`: 고정된 시간 teacher 코드에 대한 예측 Gaussian NLL.
* `L_feature_prefix_NLL`: 고정된 feature teacher 코드에 대한 예측 Gaussian NLL. ON에서 가중치 1이다.

결합 **OFF**는 동일 구조와 표본화 조건을 유지하고 마지막 feature 예측 손실만 0으로 둔다. OFF에서도 feature 분기는 생성 손실을 통해 학습된다. 이 실험은 시간 JEPA를 제거한 대조군이 아니다. 금액 0 비율이나 꼬리를 직접 맞추는 손실, 0·양수 분리 규칙, 후처리 할당량은 없다.

두 teacher의 사전학습 손실은 위 생성기 목적과 별개다. Teacher 사전학습에서 사용한 실제 미래 정보와 teacher decoder는 자유 생성 과정에 들어가지 않는다.

## 9. 자유 생성 순서

1. 고객 속성과 관측 시작·기간을 받는다. 주어진 과거 이력이 있다면 인코딩하며, 전체 이력 생성은 빈 과거(BOS)에서 시작한다.
2. 모델이 이산화된 log-normal 혼합 분포에서 고객의 이벤트 수 N을 생성한다. 데이터의 실제 N을 복사하거나 모든 고객의 이벤트 수를 고정하지 않는다.
3. 생성한 N에 맞춰 N+1 간격의 face를 샘플링한다.
4. 유효 시간·feature 좌표에 노이즈를 생성하고 역확산을 수행한다. 각 단계에서 student는 현재 noisy prefix를 읽어 두 표현의 예측값을 만든다.
5. 최종 시간 좌표를 간격과 timestamp로 복원하고, feature를 역분위수 변환해 행을 내보낸다. 생성 중 teacher를 호출하거나 실제 미래 코드를 넣지 않는다.

구현에는 생성 N의 안전 상한 4096이 있다. 이를 초과하는 예측은 해당 실행을 오류 처리하며, 상한으로 자르거나 재샘플링하지 않는다. 이는 N을 고정하거나 학습하지 않는다는 뜻이 아니다.

## 10. 실험 기준과 해석

현재 `COMB_FEATURE_ON`의 Retail 실험은 원본 A9과 같은 seed 9 초기 생성기 텐서에서 시작해 **20,000 업데이트·batch 16**을 사용했다. 시간·feature teacher는 각각 5,000 업데이트로 사전학습했다. 최종 checkpoint 하나에서 생성 seed 1·2·3을 평가했다. 3회 생성은 **독립 학습 3회가 아니다.**

Retail 원본의 417명 고객은 학습에 포함되어 있으므로 현재 결과는 **같은 모집단의 생성 분포 평가**다. 새로운 고객에 대한 held-out 일반화를 주장하지 않는다. 평가에서 모델은 N을 자유 생성하며, 실험 범위와 출처는 `outputs/paper_reproduction/a9_combined_comparison_v1/report.md`를 참조한다.

결합 ON/OFF는 같은 결합 구조에서 feature 예측 손실의 효과를 비교한다. **기존 시간 teacher 예측 손실만 끈 결합 구조 대조군은 아직 없으므로 두 JEPA의 독립 기여가 모두 입증된 것은 아니다.** 또한 기존 A9과 결합 실험은 teacher·조건화 및 난수 흐름이 달라 차이를 어느 한 요소의 단독 효과로 확정할 수 없다.

## 11. 동료에게 설명할 때 주의할 주장

* 두 표현은 **역할이 다르지만 완전히 독립이라는 보장은 없다.** 시간 teacher에도 N이 드러나고, feature teacher도 미래 시간과 feature를 함께 읽는다.
* 생성 시 실제 미래 이력을 모델에 제공하지 않는다. Teacher는 학습에 쓰이는 미래 표현의 목표를 제공한다.
* JEPA식 미래 표현 예측을 사용하지만, 이 teacher들은 미래 복원 목적을 사용해 **사전학습 후 고정**된다. EMA teacher를 사용한 원본 I-JEPA와 동일한 학습 방식이라고 주장하지 않는다.
* μ+σε는 선택한 확률적 조건화이지 필수적이거나 검증된 posterior가 아니다. 기존 생성 시점 개입에서 ε만 제거했을 때 변화는 작았으나, 이는 학습부터 ε를 제거해도 같다는 증거는 아니다.
* 시간과 feature를 공동 생성하지만, 그 관계의 인과 효과를 식별하거나 장기 동역학을 보장한다고 주장하지 않는다. 누적금액의 시간순 일관성도 구조적으로 보장하지 않는다.

## 12. 주요 구현 파일

| 역할 | 파일 |
|---|---|
| 시간 teacher | `pgjepa/a9_conditional_time_teacher.py` |
| Feature teacher 및 encoder | `pgjepa/a7_feature_teacher.py`, `pgjepa/a7_distribution_teacher.py` |
| A9 예측 분기·결합·손실 | `pgjepa/a9_model.py` |
| 확률적 feature 조건 | `pgjepa/a9_generic_model.py` |
| N·face·공동 diffusion·생성 | `pgjepa/a7_model.py` |
| 간격 CLR 표현 | `pgjepa/a7_spacing.py` |
| 분위수 전처리 | `pgjepa/quantile.py`, `pgjepa/a7_data.py` |
| 결합 ON/OFF 학습·평가 | `scripts/run_a9_combined_experiment.py` |
| Retail 비교 결과 | `outputs/paper_reproduction/a9_combined_comparison_v1/report.md` |
