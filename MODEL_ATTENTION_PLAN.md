# Clean ResNet + 1x Self-Attention 추가 계획

## 목표

현재 `strong` 모델을 기준으로, residual CNN backbone 뒤쪽에 self-attention block 1개를 추가한 비교군을 만든다.

핵심 비교는 다음 두 모델로 제한한다.

- `strong_clean`: 모델 구조 trick을 줄인 기준 ResNet
- `strong_attn`: `strong_clean` 뒤쪽에 board self-attention block 1개를 추가한 실험군

SE block은 이번 실험에 포함하지 않는다. attention 효과를 먼저 분리해서 보기 위함이다.

전제: `strong_clean` 자체는 별도 목표로 만든다. 따라서 1차 유료 실험의 질문은 "현재 legacy `strong` 대비 attention이 좋은가"가 아니라, "정리된 ResNet 기준형에 attention block 1개를 추가할 가치가 있는가"로 둔다.

## 현재 모델에서 정리할 부분

대상 파일: `python/great_kingdom_ai/model.py`

현재 `ModelConfig`에는 다음 구조 옵션이 섞여 있다.

- `policy_kernel_size`
- `spatial_value_head`
- `value_spatial_channels`
- `policy_channels`

이 중 `policy_kernel_size`는 제거 우선순위가 높다. policy head에서만 3x3 context를 더 주는 방식이라, backbone/attention 효과와 해석이 섞인다.

`spatial_value_head`는 옵션으로 두기보다 기준 구조로 고정하는 편이 낫다. 보드 게임 value는 전체 공간 배치에 민감하므로 global average pooling value head보다 spatial value head가 더 자연스럽다.

## 제안 구조

공통 backbone:

```text
input [B, C_in, 9, 9]
-> stem Conv3x3-BN-ReLU
-> N x ResidualBlock
-> optional BoardSelfAttentionBlock
-> policy head
-> value head
```

policy head:

```text
Conv1x1-BN-ReLU-Conv1x1 -> 81 board logits
AdaptiveAvgPool2d-Linear -> 1 pass logit
concat -> 82 policy logits
```

value head:

```text
Conv1x1-BN-ReLU
Flatten
Linear-ReLU-Linear-Tanh
```

## Self-Attention block 설계

입력/출력 shape는 동일하게 유지한다.

```text
[B, C, 9, 9] -> [B, C, 9, 9]
```

내부 구조:

```text
x
-> flatten board cells: [B, 81, C]
-> LayerNorm
-> qkv projection
-> attention score: QK^T / sqrt(head_dim)
-> add full 2D relative position bias
-> softmax
-> attention value aggregation
-> output projection
-> residual add
-> LayerNorm
-> small FFN: Linear(C, 4C)-GELU-Linear(4C, C)
-> residual add
-> reshape back to [B, C, 9, 9]
```

권장 초기값:

- `num_heads=4` for `channels=128`
- `ffn_multiplier=4`
- dropout 없음
- attention/FFN residual branch에 작은 LayerScale 적용

dropout은 self-play 학습에서 효과 해석을 흐릴 수 있으므로 초기 실험에서는 넣지 않는다.

`nn.MultiheadAttention`은 사용하지 않는다. PyTorch 기본 모듈에 head별 relative bias를 끼워 넣는 것보다, 9x9 고정 토큰 수를 활용해 custom attention을 직접 구현하는 편이 테스트와 수정이 쉽다.

LayerScale은 attention block이 학습 초반부터 backbone feature를 크게 흔들지 않도록 넣는다.

```text
x = x + attn_scale * attention(norm1(x))
x = x + ffn_scale * ffn(norm2(x))
```

권장 초기값:

```text
attn_scale: 1e-3
ffn_scale: 1e-3
```

output projection zero-init도 가능하지만, 첫 step에서 qkv 쪽 gradient가 막힐 수 있다. 이 실험에서는 작은 residual scale을 쓰고 projection weight는 PyTorch 기본 초기화를 유지한다.

## Full 2D Relative Position Bias

attention block에는 위치 정보가 필요하다. 1차 구현은 learned absolute positional embedding 대신 full 2D relative position bias를 사용한다.

9x9 보드에서 두 칸 사이의 상대 offset은 다음 범위로 고정된다.

```text
dr: -8..8
dc: -8..8
offset count: 17 * 17 = 289
```

각 relative offset마다 head별 bias를 학습한다.

```text
relative_bias_table: nn.Parameter shape [289, num_heads]
relative_index: registered buffer shape [81, 81]
```

forward에서는 `relative_index`로 bias table을 조회해 attention score에 더한다.

```text
bias_table[relative_index] -> [81, 81, num_heads]
permute -> [1, num_heads, 81, 81]
scores += bias
```

Manhattan distance bias는 사용하지 않는다. 구현은 더 단순하지만 `(2, 0)`, `(1, 1)`, `(0, 2)` 같은 서로 다른 board relation을 모두 같은 거리로 묶어 버린다. full 2D relative bias는 비용이 거의 늘지 않으면서 방향과 offset 형태를 보존한다.

## Preset 정리안

초기 구현에서는 기존 preset을 바로 삭제하지 않는다. checkpoint 호환성과 테스트 폭발을 피하기 위해 새 preset만 추가한다.

추가 후보:

```python
"strong_clean": ModelConfig(
    channels=128,
    residual_blocks=10,
    value_hidden=256,
    policy_channels=16,
    attention_blocks=0,
)

"strong_attn": ModelConfig(
    channels=128,
    residual_blocks=10,
    value_hidden=256,
    policy_channels=16,
    attention_blocks=1,
    attention_heads=4,
)
```

이후 arena 결과가 괜찮으면 기존 `large_policy`, `large_plus`, `strong`의 의미를 재정리한다.

## 구현 단계

1. `ModelConfig`에 attention 관련 필드 추가
   - `attention_blocks: int = 0`
   - `attention_heads: int = 4`
   - `attention_ffn_multiplier: int = 4`
   - `attention_residual_scale_init: float = 1e-3`

2. `BoardSelfAttentionBlock` 추가
   - 파일이 커지면 `python/great_kingdom_ai/model_blocks.py`로 분리
   - 1차 구현은 `model.py` 안에 두되, 파일 크기가 부담되면 즉시 분리
   - `nn.MultiheadAttention` 대신 custom scaled dot-product attention 사용

3. `Full2DRelativePositionBias` 추가
   - `relative_bias_table: [289, num_heads]`
   - `relative_index: [81, 81]`는 `register_buffer`로 관리
   - bias table은 zero init
   - shape와 device 이동을 별도 테스트

4. `PolicyValueNetwork`에 attention stage 추가
   - `self.attention = nn.Sequential(...)`
   - forward에서 `features = self.attention(features)` 적용

5. `strong_clean`, `strong_attn` preset 추가

6. 테스트 추가
   - preset 출력 shape 유지
   - attention block 입력/출력 shape 유지
   - relative bias 출력 shape가 `[1, heads, 81, 81]`인지 확인
   - relative index가 `0..288` 범위를 벗어나지 않는지 확인
   - `attn_scale`, `ffn_scale` 초기값이 설정값과 같은지 확인
   - `strong_attn`이 `BoardSelfAttentionBlock`을 포함하는지 확인
   - ONNX export smoke test가 깨지지 않는지 확인

## 검증 계획

로컬 CPU:

```bash
.venv/bin/python -m pytest tests/test_model.py tests/test_onnx_export.py
```

가능하면 smoke:

```bash
.venv/bin/python scripts/run_m8_train_smoke.py
```

Runpod:

- 동일 replay 조건에서 `strong_clean`과 `strong_attn`을 각각 학습
- 같은 train config, 같은 replay snapshot, 같은 seed를 사용
- 같은 arena config로 후보 매트릭스 평가
- 단일 run 승패보다 다음 지표를 같이 본다
  - policy KL
  - value loss
  - arena win rate
  - inference latency
  - self-play throughput
  - ONNX export/runtime 문제

권장 진행 순서:

1. local shape/ONNX test
2. Runpod short smoke
   - 둘 다 같은 작은 step 수로 학습
   - loss NaN, throughput 급락, export 실패 여부만 확인
3. Runpod medium probe
   - 같은 replay snapshot에서 `strong_clean`, `strong_attn` 학습
   - 같은 후보 평가 config로 arena 비교
4. full run
   - medium probe에서 attention이 명확히 나쁘지 않을 때만 진행

가능하면 wall-clock 기준도 같이 기록한다. attention 모델이 update당 약간 강해도 self-play throughput이 크게 떨어지면 전체 파이프라인에서는 손해일 수 있다.

## 성공 기준

`strong_attn`을 유지할 조건:

- 학습이 불안정해지지 않는다.
- ONNX export와 Rust ONNX self-play 경로가 깨지지 않는다.
- 같은 훈련 예산에서 `strong_clean` 대비 arena 결과가 의미 있게 좋거나, 비슷한 기력에서 정책/가치 품질 지표가 안정적으로 좋아진다.
- update throughput 또는 inference latency 손실이 arena 이득을 상쇄할 정도로 크지 않다.

삭제 또는 보류 조건:

- latency 증가가 self-play throughput을 크게 떨어뜨린다.
- arena 결과가 비슷한데 구현 복잡도만 늘어난다.
- ONNX/runtime 호환 이슈가 자주 발생한다.

## 추가 비교군

돈을 아끼는 쪽이면 1차 비교군은 `strong_clean`과 `strong_attn` 두 개로 충분하다.

더 정석적으로 보려면 후속으로 다음 비교군을 둔다.

- `strong_clean_wide`: attention 파라미터 증가분에 맞춘 약간 넓은 CNN
- `strong_se`: SE block만 추가한 CNN

하지만 첫 유료 실험에 셋 이상을 넣으면 비용이 커지고 해석도 늦어진다. 1차 목표는 attention이 "해볼 만한 신호"를 주는지 확인하는 것이다.

## 주의할 점

attention을 넣으면 기존 checkpoint와 shape가 달라진다. 기존 `strong` checkpoint를 그대로 resume하는 실험과 섞으면 안 된다.

첫 실험에서는 SE, absolute positional embedding, dropout을 같이 넣지 않는다. `strong_attn`이 이긴다면 그 다음 단계에서만 조합 실험을 한다.
