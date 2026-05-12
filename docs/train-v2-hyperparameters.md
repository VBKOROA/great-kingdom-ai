# Train v2 Hyperparameter Specification

이 문서는 v2 학습 경로인 `great-kingdom-train-v2`에서 JSON 설정값이 실제 코드에서
어떤 동작을 하는지 정리한다. 튜닝 권장값이 아니라 코드 기준 명세다.

기준 코드:

- `python/great_kingdom_ai/train_v2_pipeline.py`
- `python/great_kingdom_ai/train.py`
- `python/great_kingdom_ai/rust_onnx_self_play.py`
- `python/great_kingdom_ai/reanalyze.py`
- `python/great_kingdom_ai/search_reanalyze.py`
- `python/great_kingdom_ai/evaluate.py`

기준 설정 파일:

- pipeline: `configs/runpod/train-v2-pipeline.json`
- learner: `configs/runpod/train.json`
- arena: `configs/runpod/arena.json`

## 전체 흐름

한 iteration은 다음 순서로 돈다.

1. `best.pt`를 ONNX로 export한다.
2. ONNX 모델로 Rust self-play를 실행하고 trajectory episode를 생성한다.
3. trajectory replay store에 episode를 추가한다.
4. replay 전체를 reanalyze해서 `targets/targets-*.npz` 학습 snapshot을 만든다.
5. snapshot에서 batch를 샘플링해 PyTorch learner를 학습한다.
6. arena를 실행하거나 생략하고, 조건에 따라 candidate를 `best.pt`로 승격한다.
7. metrics를 기록하고, 설정에 따라 재생성 가능한 artifact를 prune한다.

주의할 점:

- `train_config.steps`는 `train_reuse_factor`가 `null`이 아닐 때 iteration마다 재계산된다.
- self-play의 기준 모델은 `best.pt`다.
- reanalyze와 learner resume의 기준 checkpoint는 `training-latest.pt`가 있으면 그것이고, 없으면 `best.pt`다.
- v2 replay capacity는 sample 수가 아니라 transition 수 기준이다.

## Pipeline Config

`TrainV2PipelineConfig`가 읽는 값이다. JSON에 없는 값은 dataclass 기본값을 쓰고, 알 수 없는
key는 dataclass 생성 시 실패한다. `self_play`는 반드시 JSON object로 지정해야 한다.

| key | 동작 |
| --- | --- |
| `work_dir` | 모든 산출물의 root directory. replay, target snapshot, checkpoint, reports가 이 아래에 저장된다. |
| `iterations` | 이번 실행에서 추가로 수행할 iteration 수. `resume=true`면 기존 metrics line 수 이후부터 이어서 돈다. 0 이하는 거부된다. |
| `replay_capacity` | trajectory replay가 보관할 최대 transition 수. 초과 시 가장 오래된 episode 단위로 제거된다. 0 이하는 거부된다. |
| `self_play_games` | iteration마다 최소 생성할 self-play game 수. `min_replay_transitions` 조건을 만족하지 못하면 더 생성할 수 있다. 음수는 거부된다. |
| `min_replay_transitions` | iteration 안에서 새로 생성한 transition의 최소 개수. 기존 replay 전체 크기가 아니라 이번 self-play batch의 transition 수를 본다. |
| `max_self_play_games` | `min_replay_transitions`를 맞추기 위해 self-play를 추가 생성할 때의 상한. `self_play_games`보다 작으면 거부된다. |
| `seed_start` | 새 run의 self-play game seed 시작값. `resume=true`면 `replay/game_logs.jsonl`의 최대 seed 다음 값부터 이어간다. |
| `onnx_device` | Rust ONNX self-play evaluator device. `"cpu"` 또는 `"cuda"`를 사용한다. |
| `onnx_max_batch_size` | Rust ONNX evaluator의 최대 inference batch size. self-play search의 leaf/root 요청이 이 한도를 넘으면 evaluator 내부에서 chunking된다. 0 이하는 거부된다. |
| `rust_self_play_batch_size` | Python wrapper가 한 번에 Rust `GumbelSelfPlayBatch`로 실행하는 game 수. self-play loop의 chunk size이기도 하다. 0 이하는 거부된다. |
| `skip_arena` | `true`면 arena 평가를 실행하지 않는다. 단, `always_promote=true`면 arena 없이 candidate를 승격한다. |
| `promote` | arena를 실행했을 때 candidate 승률이 threshold 이상이면 `best.pt`로 복사할지 결정한다. `false`여도 arena 자체는 돈다. |
| `always_promote` | `true`면 arena를 실행하지 않고 매 iteration candidate를 무조건 `best.pt`로 복사한다. |
| `resume` | `true`면 기존 trajectory replay와 metrics를 로드하고 seed도 이어간다. `false`면 새 replay를 만들고 `seed_start`부터 시작한다. |
| `train_checkpoint_mode` | `"resume"`이면 optimizer/scheduler/step까지 checkpoint에서 이어간다. `"bootstrap"`이면 model weight만 가져오고 optimizer/scheduler/step은 새로 만든다. |
| `train_reuse_factor` | 새 transition 1개당 learner가 몇 sample만큼 재사용할지 정한다. `steps = ceil(new_transitions * train_reuse_factor / train_config.batch_size)`. `null`이면 `train_config.steps`를 그대로 쓴다. |
| `min_train_steps` | `train_reuse_factor`로 계산된 step의 하한. `train_reuse_factor=null`이면 적용되지 않는다. 0 이하는 거부된다. |
| `max_train_steps` | `train_reuse_factor`로 계산된 step의 상한. `null`이면 상한이 없다. `min_train_steps`보다 작으면 거부된다. |
| `reanalyze_batch_size` | reanalyze에서 PyTorch 모델로 replay feature를 평가할 때 쓰는 batch size. target snapshot 샘플링 batch가 아니다. |
| `reanalyze_device` | reanalyze 평가 device. `null`이면 learner `device`를 따른다. |
| `bootstrap_td_steps` | value target 생성 시 몇 step 뒤 network value를 bootstrap할지 정한다. 0이면 모든 row가 최종 승패 value target을 쓴다. |
| `gamma` | bootstrap value에 곱하는 discount. target은 `(gamma ** bootstrap_td_steps) * bootstrap_value`다. `[0, 1]` 밖은 거부된다. |
| `search_reanalyze_fraction` | target snapshot row 중 search로 policy target을 다시 만들 비율. `ceil(row_count * fraction)`개를 고른다. `[0, 1]` 밖은 거부된다. |
| `search_reanalyze_budget` | search reanalyze row 수 상한. `fraction > 0`이면 `min(fraction_count, budget)`, `fraction == 0`이면 budget 단독으로 개수를 정한다. |
| `search_reanalyze_simulations` | search reanalyze에 쓰는 Gumbel simulation 수. self-play simulation과 별개다. |
| `search_reanalyze_max_considered_actions` | search reanalyze에서 root 후보로 고려할 최대 action 수. |
| `search_reanalyze_leaf_batch_size` | search reanalyze leaf evaluation batch size. |
| `search_reanalyze_root_batch_size` | search reanalyze가 한 chunk에서 재구성해 search하는 root state 수. |
| `search_reanalyze_seed` | search reanalyze Rust search seed의 base 값. 실제 batch seed는 `seed + min(selected_row_index_in_chunk)`로 만든다. |
| `prune_artifacts` | iteration 마지막에 재생성 가능한 target/candidate/ONNX artifact를 삭제할지 결정한다. |
| `prune_keep_targets` | prune 시 유지할 최신 `targets-*.npz` 개수. 음수는 거부된다. |
| `prune_keep_candidates` | prune 시 유지할 최신 `candidate-*.pt` 개수. 음수는 거부된다. |
| `prune_keep_onnx` | prune 시 유지할 최신 `best-*.onnx` 개수. 음수는 거부된다. |

## Self-play Config

`pipeline.self_play`는 `SelfPlayConfig`로 읽힌 뒤 Rust ONNX self-play에 전달된다.

| key | 동작 |
| --- | --- |
| `max_turns` | self-play 한 game의 최대 turn 수. 이 안에 terminal outcome이 나오지 않으면 runtime error가 난다. |
| `gumbel_simulations` | 기본 Gumbel search simulation 수. `playout_cap_randomization=true`이면 각 turn에서 full/fast simulation 값으로 덮어쓴다. |
| `gumbel_max_considered_actions` | 기본 root 후보 action 수. playout cap의 full/fast max action 값이 `null`이면 fallback으로 쓰인다. |
| `gumbel_c_visit` | Rust Gumbel search 생성자에 전달되는 visit scaling 상수. 0 이하는 거부된다. |
| `gumbel_c_scale` | Rust Gumbel search 생성자에 전달되는 score scaling 상수. 0 이하는 거부된다. |
| `policy_target_c_visit` | search 결과의 policy target 분포를 만들 때 쓰는 visit scaling 상수. self-play와 search reanalyze에 전달된다. |
| `policy_target_c_scale` | search 결과의 policy target 분포를 만들 때 쓰는 score scaling 상수. self-play와 search reanalyze에 전달된다. |
| `policy_target_temperature` | Rust search가 반환하는 policy target의 temperature. 0 이하는 거부된다. |
| `gumbel_seed` | Rust Gumbel self-play seed base. 실제 seed는 `gumbel_seed + game_seed_start` 형태로 batch마다 전달된다. |
| `temperature_turns` | v2 Rust ONNX self-play path에서는 사용되지 않는다. `SelfPlayConfig`에는 남아 있어 validation만 받는다. |
| `sampling_temperature` | v2 Rust ONNX self-play path에서는 사용되지 않는다. `SelfPlayConfig`에는 남아 있어 validation만 받는다. |
| `playout_cap_randomization` | `true`면 각 active game turn마다 game seed 기반 RNG로 full 또는 fast search를 고른다. v2 trajectory replay에는 full/fast와 무관하게 모든 turn transition이 저장된다. |
| `playout_cap_full_search_fraction` | playout cap이 켜졌을 때 full search를 선택할 확률. `(0, 1]` 밖은 거부된다. |
| `playout_cap_full_simulations` | full search turn에서 사용할 simulation 수. |
| `playout_cap_fast_simulations` | fast search turn에서 사용할 simulation 수. |
| `playout_cap_full_max_considered_actions` | full search turn의 후보 action 수. `null`이면 `gumbel_max_considered_actions`를 쓴다. |
| `playout_cap_fast_max_considered_actions` | fast search turn의 후보 action 수. `null`이면 `gumbel_max_considered_actions`를 쓴다. |
| `leaf_batch_size` | self-play search 중 leaf node ONNX evaluation batch size. 너무 작으면 Python/Rust 호출과 ONNX overhead가 늘고, 너무 크면 GPU memory 사용이 늘 수 있다. |

## Reanalyze Target

reanalyze는 trajectory replay 전체를 학습용 snapshot으로 변환한다.

| 설정 | 코드 동작 |
| --- | --- |
| `reanalyze_batch_size` | `_evaluate_policy_logits_values()`가 replay feature를 이 크기씩 잘라 모델 logits/value를 계산한다. |
| `bootstrap_td_steps` | `td_steps == 0`, terminal transition, 또는 bootstrap 위치가 terminal 직전 이후이면 최종 winner target을 쓴다. 그 외에는 `td_steps` 뒤의 network value를 사용한다. |
| `gamma` | bootstrap value에만 적용된다. 최종 winner target에는 discount가 붙지 않는다. |
| `search_reanalyze_*` | 일부 row의 policy target만 Rust search 결과로 교체한다. value target 계산 자체는 위 bootstrap 규칙을 따른다. |

search reanalyze row 선택은 `search_reanalyze.py` 기준으로 다음 점수를 쓴다.

- 기본 priority 점수 1.0
- value target과 network value 차이
- policy target과 current policy logits의 KL
- target age
- opening bonus: timestep이 40 이하인 row에 기본 `0.25`

v2 pipeline JSON으로 직접 노출되지 않은 search reanalyze 내부 기본값도 있다.

| 내부값 | 기본값 | 동작 |
| --- | --- | --- |
| `c_visit` | `50.0` | search reanalyze Rust search의 Gumbel visit 상수. |
| `c_scale` | `1.0` | search reanalyze Rust search의 Gumbel scale 상수. |
| `policy_target_temperature` | `1.0` | search reanalyze policy target temperature. |
| `value_error_weight` | `1.0` | search reanalyze row 선택 priority에서 value error 항의 가중치. |
| `policy_kl_weight` | `1.0` | search reanalyze row 선택 priority에서 policy KL 항의 가중치. |
| `target_age_weight` | `0.25` | search reanalyze row 선택 priority에서 target age 항의 가중치. |
| `opening_weight` | `0.25` | 초반 timestep row에 추가하는 점수. |
| `opening_max_timestep` | `40` | opening bonus를 주는 최대 timestep. |
| `max_priority` | `64.0` | row 선택 priority score cap. |

## Learner Config

`TrainingConfig`가 읽는 값이다. v2에서는 `targets/latest.npz` 같은 reanalyze target snapshot을
`ReplayDataset`으로 보고 학습한다.

| key | 동작 |
| --- | --- |
| `batch_size` | 한 gradient step에서 샘플링할 row 수. replay snapshot 길이보다 크면 학습이 실패한다. |
| `steps` | 학습 step 수. 단, pipeline `train_reuse_factor`가 `null`이 아니면 iteration마다 재계산된 값으로 덮인다. |
| `learning_rate` | AdamW optimizer의 base learning rate. 새 train state 또는 bootstrap mode에서 직접 적용된다. resume mode에서는 checkpoint의 optimizer/scheduler state도 함께 복원된다. |
| `weight_decay` | AdamW optimizer의 weight decay. resume mode에서는 checkpoint optimizer state의 영향을 받는다. |
| `value_loss_weight` | total loss에서 value MSE 항에 곱하는 계수. |
| `policy_loss_weight` | total loss에서 policy cross entropy 항에 곱하는 계수. |
| `l2_loss_weight` | 모델의 모든 trainable parameter 제곱합에 곱하는 추가 L2 regularization 계수. AdamW `weight_decay`와 별개다. |
| `lr_schedule` | `"step"`, `"constant_with_warmup"`, `"warmup_cosine"` 중 하나. 그 외 값은 거부된다. |
| `lr_decay_gamma` | `lr_schedule="step"`일 때 `StepLR`의 gamma. |
| `lr_decay_steps` | `lr_schedule="step"`일 때 몇 step마다 decay할지 정한다. 0 이하는 거부된다. |
| `lr_warmup_steps` | warmup scheduler에서 learning rate를 선형으로 올리는 step 수. |
| `lr_min_factor` | `warmup_cosine`에서 cosine decay가 도달할 최저 learning-rate factor. `[0, 1]` 밖은 거부된다. |
| `lr_cosine_steps` | `warmup_cosine`의 전체 decay 길이. 0이면 `steps`를 쓴다. |
| `seed` | `torch.manual_seed()`와 Python `random.Random` sampling seed로 쓰인다. batch sampling과 augmentation 재현성에 영향을 준다. |
| `device` | PyTorch learner device. CLI `--device`를 주면 train config와 arena config의 device가 모두 덮인다. |
| `model_preset` | 새 checkpoint를 만들 때 사용할 모델 크기. `small`, `medium`, `medium_plus`, `strong`, `large` 중 하나. checkpoint를 resume하면 checkpoint 안의 model config로 모델을 만든다. |
| `symmetry_augmentation` | `true`면 batch sampling 후 board symmetry augmentation을 랜덤 적용한다. feature, policy, legal mask가 같이 변환된다. |
| `mask_policy_loss` | `true`면 illegal action logit을 `-inf`에 가깝게 mask한 뒤 policy loss를 계산하고, target policy가 illegal action에 mass를 주면 error를 낸다. |
| `amp` | CUDA 사용 가능하고 device가 cuda일 때 `torch.amp.autocast`와 `GradScaler`를 쓴다. CPU에서는 켜도 비활성화된다. |
| `recent_sample_fraction` | batch 중 최신 window에서 뽑을 비율. 0이면 recency bias가 꺼진다. priority sampling과 같이 쓸 수 있다. |
| `recent_sample_window` | 최신 row로 볼 window 크기. `recent_sample_fraction > 0`이면 양수여야 한다. |
| `priority_enabled` | `true`면 snapshot priority score로 biased sampling을 한다. snapshot이 priority-aware sampling을 지원하지 않으면 실패한다. |
| `priority_alpha` | priority score에 적용하는 exponent. 0이면 priority 차이가 사라진다. |
| `priority_beta` | importance sampling correction 강도. 0이면 보정 없음, 1이면 가장 강한 보정. |
| `priority_value_error_weight` | priority score에서 value target과 refreshed value 차이에 곱하는 가중치. |
| `priority_policy_kl_weight` | priority score에서 target policy와 current policy logits KL에 곱하는 가중치. |
| `priority_target_age_weight` | priority score에서 target age 항에 곱하는 가중치. |
| `priority_search_reanalyzed_boost` | search reanalyze된 row의 priority score 배수. 1 이상이어야 한다. |
| `priority_max_priority` | priority score cap. `null`이면 cap이 없다. 값이 있으면 1보다 커야 한다. |
| `prefetch_batches` | CUDA 학습에서 CPU batch 준비를 background prefetch할 최대 batch 수. 0이면 prefetch를 끈다. 음수는 거부된다. |

loss 계산은 다음과 같다.

```text
policy_loss = weighted cross entropy(target_policy, model_policy)
value_loss = weighted mse(model_value, target_value)
regularization = l2_loss_weight * sum(parameter ** 2)
total = policy_loss_weight * policy_loss + value_loss_weight * value_loss + regularization
```

sample weight는 reanalyze snapshot의 `sample_weights`와 priority sampling의 importance weight를
곱한 값이다.

## Model Presets

`model_preset`은 `model.py`의 `MODEL_PRESETS`를 따른다.

| preset | channels | residual blocks | value hidden | policy channels | 기타 |
| --- | ---: | ---: | ---: | ---: | --- |
| `small` | 32 | 2 | 64 | 2 | 기본 policy kernel 1 |
| `medium` | 64 | 4 | 128 | 2 | 기본 policy kernel 1 |
| `medium_plus` | 96 | 6 | 192 | 16 | 기본 policy kernel 1 |
| `strong` | 128 | 10 | 256 | 32 | policy kernel 3, spatial value head |
| `large` | 128 | 8 | 256 | 2 | 기본 policy kernel 1 |

## Arena Config

arena는 candidate와 current best를 대국시킨 뒤 promotion 여부를 판단한다. v2 pipeline에서
arena seed는 iteration마다 `arena.seed_start + (iteration - 1) * arena.games`로 offset된다.

| key | 동작 |
| --- | --- |
| `games` | arena game 수. 0이면 승률은 0으로 요약된다. 음수는 거부된다. |
| `batch_size` | 동시에 진행할 arena game 수. 1이면 순차 path, 2 이상이면 batched arena path를 쓴다. |
| `seed_start` | arena 첫 game seed. pipeline에서 iteration offset이 더해진다. |
| `max_turns` | arena game 최대 turn 수. 초과하면 runtime error. |
| `gumbel_simulations` | arena search simulation 수. self-play 설정과 별개다. |
| `gumbel_max_considered_actions` | arena search의 root 후보 action 수. |
| `gumbel_c_visit` | arena Rust Gumbel search의 visit scaling 상수. |
| `gumbel_c_scale` | arena Rust Gumbel search의 score scaling 상수. |
| `policy_target_c_visit` | arena Rust search 생성자에 전달되는 policy target visit 상수. arena의 최종 action은 deterministic하게 고르지만 search 내부 result 생성에 들어간다. |
| `policy_target_c_scale` | arena Rust search 생성자에 전달되는 policy target scale 상수. |
| `policy_target_temperature` | arena Rust search 생성자에 전달되는 policy target temperature. |
| `gumbel_seed` | arena search seed base. 순차 arena는 player별 seed offset을 더하고, batched arena는 batch 생성자에 그대로 전달한다. |
| `leaf_batch_size` | arena search leaf evaluation batch size. |
| `device` | candidate/best PyTorch 모델 평가 device. |
| `promotion_threshold` | `candidate_win_rate >= promotion_threshold`이면 report summary의 `promoted`가 true가 된다. pipeline `promote=true`일 때만 실제 `best.pt` 복사가 일어난다. |

## Checkpoint Mode

v2에서 checkpoint 관련 동작은 다음과 같다.

| mode | 동작 |
| --- | --- |
| `resume` | `training-latest.pt`가 있으면 그것을, 없으면 `best.pt`를 `resume_path`로 넘긴다. optimizer, scheduler, scaler, global step까지 복원한다. |
| `bootstrap` | 같은 source checkpoint에서 model weight만 읽고 optimizer, scheduler, scaler, step은 새로 만든다. |

iteration 학습이 끝나면 candidate checkpoint를 `checkpoints/candidate.pt`와
`checkpoints/training-latest.pt`에 복사한다. 따라서 candidate가 arena에서 승격되지 않아도 다음
iteration의 reanalyze와 learner resume 기준은 최신 learner checkpoint가 된다.

## Promotion Modes

| 설정 조합 | 동작 |
| --- | --- |
| `skip_arena=false`, `always_promote=false`, `promote=true` | arena를 실행하고 threshold 이상이면 `best.pt`로 승격한다. |
| `skip_arena=false`, `always_promote=false`, `promote=false` | arena를 실행하지만 `best.pt`는 바꾸지 않는다. |
| `always_promote=true` | arena를 실행하지 않고 candidate를 무조건 `best.pt`로 복사한다. |
| `skip_arena=true`, `always_promote=false` | arena도 승격도 하지 않는다. 다음 self-play 기준 `best.pt`는 그대로다. |

## CLI Override

`great-kingdom-train-v2` CLI는 일부 값만 override한다.

- `--device`: train/arena device를 덮고, `--onnx-device` 또는 `--reanalyze-device`가 없으면 그쪽도 같은 값으로 덮는다.
- `--work-dir`, `--iterations`, `--self-play-games`, `--min-replay-transitions`,
  `--max-self-play-games`, `--bootstrap-td-steps`, `--train-reuse-factor`,
  `--min-train-steps`, `--max-train-steps`, `--search-reanalyze-fraction`,
  `--search-reanalyze-budget`는 pipeline config를 덮는다.
- `--skip-arena`, `--always-promote`, `--prune-artifacts` 계열도 pipeline config를 덮는다.

CLI로 노출되지 않은 값은 JSON을 바꾸거나 코드를 수정해야 한다.
