# Literature-Based Hyperparameter Tuning Guide

이 문서는 기존 `arena-anchor-playbook.md`를 근거로 삼지 않는다. 목적은 보드게임 계열
self-play RL 문헌과 공개 재현 구현에서 반복적으로 다루는 튜닝 축만 기준선으로 삼고,
우리 코드의 커스텀 파라미터는 별도로 분리하는 것이다.

## 결론

현재 선택한 방향과 값은 논문이 보장해주는 것이 아니다. 문헌이 보장하는 것은 특정
숫자 자체가 아니라, 어떤 종류의 파라미터가 AlphaGo Zero / AlphaZero / Gumbel
AlphaZero / KataGo / ELF OpenGo 계열에서 실제 튜닝 대상으로 다뤄졌는지다.

문헌 근거가 강한 튜닝 축은 다음이다.

- learning rate 및 learning rate schedule
- batch size, optimizer, momentum, weight decay
- update pressure: 새 self-play 데이터 대비 학습 step, minibatch, epoch, reuse 비율
- replay buffer 크기, 최근성, moving window, 최소 replay 양
- MCTS/Gumbel search budget: simulations, sampled actions, cvisit/cscale/cpuct
- exploration: temperature, Dirichlet/Gumbel noise, opening randomness
- playout cap randomization 또는 search budget randomization
- checkpoint gating, candidate selection, weight averaging 류의 안정화 장치

문헌 근거가 약하거나 구현 커스텀으로 봐야 하는 축은 다음이다.

- `policy_target_c_scale`을 acting/search의 `gumbel_c_scale`과 별도로 두고 낮추는 전략
- 특정 arena anchor 결과를 기준으로 장기 학습 안정성을 확정하는 전략
- 단기 arena 승률만으로 장기 self-play 분포 안정성을 판단하는 전략

따라서 불안정성이 보이면 먼저 문헌 근거가 있는 `learning_rate`, LR schedule,
`train_reuse_factor`에 해당하는 update pressure, replay freshness/window, search budget,
exploration 쪽을 조절한다. `policy_target_c_scale`을 `0.25`, `0.1`, `0.025` 같은 값으로
낮추는 것은 논문 기반 기본 처방이 아니라 별도 커스텀 실험으로만 취급한다.

## 현재 train-v4 기준선

현재 설정은 논문이 증명한 최적값이 아니라, 문헌에서 반복적으로 등장하는 축에 맞춘
새 기준선이다.

- `work_dir`: `data/runpod/train-v4`
- `learning_rate`: `0.0125`
- `train_reuse_factor`: `8.0`
- `gumbel_c_visit`: `50.0`
- `gumbel_c_scale`: `1.0`
- `policy_target_c_visit`: `50.0`
- `policy_target_c_scale`: `1.0`
- `ema_decay`: `0.999`

`gumbel_c_visit=50`, `gumbel_c_scale=1.0`은 Gumbel AlphaZero/MuZero 문헌의 보드게임
실험값과 같은 방향이다. 반면 `policy_target_c_scale=1.0`은 "별도 target scale을 낮춰야
한다"는 근거가 있어서가 아니라, search Q signal을 별도 축으로 약화하지 않는 기준선으로
둔 것이다.

## 문헌 기준 튜닝 축

### 1. Learning rate

가장 먼저 조정할 축이다. AlphaGo Zero, AlphaZero, ELF OpenGo, KataGo 모두 optimizer,
LR, LR schedule을 명시적인 학습 하이퍼파라미터로 둔다.

- AlphaGo Zero: SGD momentum, LR annealing, L2 regularization을 사용한다.
- AlphaZero: mini-batch 4096, 대규모 self-play와 병렬 learner를 쓰며 LR schedule을 둔다.
- ELF OpenGo supplement: AGZ/AZ/ELF의 LR 범위를 표로 비교한다. ELF는 `1e-2`,
  `1e-3`, `1e-4` 계열을 사용했다.
- KataGo: batch 256, per-sample LR, early instability를 줄이기 위한 초기 LR downscale,
  후반 final tuning을 위한 LR drop을 명시한다.

실무 판단:

- arena가 불안정하거나 새 checkpoint가 자주 후퇴하면 LR을 낮추는 것이 문헌 기반의
  1차 조정이다.
- LR을 내릴 때는 `train_reuse_factor`와 같이 봐야 한다. 같은 데이터에 더 많이 업데이트하면
  실질 update pressure가 올라간다.
- `0.0125`는 ELF/AGZ 계열의 `1e-2` 근처에 있는 값이다. 단, 우리 batch size와 모델/게임이
  다르므로 논문 보장값은 아니다.

### 2. Update pressure / train reuse

`train_reuse_factor`라는 이름은 논문 표준 용어가 아니다. 하지만 같은 개념권인
"새 self-play 데이터 대비 몇 번 학습하느냐"는 문헌과 구현에서 반복적으로 등장한다.

관련 표현:

- training steps / minibatches per generated self-play data
- epochs per self-play iteration
- self-play games to training minibatches ratio
- replay ratio 또는 update-to-data ratio

ELF OpenGo supplement는 비동기 AlphaZero 방식으로 바꾸면서 self-play와 training
minibatch의 비율이 바뀌고, 이것이 overfitting 방지에 도움이 되었다고 설명한다. Small
games AlphaZero hyperparameter 분석은 epoch, game episodes, MCTS simulations, outer
self-play iterations가 서로 얽혀 있으며, inner-loop를 무작정 키우는 것이 항상 좋지 않다고
보고한다.

우리 continuous learner 기준:

- 한 learner cycle은 "8 actors = 8 shards"가 아니다.
- learner는 cycle 시작 시점에 완료되어 pending 상태인 shard 전부를 가져온다.
- imported transitions에 `train_reuse_factor`를 곱해 sample budget을 만들고,
  `floor(budget / batch_size)`만큼 train step을 돈다.
- shard 하나가 1.7k-2k transitions라면 reuse 8은 shard 하나당 약 53-62 steps다.
- 8 shards가 한꺼번에 들어오면 reuse 8은 약 425-500 steps로 현재 `steps=512` cap에 가깝다.

실무 판단:

- reuse 8은 공격적인 편이지만, 현재 actor 8개와 shard 크기를 고려하면 시작 기준선으로는
  말이 된다.
- reuse 16은 8 shards가 한 번에 들어올 때 거의 항상 cap/backlog를 만들 수 있으므로 더
  공격적이다.
- 불안정하면 `learning_rate`와 함께 `train_reuse_factor`를 먼저 낮춘다.

### 3. Replay buffer / freshness

AlphaGo Zero는 최근 self-play game window에서 샘플링한다. ELF OpenGo도 큰 replay
buffer와 최소 queue 크기를 둔다. KataGo는 moving window를 두고 학습 진행에 따라 window를
키운다.

실무 판단:

- replay가 너무 작으면 새 shard 몇 개에 과적합하기 쉽다.
- replay가 너무 오래되면 최신 policy 개선이 target에 늦게 반영된다.
- 새 workdir에서는 `min_replay_transitions`를 넘기기 전까지 학습이 시작되지 않는 점을
  감안해야 한다.

### 4. Search budget and search scale

AlphaGo Zero/AlphaZero 계열은 MCTS simulations, cpuct/search constant, exploration
noise를 중요한 하이퍼파라미터로 둔다. Gumbel AlphaZero/MuZero는 root action sampling,
Sequential Halving, `cvisit`, `cscale`, simulations 수를 명시적으로 다룬다.

Gumbel paper에서 보드게임 Go/chess 실험은 normalized Q-values에 대해 `cvisit=50`,
`cscale=1.0`을 사용한다. Atari에서는 reward scale과 부분관측성 때문에 `cscale=0.1`도
언급되지만, 이것을 보드게임 policy target scale을 낮추는 근거로 가져오면 안 된다.

실무 판단:

- `gumbel_c_scale=1.0`은 문헌 근거가 있는 보드게임 기준선이다.
- `policy_target_c_scale`은 우리 구현의 분리 축이다. 논문이 "target용 cscale은 낮춰야 한다"고
  보장하지 않는다.
- search Q signal이 너무 강해 보인다는 이유만으로 `policy_target_c_scale`을 낮추는 것은
  문헌 기반 처방이 아니라 커스텀 ablation이다.

### 5. Exploration and playout cap randomization

AlphaGo Zero/AlphaZero는 초반 move sampling temperature와 Dirichlet noise를 사용한다.
Gumbel AlphaZero는 Dirichlet noise 대신 Gumbel sampling without replacement와
Sequential Halving을 사용한다. KataGo는 playout cap randomization을 명시적인 개선으로
제안하고, 일부 move만 full search로 기록해 value/policy 학습의 compute tradeoff를 맞춘다.

실무 판단:

- playout cap randomization은 문헌 근거가 있는 축이다.
- full/fast simulation 비율과 simulations 수는 튜닝 대상이다.
- 불안정할 때는 exploration을 무작정 줄이는 것보다, LR/update pressure/replay 상태를 먼저
  확인한다.

### 6. EMA / weight averaging

AlphaGo Zero/AlphaZero의 핵심 표준 축은 EMA decay가 아니다. 다만 KataGo는 snapshot을
저장하고 일정 snapshot마다 exponential moving average로 candidate net을 만드는 방식의
stochastic weight averaging 류 안정화를 사용한다.

실무 판단:

- `ema_decay=0.999`는 문헌의 핵심 AlphaZero 표준값이라기보다 구현 안정화 장치다.
- EMA는 유지해도 되지만, 불안정성 대응의 1차 손잡이는 LR과 update pressure로 둔다.
- EMA decay를 바꾸려면 별도 실험으로 취급하고, arena뿐 아니라 loss/entropy/KL/value error도
  같이 본다.

## 불안정할 때 조정 순서

1. `learning_rate`를 낮추거나 schedule을 보수적으로 바꾼다.
2. `train_reuse_factor`를 낮춰 새 데이터 대비 update pressure를 줄인다.
3. replay 최소량, moving window, recent sampling/freshness를 확인한다.
4. search budget과 playout cap randomization을 조정한다.
5. exploration noise/temperature를 확인한다.
6. 그래도 설명이 안 되면 `policy_target_c_scale` 같은 커스텀 축을 ablation으로 실험한다.

이 순서를 뒤집어 `policy_target_c_scale`부터 낮추면, 논문에서 자주 쓰이는 축을 확인하기 전에
우리 구현 고유의 완충 장치를 먼저 건드리는 셈이 된다.

## 관측 지표

단기 arena 하나만으로 결론을 내리지 않는다. 최소한 다음을 같이 본다.

- 최신 checkpoint vs 이전 checkpoint arena
- fixed opponent / anchor arena
- policy entropy
- policy target KL 또는 cross entropy
- value loss 및 value calibration
- legal move mass / illegal masking 이상 여부
- replay age distribution
- shard별 color/side/result 분포
- update steps per imported transition

중요한 판단 기준:

- "일시적으로 arena가 좋아졌다"와 "장기 self-play가 안정적으로 강해진다"는 같은 말이 아니다.
- "논문에서 다루는 축이다"와 "현재 값이 증명됐다"도 같은 말이 아니다.
- train-v4의 목적은 문헌에서 반복적으로 쓰인 축으로 기준선을 다시 잡고, 커스텀 target-scale
  실험을 뒤로 미루는 것이다.

## References

- Silver et al., 2017, "Mastering the game of Go without human knowledge"
  - https://www.nature.com/articles/nature24270
  - PDF mirror used for line lookup: https://gwern.net/doc/reinforcement-learning/model/alphago/2017-silver.pdf
- Silver et al., 2017, "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
  - https://arxiv.org/abs/1712.01815
- Tian et al., 2019, "ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero"
  - https://proceedings.mlr.press/v97/tian19a.html
  - Supplement: https://proceedings.mlr.press/v97/tian19a/tian19a-supp.pdf
- Wu, 2019, "Accelerating Self-Play Learning in Go"
  - https://arxiv.org/abs/1902.10565
- Danihelka et al., 2022, "Policy Improvement by Planning with Gumbel"
  - https://openreview.net/forum?id=bERaNdoegnO
- Wang et al., 2020, "Analysis of Hyper-Parameters for Small Games: Iterations or Epochs in Self-Play?"
  - https://arxiv.org/abs/2003.05988
