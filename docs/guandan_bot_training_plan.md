# Guandan Bot Deep Learning and Reinforcement Learning Plan

## 目标

当前系统已有 `dummy_bot` 和 `llm_agent` 两类 NPC。`dummy_bot` 响应快但策略弱，`llm_agent` 策略表达力更强但延迟高且成本不可控。新的训练目标是构建一个本地推理、低延迟、可持续自我改进的 Guandan bot。

目标能力：

- 单步推理延迟稳定低于线上行动超时，目标 p95 小于 50 ms。
- 只使用本座可见信息决策，不读取对手私有手牌。
- 动作必须由合法动作集合约束，线上非法动作率为 0。
- 强度显著超过 `dummy_bot`，并逐步超过启发式 bot。
- 训练、评测、线上推理共享同一套规则与动作编码。

## 原则

训练内循环不走 HTTP。HTTP、WebSocket、broker 和 external agent 协议用于线上接入与集成验收；训练环境直接复用 `server.domain` 的 reducer、state、commands、hand parser 和 comparator。

原因：

- 强化学习需要大量自博弈步数，进程内调用比 HTTP JSON 往返快得多。
- 训练需要高频 reset 和多环境并行，直接构造 `MatchState` 更简单、可复现。
- actor 只能看本座 snapshot，但 critic、reward 和调试需要访问完整状态。
- reducer 仍然是唯一权威规则入口，可以避免训练环境和线上规则分叉。

## 系统分层

建议新增训练与推理模块：

```text
server/domain/legal_actions.py       # 合法动作枚举，训练和推理共用
training/
  env.py                             # 进程内自博弈环境
  encode.py                          # observation/action 特征编码
  model.py                           # PyTorch 模型
  heuristic.py                       # 强启发式基线
  collect.py                         # 生成行为克隆样本
  bc_train.py                        # 行为克隆训练
  ppo_train.py                       # 自博弈强化学习
  eval.py                            # 固定种子评测
npc/rl_agent/
  player.py                          # 实现 Player.choose_action
  model_loader.py                    # 模型加载与 fallback
  server.py                          # 可选 HTTP policy server
```

## 执行步骤

### 第 1 阶段：合法动作生成器

交付物：

- `server/domain/legal_actions.py`
- `tests/domain/test_legal_actions.py`

要求：

- 支持 `lead`、`play_or_pass`、`tribute`、`return_tribute`。
- `lead` 枚举当前手牌所有可被 `parse_hand` 接受的组合。
- `play_or_pass` 返回 `pass` 和所有能通过 `can_beat` 压过当前 trick 的组合。
- 对同花顺/顺子、红心级牌等可能歧义的组合携带 `declared_type`。
- 贡牌和还贡动作遵守当前 reducer 的约束。
- 测试覆盖单张、对子、炸弹、同花顺压炸弹、pass、贡牌动作。

验收：

- 对每个枚举出的动作调用 reducer 不应产生拒绝。
- 没有轮到行动的座位不产生出牌动作。

### 第 2 阶段：训练环境

交付物：

- `training/env.py`
- `tests/training/test_env.py`

要求：

- 环境直接持有 `MatchState`。
- `reset(seed)` 创建四个训练 controller 并开始牌局。
- `observe(seat)` 返回等价于 `SeatSnapshot` 的 actor 输入。
- `legal_actions(seat)` 复用第 1 阶段生成器。
- `step(action)` 调用 reducer。
- `terminal_result()` 输出 deal/match 结果。
- 支持固定 seed 完全复现。

### 第 3 阶段：编码与启发式基线

交付物：

- `training/encode.py`
- `training/heuristic.py`
- `training/eval.py`

状态特征：

- 本座 54 种牌面计数。
- 当前级牌、双方等级、自己座位、同伴座位。
- 四家剩余牌数。
- 当前 trick 的最后出牌座位、牌型、主 rank、长度、pass 数。
- 已出牌统计。
- finish order、phase、贡牌状态。

候选动作特征：

- 动作类型、牌型、长度、主 rank。
- 是否炸弹、是否同花顺、是否使用红心级牌。
- 出后剩余牌数。
- 是否压同伴、是否压对手。

启发式基线目标：

- 能合理拆牌。
- 不随便浪费炸弹。
- 同伴领先时减少压牌。
- 对手快跑时积极拦截。
- 贡牌选择稳定。

### 第 4 阶段：行为克隆预训练

交付物：

- `training/collect.py`
- `training/bc_train.py`
- 初版模型 checkpoint

流程：

1. 用启发式 bot 自博弈生成样本。
2. 每条样本包含 observation、legal action candidates、chosen action。
3. 训练候选动作打分模型。
4. 与 `dummy_bot` 和启发式 bot 固定种子评测。

建议起步规模：

- 10 万到 50 万个决策样本验证管线。
- 稳定后扩展到 200 万以上样本。

### 第 5 阶段：自博弈强化学习

交付物：

- `training/ppo_train.py`
- 模型池与评测报告

推荐算法：

- 参数共享 actor。
- centralized critic，可访问完整状态。
- actor 执行时只看本座可见信息。
- legal action mask 约束 softmax。
- 对手池混合当前模型、历史模型、启发式 bot 和 `dummy_bot`。

奖励：

- match 胜方 `+1`，负方 `-1`。
- deal 结束按升级数给中间奖励，例如胜方 `+advance_count / 3`。
- 初期可少量 shaping，例如先走完、双下、拦截成功；后期逐步衰减。

### 第 6 阶段：线上接入

交付物：

- `npc/rl_agent/player.py`
- `npc/rl_agent/server.py`
- 集成测试

要求：

- 实现现有 `Player.choose_action(ActionRequest)` 协议。
- 本地生成合法候选动作。
- 模型对候选动作打分，选择最高分。
- 模型异常、候选为空或超时前风险过高时 fallback 到启发式策略。
- 不记录或输出其他座位私有手牌。

## 模型建议

采用候选动作打分模型，而不是固定全局动作空间：

```text
state_encoder(observation) -> h
action_encoder(candidate_i) -> a_i
score_i = MLP([h, a_i, h * a_i])
policy = softmax(score over legal candidates)
value = value_head(h)
```

第一版模型控制在 5M 到 30M 参数。RTX 4070 16GB 足够支撑该规模模型的行为克隆和中等吞吐自博弈。

## 评测指标

- vs `dummy_bot` 胜率。
- vs 启发式 bot 胜率。
- vs 上一版模型胜率。
- 平均升级数。
- 双下率。
- 非法动作率。
- 单步推理 p50/p95/p99。
- 四个座位轮换后的胜率差异。
- 炸弹使用率、pass 率、贡牌错误率。

## 当前启动项

立即启动第 1 阶段：

1. 新增 domain 合法动作生成器。
2. 添加单元测试。
3. 确认枚举动作能被 reducer 接受。
4. 之后再进入训练环境实现。

## 执行进度

- 已新增 `server/domain/legal_actions.py`，提供训练和推理共用的合法动作枚举。
- 已新增 `tests/domain/test_legal_actions.py`，验证枚举动作能被 reducer 接受。
- 已新增 `training/env.py`，提供进程内 `reset`、`observe`、`legal_actions`、`step`、`start_next_deal` 和基础 reward。
- 已新增 `tests/training/test_env.py`，验证固定 seed、私有 snapshot、合法动作 step、deal 奖励和下一局启动。
- 已新增 `training/encode.py`，提供纯 Python observation/action 特征编码。
- 已新增 `training/heuristic.py`，提供行为克隆教师、评测基线和未来推理 fallback 可复用的确定性启发式策略。
- 已新增 `training/eval.py`，提供固定 seed 本地自博弈评测 runner。
- 已新增 `tests/training/test_encode.py` 和 `tests/training/test_heuristic_eval.py`，验证编码隐私、启发式协作行为和一局评测无拒绝。
- 已新增 `training/collect.py`，提供启发式行为克隆样本采集、JSONL 读写和 `guandan-bc-collect` 命令。
- 已新增 `training/model.py` 和 `training/bc_train.py`，提供候选动作打分模型工厂和惰性 PyTorch 行为克隆训练入口 `guandan-bc-train`。
- 已新增 `tests/training/test_collect_bc.py`，验证样本结构、JSONL round trip、采集 CLI 和 BC 维度校验。
- 已新增 `training/rollout.py`，提供按座位 reward 回填的自博弈 rollout 采集和 discounted return 计算。
- 已扩展 `training/model.py`，提供候选动作 actor-critic 模型工厂。
- 已新增 `training/ppo_train.py`，提供惰性 PyTorch self-play PPO 训练入口 `guandan-ppo-train`。
- 已新增 `tests/training/test_rollout_ppo.py`，验证 rollout reward/done 回填、按座位 return 计算和 PPO 参数解析。
- 已将 `training` 加入 `pyproject.toml` 的 setuptools 包列表。

当前可运行的行为克隆命令：

```bash
guandan-bc-collect data/bc/heuristic.jsonl --seed 1 --seed 2 --max-deals 1
guandan-bc-train data/bc/heuristic.jsonl data/models/bc_ranker.pt --epochs 3
guandan-ppo-train data/models/ppo_actor_critic.pt --seed 1 --seed 2 --updates 10 --max-deals 1
```

`guandan-bc-train` 和 `guandan-ppo-train` 需要安装 PyTorch；当前代码使用惰性导入，所以没有安装训练依赖时仍可运行普通单元测试。

GPU smoke run 记录：

- 已通过 `nvidia-smi` 验证 GPU：NVIDIA GeForce RTX 4070 Ti SUPER，16GB 显存级别。
- 已通过 `uv run --extra train` 验证 PyTorch CUDA：`torch 2.12.1+cu130`，`torch.cuda.is_available() == True`。
- 已采集 `data/bc/gpu_smoke.jsonl`：20 局、1723 条决策样本。
- 已完成 CUDA BC 训练：`data/models/gpu_smoke_bc_ranker.pt`，3 epochs，loss `0.5954`，accuracy `0.815`。
- 已完成 CUDA PPO smoke 训练：`data/models/gpu_smoke_ppo_actor_critic.pt`，1 update，250 transitions，loss `0.1569`。
