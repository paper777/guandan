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
3. 大规模采集使用 compact gzip JSONL，训练只依赖 feature values、chosen index 和基础分类字段。
4. 按 seed 切分 validation，训练候选动作打分模型，并输出 overall、legal action、chosen kind 和候选数桶指标。
5. 将候选动作打分模型作为 PPO actor 的初始化权重。
6. 与 `dummy_bot` 和启发式 bot 固定种子评测。

建议起步规模：

- 10 万到 50 万个决策样本验证管线。
- 稳定后扩展到 200 万以上样本。

### 第 5 阶段：自博弈强化学习

交付物：

- `training/ppo_train.py`
- 模型池与评测报告

推荐算法：

- 参数共享 actor。
- 首次 PPO 可从第 4 阶段 BC checkpoint warm start；后续训练必须从已有 PPO actor-critic checkpoint 继续，完整加载 actor 和 critic，避免 RL 迭代重新回到启发式教师策略。
- centralized critic，可访问完整状态。
- actor 执行时只看本座可见信息。
- legal action mask 约束 softmax。
- 对手池混合当前模型、历史模型、启发式 bot 和 `dummy_bot`。

奖励：

- match 胜方 `+1`，负方 `-1`。
- deal 结束按升级数给中间奖励，例如胜方 `+advance_count / 3`。
- 终局奖励按开局手牌强度做难度系数调整：大小王、炸弹、高张和更少预估手数代表更强开局牌；强牌赢的奖励略降，弱牌赢的奖励略升。
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
- 已新增 `training/ppo_train.py`，提供惰性 PyTorch self-play PPO 训练入口 `guandan-ppo-train`，支持首次 BC warm start、从已有 PPO actor-critic checkpoint 继续训练、GAE、mini-batch PPO 更新、梯度裁剪、KL early stop 和逐 update 日志。
- 已新增 `tests/training/test_rollout_ppo.py`，验证 rollout reward/done 回填、按座位 return/GAE 计算和 PPO 参数解析。
- 已新增 `npc/rl_agent/player.py` 和 `npc/rl_agent/model_loader.py`，实现现有 `Player.choose_action(ActionRequest)` 协议、本地 snapshot 合法候选动作生成、PPO/BC checkpoint 候选打分和启发式 fallback。
- 已新增 `npc/rl_agent/server.py` 和命令 `guandan-rl-agent-server`，可作为独立 HTTP policy server 运行。
- 已将默认 NPC profile 和 CLI/broker 默认 lineup 切换为 `rl`；没有 checkpoint 或 PyTorch 不可用时默认走启发式 fallback。
- 已新增 `tests/npc_tests/rl_agent/test_player.py`，验证 RL agent fallback、模型候选选择和 ActionRequest 到 SeatSnapshot 的转换。
- 已将 `training` 和 `npc.rl_agent` 加入 `pyproject.toml` 的 setuptools 包列表。

当前可运行的训练与接入命令：

```bash
guandan-bc-collect data/bc/heuristic.compact.jsonl.gz --seed-count 8 --max-deals 1 --workers 4 --compact
guandan-bc-cache data/bc/heuristic.compact.jsonl.gz data/bc/heuristic.bc-cache --shard-size 2048
guandan-bc-train data/bc/heuristic.compact.jsonl.gz data/models/bc_ranker.pt --epochs 3 --validation-fraction 0.1 --cache-dir data/bc/heuristic.bc-cache --batch-size 128
guandan-ppo-train data/models/ppo_actor_critic.next.pt --init-policy data/models/bc_ranker.pt --seed-count 10 --updates 10 --epochs-per-update 3 --max-deals 24 --batch-size 1024 --opponent-pool self,heuristic,previous --rollout-processes 16 --inference-batch-size 16
guandan-ppo-train data/models/ppo_actor_critic.continue.pt --init-policy data/models/ppo_actor_critic.next.pt --seed-count 10 --updates 10 --epochs-per-update 3 --max-deals 24 --batch-size 1024 --opponent-pool self,heuristic,previous --rollout-processes 16 --inference-batch-size 16
guandan-rl-agent-server --model-path data/models/ppo_actor_critic.pt --device cuda
```

`--init-policy` 可接收 BC ranker checkpoint 或 PPO actor-critic checkpoint：传入 BC ranker 时只初始化 PPO actor，传入 PPO actor-critic 时完整恢复 actor 和 critic。持续训练建议写入新的 `.next.pt` checkpoint，固定评测通过后再提升为线上 `data/models/ppo_actor_critic.pt`。

`guandan-bc-train`、`guandan-ppo-train` 和 RL agent 模型推理需要安装 PyTorch；当前代码使用惰性导入，所以没有安装训练依赖时仍可运行普通单元测试，RL agent 会 fallback 到启发式策略。

GPU smoke run 记录：

- 已通过 `nvidia-smi` 验证 GPU：NVIDIA GeForce RTX 4070 Ti SUPER，16GB 显存级别。
- 已通过 `uv run --extra train` 验证 PyTorch CUDA：`torch 2.12.1+cu130`，`torch.cuda.is_available() == True`。
- 已采集 `data/bc/gpu_smoke.jsonl`：20 局、1723 条决策样本。
- 已完成 CUDA BC 训练：`data/models/gpu_smoke_bc_ranker.pt`，3 epochs，loss `0.5954`，accuracy `0.815`。
- 已完成 CUDA PPO smoke 训练：`data/models/gpu_smoke_ppo_actor_critic.pt`，1 update，250 transitions，loss `0.1569`。

## 当前训练代码技术分析（2026-06-21）

本节基于当前 `training/`、`server/domain/legal_actions.py` 和 `npc/rl_agent/` 代码审查，记录已经实现的训练闭环、编码细节、模型结构、训练流水线和后续优化建议。

### 总体架构

当前实现已经形成一条完整的本地 learned bot 流水线：

- `server/domain/legal_actions.py` 是训练和线上推理共用的合法动作候选生成器。训练侧从完整 `MatchState` 生成候选，线上侧从 `SeatSnapshot` 生成候选。
- `training/env.py` 直接复用 domain reducer 创建四个训练 controller、发牌、执行 `ActionCandidate`，并在局结束或比赛结束时生成队伍奖励。
- `training/encode.py` 将 `SeatSnapshot` 和每个合法候选动作编码为固定长度 float 特征。
- `training/heuristic.py` 提供确定性启发式教师，供 BC 采样、评测和线上 fallback 复用。
- `training/collect.py` 生成行为克隆 JSONL/JSONL.GZ 数据集；`training/bc_cache.py` 可将 JSONL 转成 tensor shard cache；`training/bc_train.py` 训练候选动作 ranker。
- `training/rollout.py` 和 `training/ppo_train.py` 基于当前 actor-critic checkpoint 做自博弈 rollout，并用 PPO 更新 actor-critic；启发式策略不参与 PPO 自博弈 actor，只保留为 BC 教师、评测基线和线上 fallback。
- `npc/rl_agent/` 支持加载 PPO actor-critic 或 BC ranker checkpoint；模型不可用、模型异常或行动 deadline 太近时 fallback 到启发式策略，再失败时 fallback 到 `DummyBotPlayer`。

这条链路的核心优点是规则单一来源明确：动作枚举、训练 step 和线上执行最终都被 reducer 约束，非法动作风险主要集中在 snapshot 重建和候选生成一致性上。

### 合法动作生成

合法动作采用候选集合方案，而不是固定全局 action id。`ActionCandidate` 覆盖 `play_cards`、`pass`、`submit_tribute` 和 `return_tribute`，并携带 `declared_type`、`hand_type`、`primary_rank`、`length` 等训练特征。

出牌候选通过组合枚举生成：

- 单张、对子、三张、满堂、顺子、同花顺、三连对、钢板、炸弹、四王。
- 红心级牌作为 wild card 参与多数非级牌组合。
- 每组候选再通过 `parse_hand()` 和 `can_beat()` 校验，确保候选能被当前规则接受。
- `play_or_pass` 总是包含 `pass`，`lead` 不包含 `pass`。
- 贡牌和还贡候选按当前 obligation 或 snapshot eligible card 生成。

当前动作枚举是正确性优先的穷举式实现。实测初始 lead 候选数量会有明显长尾：seed `1` 的第一手有 602 个候选；同一局 84 个决策的候选数分布为 min `1`、median `3`、mean `19.63`、max `602`。这说明大部分决策很轻，但首攻和复杂手牌的候选生成、编码、模型打分会成为训练和线上 p95 延迟的主要来源。

### Observation 编码

`encode_observation(snapshot)` 采用当前 `v2` schema，输出 `195` 维 float 特征，全部来自当前座位可见的 `SeatSnapshot`：

- 座位、phase、当前级牌、current turn、acting seat、legal action、贡牌来源/目标等 one-hot。
- 双方等级归一化到 `[0, 1]`。
- 四家剩余手牌数量，按 `27` 归一化。
- 四家 finish position，未完成为 `0`。
- 本座 54 种牌面计数，按双副牌最大计数 `2` 归一化。
- 当前 trick 的最后出牌座位、牌型、主 rank、长度和 pass 数。
- public 已出牌牌面计数。
- `return_rank_at_most_ten` 贡牌约束标记。

隐私边界是合理的：编码不读取其他座位私有手牌，测试也覆盖了“替换对手手牌不改变本座 observation 编码”的场景。

当前剩余缺口：

- 没有历史 trick 序列或 recurrent state，当前模型是单步静态策略。
- public 已出牌统计只有按牌面聚合的 count，还没有按 trick 序列建模。

### Action 编码

`encode_action(action, snapshot)` 同样采用当前 `v2` schema，输出 `96` 维特征：

- 动作类型 one-hot：`pass`、`play_cards`、`submit_tribute`、`return_tribute`。
- 出牌牌型 one-hot、主 rank one-hot。
- 候选动作 54 种牌面计数，按 `2` 归一化。
- 动作长度、是否有 declared type、是否炸弹类、是否使用红心级牌。
- 出完该动作后本座剩余牌数。
- 是否压同伴、是否压对手、是否能直接走完、对手危险位、相对当前 trick 的 rank margin。
- 是否拆掉炸弹、顺子或连对结构。

当前编码能表达“这手牌是什么”“打完还剩多少”和一组轻量上下文派生特征。剩余缺口：

- 结构代价仍是粗粒度二值标记，没有估算完整最少手数或最佳拆牌路径。
- rank margin 只表达主 rank 差距，没有完整比较炸弹长度、同花顺、四王等高阶强度关系。

这些派生特征不改变规则正确性，但会显著降低小模型从稀疏样本中学习协作和拆牌策略的难度。

### 模型结构

当前模型是候选动作打分，而不是固定动作空间分类。

BC ranker：

```text
state = state_encoder(observation)
action = action_encoder(candidate)
score = MLP([state, action, state * action, abs(state - action)])
softmax(score over legal candidates)
CrossEntropy(chosen_index)
```

PPO actor-critic：

```text
policy_net: same dual-tower candidate scorer
value_net:  MLP(critic_observation -> hidden -> hidden -> 1)
```

训练和线上加载统一使用 `dual_tower_v1`。checkpoint 必须携带 `model_architecture=dual_tower_v1`、当前 `encoding_schema` 和匹配的 schema hash；不再保留旧 concat MLP 或无 manifest checkpoint 的兼容路径。默认 `hidden_dim=256` 时，模型规模仍偏轻量，适合低延迟 smoke run 和线上 fallback 验证，但仍明显小于最初建议的 5M 到 30M 参数区间。

当前仍保留的差异：

- actor/critic 仍是浅层 MLP，没有注意力、recurrent state 或更大参数量。
- centralized critic 目前使用固定手写全状态特征，还没有学习式 public/private state encoder。
- opponent pool 已接入训练入口，但历史 checkpoint 的采样比例和晋级阈值还需要长期评测数据校准。

这些差异让第一版实现更简单、更容易上线，但 PPO 方差和策略泛化能力会受到限制。

### 行为克隆流水线

BC 数据采集流程：

1. `collect_heuristic_samples()` 按 seed 启动 `GuandanTrainingEnv`。
2. 当前座位用 `HeuristicPolicy` 从合法候选中选择动作。
3. 每条 `BcSample` 记录 seed、deal_id、event_seq、seat、legal_action、observation values、candidate values、chosen_index 和可选 debug payload。
4. `--compact` 模式只保留训练必要 feature values、chosen index 和 chosen kind，适合大规模 gzip JSONL。
5. `--workers` 按 seed 并行采集，先写临时 shard，再按 seed index 合并，保证输出顺序稳定。

BC 训练流程：

1. 可直接流式读 JSONL，也可先用 `guandan-bc-cache` 建 tensor shard cache。
2. validation 按完整 seed 切分，避免同一局样本同时进入 train 和 validation。
3. cached 训练把一个 batch 内的候选展开打分，再按样本 padding 成 `[batch, max_candidates]` 做 cross entropy。
4. 指标按 overall、legal_action、chosen_kind 和 candidate_count bucket 汇总。
5. checkpoint 保存模型 state、维度、训练参数、最终指标和最佳 validation epoch。

技术评价：

- seed 级验证切分和 tensor cache 是正确方向，避免了明显的数据泄漏和重复 JSON decode 成本。
- streaming 模式每个样本做一次 optimizer step，只适合小数据调试；正式训练应默认走 cache + batch。
- compact 数据会丢失 feature names，训练维度仍然可用，但后续调试、schema 迁移和 checkpoint 可解释性会变差。建议在 compact 数据旁保留一个 schema manifest。

### PPO 流水线

PPO 当前流程：

1. 用 `_initial_dimensions()` 从训练环境首个合法决策推导 observation/action 维度。
2. 可用 BC checkpoint 初始化 `policy_net`，或用已有 PPO actor-critic checkpoint 完整恢复 `policy_net` 和 `value_net`，并校验 observation/action/hidden 维度。
3. 每个 update 对每个 rollout seed 做一局或多局 rollout；可纯 self-play，也可让当前模型按队伍轮换对战 heuristic、dummy、previous 或历史 checkpoint。
4. rollout 记录 observation、候选动作、采样动作 index、old log prob、critic value、reward、done 和可选 centralized critic observation。
5. reward 在 deal/match 结束时结合开局手牌强度系数回填到每个座位最近一次 transition；PPO 可叠加小权重 shaping 并按 update 线性衰减，再按 seat 反向计算 GAE。
6. PPO loss 使用 clipped policy loss、MSE value loss、entropy bonus、gradient clipping 和 target KL early stop。

PPO 初始化约束（2026-06-27）：

- `HeuristicPolicy` 只用于 BC 采样、固定评测和线上安全 fallback；PPO rollout actor 必须来自 `TorchRolloutPolicy` 包装的模型。
- `--init-policy` 可接收当前格式的 BC ranker 或 PPO actor-critic checkpoint；继续 PPO 训练也使用该参数。
- BC ranker checkpoint 使用当前 `policy_net.*` state 初始化 PPO actor，critic 仍在 PPO 中学习；PPO checkpoint 则完整加载 actor 和 critic，使持续训练基于已训练模型而不是重新依赖启发式教师。
- checkpoint 必须保存 `model_architecture`、`centralized_critic=True`、`critic_observation_dim` 和当前 `encoding_schema`；缺失这些字段的旧 checkpoint 不再加载。

当前 PPO 是可运行 scaffold，但还不是高吞吐或强评测版本：

- rollout 可按 job 并行；训练脚本默认 `ROLLOUT_WORKERS=16`、`ROLLOUT_PROCESSES=16`，主进程集中 batched inference，主要长尾仍来自 Python 候选枚举和 reducer step，CPU opponent-pool smoke 会比较慢。
- opponent pool 已具备 self、heuristic、dummy、previous 和额外 checkpoint 输入，但还没有 Elo/胜率驱动的自动历史池采样权重。
- centralized critic 已能看完整训练 state；后续需要验证它对 value loss 和策略稳定性的实际收益。
- reward 仍以局末/比赛末为主，shaping 只做小权重辅助并默认衰减，避免模型锁死在局部启发式。
- 已有固定评测门禁入口和脚本集成，但当前默认 seed/deal 数较小，只适合作为训练产物 smoke gate；正式模型晋级还需要更大 seed 池和统计阈值。

### 线上推理接入

`RlAgentPlayer` 的线上路径是：

1. 从 `ActionRequest` 重建 `SeatSnapshot`。
2. 基于 snapshot 生成合法候选。
3. 如果 deadline 充足，尝试模型打分选择最高分候选。
4. 模型缺失、加载失败、维度不匹配、推理异常或 deadline 太近时 fallback 到启发式策略。
5. 启发式也失败时 fallback 到 protocol 级 `DummyBotPlayer`。

这套失败降级路径符合线上安全要求。需要重点持续测试的是 `seat_snapshot_from_request()` 与服务端原生 `seat_snapshot()` 的字段一致性，尤其是 `current_trick.card_ids`、贡牌 eligible cards、return tribute 限制和 hand_counts。

### 测试覆盖评价

已有测试覆盖了核心训练闭环：

- 环境 deterministic reset、private snapshot、合法 action step、局末 reward、下一局启动。
- 编码维度稳定性和 opponent private hand 隐私。
- 启发式协作行为和一局评测无拒绝。
- BC 样本结构、compact JSONL round trip、并行采集、seed validation split、tensor cache、batch 训练。
- rollout reward/done 回填、per-seat returns/GAE、PPO 参数解析、BC warm start 参数映射和已有 PPO checkpoint 继续训练参数映射。
- RL agent fallback、模型候选选择和 ActionRequest 到 SeatSnapshot 的转换。

建议补充的测试：

- 对随机 seed 的所有枚举候选执行 reducer acceptance property test，覆盖 lead、play_or_pass、tribute、return_tribute。
- Snapshot 候选与 state 候选的一致性测试：同一状态下 `legal_actions_for_state()` 与 `legal_actions_for_snapshot(seat_snapshot(...))` 应一致或只在明确字段缺失时有可解释差异。
- 编码 schema snapshot test：保存 observation/action feature names，防止无意改变维度或顺序导致当前 checkpoint 静默不可用。
- 推理延迟 benchmark：按 candidate_count bucket 记录候选生成、编码、模型打分 p50/p95/p99。

## 优化建议

优先级从高到低：

1. 建立训练评测门禁。每次 BC/PPO checkpoint 输出后自动跑固定 seed 评测，至少记录 vs `dummy_bot`、vs `HeuristicPolicy`、vs previous checkpoint 的胜率、平均升级数、双下率、pass 率、炸弹使用率和非法动作率。
2. 给编码加 schema manifest。即使使用 compact JSONL，也应保存 observation/action feature names、版本号和归一化规则；checkpoint 加载时除维度外再校验 schema hash。
3. 优化候选生成长尾。对 lead 阶段增加候选缓存、组合上限统计、可配置候选剪枝或两阶段策略：先用启发式保留高价值候选，再由模型精排。目标是控制线上 p95，而不是只优化平均延迟。
4. 增加 public 已出牌统计和 `pass_count`。这些都是公开信息，不破坏隐私，能显著提升控牌、记牌和压制决策质量。
5. 增加动作上下文派生特征。优先加入是否压同伴/对手、是否终结出完、对手危险位、相对当前 trick 的强度差、炸弹/顺子/连对拆牌成本。
6. 将模型从 concat MLP 升级为 state/action 双塔加交互项。保持候选 ranker 形式，但改为 `state_encoder(obs)`、`action_encoder(action)`、`MLP([h, a, h*a, abs(h-a)])`，再按候选 softmax。
7. PPO 引入 centralized critic。actor 继续只看 `SeatSnapshot`，critic 在训练时可看完整 public+private state 或更丰富的训练专用特征，checkpoint 推理时只需要 actor。
8. 增加 opponent pool。PPO rollout 混合当前模型、历史 checkpoint、启发式和 dummy，避免同策略自博弈过拟合。
9. 提升 rollout 吞吐。按 seed 并行收集 rollout，或者先做多进程环境采样再集中 PPO update；同时统计 reducer step/s、candidate/s、GPU utilization。
10. 分阶段放大奖励设计。保留最终胜负和升级奖励为主，先增加小权重 shaping（终结出完、拦截危险对手、同伴领先时 pass、保留炸弹），后续随 PPO 稳定后衰减，避免模型只学局部启发式。

## 优化建议前五条执行设计与进度（2026-06-28）

本轮只执行上面优化建议的前五条，目标是先提高训练产物可验收性、checkpoint 严格校验、线上输入信息量和候选生成稳定性，不提前改动模型结构。

### 1. 训练评测门禁

设计：

- 新增固定 seed 的 checkpoint 评测入口，候选 checkpoint 作为 candidate team，轮换东西/南北两队，避免座位偏置。
- 固定对手至少包含 `dummy` 和 `HeuristicPolicy`；当传入 previous checkpoint 时，再加入 `previous` 对手。
- 输出机器可读 JSON，记录胜率、平均升级数、双下率、pass 率、炸弹使用率、非法动作率和停止原因。
- BC/PPO 脚本训练完成后默认运行小规模评测；大规模训练可通过 `EVAL_SEED_COUNT=0` 关闭门禁。

已执行：

- 新增 `training/eval_gate.py` 和命令 `guandan-eval-gate`。
- `scripts/bc_train.sh` 在 BC checkpoint 输出后自动评测 vs dummy/heuristic。
- `scripts/ppo_train.sh` 默认从 `data/models/bc_ranker.pt` 启动第一版 PPO 到 `.next.pt` 后自动评测；继续训练时设置 `INIT_CHECKPOINT` 或 `BASE_MODEL` 为已有 PPO actor-critic checkpoint。
- 新增 `tests/training/test_eval_gate.py` 覆盖评测摘要生成。

命令：

```bash
uv run --extra train guandan-eval-gate data/models/ppo_actor_critic.next.pt --previous-checkpoint data/models/ppo_actor_critic.pt --seed-count 4 --max-deals 1 --device cuda
```

### 2. 编码 schema manifest

设计：

- 编码 schema 明确为当前 `v2`，包含 public 已出牌统计和动作上下文特征。
- schema manifest 包含 observation/action feature names、归一化规则和 hash。
- BC cache、BC checkpoint、PPO checkpoint 都保存 `encoding_schema`。
- checkpoint 加载时按 schema hash 严格校验；缺少 manifest 或 hash 不匹配时直接失败，避免静默使用错误特征。

已执行：

- `training/encode.py` 新增 `encoding_schema()`、`validate_encoding_schema()` 和 `ENCODING_SCHEMA_VERSION`，并在旧模型清理后移除 legacy 兼容。
- `training/bc_cache.py`、`training/bc_train.py` 写入 schema manifest。
- `training/ppo_train.py`、`training/rollout.py` 和 `npc/rl_agent/model_loader.py` 按 checkpoint schema 编码，避免 runtime 静默加载错误模型。
- 新增/更新编码测试，覆盖当前 v2 schema 和新增特征名。

### 3. 候选生成长尾优化

设计：

- 先做低风险缓存和可观测性，不引入会改变策略空间的剪枝。
- 缓存 `_candidate_card_groups(hand, level)` 的纯组合枚举结果；后续仍由当前 trick 的 `parse_hand()`/`can_beat()` 过滤，规则正确性不变。
- 暴露 cache hit/miss/currsize，供后续 benchmark 和 p95 调优使用。
- 暂不做候选上限或启发式剪枝，避免改变 BC/PPO 的合法候选集合；剪枝留到第 6 条模型升级后以两阶段 ranker 方式评估。

已执行：

- `server/domain/legal_actions.py` 对候选牌组组合加 `lru_cache(maxsize=8192)`。
- cache key 保留原始 hand tuple，避免改变候选 `card_ids` 顺序和既有样本 payload。
- 新增 `candidate_generation_cache_info()`。
- `tests/domain/test_legal_actions.py` 覆盖重复生成时 cache hit 增长，并继续验证 reducer 接受候选。

### 4. Public 已出牌统计和 `pass_count`

设计：

- reducer 在 `DealState` 保存已打出的 public card ids。
- public snapshot 只暴露按牌面聚合后的已出牌计数，不暴露任何未公开私有手牌。
- current trick snapshot 增加 `pass_count`，用于区分当前轮已有几家放弃压制。
- observation v2 将 54 个 public played face counts 和 `trick_pass_count` 加入模型输入。

已执行：

- `server/domain/state.py` 的 `DealState` 增加 `played_card_ids`。
- `server/domain/reducer.py` 在成功 `PlayCards` 时追加已出牌 ids。
- `server/services/snapshots.py` 的 `PublicTableSnapshot` 增加 `played_card_counts`，`current_trick` 增加 `pass_count`。
- `training/encode.py` 在 v2 observation 中加入 `played_face/*` 和 `trick_pass_count`。
- `tests/services/test_snapshots.py` 覆盖出牌计数和 pass_count；编码隐私测试继续保证修改对手私有手牌不会影响本座 observation。

### 5. 动作上下文派生特征

设计：

- 不改变 action candidate 本身，只在 v2 action encoding 中加入上下文派生特征。
- 优先表达协作和拆牌成本：是否压同伴、是否压对手、是否一手走完、对手是否危险位、相对当前 trick 的 rank margin、是否拆炸弹/顺子/连对。
- 这些特征全部从 `SeatSnapshot` 和本座手牌计算，保持线上推理隐私边界。

已执行：

- `training/encode.py` 的 v2 action features 增加：
  `action_beats_partner`、`action_beats_opponent`、`action_finishes_hand`、`action_opponent_danger`、`action_rank_margin`、`action_breaks_bomb`、`action_breaks_sequence`、`action_breaks_pair_run`。
- BC/PPO/线上模型加载路径统一按 checkpoint schema 校验当前 v2 编码。
- `tests/training/test_encode.py` 覆盖新增 action feature names 和字段存在性。

## 优化建议后五条执行设计与进度（2026-06-28）

本轮执行优化建议 6-10。旧模型清理后，新增能力默认用于当前 `dual_tower_v1` + v2 schema 训练；从 PPO checkpoint 继续训练时要求 checkpoint 已是当前格式。

### 6. State/action 双塔模型

设计：

- 新增 `dual_tower_v1`：`state_encoder(obs)`、`action_encoder(action)`，再用 `[h, a, h*a, abs(h-a)]` 打分。
- BC ranker 和 PPO actor 共用同一种 dual tower 候选 scorer。
- checkpoint 写入并严格校验 `model_architecture=dual_tower_v1`。

已执行：

- `training/model.py` 保留 `DEFAULT_MODEL_ARCHITECTURE=dual_tower_v1` 和双塔 scorer，删除旧 concat MLP builder。
- `training/bc_train.py` 固定输出 dual tower checkpoint，不再暴露 `--model-architecture`。
- `training/ppo_train.py` 从 checkpoint 推断 architecture；无 init 时默认 dual tower，从旧 PPO 继续时保持旧 concat 结构。
- `npc/rl_agent/model_loader.py` 可加载 dual tower BC/PPO checkpoint。

### 7. Centralized critic

设计：

- actor 仍只看 `SeatSnapshot` observation/action features。
- critic 在训练时可看完整 `MatchState`，包括四家手牌、active/finish、已出牌统计和当前 trick。
- checkpoint 写入并严格校验 `centralized_critic=True`、`critic_observation_dim` 和 feature names；不再按 decentralized critic 兼容旧 checkpoint。

已执行：

- `training/encode.py` 新增 `encode_critic_observation(state, actor)`。
- `training/rollout.py` 在 centralized critic policy 采样时保存 critic observation values。
- `training/ppo_train.py` 的 value head 可使用 critic observation；PPO batch loss 也按该输入计算 value。
- `tests/training/test_encode.py` 验证 actor observation 不看对手私有手牌，而 critic encoding 可以使用训练专用 full state。

### 8. Opponent pool

设计：

- `--opponent-pool` 接收 `self`、`heuristic`、`dummy`、`previous`。
- `--opponent-checkpoint` 可重复传入历史 checkpoint。
- 非 self 对手时，当前模型按东西/南北两队轮换，只记录当前模型座位的 PPO transitions，避免把 frozen opponent 的动作拿来更新当前策略。

已执行：

- `training/ppo_train.py` 新增 opponent pool job 构建、dummy/heuristic adapter 和 frozen checkpoint adapter。
- `scripts/ppo_train.sh` 默认使用 `self,heuristic,previous`，`previous` 指 `INIT_CHECKPOINT`/`BASE_MODEL`；CLI 仍支持手动加入 `dummy` 或重复传入 `--opponent-checkpoint`。
- `tests/training/test_rollout_ppo.py` 覆盖 opponent pool 按 candidate team 展开 rollout jobs。

### 9. Rollout 吞吐

设计：

- 将 rollout 抽象成 jobs，支持按 seed/opponent/team 组合拆分。
- `--rollout-workers` 保留线程并行 fallback；`--rollout-processes` 大于 0 时使用多进程环境 actor。命令默认仍为单线程/单进程，`scripts/ppo_train.sh` 默认提升为 `ROLLOUT_WORKERS=16`、`ROLLOUT_PROCESSES=16`，匹配 opponent pool 展开的多 rollout jobs。
- `TorchRolloutPolicy` 用 lock 包住模型 eval/inference，防止多线程下 train/eval 状态竞争。

已执行：

- `training/ppo_train.py` 新增 `_collect_rollout_jobs()`、多进程 rollout actor、batched inference server、`--rollout-workers`、`--rollout-processes`、`--inference-batch-size` 和 `--inference-batch-wait-ms`。
- `scripts/ppo_train.sh` 暴露 `ROLLOUT_WORKERS`、`ROLLOUT_PROCESSES`、`INFERENCE_BATCH_SIZE`、`INFERENCE_BATCH_WAIT_MS` 和 `CANDIDATE_BUCKET_BATCHES`；CPU/GPU 争用明显时可下调进程数或 batch size，候选枚举仍是主要长尾时可继续上调观察。
- CPU 上 opponent pool 完整对局仍可能被候选枚举长尾拖慢；真实吞吐提升需要在 CUDA/长时训练中结合 `candidate_generation_cache_info()` 继续观察。

### 10. 分阶段奖励 shaping

设计：

- 环境默认 `reward_shaping_weight=0`，普通测试和非 PPO 调用不改变。
- PPO 通过 `--reward-shaping-start`、`--reward-shaping-end` 按 update 线性衰减 shaping。
- 第一版 shaping 只给小权重局部信号：终结出完、拦截危险对手、同伴领先时 pass；非关键时机乱用炸弹略扣分。

已执行：

- `training/env.py` 新增 action-level shaping reward。
- `training/rollout.py` 将 `reward_shaping_weight` 传入环境。
- `training/ppo_train.py` 每个 update 计算线性衰减权重。
- `tests/training/test_env.py` 覆盖显式开启 shaping 后的终结出完奖励；`tests/training/test_rollout_ppo.py` 覆盖线性衰减计算。

## Rollout 吞吐优化 TODO（2026-06-28）

当前 PPO 计时显示训练耗时主要集中在 rollout，而不是 62 万参数模型的反向传播。优化顺序按“不改变训练语义、先观测再扩并发”的原则推进。

1. [x] 增加 rollout profile。每个 rollout 汇总 `legal_actions`、policy 决策、critic 编码、transition 编码和 reducer step 用时，并记录候选数均值/最大值、encoded feature 复用率。PPO update 日志打印聚合 profile，用于判断后续瓶颈在候选生成、编码、模型推理还是 reducer。
2. [x] 消除 PPO 当前策略的重复编码。`TorchRolloutPolicy` 选动作时已经完成 observation/action/critic 编码，`RolloutDecision` 携带这些 encoded features；`training/rollout.py` 写 transition 时优先复用，启发式和 frozen opponent 仍保留原编码路径。
3. [x] 缩小模型推理锁影响。PPO rollout 阶段在 update 级别统一将模型置为 eval；单步决策使用 `torch.inference_mode()`，不再每个 action 反复切换 train/eval，也不再用锁包住候选编码和状态切换。
4. [x] 基于 profile 做 batched inference。多个 rollout worker 先本地完成特征编码，再把已编码请求送入单个 inference loop；该 loop 在 `--inference-batch-wait-ms` 等待窗口内最多聚合 `--inference-batch-size` 个 decision 后统一 forward，减少 CUDA launch 和线程间模型争用。`scripts/ppo_train.sh` 默认启用 `INFERENCE_BATCH_SIZE=16`、`INFERENCE_BATCH_WAIT_MS=1.0`。
5. [x] 将 rollout worker 从线程升级为多进程环境 actor。`--rollout-processes` 大于 0 时，每个 spawned worker 进程持有独立 `GuandanTrainingEnv`，在进程内执行合法候选生成、observation/action/critic 编码、transition 组装和 reducer step；当前 policy 的模型 forward 仍通过主进程集中 batched inference 完成，避免每个 worker 复制 CUDA context。checkpoint opponent 在 process 模式默认使用 CPU 加载，减少 GPU 争用。`scripts/ppo_train.sh` 默认 `ROLLOUT_PROCESSES=16`，需要回退线程模式时设为 `0`。
6. [x] 做 candidate-count bucketing 或 segmented softmax。已先实现 candidate-count bucketing：PPO 更新阶段默认按候选动作数量的 power-of-two 桶组 minibatch，避免普通随机 batch 被少数首攻长尾候选 padding 到过宽矩阵；CLI 可用 `--no-candidate-bucket-batches` 回退，`scripts/ppo_train.sh` 可设 `CANDIDATE_BUCKET_BATCHES=0` 做 A/B。若后续 profile 仍显示训练 forward 被 padding 主导，再评估 segmented softmax。
7. [ ] 增加 partial rollout 容错。`max_steps` 作为 truncated episode 记录 profile 和告警，避免单个长尾 job 让整个 update 已完成样本报废。
