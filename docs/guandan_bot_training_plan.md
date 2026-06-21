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
- actor 从第 4 阶段 BC checkpoint warm start；critic 单独初始化并在 PPO 中学习。
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
- 已新增 `training/ppo_train.py`，提供惰性 PyTorch self-play PPO 训练入口 `guandan-ppo-train`，支持 BC warm start、GAE、mini-batch PPO 更新、梯度裁剪、KL early stop 和逐 update 日志。
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
guandan-ppo-train data/models/ppo_actor_critic.pt --init-policy data/models/bc_ranker.pt --seed-count 8 --updates 100 --epochs-per-update 3 --max-deals 4 --batch-size 256
guandan-rl-agent-server --model-path data/models/ppo_actor_critic.pt --device cuda
```

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
- `training/rollout.py` 和 `training/ppo_train.py` 基于当前模型做自博弈 rollout，并用 PPO 更新 actor-critic。
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

`encode_observation(snapshot)` 当前输出 `140` 维 float 特征，全部来自当前座位可见的 `SeatSnapshot`：

- 座位、phase、当前级牌、current turn、acting seat、legal action、贡牌来源/目标等 one-hot。
- 双方等级归一化到 `[0, 1]`。
- 四家剩余手牌数量，按 `27` 归一化。
- 四家 finish position，未完成为 `0`。
- 本座 54 种牌面计数，按双副牌最大计数 `2` 归一化。
- 当前 trick 的最后出牌座位、牌型、主 rank 和长度。
- `return_rank_at_most_ten` 贡牌约束标记。

隐私边界是合理的：编码不读取其他座位私有手牌，测试也覆盖了“替换对手手牌不改变本座 observation 编码”的场景。

当前缺口：

- 没有 public 已出牌统计，模型无法直接知道每个 rank/suit 已消耗多少，只能从当前 trick 和剩余牌数间接推断。
- snapshot 的 public trick 未暴露 `pass_count`，observation 也没有当前轮已经 pass 的人数。
- 缺少明确的“最后出牌者是同伴/对手”“对手是否只剩 1-2 张”“自己/同伴/对手是否接近走完”等派生特征；模型可以从座位和 hand_counts 间接学习，但样本效率较低。
- 没有历史 trick 序列或 recurrent state，当前模型是单步静态策略。

### Action 编码

`encode_action(action, snapshot)` 当前输出 `88` 维 float 特征：

- 动作类型 one-hot：`pass`、`play_cards`、`submit_tribute`、`return_tribute`。
- 出牌牌型 one-hot、主 rank one-hot。
- 候选动作 54 种牌面计数，按 `2` 归一化。
- 动作长度、是否有 declared type、是否炸弹类、是否使用红心级牌。
- 出完该动作后本座剩余牌数。

当前编码能表达“这手牌是什么”和“打完还剩多少”，但缺少上下文派生特征：

- 是否压同伴、是否压对手、是否能直接走完、是否拆掉炸弹或关键牌组。
- 候选动作相对当前 trick 的增量强度，例如只大一点还是高很多。
- 候选动作在剩余手牌中的结构代价，例如是否破坏顺子、连对、三带等潜在组合。

这些派生特征不改变规则正确性，但会显著降低小模型从稀疏样本中学习协作和拆牌策略的难度。

### 模型结构

当前模型是候选动作打分，而不是固定动作空间分类。

BC ranker：

```text
pair_features = concat(observation[140], action[88])  # 228 dims
MLP(228 -> hidden -> hidden -> 1)
softmax(score over legal candidates)
CrossEntropy(chosen_index)
```

PPO actor-critic：

```text
policy_net: MLP(pair_features -> hidden -> hidden -> 1)
value_net:  MLP(observation -> hidden -> hidden -> 1)
```

默认 `hidden_dim=256` 时，BC ranker 约 `125k` 参数，PPO actor-critic 约 `227k` 参数。这个规模非常适合低延迟 smoke run 和线上 fallback 验证，但明显小于最初建议的 5M 到 30M 参数区间，表达能力可能不足以稳定学会复杂拆牌、控牌、让牌和进贡策略。

当前实现与初始设计的差异：

- 初始设计建议 `state_encoder`、`action_encoder` 和 `h * a` 交互项；当前实现是简单 concat 后 MLP。
- 初始设计建议 centralized critic 可看完整训练 state；当前 value head 只看 actor observation，是 decentralized critic。
- 当前没有模型池或历史对手池，PPO 自博弈四个座位都使用同一个最新策略。

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
2. 可用 BC checkpoint 初始化 `policy_net`，并校验 observation/action/hidden 维度。
3. 每个 update 对每个 rollout seed 做一局或多局自博弈，四个座位共享同一个 `TorchRolloutPolicy`。
4. rollout 记录 observation、候选动作、采样动作 index、old log prob、value、reward 和 done。
5. reward 在 deal/match 结束时回填到每个座位最近一次 transition，再按 seat 反向计算 GAE。
6. PPO loss 使用 clipped policy loss、MSE value loss、entropy bonus、gradient clipping 和 target KL early stop。

当前 PPO 是可运行 scaffold，但还不是高吞吐或强评测版本：

- rollout 单进程串行，吞吐主要受动作枚举和 Python reducer 限制。
- 没有 opponent pool，容易产生同策略自博弈的过拟合和循环策略。
- critic 只看 actor observation，优势估计信息少；如果训练中允许 critic 看全状态，可以加 centralized critic 提升稳定性。
- reward 主要是局末/比赛末稀疏奖励，早期 PPO 对 BC 初始化质量依赖较强。
- 没有固定评测门禁，训练输出 checkpoint 前未自动跑 vs heuristic、vs dummy、vs previous model 的胜率评估。

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
- rollout reward/done 回填、per-seat returns/GAE、PPO 参数解析、BC warm start 参数映射。
- RL agent fallback、模型候选选择和 ActionRequest 到 SeatSnapshot 的转换。

建议补充的测试：

- 对随机 seed 的所有枚举候选执行 reducer acceptance property test，覆盖 lead、play_or_pass、tribute、return_tribute。
- Snapshot 候选与 state 候选的一致性测试：同一状态下 `legal_actions_for_state()` 与 `legal_actions_for_snapshot(seat_snapshot(...))` 应一致或只在明确字段缺失时有可解释差异。
- 编码 schema snapshot test：保存 observation/action feature names，防止无意改变维度或顺序导致旧 checkpoint 静默不可用。
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
