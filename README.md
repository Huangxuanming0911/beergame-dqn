# 啤酒游戏 Baseline 与算法消融实验

本项目基于课程提供的 Beergame 供应链任务，整理了带有 `t+1` 到货延迟的实验环境，并实现了 DQN 系列算法消融实验。

## 任务设定

供应链包含 3 个串行企业：

```text
外部顾客 -> 企业0 -> 企业1 -> 企业2
```

当前实验固定优化第 2 个企业：

```text
firm_id = 1
```

每个企业使用课程原始 3 维局部观测：

```text
[上一期订货量, 上一期满足需求量, 当前库存]
```

本项目不加入 `pipeline` 观测，保证本轮实验只比较算法和背景策略差异。

## 环境机制

本项目实现了 `t+1` 到货机制：

```text
t 时刻发出的订单不会立刻进入库存
t+1 时刻才会作为在途货物到达
```

每个企业的单步 reward 按当期利润计算：

```text
reward = 销售收入 - 采购成本 - 库存持有成本 - 缺货惩罚
```

对应公式：

```text
p_i * satisfied_demand_i
- p_{i+1} * order_i
- h * inventory_i
- c * lost_sales_i
```

最后一个企业没有更上游供应商，因此采购成本项为 0。

## 已实现方法

规则 baseline：

- `random`：随机订货。
- `base_stock`：库存补足策略，即根据当前库存补到目标库存水位。

DQN 系列算法消融：

- `dqn`：普通 DQN。
- `double_dqn`：使用在线网络选动作、目标网络估值，缓解 Q 值高估。
- `dueling_dqn`：使用 `V(s) + A(s,a)` 分解的 Dueling 网络。
- `dueling_double_dqn`：同时使用 Double DQN 目标和 Dueling 网络。

背景策略实验：

- `random_background`：除目标企业外，其他企业随机订货。
- `base_stock_background`：除目标企业外，其他企业使用库存补足策略。

背景策略实验用于回答：目标企业的学习效果在“随机上下游”和“规则化上下游”中有什么差异。

## 代码结构

```text
beergame/
  env.py                     # Beergame环境与t+1到货逻辑
  policies.py                # random和base_stock规则策略
  dqn.py                     # DQN、Double DQN、Dueling DQN实现
  experiments.py             # 训练、评估、画图工具函数
  run_baselines.py           # 一键运行baseline与算法消融
  run_background_experiments.py # 其他企业随机/库存补足背景对比
  run_multiagent.py          # 三个企业独立Dueling Double DQN训练

configs/
  default.json               # 环境、算法和训练配置
```

## 运行方式

训练并评估全部 baseline 与算法消融：

```powershell
cd C:\homework\multiagent\finalwork
python -m beergame.run_baselines --config configs/default.json
```

如果已有模型，只想跳过已存在模型的训练并重新评估：

```powershell
python -m beergame.run_baselines --config configs/default.json --skip-train
```

运行其他企业背景策略对比实验：

```powershell
python -m beergame.run_background_experiments --config configs/default.json
```

该实验会分别训练两个 `Dueling Double DQN`：

- 一个在其他企业随机订货的背景下训练。
- 一个在其他企业库存补足的背景下训练。

两个模型互不覆盖，单独保存到 `models/background/`。

运行多智能体实验：

```powershell
python -m beergame.run_multiagent --config configs/default.json
```

该实验采用 Independent Dueling Double DQN：

```text
企业0 -> agent_0
企业1 -> agent_1
企业2 -> agent_2
```

每个企业都有独立网络、经验池和模型文件。每一步三个 agent 同时根据自己的 3 维局部观测下单，环境统一执行动作向量。

如果已有多智能体模型，只想重新评估和重画图：

```powershell
python -m beergame.run_multiagent --config configs/default.json --skip-train
```

## 当前算法消融结果

当前配置下，每个算法训练 300 个 episode、评估 20 个 episode。展示用 seed=42 的结果如下：

| 方法 | 平均 reward | 标准差 | seed均值 |
| --- | ---: | ---: | --- |
| `random` | -2659.18 | 1980.17 | -2659.18 |
| `base_stock` | -3979.18 | 833.86 | -3979.18 |
| `dqn` | 758.08 | 90.27 | 758.08 |
| `double_dqn` | 819.15 | 83.34 | 819.15 |
| `dueling_dqn` | 763.00 | 100.82 | 763.00 |
| `dueling_double_dqn` | 831.95 | 104.36 | 831.95 |

完整结果保存在：

```text
results/baselines/baseline_summary.json
figures/baselines/baseline_comparison.png
```

## 背景策略对比结果

当前 seed=42、每种背景训练 300 个 episode、评估 20 个 episode 的结果如下：

| 背景策略 | 平均 reward | 标准差 |
| --- | ---: | ---: |
| `random_background` | 831.95 | 104.36 |
| `base_stock_background` | 824.00 | 73.38 |

在该结果下，两种背景的平均 reward 接近；`base_stock_background` 的标准差更低，说明规则化上下游背景下评估波动更小。

背景策略实验会输出：

```text
results/background/background_policy_summary.json
figures/background/background_policy_comparison.png
```

训练曲线保存为：

```text
figures/background/dueling_double_dqn_random_background_training_rewards.png
figures/background/dueling_double_dqn_base_stock_background_training_rewards.png
```

模型保存为：

```text
models/background/dueling_double_dqn_random_background_seed_42_firm_1_tplus1.pt
models/background/dueling_double_dqn_base_stock_background_seed_42_firm_1_tplus1.pt
```

## 多智能体实验结果

多智能体实验同时统计每个企业的 reward 和全链路 total reward：

```text
total_chain_reward = firm_0_reward + firm_1_reward + firm_2_reward
```

对比对象包括：

- `random_all`：所有企业随机订货。
- `base_stock_all`：所有企业库存补足。
- `single_agent_ddqn`：只训练企业1，其余企业随机。
- `multiagent_ddqn`：三个企业都使用独立 Dueling Double DQN。

当前 seed=42、训练 600 个 episode、评估 20 个 episode 的结果如下：

| 方法 | 企业0 reward | 企业1 reward | 企业2 reward | 全链路 total reward |
| --- | ---: | ---: | ---: | ---: |
| `random_all` | -3763.38 | -4023.73 | 3957.10 | -3830.00 |
| `base_stock_all` | -3509.93 | -3521.32 | 3414.05 | -3617.20 |
| `single_agent_ddqn` | -3057.18 | 831.95 | -2438.50 | -4663.73 |
| `multiagent_ddqn` | 468.77 | 538.60 | 2419.18 | 3426.55 |

该结果说明：单智能体 DDQN 能显著提高目标企业1的局部收益，但全链路收益仍为负；多智能体 DDQN 的企业1局部收益较低，但三个企业共同学习后，全链路 total reward 明显提升为正。

多智能体输出文件：

```text
models/multiagent/dueling_double_dqn_firm_0_seed_42_tplus1.pt
models/multiagent/dueling_double_dqn_firm_1_seed_42_tplus1.pt
models/multiagent/dueling_double_dqn_firm_2_seed_42_tplus1.pt
results/multiagent/multiagent_training_scores.npy
results/multiagent/multiagent_eval_points.npy
results/multiagent/multiagent_eval_scores.npy
results/multiagent/multiagent_summary.json
figures/multiagent/multiagent_training_rewards.png
figures/multiagent/multiagent_eval_curve.png
figures/multiagent/multiagent_comparison.png
```

其中 `multiagent_eval_curve.png` 是每 50 轮进行一次无探索评估得到的曲线，包含企业0、企业1、企业2和全链路 total reward。它比训练 reward 曲线更适合判断“当前学到的最终策略是否变好”。

## 输出文件

算法消融模型：

```text
models/baselines/{algorithm}_seed_{seed}_firm_1_tplus1.pt
```

算法消融训练奖励：

```text
results/baselines/{algorithm}_seed_{seed}_training_scores.npy
results/baselines/{algorithm}_training_scores.npy
figures/baselines/{algorithm}_training_rewards.png
```

背景策略实验输出：

```text
models/background/
results/background/
figures/background/
```

多智能体实验输出：

```text
models/multiagent/
results/multiagent/
figures/multiagent/
```

## 快速检查

```powershell
python -m py_compile beergame\dqn.py beergame\experiments.py beergame\run_baselines.py beergame\run_background_experiments.py beergame\run_multiagent.py
python -m beergame.run_baselines --config configs/default.json --skip-train
python -m beergame.run_background_experiments --config configs/default.json
python -m beergame.run_multiagent --config configs/default.json
```
