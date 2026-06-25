# 啤酒游戏 Baseline 与算法消融实验

本项目基于课程提供的 Beergame 供应链任务，整理了带有 `t+1` 到货延迟的实验环境，并实现了 DQN 系列算法消融实验。

## 任务设定

供应链包含 3 个串行企业。企业 0 面对外部顾客需求，企业 1 和企业 2 面对下游企业订单。当前实验固定优化第 2 个企业，即：

```text
firm_id = 1
```

其余企业使用随机订货策略，作为环境背景。

每个企业使用课程原始 3 维局部观测：

```text
[上一期订货量, 上一期满足需求量, 当前库存]
```

本项目不加入 `pipeline` 观测，保证本轮实验只比较算法差异。

## 环境机制

本项目修正并实现了 `t+1` 到货机制：

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

四个学习算法使用相同训练配置，包括训练轮数、隐藏层规模、学习率、batch size、随机种子和评估 episode 数，便于公平比较。

## 代码结构

```text
beergame/
  env.py             # Beergame环境与t+1到货逻辑
  policies.py        # random和base_stock规则策略
  dqn.py             # DQN、Double DQN、Dueling DQN实现
  experiments.py     # 训练、评估、画图工具函数
  run_baselines.py   # 一键运行baseline与算法消融

configs/
  default.json       # 环境、算法和训练配置
```

原始仓库文件 `course.py` 和 `course_dqn_example.py` 保留作为参考。

## 运行方式

训练并评估全部方法：

```powershell
cd C:\homework\multiagent\finalwork
python -m beergame.run_baselines --config configs/default.json
```

如果已有模型，只想跳过已存在模型的训练并重新评估：

```powershell
python -m beergame.run_baselines --config configs/default.json --skip-train
```

`--skip-train` 会加载已经存在的模型；如果某个算法模型不存在，则会训练该算法并保存模型。

## 当前实验结果

当前配置下，每种方法评估 20 个 episode，结果如下：

| 方法 | 平均 reward | 标准差 |
| --- | ---: | ---: |
| `random` | -3964.20 | 2280.98 |
| `base_stock` | -3787.88 | 816.81 |
| `dqn` | 758.08 | 90.27 |
| `double_dqn` | 819.15 | 83.34 |
| `dueling_dqn` | 763.00 | 100.82 |
| `dueling_double_dqn` | 831.95 | 104.36 |

在当前实验中，`dueling_double_dqn` 的平均 reward 最高，`double_dqn` 也相比普通 DQN 有明显提升。

## 输出文件

模型：

```text
models/baselines/{algorithm}_firm_1_tplus1.pt
```

训练奖励：

```text
results/baselines/{algorithm}_training_scores.npy
figures/baselines/{algorithm}_training_rewards.png
```

汇总结果：

```text
results/baselines/baseline_summary.json
figures/baselines/baseline_comparison.png
```

## 快速检查

```powershell
python -m py_compile beergame\dqn.py beergame\experiments.py beergame\run_baselines.py
python -m beergame.run_baselines --config configs/default.json --skip-train
```
