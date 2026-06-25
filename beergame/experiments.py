from __future__ import annotations

from pathlib import Path

import matplotlib

# 使用非交互式后端，避免服务器或本机Tk配置异常时无法保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .dqn import DQNAgent
from .env import BeerGameEnv
from .policies import build_policy


def setup_chinese_font():
    """为Matplotlib注册中文字体，避免保存图片时中文显示为方块。"""

    candidates = [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


setup_chinese_font()


def make_background_actions(env: BeerGameEnv, state: np.ndarray, firm_id: int, rng: np.random.Generator):
    # 未被控制的企业视为环境的一部分，这里使用随机订货。
    actions = rng.integers(0, env.config.max_order + 1, size=env.num_firms).astype(np.float32)
    return actions


def evaluate_policy(env: BeerGameEnv, policy, firm_id: int, episodes: int, seed: int | None = 123):
    rng = np.random.default_rng(seed)
    scores = []
    histories = {"orders": [], "inventory": [], "demand": [], "satisfied": [], "rewards": []}
    for episode in range(episodes):
        state = env.reset(seed=None if seed is None else seed + episode)
        done = False
        score = 0.0
        ep = {key: [] for key in histories}
        while not done:
            actions = make_background_actions(env, state, firm_id, rng)
            # 只把目标企业的随机动作替换为待评估策略的动作。
            actions[firm_id] = policy.act(state, firm_id)
            next_state, rewards, done, info = env.step(actions)
            score += float(rewards[firm_id, 0])
            ep["orders"].append(float(info["actions"][firm_id]))
            ep["inventory"].append(float(info["inventory"][firm_id]))
            ep["demand"].append(float(info["demand"][firm_id]))
            ep["satisfied"].append(float(info["satisfied_demand"][firm_id]))
            ep["rewards"].append(float(rewards[firm_id, 0]))
            state = next_state
        scores.append(score)
        for key in histories:
            histories[key].append(ep[key])
    return {"scores": np.asarray(scores, dtype=np.float32), "histories": histories}


def train_dqn(env: BeerGameEnv, agent: DQNAgent, cfg: dict):
    rng = np.random.default_rng(cfg.get("seed", 42))
    scores = []
    eps = float(cfg.get("eps_start", 1.0))
    eps_end = float(cfg.get("eps_end", 0.01))
    eps_decay = float(cfg.get("eps_decay", 0.995))
    episodes = int(cfg.get("episodes", 500))

    for episode in range(1, episodes + 1):
        state = env.reset(seed=int(cfg.get("seed", 42)) + episode)
        done = False
        score = 0.0
        while not done:
            actions = make_background_actions(env, state, agent.firm_id, rng)
            firm_state = state[agent.firm_id]
            # DQN只控制一个企业，其余企业保持随机行为作为背景环境。
            action = agent.act(firm_state, eps)
            actions[agent.firm_id] = action
            next_state, rewards, done, _ = env.step(actions)
            reward = float(rewards[agent.firm_id, 0])
            agent.step(firm_state, action, reward, next_state[agent.firm_id], done)
            state = next_state
            score += reward

        eps = max(eps_end, eps_decay * eps)
        scores.append(score)
        if episode % int(cfg.get("log_every", 50)) == 0:
            avg = np.mean(scores[-int(cfg.get("log_every", 50)):])
            print(f"episode={episode} avg_score={avg:.2f} epsilon={eps:.3f}")
    return np.asarray(scores, dtype=np.float32)


def plot_training(scores: np.ndarray, output_path: str | Path, window: int = 50):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=np.float32)

    if scores.ndim == 1:
        moving = np.array([np.mean(scores[max(0, i - window + 1): i + 1]) for i in range(len(scores))])
        plt.figure(figsize=(9, 5))
        plt.plot(scores, alpha=0.35, label="单轮奖励")
        plt.plot(moving, label=f"{window}轮滑动平均")
    else:
        moving = np.array(
            [
                [np.mean(row[max(0, i - window + 1): i + 1]) for i in range(scores.shape[1])]
                for row in scores
            ]
        )
        mean_curve = moving.mean(axis=0)
        std_curve = moving.std(axis=0)
        x = np.arange(scores.shape[1])
        plt.figure(figsize=(9, 5))
        plt.plot(mean_curve, label=f"{window}轮滑动平均（多seed均值）")
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="seed间标准差")

    plt.xlabel("训练轮次")
    plt.ylabel("奖励")
    plt.title("DQN训练奖励曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


DISPLAY_NAMES = {
    "random": "随机策略",
    "base_stock": "库存补足",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "dueling_double_dqn": "Dueling Double DQN",
}

COLORS = {
    "random": "#9aa0a6",
    "base_stock": "#6f7782",
    "dqn": "#4e79a7",
    "double_dqn": "#59a14f",
    "dueling_dqn": "#f28e2b",
    "dueling_double_dqn": "#e15759",
}


def _collect_plot_data(results: dict, names: list[str]):
    labels = [DISPLAY_NAMES.get(name, name) for name in names]
    means = np.array([float(np.mean(results[name]["scores"])) for name in names])
    stds = np.array([float(np.std(results[name]["scores"])) for name in names])
    colors = [COLORS.get(name, "#4e79a7") for name in names]
    return labels, means, stds, colors


def _annotate_bars(ax, bars, means: np.ndarray):
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1.0)
    for bar, value in zip(bars, means):
        offset = 0.025 * span
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.1f}",
            ha="center",
            va=va,
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )


def _annotate_horizontal_bars(ax, means: np.ndarray):
    x_min, x_max = ax.get_xlim()
    span = max(x_max - x_min, 1.0)
    for idx, value in enumerate(means):
        offset = 0.025 * span
        if value >= 0:
            x = value + offset
            ha = "left"
        else:
            x = min(value + 0.10 * span, -offset)
            ha = "left"
        ax.text(
            x,
            idx,
            f"{value:+.1f}",
            va="center",
            ha=ha,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )


def _draw_horizontal_comparison(ax, results: dict, names: list[str], title: str):
    labels, means, stds, colors = _collect_plot_data(results, names)
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.9, capsize=4)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("平均评估奖励")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    x_min = float(np.min(means - stds))
    x_max = float(np.max(means + stds))
    span = max(x_max - x_min, 1.0)
    ax.set_xlim(x_min - 0.12 * span, x_max + 0.18 * span)
    _annotate_horizontal_bars(ax, means)


def _plot_vertical_comparison(results: dict, names: list[str], output_path: str | Path, title: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels, means, stds, colors = _collect_plot_data(results, names)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.92, capsize=5, width=0.62)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("平均评估奖励")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    y_min = float(np.min(means - stds))
    y_max = float(np.max(means + stds))
    span = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min - 0.12 * span, y_max + 0.18 * span)
    _annotate_bars(ax, bars, means)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def plot_policy_baseline_comparison(results: dict, output_path: str | Path):
    names = ["random", "base_stock", "dqn"]
    _plot_vertical_comparison(results, names, output_path, "基础策略对比：Random / Base-stock / DQN")


def plot_dqn_ablation_comparison(results: dict, output_path: str | Path):
    names = ["dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn"]
    _plot_vertical_comparison(results, names, output_path, "DQN算法消融对比")


def plot_baseline_comparison(results: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_names = ["random", "base_stock", "dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn"]
    dqn_names = ["dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    _draw_horizontal_comparison(axes[0], results, all_names, "所有方法整体对比")
    _draw_horizontal_comparison(axes[1], results, dqn_names, "DQN系列局部放大")
    fig.suptitle("Baseline 与算法消融评估结果（误差线为标准差）", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close()


def run_rule_baselines(env: BeerGameEnv, cfg: dict):
    firm_id = int(cfg["experiment"].get("firm_id", 1))
    eval_episodes = int(cfg["experiment"].get("eval_episodes", 20))
    target = int(cfg["baselines"].get("base_stock_target", env.config.initial_inventory))
    results = {}
    for name in ["random", "base_stock"]:
        policy = build_policy(name, env.config.max_order, seed=cfg["env"].get("seed", 42), target_inventory=target)
        results[name] = evaluate_policy(env, policy, firm_id, eval_episodes)
    return results
