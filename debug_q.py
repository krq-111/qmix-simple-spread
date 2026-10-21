# -*- coding: utf-8 -*-
"""诊断脚本: 检查训练中 Q_tot 的量级是否与真实回报一致(排查 Q 值发散)。

用法: python debug_q.py --checkpoint results/xxx/latest.pth
"""
import argparse

import numpy as np
import torch

from buffer import ReplayBuffer
from envs import SimpleSpreadEnv
from qmix import QMIX
from runner import run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=200, help="先采集的随机局数")
    args = p.parse_args()

    env = SimpleSpreadEnv()
    env.reset(seed=7)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    info = ckpt["info"]
    agent = QMIX(info["n_agents"], info["n_actions"], info["obs_dim"],
                 info["state_dim"])
    agent.load(args.checkpoint)
    buffer = ReplayBuffer(200)

    # 采集一批随机数据并记录真实回报
    true_returns = []
    for _ in range(args.episodes):
        ret, _, ep = run_episode(env, agent, epsilon=1.0, collect=True)
        buffer.push(ep)
        true_returns.append(ret)
    print(f"真实回报(随机策略): mean={np.mean(true_returns):.2f} "
          f"std={np.std(true_returns):.2f}")

    # 用网络计算这些 episode 的 Q_tot(第 0 步), 与真实折扣回报对比
    batch = buffer.sample(64)
    obs = batch["obs"]
    state = batch["state"]
    actions = batch["actions"]
    B, T, NA = obs.shape[0], obs.shape[1], info["n_agents"]
    onehot = torch.nn.functional.one_hot(actions, info["n_actions"]).float()
    h = agent.init_hidden(B)
    qs = []
    for t in range(T):
        prev = onehot[:, t - 1] if t > 0 else torch.zeros_like(onehot[:, 0])
        inp = agent._build_inputs(obs[:, t], prev).reshape(B * NA, -1)
        q, h = agent.rnn_eval(inp, h.reshape(B * NA, -1))
        qs.append(q.view(B, NA, -1))
    q_all = torch.stack(qs, 1)
    q_taken = q_all.gather(3, actions.unsqueeze(-1)).squeeze(-1)
    q_tot = agent.mixer_eval(q_taken, state).squeeze(-1)  # [B, T]

    gamma = 0.99
    rewards, dones = batch["rewards"], batch["dones"]
    # 手工算真实折扣回报 G_0
    disc = gamma ** torch.arange(T, dtype=torch.float32)
    g0 = (rewards * disc).sum(1)
    print(f"网络预测 Q_tot(s0):     mean={q_tot[:, 0].mean():.2f} "
          f"min={q_tot[:, 0].min():.2f} max={q_tot[:, 0].max():.2f}")
    print(f"真实折扣回报 G_0:      mean={g0.mean():.2f} "
          f"min={g0.min():.2f} max={g0.max():.2f}")
    ratio = q_tot[:, 0].mean() / g0.mean()
    print(f"Q_tot/G0 比值: {ratio:.3f}(健康应接近 1, 远大于 1 = Q 值发散)")


if __name__ == "__main__":
    main()
