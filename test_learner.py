# -*- coding: utf-8 -*-
"""合成环境单元测试: 验证 QMIX 学习器本身的正确性。

构造一个已知最优解的合作型 MDP:
  - 2 个智能体、2 个动作、链式 T 步、无观测噪声(每步 obs 相同)
  - 奖励: 两智能体动作相同则 +1, 否则 0(纯协同问题)
  - 最优策略: 两智能体都固定选动作 1, G* = sum(0.99^t) ≈ 8.9(T=10)

如果 QMIX 实现正确, 训练后应有: Q_tot(s0) ≈ G*, 且贪心策略两智能体动作一致。
用法: python test_learner.py
"""
import numpy as np
import torch

from buffer import Episode, ReplayBuffer
from qmix import QMIX

NA, N_ACT, OBS_D, STATE_D, T = 2, 2, 3, 6, 10
G_STAR = sum(0.99 ** t for t in range(T))  # 每步 +1 的折扣和


def make_episode(actions, obs):
    """按动作序列生成一条轨迹(奖励 = 两智能体动作一致)。"""
    ep = Episode()
    state = obs.reshape(-1)                                # 全局状态 = 拼平
    for t in range(T):
        a = actions[t]
        r = 1.0 if a[0] == a[1] else 0.0
        ep.push(obs, state, a, r, t == T - 1)
    return ep


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dq", type=int, default=1, help="1=开 Double-Q")
    ap.add_argument("--rn", type=int, default=1, help="1=开奖励归一化")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--tau", type=float, default=0.0, help=">0 用软更新")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    agent = QMIX(n_agents=NA, n_actions=N_ACT, obs_dim=OBS_D,
                 state_dim=STATE_D, hidden_dim=32, mix_embed_dim=16,
                 gamma=args.gamma,
                 use_double_q=bool(args.dq), use_reward_norm=bool(args.rn))
    buffer = ReplayBuffer(500)

    eps = 1.0
    for ep_i in range(1, args.episodes + 1):
        # 用当前策略 + ε 探索生成一条轨迹(acting 与存储使用同一份 obs)
        actions = []
        hidden = agent.init_hidden()
        last = None
        obs = np.random.normal(size=(NA, OBS_D)).astype(np.float32)
        for t in range(T):
            acts, hidden = agent.act(obs, hidden, last, epsilon=eps)
            actions.append(acts)
            last = acts
        buffer.push(make_episode(actions, obs))
        if len(buffer) >= 32:
            for _ in range(2):
                agent.learn(buffer.sample(32))
            if args.tau > 0:
                agent.soft_update_target(args.tau)
            elif ep_i % 50 == 0:
                agent.update_target()
        eps = max(0.05, eps - 0.95 / 1500)

        if ep_i % 1000 == 0:
            # 贪心跑一局看协同是否达成
            hidden = agent.init_hidden()
            last, greedy = None, []
            for t in range(T):
                acts, hidden = agent.act(np.zeros((NA, OBS_D), np.float32),
                                         hidden, last, epsilon=0.0)
                greedy.append(acts)
                last = acts
            coord = all(a[0] == a[1] for a in greedy)
            ret = sum(1.0 if a[0] == a[1] else 0.0 for a in greedy)
            # Q_tot(s0)
            batch = buffer.sample(32)
            with torch.no_grad():
                obs0 = batch["obs"][:, 0]
                a0 = batch["actions"][:, 0]
                onehot = torch.nn.functional.one_hot(a0, N_ACT).float()
                inp = agent._build_inputs(obs0, torch.zeros_like(onehot))
                inp = inp.reshape(-1, inp.shape[-1])
                h = agent.init_hidden(32).reshape(-1, agent.hidden_dim)
                q, _ = agent.rnn_eval(inp, h)
                q = q.view(32, NA, N_ACT)
                q_taken = q.gather(2, a0.unsqueeze(-1)).squeeze(-1)
                q_tot0 = agent.mixer_eval(q_taken.unsqueeze(1),
                                          batch["state"][:, 0].unsqueeze(1)).mean()
            print(f"ep {ep_i:5d}: greedy return {ret:.0f}/{T}  coordinated={coord}"
                  f"  Q_tot(s0)={float(q_tot0):7.3f}  (最优 G*={G_STAR:.2f})")

    ok = ret >= T - 1 and abs(float(q_tot0) - G_STAR) < 2
    print(f"\n[dq={args.dq} rn={args.rn} gamma={args.gamma}] 结论:",
          "学习器正常" if ok else f"Q 漂移(最终 Q={float(q_tot0):.1f} vs G*={G_STAR:.1f})")


if __name__ == "__main__":
    main()
