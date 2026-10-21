# -*- coding: utf-8 -*-
"""QMIX 训练入口。

用法:
    python train.py                       # 默认超参数训练 50000 局
    python train.py --episodes 20000 --tag test
    python train.py --help                # 查看全部超参数

产出(在 results/<tag>/ 下):
    progress.csv    训练日志(每局一行: return / loss / epsilon / 评估分)
    latest.pth      最新检查点
    best.pth        评估回报最高的检查点
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from buffer import ReplayBuffer
from envs import SimpleSpreadEnv
from qmix import QMIX
from runner import evaluate, run_episode


def parse_args():
    p = argparse.ArgumentParser("QMIX on PettingZoo simple_spread")
    # 环境
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--max-cycles", type=int, default=25, help="每局步数")
    p.add_argument("--local-ratio", type=float, default=0.5)
    # 训练
    p.add_argument("--episodes", type=int, default=50000, help="总训练局数")
    p.add_argument("--buffer-size", type=int, default=5000, help="缓冲区容量(局)")
    p.add_argument("--batch-size", type=int, default=32, help="每次学习的 episode 数")
    p.add_argument("--train-iters", type=int, default=2, help="每局结束后学习几次")
    p.add_argument("--warmup", type=int, default=32, help="缓冲区攒够多少局后开始学习")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="rmsprop", choices=["rmsprop", "adam"],
                   help="adam 是 marlbenchmark/off-policy 在 MPE 上用的配置")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-update", type=int, default=200, help="每隔多少局硬更新目标网络")
    p.add_argument("--tau", type=float, default=0.0,
                   help=">0 时改用软更新(Polyak, 每局训练后执行), off-policy 在 MPE 用 0.005")
    # 探索
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-finish", type=float, default=0.05)
    p.add_argument("--eps-anneal", type=int, default=5000, help="探索率在前多少局内退火")
    # 网络
    p.add_argument("--hidden-dim", type=int, default=64, help="GRU 隐藏维度")
    p.add_argument("--mix-embed-dim", type=int, default=32, help="混合网络隐藏维度")
    # 杂项
    p.add_argument("--no-reward-norm", action="store_true", help="关闭奖励归一化")
    p.add_argument("--no-double-q", action="store_true", help="关闭 Double-Q")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-interval", type=int, default=500, help="每隔多少局贪心评估一次")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--tag", type=str, default="run1", help="结果目录名")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.join("results", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[config] {vars(args)}")
    print(f"[output ] {out_dir}")

    env = SimpleSpreadEnv(args.n_agents, args.n_agents, args.max_cycles,
                          args.local_ratio)
    eval_env = SimpleSpreadEnv(args.n_agents, args.n_agents, args.max_cycles,
                               args.local_ratio)
    info = env.info
    print(f"[env    ] {info}")

    agent = QMIX(n_agents=info["n_agents"], n_actions=info["n_actions"],
                 obs_dim=info["obs_dim"], state_dim=info["state_dim"],
                 hidden_dim=args.hidden_dim, mix_embed_dim=args.mix_embed_dim,
                 lr=args.lr, gamma=args.gamma, optimizer=args.optimizer,
                 use_double_q=not args.no_double_q,
                 use_reward_norm=not args.no_reward_norm)
    buffer = ReplayBuffer(args.buffer_size)

    # 首次 reset 传种子, 之后每局环境自动换新地图
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 5000000)  # 评估环境用不同的地图序列

    csv_path = os.path.join(out_dir, "progress.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "env_steps", "epsilon", "loss",
                         "train_return", "eval_return"])

    best_eval = -float("inf")
    t0, env_steps, loss_acc, loss_n = time.time(), 0, 0.0, 0

    for ep in range(1, args.episodes + 1):
        # 探索率线性退火
        frac = min(1.0, (ep - 1) / max(1, args.eps_anneal))
        epsilon = args.eps_start + frac * (args.eps_finish - args.eps_start)

        ret, steps, episode = run_episode(env, agent, epsilon=epsilon, collect=True)
        buffer.push(episode)
        env_steps += steps

        row_extra, loss_val = [], float("nan")
        if len(buffer) >= args.warmup:
            losses = [agent.learn(buffer.sample(args.batch_size))
                      for _ in range(args.train_iters)]
            loss_val = float(np.mean(losses))
            loss_acc += loss_val
            loss_n += 1
            if args.tau > 0:
                agent.soft_update_target(args.tau)   # 软更新: 每局训练后
            elif ep % args.target_update == 0:
                agent.update_target()                # 硬更新: 每 N 局

        # 周期性贪心评估
        if ep % args.eval_interval == 0:
            mean_r, std_r, _ = evaluate(eval_env, agent, args.eval_episodes)
            row_extra = [f"{mean_r:.2f}±{std_r:.2f}"]
            if mean_r > best_eval:
                best_eval = mean_r
                agent.save(os.path.join(out_dir, "best.pth"),
                           extra={"episode": ep, "eval_return": mean_r,
                                  "info": info, "args": vars(args)})
            eval_msg = f" | eval {mean_r:8.2f} ± {std_r:5.2f}"
        else:
            row_extra, eval_msg = [""], ""

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, env_steps, f"{epsilon:.3f}",
                                    f"{loss_val:.4f}", f"{ret:.2f}"] + row_extra)

        if ep % args.log_interval == 0:
            speed = env_steps / max(1e-9, time.time() - t0)
            avg_loss = loss_acc / max(1, loss_n)
            loss_acc, loss_n = 0.0, 0
            print(f"[ep {ep:6d}] return {ret:8.2f} | eps {epsilon:.3f} "
                  f"| loss {avg_loss:7.4f} | {speed:6.0f} steps/s{eval_msg}",
                  flush=True)

        if ep % 5000 == 0 or ep == args.episodes:
            agent.save(os.path.join(out_dir, "latest.pth"),
                       extra={"episode": ep, "info": info, "args": vars(args)})

    agent.save(os.path.join(out_dir, "latest.pth"),
               extra={"episode": args.episodes, "info": info, "args": vars(args)})
    env.close()
    eval_env.close()
    print(f"[done  ] {args.episodes} episodes, {env_steps} env steps, "
          f"{time.time() - t0:.0f}s total. best eval return = {best_eval:.2f}")


if __name__ == "__main__":
    main()
