# -*- coding: utf-8 -*-
"""评估训练好的 QMIX 模型, 并可生成智能体行为的演示 GIF。

用法:
    python evaluate.py --checkpoint results/run1/best.pth            # 贪心评估 10 局
    python evaluate.py --checkpoint results/run1/best.pth --gif demo.gif
        (生成 3 局的演示动图, 保存到 results/run1/demo.gif)
"""
import argparse
import os

import numpy as np
import torch

from envs import SimpleSpreadEnv
from qmix import QMIX
from runner import evaluate


def load_agent(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    info, hparams = ckpt["info"], ckpt["args"]
    agent = QMIX(n_agents=info["n_agents"], n_actions=info["n_actions"],
                 obs_dim=info["obs_dim"], state_dim=info["state_dim"],
                 hidden_dim=hparams.get("hidden_dim", 64),
                 mix_embed_dim=hparams.get("mix_embed_dim", 32),
                 device=device)
    agent.load(ckpt_path)
    return agent, info, hparams


def record_gif(agent, info, out_path, n_episodes=3, fps=10):
    """用贪心策略跑几局并录制成 GIF。"""
    import imageio
    env = SimpleSpreadEnv(info["n_agents"], info["n_agents"],
                          info["episode_limit"], 0.5, render_mode="rgb_array")
    env.reset(seed=2024)
    frames = [env.render()]
    for _ in range(n_episodes):
        hidden = agent.init_hidden()
        last_actions, done = None, False
        obs = env.reset()
        while not done:
            actions, hidden = agent.act(obs, hidden, last_actions, epsilon=0.0)
            obs, _, done = env.step(actions)
            frames.append(env.render())
            last_actions = actions
    env.close()
    imageio.mimsave(out_path, frames, duration=1000 / fps, loop=0)
    print(f"[gif   ] saved {len(frames)} frames -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--gif", type=str, default="", help="GIF 文件名(留空则不生成)")
    p.add_argument("--gif-episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    agent, info, hparams = load_agent(args.checkpoint)
    env = SimpleSpreadEnv(info["n_agents"], info["n_agents"],
                          info["episode_limit"], 0.5)
    env.reset(seed=2024)
    mean_r, std_r, returns = evaluate(env, agent, args.episodes)
    env.close()
    trained_for = hparams.get("episode", hparams.get("episodes", "?"))
    print(f"[eval  ] checkpoint={args.checkpoint} (trained {trained_for} episodes)")
    print(f"[eval  ] greedy return over {args.episodes} episodes: "
          f"{mean_r:.2f} ± {std_r:.2f}")
    print(f"[eval  ] per-episode: {[round(r, 1) for r in returns]}")

    if args.gif:
        out = args.gif if os.path.isabs(args.gif) else \
            os.path.join(os.path.dirname(args.checkpoint), args.gif)
        record_gif(agent, info, out, args.gif_episodes, args.fps)


if __name__ == "__main__":
    main()
