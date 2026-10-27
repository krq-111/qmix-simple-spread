# -*- coding: utf-8 -*-
"""生成"随机策略 vs 训练好的 QMIX"左右对比 GIF。

两边的环境用同一个随机种子(同一张初始地图), 确保差异只来自策略本身:
    左: 随机策略(未训练)
    右: QMIX 训练后的贪心策略

用法:
    python make_comparison.py --checkpoint results/qmix_final/best.pth \
        --gif compare_demo.gif --episodes 2
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

from envs import SimpleSpreadEnv
from evaluate import load_agent


def label_frame(frame, text):
    """在帧顶部写标签文字。"""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    d.text((14, 8), text, fill=(255, 255, 255))
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gif", default="compare_demo.gif")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    agent, info, _, _ = load_agent(args.checkpoint)
    rng = np.random.default_rng(0)

    env_rand = SimpleSpreadEnv(info["n_agents"], info["n_agents"],
                               info["episode_limit"], 0.5,
                               render_mode="rgb_array")
    env_qmix = SimpleSpreadEnv(info["n_agents"], info["n_agents"],
                               info["episode_limit"], 0.5,
                               render_mode="rgb_array")

    frames = []
    for ep in range(args.episodes):
        # 两个环境用同一 seed 重置 => 同一张初始地图, 公平对比
        env_rand.reset(seed=args.seed + ep)
        obs_q = env_qmix.reset(seed=args.seed + ep)

        done_r = done_q = False
        hidden, last = agent.init_hidden(), None
        frames.append(np.hstack([
            label_frame(env_rand.render(), "Random policy  (return ~ -27.7)"),
            label_frame(env_qmix.render(), "QMIX trained  (return ~ -19)")]))

        while not (done_r and done_q):
            if not done_r:
                _, _, done_r = env_rand.step(
                    rng.integers(0, info["n_actions"], info["n_agents"]))
            if not done_q:
                acts_q, hidden = agent.act(obs_q, hidden, last, epsilon=0.0)
                obs_q, _, done_q = env_qmix.step(acts_q)
                last = acts_q
            frames.append(np.hstack([
                label_frame(env_rand.render(), "Random policy  (return ~ -27.7)"),
                label_frame(env_qmix.render(), "QMIX trained  (return ~ -19)")]))
    env_rand.close()
    env_qmix.close()

    import imageio
    out = args.gif if args.gif.startswith(("results/", "/")) else \
        "results/qmix_final/" + args.gif
    imageio.mimsave(out, frames, duration=1000 / args.fps, loop=0)
    print(f"[gif   ] {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
