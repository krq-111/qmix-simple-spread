# -*- coding: utf-8 -*-
"""与环境交互的公共逻辑: 跑一局收集经验 / 贪心评估。"""
import numpy as np
import torch

from envs import SimpleSpreadEnv


def run_episode(env, agent, epsilon=0.0, collect=False):
    """跑完整的一局。

    args:
        env:     SimpleSpreadEnv
        agent:  QMIX
        epsilon: 探索率(评估时传 0 即纯贪心)
        collect: True 时返回 Episode 经验对象(训练用)
    return:
        (episode_return, steps, episode or None)
    """
    from buffer import Episode  # 局部导入避免循环依赖

    ep = Episode() if collect else None
    obs = env.reset()
    hidden = agent.init_hidden()
    last_actions = None
    ret, steps, done = 0.0, 0, False

    while not done:
        state = env.state()
        actions, hidden = agent.act(obs, hidden, last_actions, epsilon)
        next_obs, reward, done = env.step(actions)
        if collect:
            ep.push(obs, state, actions, reward, done)
        ret += reward
        steps += 1
        last_actions, obs = actions, next_obs

    return ret, steps, ep


def evaluate(env, agent, n_episodes=10, seed_base=10000):
    """用纯贪心策略跑 n 局, 返回 (平均回报, 标准差, 每局回报列表)。"""
    returns = []
    for k in range(n_episodes):
        ret, _, _ = run_episode(env, agent, epsilon=0.0)
        returns.append(ret)
    return float(np.mean(returns)), float(np.std(returns)), returns
