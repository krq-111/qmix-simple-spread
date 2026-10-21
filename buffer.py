# -*- coding: utf-8 -*-
"""经验回放缓冲区(episode 级)。

QMIX 的智能体网络是 RNN(GRU), 隐藏状态在一条 episode 内部前后依赖,
所以不能像 DQN 那样存单步转移, 而是**按整条 episode 存储**,
采样时每次取出若干条完整 episode 做批训练。

simple_spread 每局固定 25 步, 其实不会出现长短不一的情况,
但这里仍写了通用的"补零 + mask"逻辑, 换环境也不用改。
"""
from collections import deque

import numpy as np
import torch


class Episode:
    """一条完整轨迹(所有智能体共享时间轴)。"""

    def __init__(self):
        self.obs = []      # 每步 [n_agents, obs_dim]
        self.state = []    # 每步 [state_dim]      (全局状态)
        self.actions = []  # 每步 [n_agents]        (int)
        self.rewards = []  # 每步标量               (团队奖励)
        self.dones = []    # 每步 0/1

    def push(self, obs, state, actions, reward, done):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.state.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.rewards.append(np.float32(reward))
        self.dones.append(np.float32(done))

    def __len__(self):
        return len(self.rewards)


class ReplayBuffer:
    """存放最近 capacity 条 episode 的环形队列。"""

    def __init__(self, capacity):
        self._buf = deque(maxlen=capacity)

    def push(self, episode):
        if len(episode) > 0:
            self._buf.append(episode)

    def __len__(self):
        return len(self._buf)

    def sample(self, batch_size):
        """随机采样 batch_size 条 episode, 返回对齐后的张量字典。

        返回形状(T 为该 batch 中最长 episode 的步数):
            obs     [B, T, n_agents, obs_dim]  float32
            state   [B, T, state_dim]          float32
            actions [B, T, n_agents]           int64
            rewards [B, T]                     float32
            dones   [B, T]                     float32
            mask    [B, T]                     float32  (1=真实数据, 0=补零)
        """
        idx = np.random.randint(0, len(self._buf), size=min(batch_size, len(self._buf)))
        eps = [self._buf[i] for i in idx]
        B, T = len(eps), max(len(e) for e in eps)

        obs = np.zeros((B, T, eps[0].obs[0].shape[0], eps[0].obs[0].shape[1]), np.float32)
        state = np.zeros((B, T, eps[0].state[0].shape[0]), np.float32)
        actions = np.zeros((B, T, eps[0].actions[0].shape[0]), np.int64)
        rewards = np.zeros((B, T), np.float32)
        dones = np.zeros((B, T), np.float32)
        mask = np.zeros((B, T), np.float32)

        for b, e in enumerate(eps):
            L = len(e)
            obs[b, :L] = e.obs
            state[b, :L] = e.state
            actions[b, :L] = e.actions
            rewards[b, :L] = e.rewards
            dones[b, :L] = e.dones
            mask[b, :L] = 1.0

        return dict(obs=torch.from_numpy(obs), state=torch.from_numpy(state),
                    actions=torch.from_numpy(actions), rewards=torch.from_numpy(rewards),
                    dones=torch.from_numpy(dones), mask=torch.from_numpy(mask))
