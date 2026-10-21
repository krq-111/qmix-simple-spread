# -*- coding: utf-8 -*-
"""QMIX 的两组神经网络。

1) RNNAgent —— 每个智能体各自的 Q 网络(所有智能体共享同一套参数,
   用 agent id one-hot 区分身份, 这是 QMIX 原文的常规做法):
       输入  = 自身观测 obs + 上一步动作 one-hot + 智能体编号 one-hot
       结构  = Linear -> GRUCell -> Linear
       输出  = 每个离散动作的 Q 值
   用 GRU 是因为环境是部分可观测的(POMDP): 每个智能体只能看到局部观测。

2) QMixer —— 混合网络, QMIX 的灵魂:
       输入 = n_agents 个 Q 值 + 全局状态 state
       输出 = Q_tot(团队总 Q 值)
   关键设计: 超网络(hypernetwork)根据全局状态生成混合权重, 且权重
   全部过 torch.abs() 保证非负 —— 这保证 dQ_tot/dQ_i >= 0(单调性),
   使得 argmax Q_tot 与各智能体独立取 argmax Q_i 等价,
   从而"分散执行"时每个智能体只看自己的 Q 网络即可选出团队最优动作。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """单个智能体的 Q 网络: Linear -> GRU -> Linear。"""

    def __init__(self, input_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, h):
        """x: [batch, input_dim]  h: [batch, hidden_dim] -> q: [batch, n_actions]"""
        h = self.rnn(torch.relu(self.fc1(x)), h)
        return self.fc2(h), h


class QMixer(nn.Module):
    """混合网络: 用全局状态生成权重, 把各智能体 Q 值加权合成 Q_tot。"""

    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # 超网络: 由全局状态生成一层的权重和偏置
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1))

    def forward(self, q_values, states):
        """
        q_values: [B, T, n_agents]   各智能体所选动作的 Q 值
        states:   [B, T, state_dim]  全局状态
        return:   [B, T, 1]          团队总 Q 值 Q_tot
        """
        B, T, _ = q_values.shape
        # 把时间维摊进 batch, 一次性前向
        q = q_values.reshape(-1, 1, self.n_agents)          # [B*T, 1, n_agents]
        s = states.reshape(-1, self.state_dim)              # [B*T, state_dim]

        # 第一层: abs() 保证权重非负 => 单调性约束
        w1 = torch.abs(self.hyper_w1(s)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(s).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)               # [B*T, 1, embed]

        # 第二层: 同样 abs() 非负
        w2 = torch.abs(self.hyper_w2(s)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(s).view(-1, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2                  # [B*T, 1, 1]

        return q_tot.view(B, T, 1)
