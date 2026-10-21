# -*- coding: utf-8 -*-
"""QMIX 算法主体(训练 + 选动作)。

QMIX (Rashid et al., ICML 2018) 属于"值分解"类合作型 MARL 算法:
  - 训练(集中式): 用全局状态 s 混合出 Q_tot, 对 Q_tot 做 DQN 式的
      TD 学习:  L = ( Q_tot(s,a) - [r + gamma * max_a' Q_tot'(s',a')] )^2
  - 执行(分散式): 每个智能体只依赖自己的 RNN Q 网络独立取 argmax。

工程要点(均为 pymarl 系实现的标准做法):
  - 目标网络(rnn_target/mixer_target)周期性硬拷贝, 稳定 TD 目标;
  - RNN 输入拼接 [obs, last_action_onehot, agent_id_onehot];
  - epsilon-greedy 探索, epsilon 随训练线性退火;
  - 梯度裁剪 + RMSprop(与原文一致)。
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from networks import QMixer, RNNAgent


class QMIX:
    def __init__(self, n_agents, n_actions, obs_dim, state_dim,
                 hidden_dim=64, mix_embed_dim=32, lr=5e-4, gamma=0.99,
                 grad_norm_clip=10.0, device="cpu",
                 use_double_q=True, use_reward_norm=True, optimizer="rmsprop"):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.device = torch.device(device)
        # 稳定化技巧(marlbenchmark/off-policy 官方 MPE 配方):
        #   double_q      : DDQN 式目标 —— 在线网络选动作, 目标网络估值, 抑制过估计发散
        #   reward_norm   : 奖励归一化(运行均值/方差), 防止 TD 目标量级越滚越大
        self.use_double_q = use_double_q
        self.use_reward_norm = use_reward_norm
        self._r_mean, self._r_var, self._r_count = 0.0, 1.0, 1e-4

        # RNN 输入 = obs + 上一步动作 onehot + 智能体 id onehot
        self.input_dim = obs_dim + n_actions + n_agents

        # 在线网络与目标网络(结构相同, 参数独立)
        self.rnn_eval = RNNAgent(self.input_dim, n_actions, hidden_dim).to(self.device)
        self.rnn_target = RNNAgent(self.input_dim, n_actions, hidden_dim).to(self.device)
        self.mixer_eval = QMixer(n_agents, state_dim, mix_embed_dim).to(self.device)
        self.mixer_target = QMixer(n_agents, state_dim, mix_embed_dim).to(self.device)
        self.update_target()  # 初始时让目标网络与在线网络一致

        self.params = list(self.rnn_eval.parameters()) + list(self.mixer_eval.parameters())
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.params, lr=lr, eps=1e-5)
        else:  # QMIX 原文用 RMSprop
            self.optimizer = torch.optim.RMSprop(self.params, lr=lr, alpha=0.99, eps=1e-5)

        # 常量: 智能体 id one-hot [n_agents, n_agents]
        self._agent_ids = torch.eye(n_agents, device=self.device)

    # ------------------------------------------------------------ 选动作
    def init_hidden(self, batch_size=1):
        """每条 episode 开始前把所有隐藏状态清零。

        shape [batch_size, n_agents, hidden_dim](训练时 batch_size>1)。
        """
        return torch.zeros(batch_size, self.n_agents, self.hidden_dim,
                           device=self.device)

    def _build_inputs(self, obs, last_action_onehot):
        """拼接 RNN 输入。

        obs:                [B, n_agents, obs_dim]
        last_action_onehot: [B, n_agents, n_actions]
        return:             [B, n_agents, input_dim]
        """
        B = obs.shape[0]
        ids = self._agent_ids.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([obs, last_action_onehot, ids], dim=-1)

    @torch.no_grad()
    def act(self, obs, hidden, last_actions=None, epsilon=0.0):
        """为所有智能体选动作(分散执行, 只用各自的 RNN Q 网络)。

        args:
            obs:          np.ndarray [n_agents, obs_dim]
            hidden:       torch.Tensor [n_agents, hidden_dim](上一时刻的)
            last_actions: list[int] 上一时刻各智能体的动作(None 表示局首)
            epsilon:      探索概率
        return:
            actions: list[int] 长度 n_agents
            new_hidden
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)  # [1, NA, obs]
        if last_actions is None:  # 局首: 上一步动作按零向量处理
            last = torch.zeros(1, self.n_agents, self.n_actions, device=self.device)
        else:
            last = torch.zeros(1, self.n_agents, self.n_actions, device=self.device)
            last[0, torch.arange(self.n_agents),
                 torch.as_tensor(last_actions, device=self.device)] = 1.0
        inputs = self._build_inputs(obs_t, last).view(self.n_agents, -1)
        h = hidden.reshape(self.n_agents, self.hidden_dim)
        q, new_hidden = self.rnn_eval(inputs, h)
        q = q.view(self.n_agents, self.n_actions)

        greedy = q.argmax(dim=-1).cpu().numpy()             # 每个智能体独立 argmax
        if epsilon > 0:
            rand = np.random.random(self.n_agents) < epsilon
            greedy[rand] = np.random.randint(self.n_actions, size=rand.sum())
        return greedy.tolist(), new_hidden

    # ------------------------------------------------------------ 训练
    def learn(self, batch):
        """一批 episode 的 TD 学习, 返回 loss 标量。"""
        obs = batch["obs"].to(self.device)              # [B, T, NA, obs]
        state = batch["state"].to(self.device)          # [B, T, S]
        actions = batch["actions"].to(self.device)      # [B, T, NA]
        rewards = batch["rewards"].to(self.device)      # [B, T]
        dones = batch["dones"].to(self.device)          # [B, T]
        mask = batch["mask"].to(self.device)            # [B, T]
        B, T, NA = obs.shape[0], obs.shape[1], self.n_agents

        # 动作 one-hot [B, T, NA, n_actions]
        onehot = torch.nn.functional.one_hot(actions, self.n_actions).float()

        # ---- 在线网络: 沿时间逐步展开 GRU, 取出"实际执行动作"的 Q 值 ----
        h_eval = self.init_hidden(B)
        h_target = self.init_hidden(B)
        q_eval_list, q_target_list = [], []
        for t in range(T):
            # RNN 输入约定(与 act() 一致): 当前观测 + 上一步动作(t=0 用零向量)
            prev_onehot = onehot[:, t - 1] if t > 0 else torch.zeros_like(onehot[:, 0])
            inputs = self._build_inputs(obs[:, t], prev_onehot).reshape(B * NA, -1)
            q_eval, h_eval = self.rnn_eval(inputs, h_eval.reshape(B * NA, -1))
            q_eval_list.append(q_eval.view(B, NA, self.n_actions))

            # 目标网络用同样的输入约定, 算出的 q_target[t] 即 Q(s_t) 的估计;
            # 对第 t 步的 bootstrap 用 q_target[t+1](见下方切片)
            inputs_tgt = self._build_inputs(obs[:, t], prev_onehot).reshape(B * NA, -1)
            q_tgt, h_target = self.rnn_target(inputs_tgt, h_target.reshape(B * NA, -1))
            q_target_list.append(q_tgt.view(B, NA, self.n_actions))

        q_eval_all = torch.stack(q_eval_list, dim=1)      # [B, T, NA, n_acts]
        q_target_all = torch.stack(q_target_list, dim=1)  # [B, T, NA, n_acts]

        # 各智能体所选动作的 Q 值 [B, T, NA]
        q_taken = q_eval_all.gather(3, actions.unsqueeze(-1)).squeeze(-1)

        # ---- 下一时刻的自举 Q 值(对齐到 t 步的 bootstrap) ----
        if self.use_double_q:
            # Double-Q(DDQN 式): 在线网络挑动作, 目标网络给该动作估值, 抑制 max 过估计
            a_star = q_eval_all[:, 1:].argmax(dim=3, keepdim=True)
            q_boot = q_target_all[:, 1:].gather(3, a_star).squeeze(-1)   # [B, T-1, NA]
        else:
            q_boot = q_target_all[:, 1:].max(dim=3)[0]                   # [B, T-1, NA]

        # ---- 奖励归一化(运行统计), 只影响训练目标, 不改变环境 ----
        if self.use_reward_norm:
            with torch.no_grad():
                flat_r = rewards[mask.bool()]
                b_mean, b_var = flat_r.mean(), flat_r.var(unbiased=False)
                b_cnt = flat_r.numel()
                delta = b_mean - self._r_mean
                tot = self._r_count + b_cnt
                self._r_mean += delta * b_cnt / tot
                m_a = self._r_mean
                self._r_var = (self._r_var * self._r_count
                               + b_var * b_cnt + delta ** 2 * self._r_count * b_cnt / tot) / tot
                self._r_count = tot
            r_std = (self._r_var + 1e-8) ** 0.5
            rewards = (rewards - self._r_mean) / r_std

        # ---- 混合网络 ----
        q_tot_eval = self.mixer_eval(q_taken, state).squeeze(-1)          # [B, T]
        q_tot_target = self.mixer_target(q_boot, state[:, 1:])            # [B, T-1, 1]

        # ---- TD 目标与损失(mask 抹掉补零的位置) ----
        r, d, m = rewards[:, :-1], dones[:, :-1], mask[:, :-1]            # [B, T-1]
        targets = r + self.gamma * (1 - d) * q_tot_target.squeeze(-1)
        td_error = q_tot_eval[:, :-1] - targets.detach()
        masked_td = m * td_error
        loss = (masked_td ** 2).sum() / m.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        """目标网络硬更新: target <- eval。"""
        self.rnn_target.load_state_dict(self.rnn_eval.state_dict())
        self.mixer_target.load_state_dict(self.mixer_eval.state_dict())

    # ------------------------------------------------------------ 存取
    def save(self, path, extra=None):
        ckpt = OrderedDict([
            ("rnn", self.rnn_eval.state_dict()),
            ("mixer", self.mixer_eval.state_dict()),
            ("rnn_target", self.rnn_target.state_dict()),
            ("mixer_target", self.mixer_target.state_dict()),
            ("optimizer", self.optimizer.state_dict()),
        ])
        if extra:
            ckpt.update(extra)
        ckpt["r_stats"] = [self._r_mean, self._r_var, self._r_count]
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.rnn_eval.load_state_dict(ckpt["rnn"])
        self.mixer_eval.load_state_dict(ckpt["mixer"])
        self.rnn_target.load_state_dict(ckpt.get("rnn_target", ckpt["rnn"]))
        self.mixer_target.load_state_dict(ckpt.get("mixer_target", ckpt["mixer"]))
        if "r_stats" in ckpt:
            self._r_mean, self._r_var, self._r_count = ckpt["r_stats"]
