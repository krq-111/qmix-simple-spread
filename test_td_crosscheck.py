# -*- coding: utf-8 -*-
"""交叉验证: 用一份独立手写的 TD 目标/损失对照 qmix.learn() 的数值。

思路: 构造一个小 agent + 一个小 batch, 关闭奖励归一化(便于手工核算),
让 learn() 返回 loss; 再用完全独立写的 forward(不复用 learn 的任何代码)
计算同样的 loss。两者应完全一致(误差 < 1e-5)。
用法: python test_td_crosscheck.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from buffer import Episode, ReplayBuffer
from networks import QMixer, RNNAgent
from qmix import QMIX

torch.manual_seed(3)
np.random.seed(3)

NA, N_ACT, OBS_D, STATE_D, T, B = 3, 5, 4, 9, 6, 4
GAMMA = 0.97

agent = QMIX(n_agents=NA, n_actions=N_ACT, obs_dim=OBS_D, state_dim=STATE_D,
             hidden_dim=16, mix_embed_dim=8, gamma=GAMMA,
             use_double_q=False, use_reward_norm=False)
agent.update_target()  # target 与 eval 完全同步, 便于核算

# 保存 learn() 执行前的权重, 供独立实现复用
import copy
w_rnn = copy.deepcopy(agent.rnn_eval.state_dict())
w_mix = copy.deepcopy(agent.mixer_eval.state_dict())

# ---- 构造一个 batch 的数据 ----
buffer = ReplayBuffer(100)
for _ in range(B):
    ep = Episode()
    for t in range(T):
        obs = np.random.randn(NA, OBS_D).astype(np.float32)
        st = np.random.randn(STATE_D).astype(np.float32)
        acts = np.random.randint(0, N_ACT, NA)
        r = float(np.random.randn())
        ep.push(obs, st, acts, r, t == T - 1)
    buffer.push(ep)
batch = buffer.sample(B)

loss_learn = agent.learn(batch)   # 注意: 这一步会更新参数!

# learn 内部更新了参数, 所以独立核算要用"更新前"的权重(agent 的初始权重)
agent2 = QMIX(n_agents=NA, n_actions=N_ACT, obs_dim=OBS_D, state_dim=STATE_D,
              hidden_dim=16, mix_embed_dim=8, gamma=GAMMA,
              use_double_q=False, use_reward_norm=False)
agent2.rnn_eval.load_state_dict(w_rnn)
agent2.mixer_eval.load_state_dict(w_mix)
agent2.rnn_target.load_state_dict(w_rnn)
agent2.mixer_target.load_state_dict(w_mix)

# ---- 独立实现: 与 learn() 相同的数学, 不同写法 ----
obs = batch["obs"]          # [B, T, NA, obs]
st = batch["state"]         # [B, T, S]
acts = batch["actions"]     # [B, T, NA]
rews = batch["rewards"]     # [B, T]
dones = batch["dones"]      # [B, T]

rnn, rnn_t = agent2.rnn_eval, agent2.rnn_target
mix, mix_t = agent2.mixer_eval, agent2.mixer_target
ids = torch.eye(NA)

def chain(net, o, prev_acts):
    """逐 t 展开: 输入 [obs_t, prev_a_{t-1}, id]"""
    Bsz = o.shape[0]
    h = torch.zeros(Bsz, NA, 16)
    out = []
    for t in range(o.shape[1]):
        x = torch.cat([o[:, t], prev_acts[:, t], ids.unsqueeze(0).expand(Bsz, -1, -1)], -1)
        x = x.reshape(Bsz * NA, -1)
        q, h = net(x, h.reshape(Bsz * NA, -1))
        out.append(q.view(Bsz, NA, N_ACT))
        h = h.view(Bsz, NA, -1)
    return torch.stack(out, 1)      # [B, T, NA, n_acts]

prev_acts = torch.zeros(B, T, NA, N_ACT)
prev_acts[:, 1:] = F.one_hot(acts[:, :-1], N_ACT).float()

with torch.no_grad():
    q_eval = chain(rnn, obs, prev_acts)                    # 在线链
    q_tgt = chain(rnn_t, obs, prev_acts)                   # 目标链(同输入约定)
    q_taken = q_eval.gather(3, acts.unsqueeze(-1)).squeeze(-1)   # [B,T,NA]
    q_next_max = q_tgt[:, 1:].max(dim=3)[0]                # [B,T-1,NA]

    # 独立实现混合(直接按公式, 不调用 QMixer.forward)
    def mix_forward(m, q_vals, s):
        Bsz, Tsz = q_vals.shape[0], q_vals.shape[1]
        q = q_vals.reshape(-1, 1, NA)
        ss = s.reshape(-1, s.shape[-1])
        w1 = torch.abs(m.hyper_w1(ss)).view(-1, NA, 8)
        b1 = m.hyper_b1(ss).view(-1, 1, 8)
        hid = F.elu(torch.bmm(q, w1) + b1)
        w2 = torch.abs(m.hyper_w2(ss)).view(-1, 8, 1)
        b2 = m.hyper_b2(ss).view(-1, 1, 1)
        return (torch.bmm(hid, w2) + b2).view(Bsz, Tsz, 1)

    q_tot_eval = mix_forward(mix, q_taken, st).squeeze(-1)         # [B,T]
    q_tot_next = mix_forward(mix_t, q_next_max, st[:, 1:]).squeeze(-1)  # [B,T-1]

    targets = rews[:, :-1] + GAMMA * (1 - dones[:, :-1]) * q_tot_next
    td = q_tot_eval[:, :-1] - targets
    loss_ref = (td ** 2).sum() / (T - 1) / B

print(f"learn() loss  = {loss_learn:.6f}")
print(f"独立实现 loss = {float(loss_ref):.6f}  (用同一初始权重)")
print("差值:", abs(loss_learn - float(loss_ref)))
