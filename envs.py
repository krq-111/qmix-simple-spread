# -*- coding: utf-8 -*-
"""PettingZoo(MPE)simple_spread 环境封装。

simple_spread 是经典的 *合作型* 多智能体任务:
    N 个智能体(reward 共享)要共同覆盖 N 个地标(landmark),
    每一步的奖励 = -(所有地标到最近智能体的距离之和) - 碰撞惩罚 - 动作惩罚,
    奖励恒为负,智能体靠得越近、碰撞越少,回报越高。

我们把 PettingZoo 的 ParallelEnv API 封装成更顺手的形式:
    obs  统一堆叠成 [n_agents, obs_dim] 的 numpy 数组
    step 接收 python list 的动作(每个元素是一个智能体的离散动作编号)
    奖励取所有智能体奖励的平均值作为"团队奖励"(QMIX 属于合作型算法,
          假设所有智能体共享同一个奖励)
"""
import numpy as np
from mpe2 import simple_spread_v3


class SimpleSpreadEnv:
    """simple_spread 环境的轻量封装(离散动作、团队奖励)。"""

    def __init__(self, n_agents=3, n_landmarks=3, max_cycles=25,
                 local_ratio=0.5, render_mode=None):
        # local_ratio: 奖励中"局部(只算自己)"成分的比例, 取平均后即团队奖励
        self.penv = simple_spread_v3.parallel_env(
            N=n_agents,  # 智能体数量(同时地标数量也是 N)
            local_ratio=local_ratio,
            max_cycles=max_cycles,      # 每局最多步数
            continuous_actions=False,   # QMIX 需要离散动作: 0~4
            render_mode=render_mode,    # None 或 'rgb_array'(生成 GIF 用)
        )
        self.agents = list(self.penv.possible_agents)
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.episode_limit = max_cycles
        self._seed_used = False

    # ------------------------------------------------------------------ API
    def reset(self, seed=None):
        """重置环境, 返回堆叠后的观测 [n_agents, obs_dim]。"""
        # 注意: 只在第一次 reset 时传 seed, 之后传 None 继续沿用随机流,
        # 否则每一局都会是同一张地图
        if seed is not None and not self._seed_used:
            obs, _ = self.penv.reset(seed=seed)
            self._seed_used = True
        else:
            obs, _ = self.penv.reset()
        return self._stack(obs)

    def step(self, actions):
        """执行所有智能体的动作。

        args:
            actions: list[int], 长度为 n_agents, 每个元素取值 0~4
                    (0 不动, 1~4 分别是 -x/+x/-y/+y 方向平移)
        return:
            obs      下一时刻观测 [n_agents, obs_dim]
            reward   团队奖励(float, 所有智能体奖励的平均值)
            done     本局是否结束(到达 max_cycles 后统一结束)
        """
        act_dict = {name: int(a) for name, a in zip(self.agents, actions)}
        obs, rewards, terms, truncs, _ = self.penv.step(act_dict)
        team_reward = float(np.mean(list(rewards.values())))
        done = any(terms.values()) or any(truncs.values())
        return self._stack(obs), team_reward, done

    def state(self):
        """全局状态 [state_dim], 由所有智能体的观测拼接而成, 只给混合网络用。"""
        return np.asarray(self.penv.state(), dtype=np.float32)

    def render(self):
        """返回当前画面 RGB 数组 [H, W, 3](需 render_mode='rgb_array')。"""
        return self.penv.render()

    def close(self):
        self.penv.close()

    # ------------------------------------------------------------- helpers
    def _stack(self, obs_dict):
        """把 {agent_name: obs} 堆叠成固定顺序的 [n_agents, obs_dim]。"""
        return np.stack([np.asarray(obs_dict[name], dtype=np.float32)
                         for name in self.agents])

    @property
    def info(self):
        """环境信息字典, 供 QMIX 构建网络用。"""
        obs_dim = int(self.penv.observation_space(self.agents[0]).shape[0])
        state_dim = int(self.penv.state_space.shape[0])
        n_actions = int(self.penv.action_space(self.agents[0]).n)
        return dict(n_agents=self.n_agents, n_actions=n_actions,
                    obs_dim=obs_dim, state_dim=state_dim,
                    episode_limit=self.episode_limit)


if __name__ == "__main__":
    # 小自测: python envs.py
    env = SimpleSpreadEnv()
    info = env.info
    print("env info:", info)
    obs = env.reset(seed=0)
    print("obs shape:", obs.shape)
    ret, done, steps = 0.0, False, 0
    rng = np.random.default_rng(0)
    while not done:
        obs, r, done = env.step(rng.integers(0, info["n_actions"], info["n_agents"]))
        ret += r
        steps += 1
    print(f"random policy: steps={steps}, return={ret:.2f}")
