# QMIX 复现:PettingZoo simple_spread 多智能体协同覆盖

用 **QMIX**(值分解多智能体强化学习算法, ICML 2018)训练 3 个智能体在
[PettingZoo / MPE](https://pettingzoo.farama.org/environments/mpe/simple_spread/) 的
`simple_spread` 环境中**协同覆盖地标**的从零复现。

纯 PyTorch + CPU 实现,约 600 行注释完备的核心代码,不依赖任何 MARL 框架,
适合逐行读懂 QMIX。

## 1. 任务与算法

### 1.1 环境:simple_spread(合作型)

```
  ▲ 地标(L)     ● 智能体(A)        目标: 3 个智能体协商"谁去哪个地标",
  L1              A1                     平均分配地覆盖所有地标,
      L2      A2     A3                  并避免相互碰撞
```

- 每局 25 步(`max_cycles=25`),观测 18 维(自身速度/位置 + 地标相对位置
  + 其他智能体相对位置),动作空间 `Discrete(5)`(不动 / 上下左右)。
- **奖励是共享的(团队奖励)**:每步 = -(各地标到最近智能体的距离之和)
  - 碰撞惩罚 - 动作惩罚,恒为负 —— 覆盖得越好回报越高。随机策略 ≈ **-27**,
  训练好的策略可达 **-12 ~ -18**。
- 每个智能体只看得到局部观测(部分可观测 POMDP),因此 Q 网络用 **GRU**。

### 1.2 算法:QMIX

QMIX 的核心问题是:合作型 MARL 有共享团队奖励,但执行时要每个智能体独立决策,
怎么把"团队的 Q"拆成"每个智能体自己的 Q"?

**做法**:训练一个混合网络,把各智能体的 Q 值合成为团队 Q_tot,并施加**单调性约束**
(混合权重全部非负):

```
∂Q_tot/∂Q_i ≥ 0    ⟹    argmax_a Q_tot = ( argmax Q_1, argmax Q_2, argmax Q_3 )
```

于是"分散执行"时每个智能体对自己的 Q 网络独立取 argmax,效果就等于团队最优:

- **集中训练**:用全局状态 `state`(54 维,仅训练时可见)通过超网络生成混合权重,
  对 Q_tot 做 DQN 式 TD 学习:
  `L = ( Q_tot(s,a) - [r + γ·Q_tot'(s', a\*)] )²`
- **分散执行**:每个智能体只依赖自己的 GRU Q 网络,`a_i = argmax Q_i`。

## 2. 代码结构(按阅读顺序)

| 文件 | 内容 | 行数 |
|---|---|---|
| `envs.py` | simple_spread 封装(obs 堆叠 / 团队奖励 / 全局状态) | ~100 |
| `buffer.py` | **episode 级**经验回放(RNN 要求整条轨迹训练) | ~90 |
| `networks.py` | `RNNAgent`(GRU Q 网络)+ `QMixer`(超网络混合) | ~80 |
| `qmix.py` | QMIX 主体:选动作 / TD 学习 / 目标网络 / 存取模型 | ~200 |
| `runner.py` | 跑一局收集经验、贪心评估 | ~50 |
| `train.py` | 训练入口(命令行超参数、日志、检查点) | ~140 |
| `evaluate.py` | 贪心评估 + 生成行为演示 GIF | ~90 |
| `plot_results.py` | 从 `progress.csv` 画 reward / loss 曲线 | ~90 |
| `test_learner.py` | 合成环境(已知最优解)验证学习器能否收敛 | ~100 |
| `test_td_crosscheck.py` | 用独立实现交叉验证 TD 损失的正确性 | ~110 |
| `debug_q.py` | 诊断 Q 值量级是否发散(排障利器) | ~80 |

### 关键实现点(也是最容易踩的坑)

1. **RNN 输入要拼 `[obs, last_action_onehot, agent_id_onehot]`**:
   所有智能体共享一套网络参数,靠 agent id 区分身份,靠 last action 让 GRU
   记住"我刚做了什么"。
2. **经验必须按整条 episode 存**(不能像 DQN 存单步转移),因为 GRU 隐藏状态
   在局内前后依赖,训练时从头展开。
3. **目标网络的 RNN 输入约定必须与在线网络一致**(`当前obs + 上一步动作`),
   再用 `q_target[:, 1:]` 错一位做自举 —— 这里错位是复现 QMIX 最典型的 bug,
   症状是"越学越差、贪心策略不如随机"。
4. **目标网络必须用软更新(Polyak, τ=0.005)而不是周期性硬拷贝**:这是本项目
   踩过的最大的坑。硬拷贝 + GRU + 混合网络的自举会形成正反馈,Q 值持续向上
   漂移(奖励恒为负 Q 却变正、loss 每 1000 局翻倍、评估回报越训越差)。
   `test_learner.py` 在一个已知最优解的合成环境上验证:硬更新 Q 漂到 2 倍
   最优值,τ=0.005 软更新则精确收敛。`--tau 0.005` 因此是默认推荐配置
   (与 [marlbenchmark/off-policy](https://github.com/marlbenchmark/off-policy)
   在 MPE 上的做法一致)。
5. **奖励归一化 + Double-Q** 是第二层稳定化(同上仓库的 MPE 配方):
   归一化防止 TD 目标量级越滚越大,Double-Q(DDQN 式:在线网络选动作、
   目标网络估值)抑制 max 过估计。`--no-reward-norm --no-double-q` 可关掉
   做消融实验。
6. **TD 语义可以用独立实现交叉验证**:`test_td_crosscheck.py` 用一份不复用
   `learn()` 代码的手写实现,在同一批数据同一份权重上算出完全相同的 loss
   (差值 0.0),确认 TD 目标构造无误 —— 这是排查"越训越差"类问题的可靠手段。

## 3. 快速开始

```bash
# 1) 安装依赖(建议 Python 3.10+,任何现代版本均可)
python -m venv venv
venv\Scripts\activate            # Windows;Linux 用 source venv/bin/activate
pip install -r requirements.txt

# 2) 冒烟自测(环境/训练/评估全链路,几百局)
python envs.py

# 3) 正式训练(约 2~3 小时 CPU;日志与模型写入 results/qmix_final/)
#    --tau 0.002 --train-iters 1(软更新 + 每局 1 次梯度更新)是稳定训练的关键,
#    详见"关键实现点"第 4、5 条
python train.py --episodes 50000 --tau 0.002 --train-iters 1 --tag qmix_final

# 4) 画 reward 曲线
python plot_results.py results/qmix_final/progress.csv --smooth 1000

# 5) 贪心评估 + 生成演示 GIF
python evaluate.py --checkpoint results/qmix_final/best.pth --gif demo.gif

# (可选)诊断 Q 值是否发散
python debug_q.py --checkpoint results/qmix_final/latest.pth
```

训练进度看控制台,每 500 局打印一行(回报 / 探索率 / loss / 速度),
每 1000 局做一次 10 局贪心评估。`best.pth` 是评估回报最高的检查点。

## 4. 超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--episodes` | 50000 | 总训练局数(每局 25 步) |
| `--buffer-size` | 5000 | 回放缓冲区容量(局) |
| `--batch-size` | 32 | 每次学习采样的 episode 数 |
| `--train-iters` | 2 | 每收集一局后学习几次 |
| `--lr` | 5e-4 | RMSprop 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--target-update` | 200 | 每多少局硬更新目标网络(用 `--tau` 时忽略) |
| `--tau` | 0(推荐 0.002) | 软更新系数;>0 时每局训练后 Polyak 更新 |
| `--train-iters` | 2(推荐 1) | 每局训练后做几次梯度更新;simple_spread 上 2 次会加速 Q 值漂移 |
| `--eps-start/finish/anneal` | 1.0 / 0.05 / 5000 | ε-greedy 退火 |
| `--hidden-dim` | 64 | GRU 隐藏维度 |
| `--mix-embed-dim` | 32 | 混合网络隐藏维度 |

## 5. 结果

实际训练 3 万局(约 75 万环境步,CPU 约 2 小时)后:

| 指标 | 数值 |
|---|---|
| 随机策略回报 | ≈ **-27.7** |
| 训练最优评估回报(ep 6000) | **-18.4 ± 5.1** |
| 最优检查点复测(32 局贪心) | **-20.5 ± 5.3**(单局最好 -9.9) |
| 相对随机策略提升 | 回报提高约 25%~33% |

训练曲线(`results/qmix_final/reward_curve.png`):
约 2000 局脱离随机水平,6000 局左右达到最优(约 -18),之后在 -20~-24 区间
震荡;后期 loss 缓慢上升、评估回报缓慢回落,是 QMIX 在该任务上典型的
"部分可观测 + 非平稳信用分配"局限,也是天然的下一步改进方向(见第 6 节)。

行为演示(`results/qmix_final/demo.gif`):3 个智能体从随机游走学到
"就近分配地标、各奔一方"的覆盖行为。

### 稳定性消融(为什么默认配置长这样)

| 配置 | 1 万局内的表现 |
|---|---|
| 硬更新目标网络(pymarl 默认做法) | loss 每 1000 局翻倍,Q 值上漂,评估退回随机水平以下 |
| + 奖励归一化 + Double-Q | 发散变慢 4 倍,但仍在 5000~8000 局后退化 |
| 软更新 τ=0.005(off-policy 做法) | 合成环境上收敛,真实环境 8000 局后仍缓慢漂移 |
| **τ=0.002 + 每局 1 次更新(本项目默认)** | **稳定:评估保持 -18~-23,无发散** |
| Adam 7e-4(off-policy 脚本参数) | 在本实现上发散最快(约 3500 局 loss 即 >100) |

> 训练产出见 `results/qmix_final/`:曲线 `reward_curve.png`、日志
> `progress.csv`、模型 `best.pth` / `latest.pth`、演示 `demo.gif`。

## 6. 建议的学习路径(两周节奏)

- **第 1~2 天**:读懂 `envs.py` + `buffer.py`,跑通冒烟;理解
  "episode 级回放"为什么是 RNN 策略的必须。
- **第 3~5 天**:精读 `networks.py` + `qmix.py`,对照 QMIX 原文第 3 节,
  弄明白超网络、`abs()` 单调性约束、TD 目标的构造。
- **第 6~8 天**:跑完整训练,用 `plot_results.py` 观察曲线,用 `debug_q.py`
  对比 Q_tot 与真实回报;试试关掉 `use_reward_norm` / `use_double_q`,
  亲眼看 Q 值怎么发散(这是最有价值的实验!)。
- **第 9~14 天**:扩展实验任选:换 `n_agents=4/5`、对比 VDN(把混合网络换成
  求和)、加课程学习(先 1 个地标再 3 个)、或迁移到 `simple_reference` 等环境。

## 7. 参考

- Rashid et al. *QMIX: Monotonic Value Function Factorisation for Deep
  Multi-Agent Reinforcement Learning*, ICML 2018([论文](https://arxiv.org/abs/1803.11485))
- [yangchen1997/Multi-Agent-Reinforcement-Learning](https://github.com/yangchen1997/Multi-Agent-Reinforcement-Learning)
  —— QMIX/VDN/IQL 在 PettingZoo simple_spread 上的 PyTorch 实现,本项目的结构参考
- [marlbenchmark/off-policy](https://github.com/marlbenchmark/off-policy)
  —— QTRAN(ICLR 2022)官方代码,其 MPE 超参配方(奖励归一化 + Double-Q)是本项目
  稳定训练的关键参考
- [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl) —— QMIX 原作者实现的
  pymarl 框架(SMAC 环境)
- [PettingZoo MPE 文档](https://pettingzoo.farama.org/environments/mpe/simple_spread/)
  / [MPE2](https://mpe2.farama.org/)

## License

MIT
