# 【强化学习】05.Actor-Critic 算法原理

Actor-Critic 是一种结合了策略梯度方法和值函数方法的强化学习算法。它通过同时学习策略和价值两个网络，既能够像策略梯度方法一样直接优化策略，又能利用值函数降低梯度估计的方差。以下是关于 Actor-Critic 算法的详细分析。

---

## 1. 算法原理

Actor-Critic 算法的核心思想是将策略优化（Actor）和价值评估（Critic）结合起来。具体来说：
- **Actor (策略网络)**：负责生成策略，给定状态 $s$，输出动作的概率分布 $\pi_{\theta}(a|s)$。通过策略梯度进行更新，使得期望回报最大化。
- **Critic (价值网络)**：负责评估当前策略的价值，给定状态 $s$，输出状态价值 $V(s)$，用来估计时间差分（TD）误差，从而指导策略的更新。

### **目标函数**
Actor-Critic 使用以下目标函数来更新 Actor 和 Critic：

1. 策略网络（Actor）的目标是最大化累积奖励：
   $$
   \nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a|s) \cdot \delta \right]
   $$
   其中 $\delta = r + \gamma V(s') - V(s)$ 是 TD 误差，用于衡量当前策略与实际奖励的偏差。

2. 价值网络（Critic）的目标是最小化 TD 误差的均方误差：
   $$
   L(\phi) = \mathbb{E}_{\pi_{\theta}} \left[ \left( r + \gamma V(s') - V(s) \right)^2 \right]
   $$

通过同时优化这两个目标，Actor-Critic 算法能够在减少梯度估计方差的同时，保持对策略的稳定优化。

---

## 2. 算法步骤

以下是 Actor-Critic 的具体算法流程：

**(1). 初始化**：
   - 初始化策略网络（Actor）和价值网络（Critic）及其参数。
   - 设置优化器和超参数（学习率、折扣因子 $\gamma$、隐藏层大小等）。

**(2). 交互采样**：
   - 在每个训练回合，从环境中采样一条轨迹：
     - 根据当前策略 $\pi_{\theta}(a|s)$ 选择动作 $a$；
     - 执行动作，获得奖励 $r$ 和下一个状态 $s'$；
     - 判断是否到达终止状态。

**(3). 计算时序差分（TD）目标**：
   - 计算 TD 目标：
     $$
     \text{TD Target: } r + \gamma V(s')
     $$
   - 计算 TD 误差：
     $$
     \delta = r + \gamma V(s') - V(s)
     $$

**(4). 更新 Critic（价值网络）**：
   - 通过最小化 TD 误差的均方误差，更新 Critic 的参数：
     $$
     L(\phi) = \left( \delta \right)^2
     $$

**(5). 更新 Actor（策略网络）**：
   - 使用策略梯度方法更新 Actor 的参数：
     $$
     \nabla_{\theta} J(\theta) = \log \pi_{\theta}(a|s) \cdot \delta
     $$

**(6). 重复训练**：
   - 不断采样数据并更新 Actor 和 Critic，直到策略收敛或达到预设的训练回合。

Actor-Critic 算法通过同时优化策略和价值函数平衡了策略梯度和价值方法的优缺点，是一种高效且适用于复杂任务的强化学习算法。它在处理连续动作空间和减少梯度估计方差方面表现出色，但仍面临训练不稳定和样本效率低的问题。与 DQN 和策略梯度相比，Actor-Critic 更适用于复杂的控制任务，但需要更多的调参和训练技巧。

---

## 3. 优缺点

### **A.优点**
- **降低梯度估计的方差**： 相比于纯策略梯度方法（如 REINFORCE），Actor-Critic 使用 Critic 估计的 TD 误差作为基线，显著降低了梯度估计的方差，训练更加稳定。
- **兼具策略优化和价值评估**： Actor-Critic 同时优化策略和价值函数，能够更高效地学习复杂任务。
- **适用于连续和离散动作空间**：  Actor-Critic 算法能够很好地处理连续动作空间，适用范围更广。
- **稳定性优于纯策略梯度**：  通过 Critic 的引入，Actor-Critic 在复杂环境中比策略梯度算法表现更稳定。

### **B. 缺点**
- **样本效率低**：  Actor-Critic 算法仍然需要大量交互数据，尤其是在高维状态空间中，样本效率较低。
- **易受高偏差影响**：   由于 Critic 是基于函数逼近的，Critic 的不准确可能会导致梯度估计偏差，最终影响策略的学习。
- **训练不稳定**：   Actor 和 Critic 的更新是相互依赖的，Critic 估计的错误可能会影响 Actor 的更新，从而导致训练振荡。

---

## 4. 游戏介绍

在代码中，Actor-Critic 算法被应用于经典的强化学习任务 **CartPole-v0**。下面是关于该任务的简要介绍：

- **目标**：通过控制推力来保持杆子直立尽可能长时间。
- **状态空间**：由 4 个连续变量组成：
  - 小车位置；
  - 小车速度；
  - 杆子角度；
  - 杆子角速度。
- **动作空间**：包含 2 个离散动作：
  - 向左施加推力；
  - 向右施加推力。
- **奖励函数**：每个时间步杆子保持直立，奖励为 +1。
- **终止条件**：
  - 杆子角度过大；
  - 小车偏离屏幕边界。

Actor-Critic 算法通过学习状态的价值函数和策略网络，能够高效地解决这一控制问题，实现稳定的杆子平衡。

---

## 5. Actor-Critic 与 DQN 和策略梯度的比较

| **维度**             | **Actor-Critic**                           | **DQN**                                  | **策略梯度（如 REINFORCE）**               |
|-----------------------|--------------------------------------------|------------------------------------------|-------------------------------------------|
| **核心思想**         | 同时学习策略和价值函数                    | 通过 Q 表学习状态-动作值函数             | 直接优化策略 $\pi(a|s)$                   |
| **网络结构**         | 两个网络（Actor 和 Critic）               | 单个 Q 网络                              | 单个策略网络                              |
| **目标函数**         | $\log \pi_{\theta}(a\|s) \cdot \delta$     | TD 误差 $(r + \gamma \max Q') - Q$       | $\log \pi_{\theta}(a\|s) \cdot G_t$        |
| **动作空间**         | 离散和连续动作空间                       | 主要用于离散动作空间                     | 离散和连续动作空间                        |
| **样本效率**         | 中等                                     | 高                                      | 低                                       |
| **梯度估计方差**     | 较低                                     | 不涉及梯度                              | 高                                       |
| **学习稳定性**       | 一般，依赖 Critic 的估计准确性             | 稳定，借助经验回放与目标网络             | 不稳定，受高方差影响                      |
| **探索机制**         | 动作概率输出策略，天然探索                | $\epsilon$-贪婪策略                      | 动作概率输出策略，天然探索                |
| **实现复杂度**       | 较高                                     | 一般                                    | 较低                                     |


---

## 6. 训练代码
```python
# --------------- train.py -----------------

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

  
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)                 # 通过神经网络（策略网络 PolicyNet）将状态映射为动作概率分布 πθ(a∣s)
        action_dist = torch.distributions.Categorical(probs)      # 构造动作分布
        action = action_dist.sample()              # 根据动作概率分布，采样动作
        return action.item()

    def update(self, transition_dict):   # update 中没有 策略梯度中的for 训练来计算 累计折扣奖励 Gt。而是利用 next_states 计算时序差分（TD）
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 预计算 Critic 的值
        V_s = self.critic(states)  # 计算 V(s_t)
        V_s_next = self.critic(next_states)  # 计算 V(s_t+1)

        # (1) 时序差分（TD）目标
        td_target = rewards + self.gamma * V_s_next * (1 - dones)  # td_target = r_t + γ⋅V(s_t+1)
    
        # (2) 时序查分（TD）误差
        td_delta = td_target - V_s  # δ = r_t+γV(s_t+1)−V(s_t) = td_target - V(s_t) 。 TD 误差 δ 衡量当前价值网络预测值 V(s_t)与实际目标值的偏差

        # (3) Critic Loss
        critic_loss = torch.mean( F.mse_loss(V_s, td_target.detach()) )    # L(ϕ) = E[(V(s_t) - r_t+γ⋅V(s_t+1))^2] = E[(V(s_t) - td_target)^2]
                                                                             # Critic 的优化目标是最小化 TD 误差的均方误差
                                                                             # detach() 的作用是防止 TD 目标的梯度反传，仅更新 Critic 的参数

        # (4) Actor Loss
        log_probs = torch.log(self.actor(states).gather(1, actions))   # 获取策略网络的输出（动作概率分布 πθ(a∣s))
                                                                       # .gather(1, actions)：选择采样动作的概率。torch.log：计算该动作的对数概率 log ⁡πθ(a∣s)
        actor_loss = torch.mean(-log_probs * td_delta.detach())    # ∇θ J(θ) = E πθ[∇θ log ⁡πθ(a∣s)⋅δ]。Actor 的优化目标是最大化以下策略梯度目标
                                                                # 代码中通过最小化 − log ⁡πθ(a∣s)⋅δ 实现
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.reset(seed=0)

torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)


```

```python
# --------------- rl_utils.py -----------------
from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()[0]
                done = False
                while not done:
                    action = agent.take_action(state)

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                
```


```python
# --------------- 打印 -----------------
Iteration 0: 100%|████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 70.66it/s, episode=100, return=21.500]
Iteration 1: 100%|████████████████████████████████████████████████████████████| 100/100 [00:02<00:00, 45.48it/s, episode=200, return=71.000]
Iteration 2: 100%|███████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 28.86it/s, episode=300, return=104.700]
Iteration 3: 100%|███████████████████████████████████████████████████████████| 100/100 [00:07<00:00, 14.02it/s, episode=400, return=165.500]
Iteration 4: 100%|███████████████████████████████████████████████████████████| 100/100 [00:09<00:00, 10.43it/s, episode=500, return=193.600]
Iteration 5: 100%|███████████████████████████████████████████████████████████| 100/100 [00:09<00:00, 10.73it/s, episode=600, return=192.700]
Iteration 6: 100%|███████████████████████████████████████████████████████████| 100/100 [00:09<00:00, 11.09it/s, episode=700, return=200.000]
Iteration 7: 100%|███████████████████████████████████████████████████████████| 100/100 [00:08<00:00, 11.92it/s, episode=800, return=200.000]
Iteration 8: 100%|███████████████████████████████████████████████████████████| 100/100 [00:08<00:00, 11.72it/s, episode=900, return=200.000]
Iteration 9: 100%|██████████████████████████████████████████████████████████| 100/100 [00:09<00:00, 10.04it/s, episode=1000, return=200.000]
```

