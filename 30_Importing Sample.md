
# 强化学习中的重要性采样：原理、应用与进阶实践

## 核心动机：离策略学习的必要性
在强化学习中，策略评估与优化通常依赖数据采样。当直接从目标策略（Target Policy）采样面临高成本、高风险或低效率时（如机器人控制、医疗决策场景），离策略（Off-Policy）学习成为必然选择。重要性采样（Importance Sampling, IS）作为离策略学习的核心工具，允许利用行为策略（Behavior Policy）生成的历史数据，通过权重修正实现对目标策略的价值估计。其核心挑战在于解决分布差异：**如何用旧分布的数据估计新分布的期望？**

为了讲解清楚 强化学习中的重要性采样，本文将按照： 蒙特卡洛方法 --》 重要性采样 --》 强化学习中的重要性采样，三个主要部分介绍。


## 一、蒙特卡洛方法 - 采样估计的基础


在计算复杂积分或期望时，蒙特卡洛方法是一种强大的工具。其核心目标是计算：  
$$\mathbb{E}_p[f(\mathbf{x})] = \int p(\mathbf{x})f(\mathbf{x})d\mathbf{x}$$  
思想是通过采样求平均来近似期望：  
$$\mathbb{E}_p[f(\mathbf{x})] \approx \frac{1}{N} \sum_{i=1}^N f(\mathbf{x}_i), \quad \mathbf{x}_i \sim p(\mathbf{x})$$  

![alt text](f4e88e6e8ee75544c39a8f9cc14940df_720.png)






从可视化角度看，图中红色曲线为 $f(\mathbf{x})$，蓝色曲线为 $p(\mathbf{x})$，紫色区域 $p(\mathbf{x})f(\mathbf{x})$ 的面积即积分结果。通过从 $p(\mathbf{x})$ 采样 $\{\mathbf{x}_i\}$，计算 $f(\mathbf{x}_i)$ 的均值，即可逼近目标期望。

### 中心极限定理的作用
蒙特卡洛估计的均值 $s = \frac{1}{N} \sum_{i=1}^N f(\mathbf{x}_i)$ 服从正态分布：  
$$s \xrightarrow{d} \mathcal{N}(\mu, \sigma^2), \quad 其中 \mu = \mathbb{E}_p[f(\mathbf{x})], \sigma^2 = \frac{1}{N}\mathbb{V}_p[f(\mathbf{x})]$$  
这意味着随着采样数 $N$ 增加，估计的方差 $\mathbb{V}_p[s]$ 会按 $\frac{1}{N}$ 衰减，宽度 $2\sqrt{\mathbb{V}_p[s]}$ 也会缩小，估计更精确。


![alt text](592d7997261f5dfc7a60c561658272b3_720.png)


## 二、重要性采样：突破采样限制的利器
### 2.1 原理推导
当从 $p(\mathbf{x})$ 直接采样困难时，引入易采样的分布 $q(\mathbf{x})$，通过数学变换：  
$$
\begin{align*}
\mathbb{E}_p[f(\mathbf{x})] &= \int p(\mathbf{x})f(\mathbf{x})d\mathbf{x} \\
&= \int q(\mathbf{x}) \frac{p(\mathbf{x})}{q(\mathbf{x})}f(\mathbf{x})d\mathbf{x} \\
&= \mathbb{E}_q\left[\frac{p(\mathbf{x})}{q(\mathbf{x})}f(\mathbf{x})\right]
\end{align*}
$$  
于是，可从 $q(\mathbf{x})$ 采样 $\{\mathbf{x}_i\}$，用以下公式估计期望：  
$$\mathbb{E}_q\left[\frac{p(\mathbf{x})}{q(\mathbf{x})}f(\mathbf{x})\right] \approx \frac{1}{N} \sum_{i=1}^N \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}f(\mathbf{x}_i), \quad \mathbf{x}_i \sim q(\mathbf{x})$$  
其中，$\frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}$ 称为重要性权重。

### 2.2 核心优势
1. **无偏性**：$\mathbb{E}_q[r] = \mathbb{E}_p[f(\mathbf{x})]$，估计结果依然无偏。  
2. **方差优化潜力**：新的方差 $\mathbb{V}_q[r] = \frac{1}{N}\mathbb{V}_q\left[\frac{p(\mathbf{x})}{q(\mathbf{x})}f(\mathbf{x})\right]$。若选择合适的 $q(\mathbf{x})$，可使 $\mathbb{V}_q\left[\frac{p(\mathbf{x})}{q(\mathbf{x})}f(\mathbf{x})\right] < \mathbb{V}_p[f(\mathbf{x})]$，提升估计效率。  
**关键策略**：让 $q(\mathbf{x})$ 在 $|p(\mathbf{x})f(\mathbf{x})|$ 高的区域也具有高概率密度，减少权重的波动。

### 2.3 直观示例：理解 $q(\mathbf{x})$ 的作用

![alt text](QQ_1742724917741.png)

通过三组图像对比：  
- **左图**：直接从 $p(\mathbf{x})$ 采样，展示 $p(\mathbf{x})$ 与 $f(\mathbf{x})$ 的分布。  
- **中图**：引入 $q(\mathbf{x})$（绿色曲线），它在 $f(\mathbf{x})$ 高值区域有更高概率密度。  
- **右图**：展示 $\frac{p(\mathbf{x})}{q(\mathbf{x})}f(\mathbf{x})$ 与 $q(\mathbf{x})$ 的关系。合适的 $q(\mathbf{x})$ 能让采样更集中于对期望贡献大的区域，降低方差。


重要性采样通过更换采样分布 $q(\mathbf{x})$，解决了直接从复杂分布 $p(\mathbf{x})$ 采样的难题。它在贝叶斯推断、罕见事件概率估计、强化学习等领域广泛应用。理解其原理的核心在于：利用数学变换转移采样分布，通过权重修正偏差，同时通过合理设计 $q(\mathbf{x})$ 优化估计方差。这一技术体现了采样方法在计算效率上的巧妙突破，是连接理论推导与实际应用的重要桥梁。


### 2.4 几何视角：概率分布的坐标变换
想象两个分布 $ p(x) $ 和 $ q(x) $ 是同一函数空间的不同坐标系：
- $ q(x) $ 是我们能采样的“坐标系”。
- **$ \rho(x) = p(x)/q(x) $ 是坐标变换的“基向量”**。
- 期望 $ \mathbb{E}_p[f] $ 是函数 $ f(x) $ 在 $ p $-坐标系的投影，通过 $ \rho(x) $ 转换为 $ q $-坐标系的加权和。





### 2.5 常见误区澄清
#### 误区1：“权重越大，样本越重要”
**事实**：权重是概率校正因子，而非样本重要性。高权重样本可能来自低概率区域，反而增加方差。





## 三、强化学习中的重要性采样

在强化学习中，策略评估与优化通常依赖数据采样。当直接从目标策略（Target Policy）采样面临高成本、高风险或低效率时（如机器人控制、医疗决策场景），离策略（Off-Policy）学习成为必然选择。重要性采样（Importance Sampling, IS）作为离策略学习的核心工具，允许利用行为策略（Behavior Policy）生成的历史数据，通过权重修正实现对目标策略的价值估计。其核心挑战在于解决分布差异：**如何用旧分布的数据估计新分布的期望？**

---
### 3.1 Off-Policy学习的核心挑战
在强化学习中，目标策略 $\pi$ 和行为策略 $b$ 的轨迹分布差异导致直接重用数据存在偏差。重要性采样的核心任务是：**通过轨迹权重修正，使 $b$ 生成的轨迹能无偏估计 $\pi$ 的价值函数**。

#### 策略评估问题形式化
设轨迹 $\tau = (s_0,a_0,s_1,a_1,...,s_T)$，其回报为 $G(\tau)$。目标策略的价值函数为：  
$$V^\pi(s) = \mathbb{E}_{\tau \sim \pi}[G(\tau)|s_0=s]$$  
利用 $b$ 的轨迹估计 $V^\pi(s)$ 时，需计算重要性权重：  
$$\rho(\tau) = \prod_{t=0}^{T-1} \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$$  
从而得到修正后的估计：  
$$\hat{V}^\pi(s) = \frac{1}{N} \sum_{i=1}^N \rho(\tau_i) G(\tau_i)$$



```python
import numpy as np
from collections import defaultdict

def off_policy_mc_evaluation(env, target_policy, behavior_policy, 
                           num_episodes, gamma=0.99):
    """
    Off-Policy蒙特卡洛策略评估
    Args:
        env: 强化学习环境
        target_policy: 目标策略 π(a|s)
        behavior_policy: 行为策略 b(a|s) 
        num_episodes: 采样轨迹数量
        gamma: 折扣因子
    Returns:
        V: 估计的状态价值函数
    """
    # 初始化价值函数和计数
    V = defaultdict(float)
    counts = defaultdict(int)
    
    for _ in range(num_episodes):
        # 生成轨迹
        trajectory = []
        state = env.reset()
        done = False
        
        while not done:
            action_probs = behavior_policy(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            trajectory.append( (state, action, reward) )
            state = next_state
        
        # 逆向计算重要性权重和回报
        G = 0.0
        rho = 1.0
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = gamma * G + reward
            
            # 更新计数和价值估计（加权重要性采样）
            counts[state] += 1
            V[state] += (rho * G - V[state]) / counts[state]
            
            # 更新重要性权重
            pi_prob = target_policy(state)[action]
            b_prob = behavior_policy(state)[action]
            rho *= pi_prob / b_prob
            
            # 提前终止权重为0的轨迹
            if rho == 0:
                break  
                
    return V
```


### 3.2 时序差分学习中的重要性采样 （Expected SARSA）
在TD学习框架下，单步更新的重要性权重简化为：  
$$\rho_t = \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$$  
Q值更新规则调整为：  
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_{t+1} + \gamma \rho_{t+1} Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) \right]$$  
这种形式在Expected SARSA和Tree Backup算法中被广泛使用。



```python
def td_importance_sampling(env, target_policy, behavior_policy, 
                          num_episodes, alpha=0.1, gamma=0.99):
    """
    单步TD重要性采样算法
    """
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 根据行为策略选择动作
            action_probs = behavior_policy(state)
            action = np.random.choice(n_actions, p=action_probs)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 计算重要性权重
            rho = target_policy(state)[action] / behavior_policy(state)[action]
            
            # 计算目标策略的期望值
            expected_next_value = np.sum(
                target_policy(next_state) * Q[next_state]
            )
            
            # TD更新
            td_target = reward + gamma * expected_next_value
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * rho * td_error
            
            state = next_state
            
    return Q
```


### 3.3 多步重要性采样的乘积权重
对于n步TD学习，需考虑连续动作的联合概率比：  
$$\rho_{t:t+n} = \prod_{k=t}^{t+n-1} \frac{\pi(a_k|s_k)}{b(a_k|s_k)}$$  
此时价值估计为：  
$$\hat{V}^\pi(s_t) = \mathbb{E}_b\left[ \rho_{t:t+n} \left( \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} \right) + \gamma^n \rho_{t:t+n} V(s_{t+n}) \right]$$  
乘积权重会指数级放大方差，需配合截断或归一化技术使用。



```python
# 多步TD重要性采样（n-step TD with IS）
class NStepISAgent:
    def __init__(self, n_steps, gamma, alpha):
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.buffer = []
        
    def update(self, Q):
        """处理n步经验"""
        states, actions, rewards, rhos = zip(*self.buffer)
        
        # 计算累积奖励和权重
        G = 0
        total_rho = 1.0
        for k in range(self.n_steps):
            G += (self.gamma**k) * rewards[k]
            total_rho *= rhos[k]
            
        # 更新起始状态的Q值
        start_state = states[0]
        start_action = actions[0]
        Q[start_state][start_action] += self.alpha * total_rho * (
            G - Q[start_state][start_action]
        )
        
    def train(self, env, target_policy, behavior_policy, num_episodes):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            self.buffer = []
            
            while not done:
                # 收集n步经验
                while len(self.buffer) < self.n_steps and not done:
                    action_probs = behavior_policy(state)
                    action = np.random.choice(len(action_probs), p=action_probs)
                    next_state, reward, done, _ = env.step(action)
                    
                    # 计算重要性权重
                    rho = target_policy(state)[action] / behavior_policy(state)[action]
                    
                    self.buffer.append( (state, action, reward, rho) )
                    state = next_state
                
                if len(self.buffer) >= self.n_steps:
                    self.update(Q)
                    self.buffer.pop(0)
                    
        return Q
```



---

## 四、方差与偏差的平衡艺术
### 4.1 重要性采样的方差困境
重要性权重的方差可形式化为：  
$$\mathbb{V}_b[\rho(\tau)] = \mathbb{E}_b[\rho^2(\tau)] - (\mathbb{E}_b[\rho(\tau)])^2$$  
当 $\pi$ 与 $b$ 差异较大时，方差会随轨迹长度指数爆炸。例如在Atari游戏中，单步权重方差约为1.2，100步后方差膨胀至 $1.2^{100} \approx 8 \times 10^7$。

### 4.2 加权重要性采样（WIS）
通过归一化权重抑制方差：  
$$\hat{V}^\pi_{WIS}(s) = \frac{\sum_{i=1}^N \rho(\tau_i) G(\tau_i)}{\sum_{i=1}^N \rho(\tau_i)}$$  
WIS以引入微小偏差为代价显著降低方差，实际应用更广泛。

### 4.3 混合策略与截断技术
- **Adaptive Mixture**：混合原始和加权估计，$\hat{V} = \lambda \hat{V}_{IS} + (1-\lambda)\hat{V}_{WIS}$
- **Clipping**：限制单个权重的最大值，如PPO中设置 $\rho_t \in [1-\epsilon,1+\epsilon]$  

PPO的代理目标函数显式包含重要性权重：  
$$L^{CLIP}(\theta) = \mathbb{E}_t\left[ \min\left( \rho_t(\theta) A_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$  
其中 $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。这种设计在策略更新中平衡了方差与稳定性。


```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 策略网络（Actor）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # 价值网络（Critic）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOTrainer:
    def __init__(self, model, lr=3e-4, gamma=0.99, epsilon=0.2, ent_coef=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # PPO超参数
        self.gamma = gamma          # 折扣因子
        self.epsilon = epsilon      # Clipping阈值
        self.ent_coef = ent_coef    # 熵系数
        
    def update(self, states, actions, old_log_probs, rewards, dones):
        """
        PPO核心更新函数
        states:        状态序列 [batch_size, state_dim]
        actions:       动作序列 [batch_size]
        old_log_probs: 旧策略的对数概率 [batch_size]
        rewards:       奖励序列 [batch_size]
        dones:         终止标记 [batch_size]
        """
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # 计算优势函数（简单实现，实际应使用GAE）
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze()
            advantages = rewards + (1 - dones) * self.gamma * values - values
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多epoch更新（通常3-4次）
        for _ in range(3):
            # 获取新策略的概率
            new_probs, _ = self.model(states)
            dist = Categorical(new_probs)
            
            # 计算新策略的对数概率
            new_log_probs = dist.log_prob(actions)
            
            # 重要性权重（概率比）
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # 两种替代损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # 策略损失（取最小值）
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失（MSE）
            _, new_values = self.model(states)
            value_loss = nn.MSELoss()(new_values.squeeze(), rewards)
            
            # 熵正则项
            entropy = dist.entropy().mean()
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
            
            # 梯度更新
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

# 使用示例
if __name__ == "__main__":
    # 初始化环境和模型
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = PPOActorCritic(state_dim, action_dim)
    trainer = PPOTrainer(model)
    
    # 数据收集（简化版）
    states, actions, rewards, dones = [], [], [], []
    state = env.reset()
    for _ in range(256):  # 收集256步数据
```

---

## 五、关键挑战与前沿方向
### 5.1 高维动作空间的困境
当动作空间维度增加时，$\pi(a|s)/b(a|s)$ 的估计误差会被放大。解决方法包括：  
- **低方差策略参数化**：如使用高斯策略保证概率密度平滑  
- **分层重要性采样**：在动作子空间分解权重

### 5.2 混合离线与在线学习
在Offline RL场景中，IS需要处理分布外（OOD）动作。最新研究如IMAP算法通过重要性权重调整行为克隆损失：  
$$L(\theta) = \mathbb{E}_{(s,a)\sim \mathcal{D}} \left[ \frac{\pi_\theta(a|s)}{b(a|s)} \|Q(s,a) - Q_{target}\|^2 \right]$$

---

## 总结
重要性采样在强化学习中的核心价值在于打破数据来源与目标策略的强耦合，但其成功依赖于精细的方差控制。未来的研究方向可能包括：  
1. **自适应权重裁剪**：根据轨迹长度动态调整截断阈值  
2. **隐式策略建模**：通过GAN等生成模型避免显式概率密度计算  
3. **物理启发的方差缩减**：借鉴分子动力学中的重加权技术  


## 引用

[1]. https://zhuanlan.zhihu.com/p/596909041

[2]. https://blog.csdn.net/qq_22866291/article/details/145560939

[3]. https://zhuanlan.zhihu.com/p/669378380

[4]. https://www.yuque.com/chenjiarui-i3tp3/sv7cbq/afns6z

[5]. https://www.bilibili.com/video/BV18M4y1p73k/





