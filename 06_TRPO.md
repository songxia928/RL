

# 【强化学习】06.信任区域策略优化(TRPO) 算法原理

TRPO（Trust Region Policy Optimization）是一种策略优化算法，属于强化学习中的策略梯度方法。它通过约束策略更新的步幅，防止策略的剧烈变化，从而提高训练的稳定性和效率。在以下部分，我们将从算法原理、算法步骤、优缺点以及与其他强化学习算法（DQN、策略梯度、Actor-Critic）的对比来分析 TRPO。

---

## 1. 算法原理

TRPO 是策略优化的改进算法，旨在解决策略梯度方法如 REINFORCE 和 Actor-Critic 中策略更新步幅过大可能导致训练不稳定的问题。核心思想是在更新策略时，限制新旧策略之间的变化幅度，使得每次策略更新保持在允许的“信任域”（Trust Region）内。




### **1.1 TRPO 的目标函数和约束条件**

TRPO 的优化目标是：
$$
\max_\theta \; L(\theta) = \mathbb{E}_{s, a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s, a) \right]
$$
同时需要满足约束条件：
$$
\mathbb{E}_{s \sim \pi_{\text{old}}} \left[ D_{\text{KL}}(\pi_{\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s)) \right] \leq \delta
$$

其中：
- $\pi_{\theta}(a|s)$：当前策略的概率；
- $\pi_{\theta_{\text{old}}}(a|s)$：旧策略的概率；
- $L(\theta)$ 是代理目标函数，衡量新策略的表现；
- $D_{\text{KL}}$ 是新旧策略间的 KL 散度，用于限制策略的更新幅度；
- $\delta$ KL 散度的约束阈值，信任域的阈值，防止策略更新过度。

这意味着，我们希望在优化 $L(\theta)$ 的同时，保证新旧策略间的变化幅度不会太大。

---

### **1.2 将目标函数展开**

为了更清楚地分析目标函数，我们对 $L(\theta)$ 进行泰勒展开。假设当前策略参数为 $\theta_{\text{old}}$，新参数为 $\theta_{\text{new}} = \theta_{\text{old}} + \Delta\theta$，则目标函数可以在 $\theta_{\text{old}}$ 附近展开为：
$$
L(\theta_{\text{new}}) \approx L(\theta_{\text{old}}) + g^T \Delta\theta
$$

其中，$g = \nabla_\theta L(\theta_{\text{old}})$ 是目标函数在当前点的梯度，表示当前策略方向上的上升斜率。

同时，对 KL 散度约束也进行二阶展开：
$$
\mathbb{E}[D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)] \approx \frac{1}{2} \Delta\theta^T H \Delta\theta
$$
其中，$H = \nabla^2_\theta D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)$ 是 KL 散度的黑塞矩阵。

---

### **1.3 二次优化问题的引入**

在保证 KL 散度约束的情况下，目标是找到一个参数更新方向 $\Delta\theta$，即优化以下问题：
$$
\max_{\Delta\theta} \; g^T \Delta\theta, \quad \text{s.t. } \frac{1}{2} \Delta\theta^T H \Delta\theta \leq \delta
$$

#### **解释约束条件**
- 这里的约束 $\frac{1}{2} \Delta\theta^T H \Delta\theta \leq \delta$ 表示新旧策略的 KL 散度不能超过阈值 $\delta$。KL 散度的二阶展开保证了这一约束是近似准确的。

#### **解释目标**
- $g^T \Delta\theta$ 是优化目标，表示在参数更新方向 $\Delta\theta$ 上，$L(\theta)$ 的增长量。

这个二次优化问题的目标就是在 KL 散度约束之内，找到让目标函数 $L(\theta)$ 增长最快的方向 $\Delta\theta$。

---

### **1.4 为什么 $x$ 是搜索方向？**

为了求解这个优化问题，使用 **拉格朗日乘子法**。构造拉格朗日函数：
$$
\mathcal{L}(\Delta\theta, \lambda) = g^T \Delta\theta - \lambda \left( \frac{1}{2} \Delta\theta^T H \Delta\theta - \delta \right)
$$

对 $\mathcal{L}$ 求导，得到：
$$
\nabla_{\Delta\theta} \mathcal{L} = g - \lambda H \Delta\theta = 0
$$

解得：
$$
\Delta\theta = \frac{1}{\lambda} H^{-1} g
$$

这表明，更新方向 $\Delta\theta$ 与 $H^{-1} g$ 成正比，而 $\Delta\theta$ 的具体大小由拉格朗日乘子 $\lambda$ 控制。

令 $x = H^{-1} g$，则方向 $x$ 是我们希望的搜索方向，表示在 KL 散度限制下，目标函数 $L(\theta)$ 的最优增长方向。

---

### **1.5 TRPO 中的搜索方向与步长调整**

在 TRPO 中，搜索方向 $x = H^{-1} g$ 是通过共轭梯度法求解的，而步长系数 $\alpha$ 通过 KL 散度约束计算：
$$
\alpha = \sqrt{\frac{2 \delta}{x^T H x}}
$$

最终的参数更新为：
$$
\Delta\theta = \alpha x
$$

这确保了：
1. 更新方向 $x$ 最大化了目标函数 $L(\theta)$ 的增长；
2. 更新幅度 $\alpha$ 符合 KL 散度限制。

---

### **1.6 为什么这个优化问题和 TRPO 的目标函数一致**

从上面的推导可以看出，TRPO 的原始目标是：
$$
\max_\theta \; L(\theta), \quad \text{s.t. } \mathbb{E}[D_{\text{KL}}] \leq \delta
$$

通过展开和近似，将其转化为二次优化问题：
$$
\max_{\Delta\theta} \; g^T \Delta\theta, \quad \text{s.t. } \frac{1}{2} \Delta\theta^T H \Delta\theta \leq \delta
$$

- **目标一致性**：优化目标 $g^T \Delta\theta$ 是 $L(\theta)$ 在当前点的线性近似，直接反映了目标函数的增长量。
- **约束一致性**：KL 散度的二阶近似 $\frac{1}{2} \Delta\theta^T H \Delta\theta$ 是原始约束条件的精确近似。

因此，这个二次优化问题是 TRPO 原始目标函数的一个合理近似解，且通过 $x$ 找到了最优的更新方向。

---

### **1.7 总结**

1. 二次优化问题的目标是找到在 KL 散度约束下 $L(\theta)$ 增长最快的方向。
2. 搜索方向 $x = H^{-1} g$ 是这个问题的最优解，因为它直接最大化了目标函数的增长。
3. 通过步长调整 $\alpha$，TRPO 保证了更新幅度符合 KL 散度约束。
4. 这个优化过程与 TRPO 的原始目标函数完全一致，是通过线性化目标和二次近似约束的合理解法。



---

## 2. 算法步骤

以下是 TRPO 的算法流程：

**(1). 初始化**
- 初始化策略网络（Actor）和价值网络（Critic）。
- 设置超参数：信任域约束 $\delta$、步长系数 $\alpha$、折扣因子 $\gamma$ 和 GAE 参数 $\lambda$。

**(2). 数据采样**
- 与环境交互，按照当前策略 $\pi_{\theta}(a|s)$ 采样一批轨迹，记录：
  - 状态 $s$、动作 $a$、奖励 $r$、下一状态 $s'$、是否终止 $done$。

**(3). 计算 GAE（广义优势估计）**
- 计算时序差分误差 $ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $；
- 计算优势函数 $A_t$：
$$
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

**(4). 更新 Critic（价值网络）**
- 计算 TD 目标：
$$
\text{TD Target: } r + \gamma V(s')
$$
- 使用均方误差（MSE）损失优化 Critic 网络：
$$
L(\phi) = \mathbb{E} \left[ \left( r + \gamma V(s') - V(s) \right)^2 \right]
$$

**(5). 更新 Actor（策略网络）**
- 计算策略目标函数的梯度：
$$
g = \nabla_{\theta} \mathbb{E} \left[ \log \pi_{\theta}(a|s) \cdot A_t \right]
$$
- 使用共轭梯度法求解梯度和黑塞矩阵的约束问题：
$$
H \cdot x = g
$$
其中 $H$ 是 KL 散度的二阶导数矩阵。

- 通过线性搜索找到最优步幅，更新策略参数：
$$
\theta_{\text{new}} = \theta_{\text{old}} + \alpha \cdot x
$$

**(6). 重复训练**
- 重复采样、优化，直到策略收敛或达到预设的训练回合。

---

## 3. 优缺点

### **优点：**
- **更新稳定性高**： TRPO 避免了策略更新幅度过大的问题，通过 KL 散度约束实现训练稳定性。
- **高性能**： 在复杂环境中，TRPO 能有效提高策略的收敛速度和最终性能。
- **适用于高维和连续动作空间**： TRPO 能在高维状态和连续动作空间表现良好，适用范围更广。
- **降低梯度估计的方差**： 借助优势函数 $A_t$，TRPO 能降低策略梯度的高方差问题。

### **缺点：**
- **实现复杂**： TRPO 涉及二阶优化（如共轭梯度法）和线性搜索，代码实现复杂度较高。
- **计算成本高**： 每次策略更新需计算 KL 散度、梯度和黑塞矩阵的乘积，共轭梯度法和线性搜索进一步增加计算成本。
- **不适合大规模环境**： 在稀疏奖励或高维离散动作环境中，TRPO 的性能可能受限，且计算开销较大。

---

## 4. 游戏介绍

在代码中，TRPO 算法被应用于经典的强化学习任务 **CartPole-v0**：

### **任务目标**
- 控制小车的左右移动以保持杆子的平衡，尽可能延长杆子直立的时间。

### **环境特征**
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
  - 杆子角度超过阈值；
  - 小车偏离边界。

---

## 5. 与 DQN、策略梯度、Actor-Critic 的比较

| **维度**             | **TRPO**                                | **DQN**                                  | **策略梯度（如 REINFORCE）**               | **Actor-Critic**                     |
|-----------------------|-----------------------------------------|------------------------------------------|-------------------------------------------|---------------------------------------|
| **核心思想**         | 限制新旧策略的 KL 散度，稳定更新       | 使用 Q 函数学习状态-动作值               | 直接优化策略 $\pi(a|s)$                   | 同时学习策略和价值函数                |
| **网络结构**         | 策略网络（Actor）+价值网络（Critic）   | 单个 Q 网络                              | 单个策略网络                              | 两个网络（Actor 和 Critic）           |
| **目标函数**         | KL 散度约束的策略目标                  | TD 误差 $(r + \gamma \max Q') - Q$       | $\log \pi_{\theta}(a\|s) \cdot G_t$        | $\log \pi_{\theta}(a\|s) \cdot \delta$ |
| **动作空间**         | 连续和离散动作空间                    | 主要用于离散动作空间                     | 连续和离散动作空间                        | 连续和离散动作空间                   |
| **梯度估计方差**     | 低                                     | 不涉及梯度                              | 高                                       | 较低                                 |
| **训练稳定性**       | 非常稳定                               | 稳定，借助目标网络与经验回放             | 不稳定，受高方差影响                      | 一般，依赖 Critic 的准确性            |
| **样本效率**         | 中等                                   | 高                                      | 低                                       | 中等                                 |
| **计算成本**         | 高，需共轭梯度法和线性搜索             | 较低                                    | 较低                                     | 中等                                 |
| **更新方式**         | 批量更新                               | 批量更新                                | 每回合更新策略                           | 每步更新 Critic 和 Actor              |


---

## 6. 训练代码
```python
# -------------- train.py ----------------

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils
import copy


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

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


class TRPO:
    """ TRPO算法 """
    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)                 # 通过神经网络（策略网络 PolicyNet）将状态映射为动作概率分布 πθ(a∣s)
        action_dist = torch.distributions.Categorical(probs)      # 构造动作分布
        action = action_dist.sample()              # 根据动作概率分布，采样动作

        return action.item()

    # 该函数用于计算 KL 散度二阶导数（黑塞矩阵 H）和一个向量的乘积，即 H⋅v。
     # TRPO 的约束条件为：E_s∼πold[ D_KL(πold(⋅∣s)∥πθ(⋅∣s)) ]  ≤  δ
     # 对应的黑塞矩阵（KL 散度二阶导数）为：H = ∇^2_θ E_s∼πold[D_KL(πold(⋅∣s)∥πθ(⋅∣s))]
    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        new_action_dists = torch.distributions.Categorical(self.actor(states))  # 新策略分布 πθ(⋅∣s)

        # KL 散度
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))  # KL 距离 D_KL(π_old || πθ)
    
        # 一阶梯度
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)  # KL 散度的一阶梯度
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])  # 拉平为向量

        # 与 vector 乘积
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)  # 一阶梯度和输入向量的点积

        # 二阶梯度
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())  # 二阶梯度
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])  # 拉平为向量
        return grad2_vector

    # 共轭梯度法用于高效求解约束优化问题中的线性方程组，目标是计算：x = H^−1·g   <=  H·x = g
        #     g = ∇θ L(θ)：目标函数的梯度
        #     H：KL 散度的二阶导数矩阵（黑塞矩阵）
        # 共轭梯度法避免了直接计算 H^−1，通过迭代优化得到结果 
    def conjugate_gradient(self, grad, states, old_action_dists):  # 共轭梯度法求解方程
        x = torch.zeros_like(grad)  # 初始解为 0
        r = grad.clone()  # 初始残差 r = g
        p = grad.clone()  # 初始方向 p = g
        rdotr = torch.dot(r, r)  # 初始残差的二次范数

        for i in range(10):  # 最大迭代次数为 10
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)  # 计算 H*p
            alpha = rdotr / torch.dot(p, Hp)  # 步长 α = r^T·r / p^T·H·p   
            x += alpha * p  # 更新解 x
            r -= alpha * Hp  # 更新残差 r
            new_rdotr = torch.dot(r, r)  # 新的残差
            if new_rdotr < 1e-10:  # 如果收敛，则停止
                break

            beta = new_rdotr / rdotr  # 更新系数 β = (r_new^T·r_new) / (r^T·r)
            p = r + beta * p  # 更新方向 p
            rdotr = new_rdotr
        return x

    # 计算策略目标： L(θ) = E[r(θ)⋅A]
    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):  # 计算策略目标 
        log_probs = torch.log(actor(states).gather(1, actions))  # 新策略的对数概率
        ratio = torch.exp(log_probs - old_log_probs)  # 概率比值 r(θ) = πθ / π_old
        return torch.mean(ratio * advantage)  # 计算策略目标  L(θ) = E[r(θ)⋅A] 。 

    # 通过线性搜索法，找到满足目标值上升且 KL 散度不超过阈值的策略参数更新幅度
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector( self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)  # 旧策略的 目标函数

        for i in range(15):  # 线性搜索主循环
            coef = self.alpha**i  # 搜索步长系数
            new_para = old_para + coef * max_vec     # 新策略参数。其中，max_vec = x * max_coef 是最大更新方向
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())  
            new_action_dists = torch.distributions.Categorical(new_actor(states))  # 新策略 分布
            
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))  # D_KL(πold(⋅∣s)∥πθ(⋅∣s)) ：新旧策略的 KL 散度，用于限制策略更新幅度
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)  # 新策略的 目标函数
            
            if new_obj > old_obj and kl_div < self.kl_constraint: # 新策略的目标函数值比旧策略更高 and 新旧策略的 KL 散度小于阈值 self.kl_constraint
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):  # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)   # L(θ) = E[πθ(a∣s) ∥ πold(a∣s) ⋅ A(s,a)]
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())   # g=∇θ L(θ)
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()   # 将梯度拉平成向量存储为 obj_grad，为后续共轭梯度法求解方向 H^−1⋅g 做准备

        # ==== 方向x。   用共轭梯度法计算 x = H^(-1)·g
        x = self.conjugate_gradient(obj_grad, states, old_action_dists)  # max_⁡θ g^T·x    s.t. 1/2·x^T·H·x ≤ δ
                                                                            # 其中，
                                                                            #     g = ∇θ L(θ)：目标函数的梯度
                                                                            #     H：KL 散度的二阶导数矩阵（黑塞矩阵）。
                                                                            #     x：搜索方向
        # ==== 求步长coef。    线性搜索
        Hx = self.hessian_matrix_vector_product(states, old_action_dists, x)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(x, Hx) + 1e-8))    # 最大步长 α = (2δ/x^T·H·x)^0.5
        
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    x * max_coef)  # 线性搜索
        
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)


        # 预计算 Critic 的值
        V_s = self.critic(states)  # V(s_t)
        V_s_next = self.critic(next_states)  # V(s_t+1)

        # (1) 计算 TD 目标
        td_target = rewards + self.gamma * V_s_next * (1 - dones)  # TD 目标: r_t + γ⋅V(s_t+1)

        # (2) 计算 TD 误差
        td_delta = td_target - V_s  # TD 误差: δ = td_target - V(s_t)

        # (3) Critic 的损失函数
        critic_loss = torch.mean(F.mse_loss(V_s, td_target.detach()))  # L(ϕ) = E[(V(s_t) - r_t+γ⋅V(s_t+1))^2] = E[(V(s_t) - td_target)^2] 
                                                                       # 和 Actor-Critic 中的 Critic 的损失函数 一样 


        # ==== 更新价值函数

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数

        # ==== 更新策略函数
        # 预计算 Actor 的输出
        actor_output = self.actor(states)  # 策略网络的输出 πθ(a|s)

        # (1) 计算优势函数
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)   #  A = A_πold(s,a)

        # (2) 计算旧策略的对数概率
        old_log_probs = torch.log(actor_output.gather(1, actions)).detach()   # log ⁡πold(a∣s)

        # (3) 构造旧策略的分布
        old_action_dists = torch.distributions.Categorical(actor_output.detach())  # πold(a∣s), 用于计算 KL 散度和目标函数

        # (4) 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)


num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
critic_lr = 1e-2
kl_constraint = 0.0005
alpha = 0.5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.reset(seed=0)

torch.manual_seed(0)
agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
             kl_constraint, alpha, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()


```

```python
# -------------- rl_utils.py ----------------
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
# -------------- 打印 ----------------
Iteration 0: 100%|███████████████████████████████████████████████████████████████| 50/50 [00:02<00:00, 17.80it/s, episode=50, return=25.500]
Iteration 1: 100%|██████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 25.92it/s, episode=100, return=35.200]
Iteration 2: 100%|██████████████████████████████████████████████████████████████| 50/50 [00:02<00:00, 22.49it/s, episode=150, return=48.200]
Iteration 3: 100%|██████████████████████████████████████████████████████████████| 50/50 [00:02<00:00, 18.35it/s, episode=200, return=77.600]
Iteration 4: 100%|██████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 13.89it/s, episode=250, return=95.500]
Iteration 5: 100%|██████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 13.90it/s, episode=300, return=99.800]
Iteration 6: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.26it/s, episode=350, return=120.400]
Iteration 7: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  8.54it/s, episode=400, return=140.700]
Iteration 8: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  9.66it/s, episode=450, return=149.000]
Iteration 9: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  8.69it/s, episode=500, return=147.700]
2025-01-08 00:52:35.094 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:52:41.481 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:52:41.497 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:52:41.569 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:52:41.586 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:52:41.787 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:52:55.402 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:53:01.511 python[87273:4278333] +[CATransaction synchronize] called within transaction
2025-01-08 00:53:15.967 python[87273:4278333] +[CATransaction synchronize] called within transaction
```
