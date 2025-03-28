

# GRPO：从PPO到群体相对策略优化的进化之路

## 一、PPO的局限性与GRPO的诞生
近端策略优化（PPO）通过截断机制（Clipping）约束策略更新幅度，在稳定性和样本效率上取得了突破。其核心目标函数为：
$$
L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]
$$

其中优势函数 $A_t$ 依赖Critic网络估计值。然而，PPO在大规模模型训练中暴露两大痛点：
1. **Critic依赖**：需维护与Actor规模相当的价值网络，显存占用增加30%以上；
2. **绝对优势偏差**：基于绝对奖励的优势估计易受单一样本波动影响，尤其在稀疏奖励场景（如数学推理）表现不稳定。

DeepSeek团队提出的**群体相对策略优化（GRPO）**，通过三大创新突破上述限制：
- **无Critic架构**：用群体采样替代价值网络，直接计算相对优势；
- **组内竞争机制**：同一问题生成多组输出，基于组内均值/方差归一化优势；
- **双重约束**：结合截断（Clipping）与KL散度惩罚，实现更柔性的策略更新控制。


## 二、GRPO核心原理：群体相对优势估计
### 1. 目标函数设计
GRPO目标函数融合策略梯度、截断约束及KL正则：
$$
\mathcal{L}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) - \beta \cdot \text{KL}[\pi_{\theta_{\text{old}}} || \pi_\theta]\right]
$$
- **相对优势 $\hat{A}_t$**：对同一状态 $s_t$，采样 $G$ 个动作 $a_{t1},...,a_{tG}$，计算组内归一化奖励：
  $$
  \hat{A}_t = \frac{r_t - \mu_G}{\sigma_G + \epsilon}, \quad \mu_G=\frac{1}{G}\sum r_t, \sigma_G=\sqrt{\frac{1}{G}\sum (r_t-\mu_G)^2}
  $$
- **KL散度惩罚**：显式约束新旧策略分布差异，避免剧烈更新。

### 2. 与PPO的核心差异
| 特性                | PPO                                  | GRPO                                 |
|---------------------|--------------------------------------|--------------------------------------|
| 优势估计            | 依赖Critic网络（绝对优势）           | 组内归一化（相对优势）               |
| 价值网络            | 必须（Actor-Critic架构）             | 无需（节省显存30%+）                 |
| 约束机制            | 单一截断（Clipping）                 | 截断+KL散度（双重软约束）            |
| 奖励类型适配        | 适合连续奖励场景                     | 更适用于稀疏/规则奖励（如数学推理） |
| 显存复杂度          | O(2N)（Actor+Critic）               | O(N)（仅Actor）                      |


## 三、GRPO算法实现：从PPO代码进化
以下是基于PyTorch的GRPO核心实现（对比PPO代码）：

### 1. 策略网络（仅保留Actor）
```python
class GRPOPolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
```

### 2. 群体采样与优势计算
```python
class GRPO:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, eps, beta, gamma, group_size, device):
        self.actor = GRPOPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.eps = eps  # 截断范围
        self.beta = beta  # KL惩罚系数
        self.gamma = gamma
        self.group_size = group_size  # 每组采样数量
        self.device = device

    def sample_group(self, state, group_size):
        """生成群体动作样本"""
        state = torch.tensor([state]*group_size, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        return actions.cpu().numpy(), probs  # 保存旧策略概率

    def update(self, transitions):
        """群体更新逻辑"""
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        old_probs = torch.tensor(transitions['old_probs'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).view(-1, 1).to(self.device)

        # 组内归一化优势计算
        group_rewards = rewards.view(-1, self.group_size)  # [序列长度, 组大小]
        mu = group_rewards.mean(dim=1, keepdim=True)
        std = group_rewards.std(dim=1, keepdim=True) + 1e-8
        A = (group_rewards - mu) / std  # 相对优势
        A = A.view(-1, 1)  # 展开为序列维度

        # 策略更新
        new_probs = self.actor(states).gather(1, transitions['actions'].view(-1, 1))
        ratio = torch.exp(torch.log(new_probs) - torch.log(old_probs))
        
        # 截断目标
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * A
        clip_loss = -torch.mean(torch.min(surr1, surr2))
        
        # KL散度惩罚（近似计算）
        kl_div = torch.sum(old_probs * (torch.log(old_probs) - torch.log(new_probs)), dim=1)
        kl_loss = torch.mean(kl_div)
        
        # 总损失
        total_loss = clip_loss + self.beta * kl_loss
        
        # 梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

### 3. 训练流程对比
```python
# PPO训练循环（参考原代码）
def ppo_train():
    while not done:
        action = agent.take_action(state)
        # 单一样本采集...
        agent.update(transition_dict)  # 含Critic更新

# GRPO训练循环（群体采样）
def grpo_train():
    for episode in range(num_episodes):
        transitions = {'states': [], 'actions': [], 'old_probs': [], 'rewards': []}
        state = env.reset()
        while not done:
            # 群体采样（G=16）
            actions, probs = agent.sample_group(state, group_size=16)
            for a, p in zip(actions, probs):
                next_state, reward, done, _ = env.step(a)
                transitions['states'].append(state)
                transitions['actions'].append(a)
                transitions['old_probs'].append(p)
                transitions['rewards'].append(reward)
            state = next_state
        agent.update(transitions)  # 仅更新Actor
```


## 四、GRPO的工程优势与应用场景
1. **显存优化**：去除Critic网络，千亿参数模型训练显存占用降低约30%（参考DeepSeek实践）。
2. **稀疏奖励适配**：通过组内竞争挖掘相对优势，在数学推理、代码生成等规则奖励场景表现更优。
3. **稳定性提升**：双重约束（截断+KL）避免策略崩溃，尤其适合长序列生成任务（如对话系统）。
4. **分布式扩展**：群体采样天然支持并行计算，结合vLLM等加速库，训练吞吐量提升2-3倍（见CSDN教程）。

> **典型应用**：DeepSeek-R1在数学推理任务中，通过GRPO将解题准确率提升18%，同时将训练成本降低40%（据MobotStone分析）。


## 五、总结：GRPO的进化意义
GRPO通过**群体智慧替代个体裁判**，在以下维度超越PPO：
- **架构革新**：无Critic设计突破传统Actor-Critic范式，适配大模型时代的显存限制；
- **机制升级**：相对优势估计解决绝对奖励偏差，KL约束提供更精细的策略控制；
- **场景拓展**：从连续控制（如CartPole）到离散生成（如代码/数学），重塑RL在复杂任务中的应用边界。

未来，随着群体采样规模（N从16到128+）的动态优化，以及与FlashAttention等加速技术的深度整合，GRPO有望成为大模型强化学习的标配算法，推动AGI在推理、创作等领域的持续突破。

```python
# 完整GRPO训练示例（伪代码）
class GRPOAgent:
    def __init__(self, model, group_size=16, beta=0.01, eps=0.2):
        self.model = model  # 大语言模型作为策略网络
        self.group_size = group_size
        self.beta = beta
        self.eps = eps

    def train(self, dataset):
        for batch in dataset:
            prompts = batch['questions']
            # 群体采样：同一问题生成G个回答
            completions = self.model.generate(prompts, num_samples=self.group_size)
            # 奖励计算（规则/模型驱动）
            rewards = compute_rewards(completions)  # 如数学题正确性评分
            # 组内归一化
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            # 策略更新
            loss = self.calculate_loss(completions, rewards)
            self.model.backward(loss)
            self.model.step()

    def calculate_loss(self, completions, rewards):
        # 计算新旧策略概率比
        log_probs = self.model.log_prob(completions)
        old_log_probs = self.model_old.log_prob(completions).detach()
        ratio = torch.exp(log_probs - old_log_probs)
        # 截断目标+KL惩罚
        surr = torch.min(ratio*rewards, torch.clamp(ratio, 1-self.eps, 1+self.eps)*rewards)
        kl_div = torch.mean(torch.sum(old_log_probs - log_probs, dim=-1))
        return -torch.mean(surr) + self.beta * kl_div
```

> **代码说明**：结合Hugging Face Transformers与vLLM，可实现分布式群体采样。实际应用中需注意：
> - 采样组大小（group_size）权衡稳定性与计算成本（大N需更高并行能力）；
> - KL系数（beta）动态调整（如使用Trust Region策略）；
> - 奖励函数设计（规则/模型驱动）需匹配任务特性（如数学题的步骤正确性评分）。

通过GRPO，强化学习正从“单一个体试错”迈向“群体智慧进化”，这种范式转变不仅提升了训练效率，更打开了大模型在复杂推理领域的潜力。正如DeepSeek的实践所示，GRPO不仅是算法创新，更是工程与理论结合的典范，为大模型时代的RLHF（基于人类反馈的强化学习）提供了可扩展的新路径。