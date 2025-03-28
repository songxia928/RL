import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import collections
import random

# -------------------------
# 1. 策略网络定义
# -------------------------
class GRPOPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(GRPOPolicyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return F.softmax(logits, dim=-1)

# -------------------------
# 2. GRPO算法实现
# -------------------------
class GRPO:
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 lr=3e-4, eps=0.2, beta=0.01, gamma=0.99, 
                 group_size=16, device='cpu'):
        self.device = device
        self.actor = GRPOPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.eps = eps        # 截断参数
        self.beta = beta      # KL惩罚系数
        self.gamma = gamma    # 折扣因子
        self.group_size = group_size  # 群体采样大小
        
        self.action_dim = action_dim
        self.state_dim = state_dim

    def sample_group(self, state, group_size=None):
        """生成群体动作样本"""
        group_size = group_size or self.group_size
        state = torch.tensor([state]*group_size, dtype=torch.float, device=self.device)
        probs = self.actor(state)
        dist = Categorical(probs)
        actions = dist.sample()
        return actions.cpu().numpy(), probs.detach()  # 返回动作和旧策略概率

    def update(self, transitions):
        """执行策略更新"""
        # 转换数据格式
        states = torch.tensor(transitions['states'], dtype=torch.float, device=self.device)
        old_probs = torch.tensor(transitions['old_probs'], dtype=torch.float, device=self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float, device=self.device).view(-1, 1)

        # 1. 计算相对优势（组内归一化）
        # 确保奖励数量是group_size的整数倍
        assert rewards.shape[0] % self.group_size == 0, \
            f"Reward count ({rewards.shape[0]}) not multiple of group_size ({self.group_size})"
        
        group_rewards = rewards.view(-1, self.group_size)  # [序列长度, 组大小]
        mu = group_rewards.mean(dim=1, keepdim=True)
        std = group_rewards.std(dim=1, keepdim=True) + 1e-8
        A = (group_rewards - mu) / std  # 相对优势
        A = A.view(-1, 1)  # 展开为序列维度

        # 2. 计算策略比率
        new_probs = self.actor(states).gather(1, actions)
        ratio = torch.exp(torch.log(new_probs) - torch.log(old_probs.gather(1, actions)))

        # 3. 计算损失
        # 截断损失
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * A
        clip_loss = -torch.mean(torch.min(surr1, surr2))

        # KL散度惩罚（精确计算）
        kl_div = torch.sum(old_probs * (torch.log(old_probs) - torch.log(new_probs)), dim=1)
        kl_loss = torch.mean(kl_div)

        # 总损失
        total_loss = clip_loss + self.beta * kl_loss

        # 4. 梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'clip_loss': clip_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }

# -------------------------
# 3. 训练辅助工具
# -------------------------
class EpisodeBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.old_probs = []
        self.rewards = []

    def add(self, state, action, old_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.old_probs.append(old_prob)
        self.rewards.append(reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.old_probs.clear()
        self.rewards.clear()

    def to_dict(self):
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'old_probs': np.array(self.old_probs),
            'rewards': np.array(self.rewards)
        }

def moving_average(a, window=5):
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

# -------------------------
# 4. 训练主循环（关键修复）
# -------------------------
def train(env_name='CartPole-v1', num_episodes=1000, render=False):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 超参数配置
    config = {
        'hidden_dim': 128,
        'lr': 3e-4,
        'eps': 0.2,
        'beta': 0.01,
        'gamma': 0.99,
        'group_size': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    agent = GRPO(state_dim, action_dim, **config)
    buffer = EpisodeBuffer()
    returns = []

    for episode in tqdm(range(1, num_episodes+1), desc='Training'):
        state, _ = env.reset()  # 正确获取初始状态
        episode_return = 0
        buffer.clear()
        done = False

        while not done:
            if render and episode % 100 == 0:
                env.render()

            # 群体采样（每个状态生成group_size个动作）
            actions, probs = agent.sample_group(state)
            
            # 处理所有group_size个动作（即使某个动作导致结束）
            for a, p in zip(actions, probs):
                next_state, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated  # 标记是否结束，但继续处理所有动作
                
                buffer.add(
                    state=state,
                    action=a,
                    old_prob=p.cpu().numpy(),
                    reward=reward
                )
                
                episode_return += reward

            # 更新状态（使用最后一个动作的next_state）
            state = next_state

            # 仅在所有动作处理完后检查是否结束
            if done:
                break

        # 执行策略更新
        transitions = buffer.to_dict()
        try:
            loss_info = agent.update(transitions)
        except AssertionError as e:
            tqdm.write(f"Skipping invalid episode: {e}")
            continue

        returns.append(episode_return)
        
        # 进度显示
        if episode % 10 == 0:
            avg_return = np.mean(returns[-10:])
            tqdm.write(f"Episode: {episode}, Return: {avg_return:.2f}, "
                       f"Loss: {loss_info['total_loss']:.4f}, KL: {loss_info['kl_loss']:.4f}")

    env.close()
    return returns

# -------------------------
# 5. 运行入口
# -------------------------
if __name__ == '__main__':
    returns = train(num_episodes=500)
    
    # 绘制训练曲线
    import matplotlib.pyplot as plt
    plt.plot(returns)
    plt.plot(moving_average(returns, window=10))
    plt.title('GRPO Training Curve (CartPole)')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Raw', 'Moving Average (10)'])
    plt.show()