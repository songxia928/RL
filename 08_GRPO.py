import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
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
# 2. GRPO算法实现（核心修复）
# -------------------------
class GRPO:
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 lr=3e-4, eps=0.2, beta=0.01, gamma=0.99, 
                 device='cpu'):
        self.device = device
        self.actor = GRPOPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.eps = eps        # 截断参数
        self.beta = beta      # KL惩罚系数
        self.gamma = gamma    # 折扣因子

        self.action_dim = action_dim
        self.state_dim = state_dim

    def sample(self, state):
        """单动作采样"""
        state = torch.tensor([state], dtype=torch.float, device=self.device)
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        return action.cpu().numpy()[0], prob.detach().cpu().numpy()[0]

    def update(self, transitions, discounted_rewards):
        """执行策略更新（修复优势和KL计算）"""
        states = torch.tensor(transitions['states'], dtype=torch.float, device=self.device)
        old_probs = torch.tensor(transitions['old_probs'], dtype=torch.float, device=self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        A = torch.tensor(discounted_rewards, dtype=torch.float, device=self.device).view(-1, 1)

        # 1. 计算策略比率
        new_probs = self.actor(states).gather(1, actions)
        ratio = torch.exp(torch.log(new_probs) - torch.log(old_probs.gather(1, actions)))

        # 2. 计算损失
        # 截断损失
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * A
        clip_loss = -torch.mean(torch.min(surr1, surr2))

        # KL散度惩罚（分布间精确计算）
        old_dist = Categorical(old_probs)
        new_dist = Categorical(self.actor(states))
        kl_div = kl_divergence(old_dist, new_dist)
        kl_loss = torch.mean(kl_div)

        # 总损失
        total_loss = clip_loss + self.beta * kl_loss

        # 3. 梯度更新
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

            # 单动作采样（修复）
            action, old_prob = agent.sample(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(
                state=state,
                action=action,
                old_prob=old_prob,
                reward=reward
            )

            state = next_state
            episode_return += reward

        # 计算折扣累积奖励（修复优势估计）
        rewards = buffer.rewards
        discounted_rewards = []
        running_reward = 0
        for r in reversed(rewards):
            running_reward = r + agent.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

        # 执行策略更新
        transitions = buffer.to_dict()
        loss_info = agent.update(transitions, discounted_rewards)

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
    returns = train(num_episodes=1000)
    
    # 绘制训练曲线
    import matplotlib.pyplot as plt
    plt.plot(returns)
    plt.plot(moving_average(returns, window=10))
    plt.title('GRPO Training Curve (CartPole)')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Raw', 'Moving Average (10)'])
    plt.show()
    