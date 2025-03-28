import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import rl_utils  # 保留用户原有的PPO工具
import PPO

# ===========================
# 0. 统一配置与工具
# ===========================
class Config:
    ENV_NAME = 'CartPole-v1'
    NUM_EPISODES = 1000
    HIDDEN_DIM = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 0

    # PPO专属配置
    PPO_CFG = {
        'actor_lr': 1e-3,
        'critic_lr': 1e-2,
        'lmbda': 0.95,
        'epochs': 10,
        'eps': 0.2,
        'gamma': 0.98,
    }

    # GRPO专属配置
    GRPO_CFG = {
        'lr': 3e-4,
        'eps': 0.2,
        'beta': 0.01,
        'gamma': 0.99,
    }

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ===========================
# 1. PPO训练函数
# ===========================
def train_ppo():
    set_seed(Config.SEED)
    env = gym.make(Config.ENV_NAME)
    env.reset(seed=Config.SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO.PPO(
        state_dim=state_dim,
        hidden_dim=Config.HIDDEN_DIM,
        action_dim=action_dim,
        actor_lr=Config.PPO_CFG['actor_lr'],
        critic_lr=Config.PPO_CFG['critic_lr'],
        lmbda=Config.PPO_CFG['lmbda'],
        epochs=Config.PPO_CFG['epochs'],
        eps=Config.PPO_CFG['eps'],
        gamma=Config.PPO_CFG['gamma'],
        device=Config.DEVICE
    )

    return rl_utils.train_on_policy_agent(
        env=env,
        agent=agent,
        num_episodes=Config.NUM_EPISODES
    )

# ===========================
# 2. GRPO训练函数
# ===========================
class GRPO:
    def __init__(self, state_dim, action_dim, hidden_dim=128, **kwargs):
        self.device = kwargs['device']
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        self.eps = kwargs['eps']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']

    def sample(self, state):
        state = torch.tensor([state], dtype=torch.float, device=self.device)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.cpu().numpy()[0], probs.detach().cpu().numpy()[0]

    def update(self, transitions, discounted_rewards):
        states = torch.tensor(transitions['states'], dtype=torch.float, device=self.device)
        old_probs = torch.tensor(transitions['old_probs'], dtype=torch.float, device=self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        A = torch.tensor(discounted_rewards, dtype=torch.float, device=self.device).view(-1, 1)

        new_probs = self.actor(states).gather(1, actions)
        ratio = torch.exp(torch.log(new_probs) - torch.log(old_probs.gather(1, actions)))

        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * A
        clip_loss = -torch.mean(torch.min(surr1, surr2))

        old_dist = Categorical(old_probs)
        new_dist = Categorical(self.actor(states))
        kl_div = kl_divergence(old_dist, new_dist)
        kl_loss = torch.mean(kl_div)

        total_loss = clip_loss + self.beta * kl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'clip_loss': clip_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }

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

def train_grpo():
    set_seed(Config.SEED)
    env = gym.make(Config.ENV_NAME)
    env.reset(seed=Config.SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    config = {
        'hidden_dim': Config.HIDDEN_DIM,
        'lr': Config.GRPO_CFG['lr'],
        'eps': Config.GRPO_CFG['eps'],
        'beta': Config.GRPO_CFG['beta'],
        'gamma': Config.GRPO_CFG['gamma'],
        'device': Config.DEVICE
    }

    agent = GRPO(state_dim, action_dim, **config)
    buffer = EpisodeBuffer()
    returns = []

    for episode in tqdm(range(1, Config.NUM_EPISODES+1), desc='GRPO Training'):
        state, _ = env.reset()
        episode_return = 0
        buffer.clear()
        done = False

        while not done:
            action, old_prob = agent.sample(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, old_prob, reward)
            state = next_state
            episode_return += reward

        rewards = buffer.rewards
        discounted_rewards = []
        running_reward = 0
        for r in reversed(rewards):
            running_reward = r + agent.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

        transitions = buffer.to_dict()
        agent.update(transitions, discounted_rewards)
        returns.append(episode_return)

    env.close()
    return returns

# ===========================
# 3. 对比实验主流程
# ===========================
if __name__ == '__main__':
    ppo_returns = train_ppo()
    grpo_returns = train_grpo()

    min_len = min(len(ppo_returns), len(grpo_returns))
    ppo_returns = ppo_returns[:min_len]
    grpo_returns = grpo_returns[:min_len]

    window = 10
    ppo_ma = rl_utils.moving_average(np.array(ppo_returns), window_size=window)
    grpo_ma = rl_utils.moving_average(np.array(grpo_returns), window_size=window)

    plt.figure(figsize=(12, 6))
    plt.plot(ppo_returns, alpha=0.3, color='#1f77b4', label='PPO (Raw)')
    plt.plot(range(window-1, len(ppo_ma)+window-1), ppo_ma, color='#1f77b4', label=f'PPO (MA-{window})')
    plt.plot(grpo_returns, alpha=0.3, color='#ff7f0e', label='GRPO (Raw)')
    plt.plot(range(window-1, len(grpo_ma)+window-1), grpo_ma, color='#ff7f0e', label=f'GRPO (MA-{window})')
    
    plt.title(f'PPO vs GRPO: {Config.ENV_NAME} Training Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n=== Final Performance ===")
    print(f"PPO: Last 10 episodes avg: {np.mean(ppo_returns[-10:]):.2f}")
    print(f"GRPO: Last 10 episodes avg: {np.mean(grpo_returns[-10:]):.2f}")