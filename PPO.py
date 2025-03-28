


#  此代码为 基于pytorch实现的 Actor-Critic 代码实现，游戏实例包选择 gym。
#  具体原理和细节，请参考博客：https://blog.csdn.net/songxia928_928/article/details/145266345

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
    

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)                 # 通过神经网络（策略网络 PolicyNet）将状态映射为动作概率分布 πθ(a∣s)
        action_dist = torch.distributions.Categorical(probs)      # 构造动作分布
        action = action_dist.sample()              # 根据动作概率分布，采样动作

        return action.item()

    def update(self, transition_dict):
        # 将采样的数据转换为张量
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

        # (3) 优势函数 和 旧策略的概率分布
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)  # 优势函数
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # (4) 计算 Loss
        for _ in range(self.epochs):   # 同一批数据训练了多次
            # Critic Loss
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # Actor Loss
            log_probs = torch.log(self.actor(states).gather(1, actions))  
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))   # L_CLIP(θ) = E[min⁡(rt(θ)At, clip(rt(θ),1−ϵ,1+ϵ)At)]
                                                                  # 第一项 rt(θ)At：表示新旧策略比值调整后的优势。
                                                                  # 第二项 clip(rt(θ),1−ϵ,1+ϵ)At)：对比值进行截断，防止更新幅度过大。
            # 梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()



if __name__ == '__main__':

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.reset(seed=0)

    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

