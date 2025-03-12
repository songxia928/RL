



#  此代码为 基于pytorch实现的 Actor-Critic 代码实现，游戏实例包选择 gym。
#  具体原理和细节，请参考博客：https://blog.csdn.net/songxia928_928/article/details/145243586




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

