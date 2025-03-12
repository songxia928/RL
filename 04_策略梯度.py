


#  此代码为 基于pytorch实现的 策略 代码实现，游戏实例包选择 gym。
#  具体原理和细节，请参考博客：https://blog.csdn.net/songxia928_928/article/details/145240895


import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils



class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)   # 通过神经网络（策略网络 PolicyNet）将状态映射为动作概率分布 πθ(a∣s)
        action_dist = torch.distributions.Categorical(probs)  # 创建一个离散的类别分布（Categorical Distribution），表示当前的策略πθ(a∣s)
        action = action_dist.sample()    # 根据动作的概率分布随机采样动作 at
                                         # 动作是基于当前策略的随机行为，而不是直接选择最大概率的动作，这种随机性是策略梯度方法的核心，旨在探索动作空间
        return action.item()

    def update(self, transition_dict): # 基于每局采样到的数据，通过策略梯度公式更新策略网络，使得策略更倾向于选择能够获得更高回报的动作
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0   # 累计折扣回报
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = reward + self.gamma * G       #  G_t = r_t + γ⋅G_t+1
            loss = -log_prob * G  # 负号是因为 PyTorch 中的优化器默认是 最小化损失函数，而策略梯度方法的目标是 最大化回报。
                                  # 即：最小化 − log πθ(a∣s)⋅G_t，等价于最大化 log πθ(a∣s)⋅G_t
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新策略网络的参数

learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = "CartPole-v0"
env = gym.make(env_name)
env.reset(seed=0)

torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                  device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }

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
            agent.update(transition_dict)    # 每一局，更新一次策略。且只有一个游戏实例
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)



