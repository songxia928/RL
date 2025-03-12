


#  此代码为 基于pytorch实现的 DQN 代码实现，游戏实例包选择 gymnasium，有别于gym，gymnasium可以实现游戏可视化。
#  具体原理和细节，请参考博客：https://zhuanlan.zhihu.com/p/696788620

import os
import time
import random
#import gym
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from config_DQN import *


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        b_s, b_a, b_r, b_ns, b_d = np.array(state), action, reward, np.array(next_state), done
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        return transition_dict

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #print(' ---- x.shape: ', x.shape)
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state, is_epsilon = True):  # epsilon-贪婪策略采取动作
        if is_epsilon  and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值

        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self, model_path):
        torch.save(self.q_net.state_dict(), model_path)

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.q_net.load_state_dict(state_dict, False)


def cal_reward(state, _reward, done, step_num, max_step_num):
    reward = _reward 
    if step_num >= max_step_num-1:  reward += reward*5      # 完成最后一步给更多奖励
    elif done : reward = -1   # 如果 提前结束， 负分            

    return reward


def plot_figure(results):
    keys = ['reward', 'success']
    for k in keys:
        iteration_list = list(range(len(results['ave_'+k])))
        plt.plot(iteration_list, results['ave_'+k], color='b', label='ave_'+k)
        plt.plot(iteration_list, results['max_'+k], color='r', label='max_'+k)
        plt.plot(iteration_list, results['min_'+k], color='g', label='min_'+k)
        plt.xlabel('Iteration')
        plt.ylabel(k)
        plt.title('DQN on {}'.format(game_name, k))
        plt.show()

        figure_path = train_figure_path.replace('.png', '_{}.png'.format(k))
        plt.savefig(figure_path)


def train():    
    # ==== env
    env = gym.make(game_name)

    # ==== seed
    random.seed(0)
    np.random.seed(0)
    #env.seed(0)
    torch.manual_seed(0)

    # ==== buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # ==== agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    results = {}
    for k in ['reward', 'success']:
        results['ave_' + k] = []
        results['max_' + k] = []

        results['min_' + k] = []

    for i in range(iteration_num):
        with tqdm(total=episode_num, desc='Iteration %d' % i) as pbar:
            rewards, successes = [], []
            for i_episode in range(episode_num):
                #state = env.reset()
                state, _ = env.reset()

                for step_num in range(max_step_num):
                    # ==== action
                    action = agent.take_action(state)

                    # ==== step
                    #next_state, reward, done, _ = env.step(action)
                    next_state, _reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # ==== reward
                    reward = cal_reward(next_state, _reward, done, step_num, max_step_num)

                    # ==== buffer
                    replay_buffer.add(state, action, reward, next_state, done)

                    # ==== 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        transition_dict = replay_buffer.sample(batch_size)
                        agent.update(transition_dict)

                    state = next_state
                    if done : break

                success = 0
                if step_num >= 499 :
                    success = 1
                successes.append(success)
                rewards.append(reward)

                ave_reward = np.mean(rewards)
                max_reward = max(rewards)
                min_reward = min(rewards)
                ave_success = np.mean(successes)
                max_success = max(successes)
                min_success = min(successes)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':  '%d' % (episode_num * i + i_episode + 1),
                        '  reward':   '%.3f' % ave_reward,
                        'max_r': max_reward,
                        'min_r': min_reward,
                        '  success':  '%.3f' % ave_success,
                        'max_s': max_success,
                        'min_s': min_success,
                    })
                pbar.update(1)

            results['ave_reward'].append(ave_reward)
            results['max_reward'].append(max_reward)
            results['min_reward'].append(min_reward)
            results['ave_success'].append(ave_success)
            results['max_success'].append(max_success)
            results['min_success'].append(min_success)


    agent.save_model(model_path)
    plot_figure(results)



def test():
    env_play = gym.make(game_name, render_mode='human')   # UI
    state, _ = env_play.reset()

    # ==== agent
    state_dim = env_play.observation_space.shape[0]
    action_dim = env_play.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)
    agent.load_model(model_path)

    time.sleep(10)

    step_num = -1
    done = False
    while not done:
        step_num += 1

        action = agent.take_action(state, is_epsilon=False)
        next_state, reward, terminated, truncated, info = env_play.step(int(action))
        done = terminated or truncated
        state = next_state

        print(' Test ---- step_num, action, reward, obs, done: ', step_num, action, reward, state, done)

        time.sleep(0.1)



if __name__ == '__main__':
    train()

    test()
