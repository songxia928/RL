


#  此代码为 基于pytorch实现的 QLearning 代码实现，游戏实例包选择 gymnasium，有别于gym，gymnasium可以实现游戏可视化。
#  具体原理和细节，请参考博客： https://zhuanlan.zhihu.com/p/688337472

import os
import time
from itertools import count
import numpy  as np
import pandas as pd

import gymnasium as gym


class QLearning(object):
    def __init__(self, env, env_play) -> None:
        # == epsilon
        self.max_epsilon = 1      # 探索系数最大值
        self.min_epsilon = 0.05   # 探索系数最小值
        self.epsilon = self.max_epsilon

        # == env
        self.episodes = 20000    # 游戏局数
        self.env = env
        self.env_play = env_play


        # == Q-learning
        self.alpha    = 0.5          
        self.gamma    = 0.95
        self.q_table_csv = './q_table_{}x{}.csv'.format(self.env.observation_space.n, self.env.action_space.n)

        self.q_table = pd.DataFrame(
            np.zeros((self.env.observation_space.n, self.env.action_space.n)),  
            index=range(0, self.env.observation_space.n),     # 对应 环境的维度，observation （0，15）， 也就是position
            columns=range(0, self.env.action_space.n)         # 对应 动作空间维度，action （0，3） 。1表示下，2表示右
            )
            # 整个q_table 表示 当处于 position 位置时，选择哪个方向最可能到达终点
        print(' -------- init. qtable： \n', self.q_table)

    def epsilon_decay(self, episode): 
        a = 7.5  # 比例系数
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-a * episode/ self.episodes)   # 指数衰减
        return epsilon

    def select_action(self, state, greedy=False):
        e = np.random.uniform()
        action = None
        if (e < self.epsilon or (self.q_table.iloc[state] == 0).all()) and not greedy:   
            action = self.env.action_space.sample()
        else:
            action = self.q_table.iloc[state].idxmax()
        return action

    def update_q_table(self, state, action, reward, next_state):    # 计算 state(s), action(a)  --》 next_state(s')  时的动作价值函数 
        # Q_{t+1}(s, a) = Q_t(s, a) + \alpha \cdot (R(s, a) + \gamma \cdot \max_{a'} Q_t(s', a') - Q_t(s, a))
        q = self.q_table.iloc[state][action]       # Q_t(s, a)
        q_new = q + self.alpha * (reward + self.gamma * self.q_table.iloc[next_state].max() - q)    #  时序差分
        self.q_table.iloc[state][action] = q_new

    def train(self):
        for episode in range(self.episodes):
            rewards = []
            successes = []

            observation, info = self.env.reset()
            #observation, info = self.env.reset(seed=42)  # 固定种子

            # ======== For each step
            for step_num in range(100):
                action = self.select_action(observation)
                observation_new, reward, terminated, truncated, info = self.env.step(action)
                #print(' -- episode, step_num, action, observation_new, reward, terminated, truncated, info: ', episode, step_num, action, observation_new, reward, terminated, truncated, info)

                # Truncated在官方定义中用于处理比如超时等特殊结束的情况。
                '''
                observation (ObsType) : 环境观察空间的一个元素，作为代理动作的下一个观察结果
                reward (SupportsFloat) : 采取行动的结果的奖励。 成功到达目的地会得到奖励 1，否则奖励为 0
                terminated (bool) : 代理是否达到最终状态，可以是正数或负数。
                truncated (bool) : 是否满足MDP范围外的截断条件。 通常，这是一个时间限制，但也可用于指示代理实际越界。 可用于在达到最终状态之前提前结束情节。
                info (dict) : 包含辅助诊断信息（有助于调试、学习和记录）。
                '''
                success = reward
                done = terminated or truncated
                if done and reward == 0:  # 调入冰窟 给负分
                    reward = -1

                successes.append(success)
                rewards.append(reward)

                self.update_q_table(observation, action, reward, observation_new)
                observation = observation_new

                if done:
                    self.epsilon = self.epsilon_decay(episode)
                    break

            ave_reward = sum(rewards)/len(rewards)
            ave_successes = sum(successes)/len(successes)
            if episode%1000 == 0: print(' Train ---- episode={}, epsilon={:.3f}, ave_successes={:.3f} ave_reward={:.3f} '.format(episode, self.epsilon, ave_successes, ave_reward))
            #print(' -- q_table:\n',qlearn.q_table)

        # save csv
        self.q_table.to_csv(self.q_table_csv, index=False)

    def test(self):
        #self.env.render(render_mode='human')  # 在这里指定渲染模式
        #self.env.render(mode='human')

        if os.path.exists(self.q_table_csv):
            dtype = dict(zip(np.array([str(x) for x in np.arange(0,self.env_play.action_space.n)]), np.array(['float64'] * self.env_play.action_space.n)))
            self.q_table = pd.read_csv(self.q_table_csv, header=0, dtype=dtype)
            print(' ---- read q_table: \n', self.q_table)

            observation, info = self.env_play.reset()
            #observation, info = self.env_play.reset(seed=42)   # 固定种子

            time.sleep(10)

            step_num = -1
            done = False
            while not done:
                step_num += 1

                action = self.select_action(observation, True)
                observation_new, reward, terminated, truncated, info = self.env_play.step(int(action))
                done = terminated or truncated
            
                observation = observation_new

                print(' Test ---- step_num, action, reward, observation: ', step_num, action, reward, observation)

                time.sleep(1)


if __name__ == '__main__':

    '''
    固定种子， 无滑动：    一直能通关
    固定种子， 有滑动：    一直能通关
    不固定种子， 无滑动：  一直能通关
    不固定种子， 有滑动：    偶尔能通关 ，且通关步长较大。 （可能是训练不充分）

    （这里的不固定种子，并不是随机开局）
    '''

    # ==== 创建环境
    env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=False)    # 无滑动
    env_play = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')

    #env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True)  # 有滑动
    #env_play = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')

    '''
    'FrozenLake-v1'：这是环境的名称，FrozenLake 是一个经典的强化学习问题，代表一个冰冻湖面上的智能体需要在湖面上移动，避开洞口，到达目标位置。这个环境有一个简单的离散状态空间和动作空间。
    desc=None：desc 参数用于指定环境的地图描述。如果设置为 None，则使用 map_name 参数指定的地图。如果提供了一个地图描述，那么将使用这个描述创建环境。
    map_name="4x4"：这个参数指定了环境的地图大小。默认情况下，FrozenLake 环境是一个 4x4 的地图，这里我们将其扩展到 4x4 的大小。
    is_slippery=True：这个参数表示地面是否有滑动效果。如果设置为 True，智能体在执行动作时有一定的概率滑动到非预期的相邻格子。这会使问题更具挑战性，因为智能体需要学会在不确定性条件下做出决策。
    render_mode='human'：这个参数指定了环境的渲染模式。'human' 表示环境将以人类可读的方式呈现，通常是通过图形界面或者命令行输出。这有助于我们观察智能体在环境中的表现。
    '''

    # ==== 初始化算法
    qlearn = QLearning(env, env_play)


    # ==== train
    qlearn.train()

    # ==== test
    qlearn.test()

