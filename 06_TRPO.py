

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

