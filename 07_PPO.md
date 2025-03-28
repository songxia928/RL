


# 【强化学习】07.近端策略优化(PPO) 算法原理

- 论文：https://arxiv.org/abs/1707.06347
- 复现代码： https://github.com/songxia928/RL/blob/main/PPO.py



["PPO（Proximal Policy Optimization）"](https://arxiv.org/abs/1707.06347)是一种强化学习算法，是策略优化方法的现代改进版本。它结合了策略梯度方法的优势，同时通过限制策略更新幅度，保持训练的稳定性和高效性。PPO 是当前深度强化学习中的主流算法，广泛应用于各种复杂任务。下面我们从以下原理和训练步骤等方面介绍PPO算法。

---

## **1. 算法原理**


### **1.0 通用变量**


| **变量**                  | **定义及含义**                                                                                                      |
|---------------------------|-------------------------------------------------------------------------------------------------------------------|
| $s$ 或 $s_t$          | 当前状态（state），表示智能体与环境交互时的状态，一般为一个向量。                                                   |
| $a$ 或 $a_t$          | 动作（action），表示智能体在状态 $s_t$ 下采取的行为。动作可以是离散的或连续的。                                      |
| $r$ 或 $r_t$          | 即时奖励（reward），智能体在状态 $s_t$ 下执行动作 $a_t$ 后从环境获得的即时反馈信号。                                  |
| $\gamma$                | 折扣因子（discount factor），用来平衡立即奖励和未来奖励的权重，取值范围为 $[0, 1]$。                                  |
| $T$                     | 轨迹的终止时间步（通常为有限时间步），表示任务结束时的时间步。                                                        |
| $\pi_\theta(a\|s)$       | 策略（policy），表示在状态 $s$ 下采取动作 $a$ 的概率，策略由参数 $\theta$ 表示。                                     |
| $\pi_{\theta_{\text{old}}}(a\|s)$ | 旧策略（旧的 $\theta$ 参数值）在某些算法（如 PPO）中会保存旧的策略来计算策略比值。                                     |
| $G_t$                   | 累积回报（return），从时间步 $t$ 开始的总奖励：<br> $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$                            |
| $V(s)$                  | 状态值函数（state value function），表示在状态 $s$ 下，按照当前策略 $\pi$ 能获得的期望总回报：<br> $V(s) = \mathbb{E}_{\pi}[G_t \| s_t=s]$ |
| $Q(s, a)$               | 状态-动作值函数（state-action value function），表示在状态 $s$ 下采取动作 $a$ 后能获得的期望总回报：<br> $Q(s, a) = \mathbb{E}_{\pi}[G_t \| s_t=s, a_t=a]$ |
| $Q_\theta(s, a)$        | Q 网络（Q-value function），表示通过神经网络近似的状态-动作值 $Q(s, a)$。                                             |
| $A(s, a)$               | 优势函数（advantage function），衡量某一个动作相对于当前策略的平均水平的好坏：<br> $A(s, a) = Q(s, a) - V(s)$        |

---



### **1.1 策略梯度方法（Policy Gradient）**

策略梯度方法直接优化策略 $\pi_\theta(a|s)$ 的参数 $\theta$，以最大化累积期望回报 $J(\theta)$：

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]
$$

通过对 $J(\theta)$ 求梯度，得到策略梯度公式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]
$$

其中：
- $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$，即从时间步 $t$ 开始的累积回报，或者叫累计折扣回报 和 实际回报。直接指出的是 $R_t$ 表示 $t$ 时刻回报。
- $\nabla_\theta \log \pi_\theta(a|s)$ 是策略的对数梯度。


#### **(1) 采样方式**：
策略梯度方法从当前策略 $\pi_\theta(a|s)$ 中直接采样完整轨迹 $(s_t, a_t, r_t)$，通过这些数据计算累积回报 $G_t$。

**数据使用**：采样到的轨迹只用于一次参数更新后丢弃。

#### **(2) 策略梯度问题**
1. **高方差**：直接使用 $G_t$ 作为回报信号，导致梯度估计的方差较大，训练不稳定。
2. **缺少基线**：未引入基线函数导致估计进一步不稳定。
3. **采样效率低**：采样量需求大，训练数据利用率低。

*为什么方差会高呢？ 因为有的时候局势很好，执行的动作都有很高的回报，但是有的时候局势很差，执行的动作回报都很低，这个时候回报的方差很高。同时，我们需要根据回报选择出相对比较好的动作，这个时候我们的回报也不能一直是一个绝对量，需要是一个相对量。*

为了解决这些问题，提出了 **Actor-Critic 方法**。

---

### **1.2 Actor-Critic 方法**

Actor-Critic 方法结合了策略梯度方法和值函数估计，通过引入 <font color="#dd0000">基线函数</font> $V(s)$，降低了梯度估计的方差。

#### **(1) 改进的目标函数**
将策略梯度公式改为以下形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A_t \right]
$$

其中，$A_t$ 是优势函数，定义为：
$$
A_t = G_t - V(s_t)
$$
- $G_t$ 是累计回报。
- $V(s_t)$ 是价值函数，估计状态 $s_t$ 的期望回报。

#### **(2) 更新规则**
1. **Actor（策略网络）**：
   更新策略网络 $\pi_\theta(a|s)$。
   
2. **Critic（价值网络）**：
   使用均方误差（MSE）优化价值网络：
   $$
   L_{\text{critic}}(\phi) = \mathbb{E} \left[ (G_t - V_\phi(s_t))^2 \right]
   $$

#### **(3) 采样方式与数据利用**
- **采样方式**：从当前策略 $\pi_\theta(a|s)$ 中采样 $(s_t, a_t, r_t, s_{t+1})$，而不需要完整轨迹。
- **数据使用**：
  1. Critic（价值网络）：
     - 使用采样数据 $(s_t, r_t, s_{t+1})$ 更新价值网络，通过最小化 TD 误差：
       $$
       L_{\text{critic}}(\phi) = \mathbb{E} \left[ (r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2 \right]
       $$
     - TD 误差显式引入了下一状态 $s_{t+1}$，降低了梯度方差。
  2. Actor（策略网络）：
     - 使用采样数据的状态 $s_t$ 和优势函数 $A_t = G_t - V(s_t)$ 更新策略。

#### **(4) 问题**
Actor-Critic 减少了梯度估计的方差，但仍存在问题：
1. **步幅问题**：策略更新幅度过大可能导致性能退化。
2. **不稳定性**：Critic 的误差会影响策略更新，导致不稳定。
3. **数据使用低效**：采样数据仍然只使用一次，无法重复利用。

为了解决步幅问题，提出了 **TRPO**。

---

### **1.3 TRPO（Trust Region Policy Optimization）**

TRPO 通过限制每次策略更新的幅度，保证策略更新在 <font color="#dd0000">“信任域”（Trust Region）</font> 内，从而提高更新的稳定性。


#### **(1) 目标函数**
TRPO 的优化目标为：

$$
\max_\theta \; \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A_t \right]
$$
在以下约束条件下：
$$
\mathbb{E}_{s \sim \pi_{\theta_{\text{old}}}} \left[ D_{\text{KL}} \left( \pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s) \right) \right] \leq \delta
$$

其中：
- $\pi_\theta(a|s)$ 是更新后的策略，$\pi_{\theta_{\text{old}}}(a|s)$ 是旧策略；
- $D_{\text{KL}}$ 是 KL 散度，用来衡量新旧策略分布的变化幅度；
- $\delta$ 是允许的最大变化。

上面是一个有约束的优化问题，TRPO详细求解过程比较复杂，详细过程可以参考上一篇文章 [【强化学习】06.信任区域策略优化(TRPO) 算法原理](https://blog.csdn.net/songxia928_928/article/details/145243619)。


#### **(2) 采样方式与数据利用**
- **采样方式**：
  - TRPO 依然采用 **on-policy 采样**，从旧策略 $\pi_{\theta_{\text{old}}}(a|s)$ 中采样数据 $(s_t, a_t, r_t, s_{t+1})$。
- **数据使用**：
  1. 采样数据 $(s_t, a_t, r_t)$ 被用来计算：
     - 策略比值 $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$。
     - 优势函数 $A_t = G_t - V(s_t)$。
  2. **多次利用采样数据**：TRPO 在一次采样后会使用共轭梯度法进行多次优化迭代，从而提高数据利用效率。
- **优点**：
  - KL 散度约束防止策略更新幅度过大，提高更新稳定性。
  - 提高了采样效率：多次利用数据，减少了采样需求。


#### **(3) 问题**
TRPO 的二阶优化（如共轭梯度法）计算复杂，难以高效实现。因此，进一步提出了 **PPO**。

---

### **1.4 PPO（Proximal Policy Optimization）**

PPO 通过引入 <font color="#dd0000"> 截断机制（Clipping）</font> 代替 KL 散度约束，简化了策略更新，同时保持了更新的稳定性。

#### **(1) 目标函数**
PPO 的核心目标函数如下：
$$
L^{CLIP}(\theta) = \mathbb{E}
\left[
\min \left( r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t \right)
\right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ 是新旧策略概率的比值；
- $A_t$ 是优势函数；
- $\epsilon$ 是截断范围的超参数。

#### **(2) 截断机制**
- 如果 $r_t(\theta)$ 超出范围 $[1-\epsilon, 1+\epsilon]$，则对其进行截断。
- 截断的目的是限制策略更新幅度，避免策略发生过大变化。

#### **(3) 完整优化目标**
PPO 的完整目标函数包括三个部分：
$$
L(\theta) = \mathbb{E} \left[ L^{CLIP}(\theta) - c_1 L_{\text{critic}}(\phi) + c_2 S[\pi_\theta](s) \right]
$$

其中：
- $L^{CLIP}(\theta)$ 是截断后的策略目标，用于更新 Actor。
- $L_{\text{critic}}(\phi) = \mathbb{E} \left[ (G_t - V_\phi(s_t))^2 \right]$ 是 Critic 的损失，用于更新价值网络。
- $S[\pi_\theta](s)$ 是策略的熵正则项，用于鼓励探索性。
- $c_1$ 和 $c_2$ 是权重超参数。


#### **(4) 采样方式与数据利用**
- **采样方式**：
  - 与 TRPO 类似，PPO 也依赖 on-policy 采样，记录数据 $(s_t, a_t, r_t, s_{t+1})$。
- **数据使用**：
  1. 采样数据可以被多次利用：PPO 使用随机小批量（minibatch SGD）训练策略网络。
  2. 截断机制限制策略比值 $r_t(\theta)$ 的范围，确保策略更新幅度受控，避免不稳定。




#### **(5) 改进与问题**
- **改进**：
  1. 优化简单：PPO 只需一阶优化（梯度下降），不需要 TRPO 的复杂二阶优化。
  2. 数据利用效率高：采样数据重复使用多次。
- **问题**：
  1. 依然需要大量 on-policy 数据，采样代价高。




### **1.5 PPO 和 TRPO 的数据使用方式的差异**

**TRPO 的数据使用方式**
- **一次性使用**：TRPO 中，采样的轨迹数据（状态、动作、奖励等）通常只被使用一次，即一次策略更新后就重新采样新的数据。
- **原因**：TRPO 构建了一个二次约束的优化问题（通过 KL 散度约束），在这个优化问题中，策略更新幅度需要非常精确地控制。如果多次重复使用同一数据，会导致目标函数和约束条件在优化过程中逐渐偏离原始设计，从而破坏算法的理论保证。

**PPO 的数据使用方式**
- **多次重复使用**：PPO 会对同一批采样数据进行多次更新，通常在 `self.epochs` 中迭代训练。
- **原因**：PPO 通过截断目标函数（clipping）来限制策略更新的幅度，即便多次更新模型也不会导致策略发生过大的变化。因此，PPO 允许在相同的采样数据上进行多轮训练，这提高了数据的使用效率。

---

#### **（1） 为什么 PPO 允许多次使用数据？**

PPO 允许多次使用数据的核心原因在于：**PPO 的目标函数设计通过截断机制（Clipping）对策略更新幅度进行了限制。**

##### **A. 截断机制确保更新的稳定性**

PPO 的目标函数如下：
$$
L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ 
\min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)
\right]
$$
其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是新旧策略的概率比值；
- $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$ 是截断机制，用于限制策略更新幅度；
- $A_t$ 是优势函数。

**核心设计：**
1. **限制更新幅度：**当比值 $r_t(\theta)$ 超过区间 $[1 - \epsilon, 1 + \epsilon]$ 时，目标函数会对比值进行截断，强制将其限制在该区间内。这样，即便在多轮迭代中，新策略也不会偏离旧策略太远。
2. **避免策略崩塌：**截断机制在优化中起到了“保护措施”的作用，防止策略更新过度。即便多次使用同一批数据，也不会导致策略发生过大的变化。

因此，PPO 的截断机制使得即便多次更新，策略优化仍然是稳定的。

---

##### **B. KL 散度约束和截断机制的对比**

**TRPO 的 KL 散度约束**
TRPO 通过优化如下约束问题：
$$
\max_\theta \; L(\theta), \quad \text{s.t. } \mathbb{E}[D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)] \leq \delta
$$
- KL 散度约束用于严格限制新旧策略之间的变化。
- 一旦完成一次策略更新，新策略和旧策略的 KL 散度已经发生了变化，如果继续使用同一批数据更新，将破坏原有的约束条件。
- 因此，TRPO 的数据只能使用一次。

**PPO 的截断机制**
PPO 放弃了精确的 KL 散度约束，改用截断机制控制策略变化：
- 截断机制对比值 $r_t(\theta)$ 的区间进行约束，使得每次更新的策略变化不会过大。
- 即便多次使用相同的数据，截断机制仍然在每一次迭代中有效，保持了策略的更新幅度的稳定。
- 这使得 PPO 能够在相同数据上进行多次更新，同时避免策略崩塌。

---

##### **C. 数据多次使用的效率优势**

**PPO 的多次优化**
在强化学习中，采样数据的代价通常很高（需要与环境交互），特别是在复杂的环境中，采样可能需要耗费大量时间和计算资源。PPO 的设计允许对同一批数据进行多次优化：
- 同一采样数据经过多轮训练，这就大幅提高了数据的利用效率；
- 每次优化都会进一步改进策略，使得在相同的采样数据上能够逐步接近最优。

**TRPO 和其他算法**
相比之下，TRPO 等只对采样数据使用一次，策略优化的效率相对较低，因为每次策略更新都需要重新采样。

---

#### **(3) 为什么 TRPO 不能多次使用数据？**

尽管 TRPO 和 PPO 都是基于策略梯度的优化方法，但它们在策略更新控制上有本质区别。

1. **严格的 KL 散度约束：**
   - TRPO 中，新旧策略之间的 KL 散度是以严格约束的形式控制的（$\mathbb{E}[D_{\text{KL}}] \leq \delta$）。
   - 在完成一次更新后，新的策略已经不满足原来的 $D_{\text{KL}}$ 约束条件。如果继续对同一批数据进行多次更新，就会破坏 TRPO 的理论基础。

2. **一次性优化方向：**
   - TRPO 使用共轭梯度法计算的搜索方向 $x = H^{-1}g$ 是针对当前策略及采样数据的最优方向。如果对同一批数据多次优化，会导致目标函数被过度调整，偏离初始的优化方向。

3. **高计算复杂度：**
   - TRPO 的计算复杂度较高（需要计算 KL 散度的黑塞矩阵和共轭梯度法），因此重复优化同一批数据在计算上也不划算。

总结来说，TRPO 的理论设计和实现方式决定了它只能对每批数据进行一次更新。




---

## **2. 算法步骤**

### **2.1 主要流程**
以下是 PPO 算法的主要流程：

**（1）初始化**  
- 初始化策略网络（Actor）和价值网络（Critic）。
- 设置超参数：学习率、折扣因子 $\gamma$、截断范围 $\epsilon$、GAE 参数 $\lambda$、训练轮数等。

**（2）采样数据**  
- 与环境交互，按照当前策略 $\pi_{\theta}(a|s)$ 采样一批轨迹，记录：
  - 状态 $s$、动作 $a$、奖励 $r$、下一状态 $s'$、是否终止 $done$。

**（3）计算 GAE（广义优势估计）**  
- 计算时序差分误差（TD Error）：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
- 计算优势函数 $A_t$：
$$
A_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
$$

**（4）更新 Critic**  
- 使用 TD 目标优化价值网络：
$$
L_{\text{critic}} = \mathbb{E} \left[ \left( r + \gamma V(s') - V(s) \right)^2 \right]
$$

**（5）更新 Actor**  
- 计算新旧策略比例：
$$
r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
$$
- 计算截断的策略目标：
$$
L^{CLIP}(\theta) = \mathbb{E}
\left[
\min \left( r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t \right)
\right]
$$
- 最大化目标函数，优化策略网络。

**（6）重复训练**  
- 重复数据采样、更新 Actor 和 Critic，直到策略收敛或达到预设训练轮数。



### **2.2 结合实际举例**

我将基于实际的 PPO 实现，结合一个具体的例子（如经典的 **CartPole-v0** 环境）来丰富描述 PPO 的过程，进一步细化每个步骤中的计算和实现细节。

---

我们以一个强化学习任务为例，例如 **CartPole-v0**：
- **任务目标**：控制小车左右移动，保持杆子的平衡尽可能长时间。
- **状态空间**：由 4 个连续变量组成（小车位置、小车速度、杆子角度、杆子角速度）。
- **动作空间**：包含 2 个离散动作（向左施加推力，向右施加推力）。
- **奖励函数**：每个时间步杆子保持直立，奖励为 +1。

以下为丰富后的步骤：

---

#### **（1）初始化**

- 初始化 **策略网络（Actor）** 和 **价值网络（Critic）**，常见结构为全连接神经网络：
  - 输入为状态 $s$。
  - Actor 网络输出动作的概率分布 $\pi_\theta(a|s)$。
  - Critic 网络输出状态值 $V(s)$。
- 设置超参数：
  - 学习率 $\alpha$（用于优化器）。
  - 折扣因子 $\gamma$（控制未来奖励的权重）。
  - 截断范围 $\epsilon$（限制策略更新幅度，如 $\epsilon = 0.2$）。
  - GAE 参数 $\lambda$（权衡偏差与方差，常用 $\lambda = 0.95$）。
  - batch size 和训练轮数等。

##### **示例（初始化）**
- 策略网络（Actor）：输入大小为 4（状态维度），输出大小为 2（动作分布）。
- Critic 网络：输入大小为 4，输出大小为 1（状态值）。
- 超参数：
  - $\alpha = 3 \times 10^{-4}$，$\gamma = 0.99$，$\epsilon = 0.2$，$\lambda = 0.95$。

---

#### **（2）采样数据**

- 与环境（如 CartPole-v0）交互：
  - 使用当前策略 $\pi_\theta(a|s)$，根据动作分布采样动作 $a$。
  - 执行动作后，记录每一步的状态 $s$、动作 $a$、奖励 $r$、下一状态 $s'$ 和是否终止 $done$。
- 收集多条轨迹（通常收集 $N$ 个时间步的数据）后，存储为以下形式：
  - 状态：$[s_1, s_2, \dots, s_N]$。
  - 动作：$[a_1, a_2, \dots, a_N]$。
  - 奖励：$[r_1, r_2, \dots, r_N]$。
  - 下一状态：$[s'_1, s'_2, \dots, s'_N]$。
  - Done 标志：$[done_1, done_2, \dots, done_N]$。

##### **示例（采样数据）**
- 使用策略 $\pi_\theta(a|s)$（如 softmax 输出动作概率），对每个状态 $s$ 采样动作 $a$：
  - 假设状态 $s_1 = [0.0, 0.0, 0.1, 0.0]$。
  - 策略网络输出 $\pi_\theta(a|s_1) = [0.8, 0.2]$。
  - 随机采样后选择动作 $a_1 = 0$。
- 与环境交互后得到：
  - 奖励 $r_1 = +1$。
  - 下一状态 $s'_1 = [0.1, -0.1, 0.15, -0.05]$。

---

#### **（3）计算 GAE（广义优势估计）**

PPO 使用 **广义优势估计（Generalized Advantage Estimation, GAE）** 来平衡偏差和方差，计算优势函数 $A_t = G_t - V(s_t)$。

##### **实现步骤**
1. **计算时序差分误差（TD Error）**：
   $$
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
   $$
   这里 $V(s_t)$ 是 Critic 网络输出的状态值。

2. **计算优势函数 $A_t$**：
   使用 GAE 来平滑 TD 误差的累加：
   $$
   A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \dots
   $$

3. **计算目标回报 $G_t$**：
   累积折扣奖励（用于 Critic 的 TD 目标）：
   $$
   G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
   $$

##### **示例（计算 GAE）**
假设从环境中采样到以下轨迹：
- $\mathbf{r} = [1, 1, 1, 1, 1]$；
- Critic 估计的状态值：$[V(s_1) = 0.5, V(s_2) = 0.6, V(s_3) = 0.7, V(s_4) = 0.8, V(s_5) = 0.0]$。

1. 计算 $\delta_t$：
   $$
   \delta_1 = 1 + 0.99 \cdot 0.6 - 0.5 = 1.094
   $$
   重复计算每个时间步的 $\delta_t$。

2. 使用 GAE 计算 $A_t$：
   $$
   A_1 = \delta_1 + (\gamma \lambda)\delta_2 + (\gamma \lambda)^2\delta_3 + \dots
   $$

---

#### **（4）更新 Critic**

Critic 网络的优化目标是最小化以下均方误差（MSE）损失：
$$
L_{\text{critic}} = \mathbb{E} \left[ \left( G_t - V(s_t) \right)^2 \right]
$$
- $G_t$ 是实际回报（从采样轨迹中计算）。
- $V(s_t)$ 是当前 Critic 估计的状态值。

##### **示例（更新 Critic）**
- 假设目标回报 $G_t = [1.5, 1.4, 1.3, 1.2, 1.1]$；
- Critic 网络输出 $[0.5, 0.6, 0.7, 0.8, 0.9]$。
- MSE 损失为：
  $$
  L_{\text{critic}} = \frac{1}{5} \sum_{t=1}^5 \left( G_t - V(s_t) \right)^2
  $$

---

#### **（5）更新 Actor**

Actor 网络的优化目标为截断的策略目标：
$$
L^{CLIP}(\theta) = \mathbb{E}
\left[
\min \left( r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t \right)
\right]
$$
- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ 是新旧策略的比值。
- $A_t$ 是从 GAE 计算的优势函数。

##### **示例（更新 Actor）**
- 假设：
  - $\pi_{\theta_{\text{old}}}(a|s) = 0.7$，$\pi_\theta(a|s) = 0.9$；
  - $A_t = 2.5$，$\epsilon = 0.2$。
- 计算比值：
  $$
  r_t(\theta) = \frac{0.9}{0.7} = 1.2857
  $$
- 进行截断：
  $$
  \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) = \text{clip}(1.2857, 0.8, 1.2) = 1.2
  $$
- 最终目标值为：
  $$
  L^{CLIP} = \min \left( 1.2857 \cdot 2.5, 1.2 \cdot 2.5 \right) = 3.0
  $$

---

#### **（6）重复训练**

- 重复采样数据，更新 Actor 和 Critic 网络，直到策略收敛或达到预设的训练轮数。
- PPO 通常对同一批数据进行多次更新，提高数据利用率。







---

## **3. 优缺点**

**优点**  
- **简洁高效**： PPO 通过简单的截断函数代替复杂的 KL 散度约束，避免了 TRPO 中昂贵的二阶优化计算，易于实现。
- **更新稳定性好**： PPO 限制了策略更新的幅度，避免策略变化过大，训练更加稳定。
- **样本效率高**： PPO 使用小批量数据重复优化，提高了样本利用率。
- **广泛适用**： 适用于离散和连续动作空间，广泛应用于各种复杂任务。

**缺点**  
- **依赖超参数调优**： 截断参数 $\epsilon$ 等超参数对训练稳定性和效率影响较大，需要精心调试。
- **计算成本较高**： 相比于 DQN 等值函数方法，PPO 的计算成本较高，需同时更新 Actor 和 Critic。

PPO 是一种高效且稳定的强化学习算法，它在策略优化中通过截断约束限制策略更新幅度，兼具简洁性和高性能。在与 DQN、传统策略梯度、Actor-Critic 和 TRPO 的对比中，PPO 在稳定性和样本效率方面表现优异，是现代强化学习应用的主流算法之一。

---

## **4. 为什么 TRPO 和 PPO 是 On-policy？**

### **4.1 采样方式的演进**

| **算法**         | **采样方式**                            | **数据使用**                           | **采样效率** | **更新稳定性** |
|------------------|-----------------------------------------|---------------------------------------|--------------|----------------|
| 策略梯度         | On-policy，采样完整轨迹                 | 数据单次使用，仅用于一次更新           | 低           | 不稳定         |
| Actor-Critic     | On-policy，采样单步数据 $(s, a, r, s')$| 数据单次使用，Critic 提高方差效率      | 低           | 较稳定         |
| TRPO             | On-policy，采样 $(s, a, r, s')$       | 数据单次使用，二阶优化，KL 限制更新幅度 | 高           | 高             |
| PPO              | On-policy，采样 $(s, a, r, s')$       | 数据多次使用，截断机制限制更新幅度      | 高           | 高             |

- 表格中将 TRPO 的采样效率标记为高 是合理的，因为它在 On-policy 算法中，是第一个通过二阶优化方法显著提高数据使用效率的算法。


### **4.2 数据来源：**
- TRPO 和 PPO 的数据都是从**当前的策略（或旧策略）**中采样而来，而这个旧策略是从最近一次更新后的策略产生的。
- 每次更新后，新策略 $\pi_\theta(a|s)$ 会被用来重新采样数据，替代之前的数据。

### **4.3 策略比值的约束：**
- TRPO 和 PPO 的核心公式依赖于策略比值：
  $$
  r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
  $$
  - 其中 $\pi_{\theta_{\text{old}}}$ 是采样数据时的策略，$\pi_\theta$ 是更新后的目标策略。
  - 在 TRPO 和 PPO 中，优化目标是基于一个**固定的旧策略** $\pi_{\theta_{\text{old}}}$，而新策略 $\pi_\theta$ 是在约束条件下被优化的。
  
  **重要的是：**
  - 这仍然是 on-policy，因为使用的采样数据是和旧策略 $\pi_{\theta_{\text{old}}}$ 一致的，优化过程中的每一步都基于相同的策略分布。
  - 即使 PPO/TRPO 多次利用了数据，这些数据仍然只对当前策略的分布有效，当策略变化较大时，这些数据就失效了。

### **4.4 为什么数据不符合 Off-policy？**
- Off-policy 方法需要利用历史数据或其他策略（行为策略）的采样结果，而 TRPO 和 PPO 的数据只能用于优化最近的策略。
- 在 TRPO 和 PPO 中，如果数据来自一个完全不同的策略分布（如很久之前的策略采样数据），将无法有效估计策略更新目标，违反 on-policy 的假设。



---

## **5. 与 DQN、策略梯度、Actor-Critic、TRPO 的比较**

| **算法**        | **改进点**                                                                                               | **缺点**                                                              | **目标函数**                                                                                                                       |
|------------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| **DQN**         | - 使用 Q-learning 的值函数方法，学习状态-动作值函数 $Q(s, a)$。<br>- 引入经验回放和目标网络提高稳定性。      | - 只能用于离散动作空间。<br>- 更新方式较高效，但无法处理策略直接优化问题。        | $L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)^2 \right]$                                 |
| **策略梯度**    | - 直接优化策略 $\pi_\theta(a\|s)$，目标是最大化期望回报 $J(\theta)$。                                      | - 高方差，可能导致训练不稳定。<br>- 无基线方法，效率较低。                    | $J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \log \pi_\theta(a\|s) \cdot G_t \right]$                                               |
| **Actor-Critic**| - 引入价值函数基线 $V(s)$，降低梯度估计的方差。<br>- 同时优化策略（Actor）和价值网络（Critic）。            | - 易受 Critic 误差影响，训练不稳定。<br>- 策略更新步幅可能过大导致退化。         | $J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \log \pi_\theta(a\|s) \cdot \left(G_t - V(s)\right) \right]$                           |
| **TRPO**        | - 引入 KL 散度约束，限制策略更新步幅，防止策略退化。<br>- 提高复杂任务中的训练稳定性。                       | - 优化过程复杂，需二阶优化（如共轭梯度法）。<br>- 计算成本较高，难以实现高效训练。 | $\max_\theta \; \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a\|s)}{\pi_{\theta_{\text{old}}}(a\|s)} A_t \right]$ <br> s.t. $\mathbb{E}_{s \sim \pi_{\theta_{\text{old}}}} \left[ D_{\text{KL}} \left( \pi_{\theta_{\text{old}}} \| \pi_\theta \right) \right] \leq \delta$ |
| **PPO**         | - 用截断方式（Clipping）代替 KL 散度约束，简化实现。<br>- 可重复利用采样数据，效率更高。<br>- 提高训练稳定性和样本利用率。 | - 较依赖超参数调优（如 $\epsilon$）。<br>- 在高维任务中仍有一定计算成本。        | $L^{CLIP}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t \right) \right]$ |



---

## **6. 训练代码**

```python
# -------------- train.py ----------------
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


```

```python
# -------------- rl_utils.py ----------------
from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
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
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                
```


```python
# -------------- 打印 ----------------
Iteration 0: 100%|██████████████████████████████████████████████████████████████| 50/50 [00:04<00:00, 10.63it/s, episode=50, return=152.100]
Iteration 1: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  8.03it/s, episode=100, return=164.400]
Iteration 2: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.45it/s, episode=150, return=200.000]
Iteration 3: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.89it/s, episode=200, return=200.000]
Iteration 4: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  8.94it/s, episode=250, return=200.000]
Iteration 5: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  8.91it/s, episode=300, return=200.000]
Iteration 6: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  8.25it/s, episode=350, return=200.000]
Iteration 7: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  8.96it/s, episode=400, return=189.500]
Iteration 8: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  9.41it/s, episode=450, return=165.700]
Iteration 9: 100%|█████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  9.39it/s, episode=500, return=200.000]
```
