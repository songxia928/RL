

import os
import torch


#game_name = 'CartPole-v0'
game_name = 'CartPole-v1'
method_name = 'DQN'

# == buffer
buffer_size = 10000
minimal_size = 500
batch_size = 64

# == model
hidden_dim = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

target_update = 10
epsilon = 0.01
gamma = 0.98
lr = 2e-3

# == iteration
iteration_num = 10  # 迭代次数
episode_num = 100    # 单次迭代的游戏局数

max_step_num = None  # 单局的最大步长， None 表示不限制
if game_name == 'CartPole-v0':
    max_step_num = 200  
elif game_name == 'CartPole-v1':
    max_step_num = 500 


# == path
dir_data = './data/' + method_name
dir_out = './output/' + method_name + '_' + game_name
dir_models = dir_out + '/models'
dir_figures = dir_out + '/figures'
model_path = dir_out + '/models/lastest.pt'
train_result_path = dir_out + '/train_result.csv'
train_figure_path = dir_out + '/train_figure.png'
test_result_path = dir_out + '/test_result.csv'


if not os.path.exists(dir_models): os.makedirs(dir_models)
if not os.path.exists(dir_figures): os.makedirs(dir_figures)
