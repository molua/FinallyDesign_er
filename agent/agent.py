import torch
import torch.nn.functional as F
import random

from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from utils import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = torch.nn.Linear(hidden_dim * 2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print('PolicyNet: x.shape is ', x.shape)
        # return F.softmax(self.fc3(x), dim=1)
        return F.softmax(x, dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# class PPO:
#     ''' PPO算法,采用截断方式 '''
#     def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
#                  lmbda, epochs, eps, gamma, device):
#         self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
#         self.critic = ValueNet(state_dim, hidden_dim).to(device)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
#                                                 lr=actor_lr)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
#                                                  lr=critic_lr)
#         self.gamma = gamma
#         self.lmbda = lmbda
#         self.epochs = epochs  # 一条序列的数据用来训练轮数
#         self.eps = eps  # PPO中截断范围的参数
#         self.device = device
#
#     def take_action(self, state, epsilon, action_dim):
#         random.seed()
#         state = torch.tensor([state], dtype=torch.float).to(self.device)
#         # state = state.to(self.device) if isinstance(state, torch.Tensor) else torch.tensor([state], dtype=torch.float).to(self.device)
#         if random.random() < epsilon:  # 添加随机性
#             action = random.choice(range(action_dim))
#             return action
#         else:
#             probs = self.actor(state)
#             # print('state: ', state, 'probs: ', probs)
#             # print('probs: ', probs, torch.distributions.constraints.simplex.check(probs))
#             action_dist = torch.distributions.Categorical(probs)
#             action = action_dist.sample()
#             return action.item()
#
#     def update(self, transition_dict):
#         states = torch.tensor(transition_dict['states'],
#                               dtype=torch.float).to(self.device)
#         # states = transition_dict['states'].to(self.device) if isinstance(transition_dict['states'], torch.Tensor) else torch.tensor([transition_dict['states']], dtype=torch.float).to(self.device)
#
#         actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
#             self.device)
#         rewards = torch.tensor(transition_dict['rewards'],
#                                dtype=torch.float).view(-1, 1).to(self.device)
#         next_states = torch.tensor(transition_dict['next_states'],
#                                    dtype=torch.float).to(self.device)
#         dones = torch.tensor(transition_dict['dones'],
#                              dtype=torch.float).view(-1, 1).to(self.device)
#         td_target = rewards + self.gamma * self.critic(next_states) * (1 -
#                                                                        dones)  # 计算的是强化学习中的TD(时间差分)目标
#         td_delta = td_target - self.critic(states)  # TD(时间差分)目标(td_target)和TD误差(td_delta)。这两个值在强化学习中经常用于更新策略和价值函数。
#         advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
#                                                td_delta.cpu()).to(self.device)
#         old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()  # 计算在给定状态states下，执行动作actions的对数概率
#
#         for i in range(self.epochs):
#             log_probs = torch.log(self.actor(states).gather(1, actions))
#             ratio = torch.exp(log_probs - old_log_probs)  # 计算重要性采样的权重
#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1 - self.eps,
#                                 1 + self.eps) * advantage  # 截断
#             actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
#             critic_loss = torch.mean(
#                 F.mse_loss(self.critic(states), td_target.detach()))
#             self.actor_optimizer.zero_grad()
#             self.critic_optimizer.zero_grad()
#             actor_loss.backward()
#             critic_loss.backward()
#             self.actor_optimizer.step()
#             self.critic_optimizer.step()

class MyCustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(MyCustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        obs_shape = get_obs_shape(observation_space)
        self.features_extractor = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
        )

    def forward(self, observations):
        return self.features_extractor(observations)

class MyPPO(PPO):
    def __init__(self, policy, env, policy_kwargs=None, features_extractor_kwargs=None, **kwargs):
        super(MyPPO, self).__init__(policy, env, policy_kwargs=policy_kwargs, **kwargs)
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        self.set_features_extractor(MyCustomFeaturesExtractor(env.observation_space, self.features_dim, **features_extractor_kwargs))
