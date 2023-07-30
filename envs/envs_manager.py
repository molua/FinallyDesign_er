import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecEnvWrapper

from envs.envs import CoEnvs

def obs_to_dict(obs):  # 将观测值转换成字典形式
    """
    Convert an observation into a dict.
    """
    if isinstance(obs, dict) or isinstance(obs, gym.spaces.Dict):
        return obs
    return {None: obs}

def dict_to_obs(obs_dict):  # 将字典形式转换成观测值
    """
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data typesVecEnvWrapper

    # 它首先调用原始并行化环境容器 venv 的 reset() 方法获取观测数据。然后将观测数据转换为 PyTorch 张量，并将其移动到指定的设备上。最后返回转换后的观测数据。
    def reset(self):
        obs = self.venv.reset()
        obs = obs_to_dict(obs)
        for k in obs:
            obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        return dict_to_obs(obs)

    # 该方法用于异步执行动作。
    def step_async(self, actions):
        if type(actions) != np.ndarray:
            actions = actions.cpu().numpy()
            # actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    # 该方法用于等待异步执行的动作并获取环境的反馈。它调用 venv 的 step_wait() 方法，获取环境的反馈结果，包括观测、奖励、是否终止和其他信息
    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = obs_to_dict(obs)
        for k in obs:
            obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return dict_to_obs(obs), reward, done, info


class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):

        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        wos = obs_to_dict(wos)
        self.stacked_obs = {}
        new_observation_spaces = {}
        self.shape_dim0 = {}
        for k in wos.spaces:

            self.shape_dim0[k] = wos.spaces[k].shape[0]
            low = np.repeat(wos.spaces[k].low, self.nstack, axis=0)
            high = np.repeat(wos.spaces[k].high, self.nstack, axis=0)

            if device is None:
                device = torch.device('cpu')
            self.stacked_obs[k] = torch.zeros((venv.num_envs,) + low.shape).to(device)

            new_observation_spaces[k] = gym.spaces.Box(
                low=low, high=high, dtype=venv.observation_space.dtype)

        if set(new_observation_spaces.keys()) == {None}:
            VecEnvWrapper.__init__(self, venv, observation_space=new_observation_spaces[None])
        else:
            VecEnvWrapper.__init__(self, venv, observation_space=gym.spaces.Dict(new_observation_spaces))


    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = obs_to_dict(obs)
        for k in obs:
            self.stacked_obs[k][:, :-self.shape_dim0[k]] = \
                self.stacked_obs[k][:, self.shape_dim0[k]:]
            for (i, new) in enumerate(news):
                if new:
                    self.stacked_obs[k][i] = 0
            self.stacked_obs[k][:, -self.shape_dim0[k]:] = obs[k]
        return dict_to_obs(self.stacked_obs), rews, news, infos

    def reset(self):

        obs = self.venv.reset()
        obs = obs_to_dict(obs)
        for k in obs:
            self.stacked_obs[k].zero_()
            self.stacked_obs[k][:, -self.shape_dim0[k]:] = obs[k]
        return dict_to_obs(self.stacked_obs)

    def close(self):
        self.venv.close()


def make_env(args, carla_port, sumo_label, csv_path):
    return lambda: CoEnvs(args, carla_port, sumo_label, csv_path)

def make_vec_envs(args, config, device):
    carla_port = range(args.carla_port, args.carla_port + 1000 * config.num_processes, 1000)
    sumo_label = range(args.sumo_label, args.sumo_label + 1 * config.num_processes)
    envs = [make_env(args, carla_port[i], sumo_label[i], args.out_csv_name + '_{}'.format(i)) for i in range(config.num_processes)]
    if len(envs) > 1 or config.apply_her:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if config.gamma is None:
        envs = VecNormalize(envs, norm_reward=False, norm_obs=config.norm_obs)
    else:
        envs = VecNormalize(envs, gamma=config.gamma, norm_reward=config.norm_reward, norm_obs=config.norm_obs)

    # envs = VecPyTorch(envs, device)
    #
    # envs = VecPyTorchFrameStack(envs, config.num_frame_stack, device)


    return envs