import argparse
import configparser
import logging
import os.path
import random
import shutil
import time
from collections import namedtuple
import datetime


import torch
import numpy as np
import yaml
from stable_baselines3 import PPO

from torch.utils.tensorboard import SummaryWriter

from agent.agent import MyPPO
from envs.envs import CoEnvs
from envs.envs_manager import make_vec_envs

from utils import rl_utils
from utils.arguements import parse_args
from utils.general import increment_path



def seed_everything(seed=10):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)  # Python
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_dir(output_path, outputs=['log', 'data', 'model']):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dirs = {}
    for path in outputs:
        cur_dir = output_path + '\%s\\' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs

def init_logs(log_path):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_path, time.time())),
                            logging.StreamHandler()
                        ])

def load_config_file(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

        # To be careful with values like 7e-5
        config['lr'] = float(config['lr'])
        config['eps'] = float(config['eps'])
        config['alpha'] = float(config['alpha'])
        return config

def get_config_and_checkpoint(args):
    config_dict, checkpoint = None, None
    if args.config and args.resume_training:
        print('ERROR: Should either provide --config or --resume-training but not both.')
        exit(1)

    if args.config:
        config_dict = load_config_file(args.config)  # 加载yaml和学习率

    if args.resume_training:
        print('Resuming training from: {}'.format(args.resume_training))
        assert os.path.isfile(args.resume_training), 'Checkpoint file does not exist'
        checkpoint = torch.load(args.resume_training)
        config_dict = checkpoint['config']

    if config_dict is None:
        print("ERROR: --config or --resume-training flag is required.")
        exit(1)

    config = namedtuple('Config', config_dict.keys())(*config_dict.values())
    return config, checkpoint

# 定义回调函数来保存模型
def save_model_callback(_locals, _globals):
    if _locals['self'].num_timesteps % 3600 == 0:
        _locals['self'].save("ppo_carla_model")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    seed_everything()
    # dirs = init_dir(args.output_dir)
    # init_logs(dirs['log'])

    # 单环境
    # args.out_csv_name = increment_path('output/data/csv/env_only_', mkdir=True).as_posix()
    # args.summary_path = increment_path('output/data/tensorboard/env_only_', mkdir=True).as_posix()
    # envs = CoEnvs(args)
    # envs = DummyVecEnv(envs)
    # envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
    # summary_path = increment_path('output/data/tensorboard', mkdir=True).as_posix()
    # summary_writer = SummaryWriter(summary_path)

    # try:
    #     # init env
    #     # model = PPO(config['MODEL_CONFIG'])
    #     model = PPO('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, n_steps=1024, n_epochs=20,
    #                 batch_size=256, clip_range=0.2, verbose=0)
    #     model.learn(total_timesteps=800000)
    #
    #     # trainer = Trainer()
    #     # trainer.run()
    # except Exception as e:
    #     logging.warning(e)
    # finally:
    #     env.close(sensor=True)
    #     summary_writer.close()

    # model = PPO('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, n_steps=1024, n_epochs=20,
    #             batch_size=256, clip_range=0.2, verbose=0)
    # model.learn(total_timesteps=800000)

    # actor_lr = 0.0005
    # critic_lr = 0.0005
    # num_episodes = 500
    # hidden_dim = 128
    # gamma = 0.98
    # lmbda = 0.95
    # epochs = 10
    # eps = 0.2
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    #     "cpu")
    # state_dim = 53
    # action_dim = 4
    # epsilon = 0.2

    # agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
    #             epochs, eps, gamma, device)

    # return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes, epsilon, action_dim, summary_writer)
    # checkpoint_callback = CheckpointCallback(save_freq=3600, save_path='output/model/', name_prefix='PPO_model')
    # model = MyPPO('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, n_steps=100, n_epochs=10,
    #             batch_size=25, clip_range=0.2, verbose=0)
    #
    # model.learn(total_timesteps=730000)

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    now = datetime.datetime.now()
    experiment_name = args.experiment_name + '_' + now.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir_model = os.path.join(args.save_dir, 'model', experiment_name)
    save_dir_config = os.path.join(args.save_dir, 'config', experiment_name)

    config, checkpoint = get_config_and_checkpoint(args)

    os.makedirs(save_dir_model)
    os.makedirs(save_dir_config)


    if args.config:
        shutil.copy2(args.config, save_dir_config)  # 为每一次实验保存超参数

    # Tensorboard Logging
    args.summary_path = os.path.join(args.save_dir, 'tensorboard', experiment_name)

    # cvs_path
    args.out_csv_name = os.path.join(args.save_dir, 'csv', experiment_name)


    device = torch.device("cuda:0" if args.cuda else "cpu")



    envs = make_vec_envs(args, config, device)

    agent = PPO(policy=config.policy, env=envs, learning_rate=config.lr, n_steps=config.n_steps,  n_epochs=config.n_epochs,
                batch_size=config.batch_size, clip_range=0.2, verbose=0)

    agent.learn(total_timesteps=720000, callback=save_model_callback)

    print('yes!')

    # env.close(sensor=True)
    # env.summary_writer.close()
