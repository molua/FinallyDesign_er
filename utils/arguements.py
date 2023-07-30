import argparse

import torch


def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--sumo_cfg_file', type=str, default='sumo_integration/examples/gong_ten_cross_one.sumocfg', help='sumo configuration file')
    # argparser.add_argument('--sumo_cfg_file', type=str, default='examples/gong_ten_cross_one.sumocfg', help='sumo configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    argparser.add_argument('--sumo-gui', default=False, help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=0.2,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'carla'],
                           help="select traffic light manager (default: none)",
                           default='none')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    argparser.add_argument('--output-dir', type=str, default='output', required=False, help='the directory to store logs, data and model')
    argparser.add_argument('--sumo-seed', type=int, default=42, help='sumo seed')
    argparser.add_argument('--carla-seed', type=int, default=21, help='sumo seed')
    argparser.add_argument('--xodr-path', type=str, default='sumo_integration/examples/opendrive/gong_ten_cross.xodr', help='xodr path')
    argparser.add_argument('--episode-length', type=int, default=3600, help='how many steps in a episode')
    # argparser.add_argument('--episode-length', type=int, default=36, help='how many steps in a episode')
    argparser.add_argument('--weights', type=str, default='best.pt', help='the path of model')
    argparser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    argparser.add_argument('--yellow-time', type=int, default=2, help='yellow light time')
    argparser.add_argument('--delta-time', type=int, default=10, help='Simulation seconds between actions')
    argparser.add_argument('--begin-time', type=int, default=0, help='Simulation begin time')
    argparser.add_argument('--min-green', type=int, default=5, help='Minimum Green Time')
    argparser.add_argument('--max-green', type=int, default=50, help='Maximum Green Time')
    # argparser.add_argument('--out-csv-name', type=str, default='output/data/run', help='Maximum Green Time')
    argparser.add_argument('--total-time', type=int, default=360000, help='time to complete all training')
    argparser.add_argument('--w-all-wait', type=float, default=6, help='the weight of all wait')
    argparser.add_argument('--w-max-wait', type=float, default=4, help='the weight of max wait')
    argparser.add_argument('--w-emergy-wait', type=float, default=0, help='the weight of emergy wait')
    argparser.add_argument('--process-number', type=int, default=1, help='')
    argparser.add_argument('--save-dir', type=str, default='output/', help='')
    argparser.add_argument('--config', type=str, default='config/ppo.yaml', help='')
    argparser.add_argument('--experiment-name', type=str, default='train', help='')
    argparser.add_argument('--resume-training', default=None, help='checkpoint file to resume training from')
    argparser.add_argument('--no-cuda', default=False, help='disables CUDA training')
    argparser.add_argument('--sumo-label', type=int, default=8813, help='disables CUDA training')
    args = argparser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args