#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module is responsible for the management of the carla simulation. """
import argparse
import datetime
# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import logging
import os
import random
import sys

import carla  # pylint: disable=import-error


# from constants import INVALID_ACTOR_ID, SPAWN_OFFSET_Z
from .constants import INVALID_ACTOR_ID, SPAWN_OFFSET_Z

# ==================================================================================================
# -- carla simulation ------------------------------------------------------------------------------
# ==================================================================================================


class CarlaSimulation(object):
    """
    CarlaSimulation is responsible for the management of the carla simulation.
    """
    def __init__(self, host, port, step_length, carla_seed):
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('gong_ten_cross')


        print('port: ', port)

        # client = carla.Client(host, port)
        # client.set_timeout(20.0)
        # self.world = client.load_world('gong_ten_cross')
        # self.client = client

        # self.world = self.client.load_world('gong_ten_cross')
        # self.world = self.load_customized_world(xodr_path)
        self.blueprint_library = self.world.get_blueprint_library()
        self.step_length = step_length

        # The following sets contain updated information for the current frame.
        self._active_actors = set()
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # Set traffic lights.
        self._tls = {}  # {landmark_id: traffic_ligth_actor}

        tmp_map = self.world.get_map()

        random.seed(carla_seed)

        for landmark in tmp_map.get_all_landmarks_of_type('1000001'):
            if landmark.id != '':
                traffic_ligth = self.world.get_traffic_light(landmark)
                if traffic_ligth is not None:
                    self._tls[landmark.id] = traffic_ligth
                else:
                    logging.warning('Landmark %s is not linked to any traffic light', landmark.id)

    def get_actor(self, actor_id):
        """
        Accessor for carla actor.
        """
        return self.world.get_actor(actor_id)

    # This is a workaround to fix synchronization issues when other carla clients remove an actor in
    # carla without waiting for tick (e.g., running sumo co-simulation and manual control at the
    # same time)
    def get_actor_light_state(self, actor_id):
        """
        Accessor for carla actor light state.

        If the actor is not alive, returns None.
        """
        try:
            actor = self.get_actor(actor_id)
            return actor.get_light_state()
        except RuntimeError:
            return None

    @property
    def traffic_light_ids(self):
        return set(self._tls.keys())

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        if landmark_id not in self._tls:
            return None
        return self._tls[landmark_id].state

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(True)
                # We set the traffic light to 'green' because 'off' state sets the traffic light to
                # 'red'.
                actor.set_state(carla.TrafficLightState.Green)

    def spawn_actor(self, blueprint, transform):
        """
        Spawns a new actor.

            :param blueprint: blueprint of the actor to be spawned.
            :param transform: transform where the actor will be spawned.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        transform = carla.Transform(transform.location + carla.Location(0, 0, SPAWN_OFFSET_Z),
                                    transform.rotation)

        batch = [
            carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False))
        ]
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error('Spawn carla actor failed. %s', response.error)
            return INVALID_ACTOR_ID

        return response.actor_id

    def destroy_actor(self, actor_id):
        """
        Destroys the given actor.
        """
        actor = self.world.get_actor(actor_id)
        if actor is not None:
            return actor.destroy()
        return False

    def synchronize_vehicle(self, vehicle_id, transform, lights=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param lights: new vehicle light state.
            :return: True if successfully updated. Otherwise, False.
        """
        vehicle = self.world.get_actor(vehicle_id)
        if vehicle is None:
            return False

        vehicle.set_transform(transform)
        if lights is not None:
            vehicle.set_light_state(carla.VehicleLightState(lights))
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param landmark_id: id of the landmark to be updated.
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        if not landmark_id in self._tls:
            logging.warning('Landmark %s not found in carla', landmark_id)
            return False

        traffic_light = self._tls[landmark_id]
        traffic_light.set_state(state)
        return True

    def tick(self):
        """
        Tick to carla simulation.
        """
        self.world.tick()

        # Update data structures for the current frame.
        current_actors = set(
            [vehicle.id for vehicle in self.world.get_actors().filter('vehicle.*')])
        self.spawned_actors = current_actors.difference(self._active_actors)
        self.destroyed_actors = self._active_actors.difference(current_actors)
        self._active_actors = current_actors

    def close(self):
        """
        Closes carla client.
        """
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(False)

    def load_customized_world(self, xodr_path):
        if os.path.exists(xodr_path):
            with open(xodr_path) as od_file:
                try:
                    data = od_file.read()
                except OSError:
                    print('file could not be readed.')
                    sys.exit()
            logging.info('load opendrive map %r.' % os.path.basename(xodr_path))
            vertex_distance = 2.0  # in meters
            max_road_length = 500.0  # in meters
            wall_height = 1.0  # in meters
            extra_width = 0.6  # in meters
            world = self.client.generate_opendrive_world(
                data, carla.OpendriveGenerationParameters(
                    vertex_distance=vertex_distance,
                    max_road_length=max_road_length,
                    wall_height=wall_height,
                    additional_width=extra_width,
                    smooth_junctions=False,
                    enable_mesh_visibility=True))
            return world

        else:
           sys.exit('xodr path not exist')

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--sumo_cfg_file', type=str, default='sumo_integration/examples/gong_ten_cross_one.sumocfg', help='sumo configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=3000,
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
    argparser.add_argument('--sumo-gui', default=True, help='run the gui version of sumo')
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
    argparser.add_argument('--config', type=str, default='config/config.ini', help='the config file')
    argparser.add_argument('--sumo-seed', type=int, default=42, help='sumo seed')
    argparser.add_argument('--carla-seed', type=int, default=21, help='sumo seed')
    argparser.add_argument('--xodr-path', type=str, default='sumo_integration/examples/opendrive/gong_ten_cross.xodr', help='xodr path')
    argparser.add_argument('--episode-length', type=int, default=3600, help='how many steps in a episode')
    # argparser.add_argument('--episode-length', type=int, default=36, help='how many steps in a episode')
    argparser.add_argument('--weights', type=str, default='best.pt', help='the path of model')
    argparser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    argparser.add_argument('--yellow-time', type=int, default=1, help='yellow light time')
    argparser.add_argument('--delta-time', type=int, default=10, help='Simulation seconds between actions')
    argparser.add_argument('--begin-time', type=int, default=0, help='Simulation begin time')
    argparser.add_argument('--min-green', type=int, default=5, help='Minimum Green Time')
    argparser.add_argument('--max-green', type=int, default=50, help='Maximum Green Time')
    argparser.add_argument('--out-csv-name', type=str, default='output/data/run', help='Maximum Green Time')
    argparser.add_argument('--total-time', type=int, default=360000, help='time to complete all training')
    argparser.add_argument('--w-all-wait', type=float, default=6, help='the weight of all wait')
    argparser.add_argument('--w-max-wait', type=float, default=4, help='the weight of max wait')
    argparser.add_argument('--w-emergy-wait', type=float, default=0, help='the weight of emergy wait')
    argparser.add_argument('--process-number', type=int, default=2, help='')
    args = argparser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    carla_env = CarlaSimulation(args.carla_host, args.carla_port, args.step_length, args.carla_seed)
    spawn_points = carla_env.world.get_map().get_spawn_points()
    # client = carla.Client(args.carla_host, args.carla_port)
    # client.set_timeout(20.0)
    # synchronous_master = False
    # # random.seed(args.seed if args.seed is not None else int(time.time()))
    #
    #
    #     # world = client.load_world('Town07')
    # world = client.load_world('gong_ten_cross')
    while True:
        print(datetime.datetime.now())
        carla_env.world.wait_for_tick()
