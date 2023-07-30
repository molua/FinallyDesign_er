#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Script to integrate CARLA and SUMO simulations
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import argparse
import logging
import time

# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================

import glob
import os
import sys

import carla
import traci

try:
    sys.path.append(
        glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================


class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    """
    def __init__(self,
                 sumo_simulation,
                 carla_simulation,
                 tls_manager='none',
                 sync_vehicle_color=False,
                 sync_vehicle_lights=False):

        self.sumo = sumo_simulation
        self.carla = carla_simulation

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights

        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.max_substep_delta_time = 0.1
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        self.image_width = 640
        self.image_height = 640
        self.fov = 90
        camera_blueprint = BridgeHelper.blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_blueprint.set_attribute('image_size_x', str(self.image_width))
        camera_blueprint.set_attribute('image_size_y', str(self.image_height))
        camera_blueprint.set_attribute('fov', str(self.fov))

        camera_blueprint_rgb = BridgeHelper.blueprint_library.find('sensor.camera.rgb')
        camera_blueprint_rgb.set_attribute('image_size_x', str(self.image_width))
        camera_blueprint_rgb.set_attribute('image_size_y', str(self.image_height))
        camera_blueprint_rgb.set_attribute('fov', str(self.fov))

        self.camera_n = self.carla.world.spawn_actor(camera_blueprint, carla.Transform(carla.Location(x=-3.63, y=-13.55, z=18.0), carla.Rotation(pitch=-40, yaw=-90.0, roll=0.0)))
        self.camera_s = self.carla.world.spawn_actor(camera_blueprint, carla.Transform(carla.Location(x=3.63, y=13.55, z=18.0), carla.Rotation(pitch=-40, yaw=90.0, roll=0.0)))  # s
        self.camera_w = self.carla.world.spawn_actor(camera_blueprint, carla.Transform(carla.Location(x=-13.55, y=3.63, z=18.0), carla.Rotation(pitch=-40, yaw=180.0, roll=0.0)))  # w
        self.camera_e = self.carla.world.spawn_actor(camera_blueprint,carla.Transform(carla.Location(x=13.55, y=-3.63, z=18.0), carla.Rotation(pitch=-40, yaw=0.0, roll=0.0)))  # e

        # self.camera_n.listen(lambda data: self.process_img(image=data, name="camera_n"))
        # self.camera_s.listen(lambda data: self.process_img(image=data, name="camera_s"))
        # self.camera_e.listen(lambda data: self.process_img(image=data, name="camera_e"))
        # self.camera_w.listen(lambda data: self.process_img(image=data, name="camera_w"))

        self.camera_n.listen(lambda image: image.save_to_disk('carla_images_segment_3/n_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette))
        self.camera_s.listen(lambda image: image.save_to_disk('carla_images_segment_3/s_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette))
        self.camera_e.listen(lambda image: image.save_to_disk('carla_images_segment_3/e_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette))
        self.camera_w.listen(lambda image: image.save_to_disk('carla_images_segment_3/w_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette))

        self.camera_n_rgb = self.carla.world.spawn_actor(camera_blueprint_rgb, carla.Transform(carla.Location(x=-3.63, y=-13.55, z=18.0), carla.Rotation(pitch=-40, yaw=-90.0, roll=0.0)))
        self.camera_s_rgb = self.carla.world.spawn_actor(camera_blueprint_rgb, carla.Transform(carla.Location(x=3.63, y=13.55, z=18.0), carla.Rotation(pitch=-40, yaw=90.0, roll=0.0)))  # s
        self.camera_w_rgb = self.carla.world.spawn_actor(camera_blueprint_rgb, carla.Transform(carla.Location(x=-13.55, y=3.63, z=18.0), carla.Rotation(pitch=-40, yaw=180.0, roll=0.0)))  # w
        self.camera_e_rgb = self.carla.world.spawn_actor(camera_blueprint_rgb,carla.Transform(carla.Location(x=13.55, y=-3.63, z=18.0), carla.Rotation(pitch=-40, yaw=0.0, roll=0.0)))  # e

        # self.camera_n_rgb.listen(lambda data: self.process_img_rgb(image=data, name="camera_n"))
        # self.camera_s_rgb.listen(lambda data: self.process_img_rgb(image=data, name="camera_s"))
        # self.camera_w_rgb.listen(lambda data: self.process_img_rgb(image=data, name="camera_e"))
        # self.camera_e_rgb.listen(lambda data: self.process_img_rgb(image=data, name="camera_w"))


        self.camera_n_rgb.listen(lambda image: image.save_to_disk('carla_images_segment_3/n_out/%06d.png' % image.frame))
        self.camera_s_rgb.listen(lambda image: image.save_to_disk('carla_images_segment_3/s_out/%06d.png' % image.frame))
        self.camera_w_rgb.listen(lambda image: image.save_to_disk('carla_images_segment_3/w_out/%06d.png' % image.frame))
        self.camera_e_rgb.listen(lambda image: image.save_to_disk('carla_images_segment_3/e_out/%06d.png' % image.frame))


        self.sensor_list = []
        self.sensor_list.append(self.camera_n)
        self.sensor_list.append(self.camera_s)
        self.sensor_list.append(self.camera_e)
        self.sensor_list.append(self.camera_w)
        self.sensor_list.append(self.camera_n_rgb)
        self.sensor_list.append(self.camera_s_rgb)
        self.sensor_list.append(self.camera_w_rgb)
        self.sensor_list.append(self.camera_e_rgb)

    def process_img(self, image, name):
        frame = image.frame
        # print('frame: ', frame)

        if frame % 5 == 0:
            if name == 'camera_n':
                image.save_to_disk('carla_images_segment_2/n_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette)
                # image.save_to_disk('carla_images_segment_1/n_out/%06d.png' % image.frame)
            if name == 'camera_s':
                image.save_to_disk('carla_images_segment_2/s_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette)
                # image.save_to_disk('carla_images_segment_1/s_out/%06d.png' % image.frame)
            if name == 'camera_w':
                image.save_to_disk('carla_images_segment_2/w_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette)
                # image.save_to_disk('carla_images_segment_1/w_out/%06d.png' % image.frame)
            if name == 'camera_e':
                image.save_to_disk('carla_images_segment_2/e_out_segment/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette)
                # image.save_to_disk('carla_images_segment_1/e_out/%06d.png' % image.frame)

    def process_img_rgb(self, image, name):
        frame = image.frame
        # print('frame: ', frame)

        if frame % 5 == 0:
            if name == 'camera_n':
                image.save_to_disk('carla_images_segment_2/n_out/%06d.png' % image.frame)
            if name == 'camera_s':
                image.save_to_disk('carla_images_segment_2/s_out/%06d.png' % image.frame)
            if name == 'camera_w':
                image.save_to_disk('carla_images_segment_2/w_out/%06d.png' % image.frame)
            if name == 'camera_e':
                image.save_to_disk('carla_images_segment_2/e_out/%06d.png' % image.frame)

    def tick(self):
        """
        Tick to simulation synchronization
        """
        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)

                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                             carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(sumo_actor.signals,
                                                                     carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()
        for x in self.sensor_list:
            # if x.is_listening():
            #     x.stop()
            self.carla.destroy_actor(x)


def synchronization_loop(args):
    """
    Entry point for sumo-carla co-simulation.
    """
    sumo_simulation = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, args.client_order)
    # carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length)
    carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length, args.xodr_path, args.carla_seed)

    synchronization = SimulationSynchronization(sumo_simulation, carla_simulation, args.tls_manager,
                                                args.sync_vehicle_color, args.sync_vehicle_lights)
    try:
        while True:
            start = time.time()

            synchronization.tick()

            end = time.time()
            elapsed = end - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning synchronization')

        synchronization.close()
    # traci.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--sumo_cfg_file', type=str, help='sumo configuration file', default='sumo_integration/examples/gong_ten_cross_one.sumocfg')
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
    argparser.add_argument('--sumo-gui', default=True, help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=1,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'carla'],
                           help="select traffic light manager (default: none)",
                           default='none')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    argparser.add_argument('--carla-seed', type=int, default=21, help='sumo seed')
    argparser.add_argument('--xodr-path', type=str, default='sumo_integration/examples/opendrive/gong_ten_cross.xodr', help='xodr path')
    arguments = argparser.parse_args()

    if arguments.sync_vehicle_all is True:
        arguments.sync_vehicle_lights = True
        arguments.sync_vehicle_color = True

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synchronization_loop(arguments)
