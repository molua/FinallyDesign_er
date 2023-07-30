import logging
import time
from pathlib import Path

import carla
import cv2
import gym
import imutils
import numpy as np
import pandas as pd
import torch
import traci
from gym import spaces
from gym.envs.registration import EnvSpec
from torch.utils.tensorboard import SummaryWriter

from envs.junction import Junc
from envs.traffic_signals import TrafficSignal
from models.common import DetectMultiBackend
from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position
from utils.general import increment_path


class CoEnvs(gym.Env):
    def __init__(self, args, carla_port, sumo_label, csv_path):
        self.sumo = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host, args.sumo_gui, args.client_order, args.sumo_seed, sumo_label, args.sumo_port)
        self.carla = CarlaSimulation(args.carla_host,  carla_port, args.step_length, args.carla_seed)
        self.tls_manager = args.tls_manager
        self.sumo_host = args.sumo_host
        self.sumo_cfg_file = args.sumo_cfg_file
        self.sumo_port = args.sumo_port
        self.sumo_gui = args.sumo_gui
        self.client_order = args.client_order
        self.sumo_seed = args.sumo_seed
        self.total_time = args.total_time
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.
        self.episode_length = args.episode_length
        self.step_length = args.step_length
        self.delta_time = args.delta_time
        self.yellow_time = args.yellow_time
        self.begin_time = args.begin_time
        self.min_green = args.min_green
        self.max_green = args.max_green
        self.run = 0  # 表示环境运行的次数
        self.out_csv_name = csv_path
        self.metrics = []  # 用于记录每次episode的数据
        self.traffic_signal = TrafficSignal(self, self.sumo.ts_id, self.delta_time, self.yellow_time, self.min_green,
                                            self.max_green, self.begin_time)
        self.spec = EnvSpec('SUMORL-v0', entry_point='envs:CoEnvs')
        self.observation_space = spaces.Box(low=np.zeros(self.traffic_signal.num_green_phases + 7 * 12 + 1, dtype=np.float32),
                                            high=np.full((self.traffic_signal.num_green_phases + 7 * 12 + 1), 9999, dtype=np.float32))
        self.action_space = spaces.Discrete(self.traffic_signal.num_green_phases)
        self.last_reward = None
        self.next_action_time = 0
        self.last_sum_all_wait = 0
        self.last_sum_all_wait_detail = [0, 0, 0, 0]
        self.last_sum_max_wait = 0
        self.last_emergy_wait = 0
        self.last_emergy_wait_0 = 0
        self.last_stop_vehicle = 0
        self.last_sum_emergy_max_wait = 0
        self.last_sum_emergy_max_wait_0 = 0
        self.last_sum_emergy_num = 0
        self.last_sum_emergy_num_0 = 0
        self.last_sum_all_wait_per_lane_ave = 0
        self.last_sum_all_wait_per_lane_ave_max = 0
        self.w_max_wait = args.w_max_wait
        self.w_all_wait = args.w_all_wait
        self.w_emergy_wait = args.w_emergy_wait
        self.summary_writer = SummaryWriter(args.summary_path)
        self.action = 0
        self.last_stop_time = 0




        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.max_substep_delta_time = 0.02
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        self.image_width = 640
        self.image_height = 640
        self.fov = 90
        camera_blueprint = BridgeHelper.blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute('image_size_x', str(self.image_width))
        camera_blueprint.set_attribute('image_size_y', str(self.image_height))
        camera_blueprint.set_attribute('fov', str(self.fov))

        self.n = Junc(name='n', world=self.carla.world, camera_blueprint=camera_blueprint, fps=1 / self.step_length)
        self.s = Junc(name='s', world=self.carla.world, camera_blueprint=camera_blueprint, fps=1 / self.step_length)
        self.e = Junc(name='e', world=self.carla.world, camera_blueprint=camera_blueprint, fps=1 / self.step_length)
        self.w = Junc(name='w', world=self.carla.world, camera_blueprint=camera_blueprint, fps=1 / self.step_length)


        self.n.camera.listen(lambda data: self.process_img(image=data, name="camera_n"))
        self.s.camera.listen(lambda data: self.process_img(image=data, name="camera_s"))
        self.e.camera.listen(lambda data: self.process_img(image=data, name="camera_e"))
        self.w.camera.listen(lambda data: self.process_img(image=data, name="camera_w"))

        self.carla.tick()

        self.sensorids = []
        self.sensorids.append(self.n.camera)
        self.sensorids.append(self.s.camera)
        self.sensorids.append(self.w.camera)
        self.sensorids.append(self.e.camera)

    def tick(self):

        self.sumo.tick()
        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)  # 订阅输入路网的车辆信息
            sumo_actor = self.sumo.get_actor(sumo_actor_id)  # 用于获取订阅车辆对象更新信息的结果，包括车辆的类型，vclass，位置（carla_transform),车辆信号灯状态0：无信号灯，1：红灯，2：黄灯，3：绿灯， carla的vector3D， 车辆颜色

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor=sumo_actor)
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

        # Updates traffic lights in carla based on sumo information.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            self.carla.synchronize_vehicle(vehicle_id=carla_actor_id, transform=carla_transform)
        self.carla.tick()


    def synchronize(self):
        # try:
        start = time.time()
        self.tick()
        self.n.c2w = np.linalg.inv(np.array(self.n.camera.get_transform().get_inverse_matrix()))
        self.s.c2w = np.linalg.inv(np.array(self.s.camera.get_transform().get_inverse_matrix()))
        self.w.c2w = np.linalg.inv(np.array(self.w.camera.get_transform().get_inverse_matrix()))
        self.e.c2w = np.linalg.inv(np.array(self.e.camera.get_transform().get_inverse_matrix()))

        # print(self.n.frameCounter, self.s.frameCounter, self.w.frameCounter, self.e.frameCounter)


        if self.n.img is not None:
            track_result = self.n.model.feedCap(junction=self.n)
            junction = self.n
            # print('n: ', 'lane_0:', junction.lane_0.all_wait, junction.lane_0.track_num,
            #       junction.lane_0.vehicle_num, junction.lane_0.max_wait,
            #       '   lane_1: ', junction.lane_1.all_wait, junction.lane_1.track_num,
            #       junction.lane_1.vehicle_num, junction.lane_1.max_wait,
            #       '   lane_2: ', junction.lane_2.all_wait, junction.lane_2.track_num,
            #       junction.lane_2.vehicle_num,  junction.lane_2.max_wait)
            # if track_result is not None:
            #     cv2.imshow('cemara_n', track_result)
            #     cv2.waitKey(1)
        if self.s.img is not None:
            track_result = self.s.model.feedCap(junction=self.s)
            junction = self.s
            # print('s: ', 'lane_0:', junction.lane_0.all_wait, junction.lane_0.track_num,
            #       junction.lane_0.vehicle_num, junction.lane_0.max_wait,
            #       '   lane_1: ', junction.lane_1.all_wait, junction.lane_1.track_num,
            #       junction.lane_1.vehicle_num, junction.lane_1.max_wait,
            #       '   lane_2: ', junction.lane_2.all_wait, junction.lane_2.track_num,
            #       junction.lane_2.vehicle_num,  junction.lane_2.max_wait)
            # if track_result is not None:
            #     cv2.imshow('cemara_s', track_result)
            #     cv2.waitKey(1)
        if self.w.img is not None:
            track_result = self.w.model.feedCap(junction=self.w)
            junction = self.w
            # print('w: ', 'lane_0:', junction.lane_0.all_wait, junction.lane_0.track_num,
            #       junction.lane_0.vehicle_num, junction.lane_0.max_wait,
            #       '   lane_1: ', junction.lane_1.all_wait, junction.lane_1.track_num,
            #       junction.lane_1.vehicle_num, junction.lane_1.max_wait,
            #       '   lane_2: ', junction.lane_2.all_wait, junction.lane_2.track_num,
            #       junction.lane_2.vehicle_num,  junction.lane_2.max_wait)
            # if track_result is not None:
            #     cv2.imshow('cemara_w', track_result)
            #     cv2.waitKey(1)
        if self.e.img is not None:
            track_result = self.e.model.feedCap(junction=self.e)
            junction = self.e
            # print('e: ', 'lane_0:', junction.lane_0.all_wait, junction.lane_0.track_num,
            #       junction.lane_0.vehicle_num, junction.lane_0.max_wait,
            #       '   lane_1: ', junction.lane_1.all_wait, junction.lane_1.track_num,
            #       junction.lane_1.vehicle_num, junction.lane_1.max_wait,
            #       '   lane_2: ', junction.lane_2.all_wait, junction.lane_2.track_num,
            #       junction.lane_2.vehicle_num,  junction.lane_2.max_wait)
            # if track_result is not None:
            #     cv2.imshow('cemara_e', track_result)
            #     cv2.waitKey(1)
        # print(self.n.img is not None, self.s.img is not None, self.w.img is not None, self.e.img is not None, self.n.frameCounter, self.s.frameCounter, self.w.frameCounter, self.e.frameCounter)

        end = time.time()
        elapsed = end - start
        if elapsed < self.step_length:
            time.sleep(self.step_length - elapsed)
        # except Exception as e:
        #     logging.info(e)
        # finally:
        #     logging.info('Cleaning synchronization')

    def process_img(self, image, name, K_inverse=None, camera=None, fps=None, lane_line=None, max_det=1000, hide_labels=False, hide_conf=False):
        frame = image.frame
        # print('frame: ', frame)

        # if frame:
        #     if name == 'camera_n':
        #         image.save_to_disk('carla_images/n_out/%06d.png' % image.frame)
        #     if name == 'camera_s':
        #         image.save_to_disk('carla_images/s_out/%06d.png' % image.frame)
        #     if name == 'camera_w':
        #         image.save_to_disk('carla_images/w_out/%06d.png' % image.frame)
        #     if name == 'camera_e':
        #         image.save_to_disk('carla_images/e_out/%06d.png' % image.frame)

        """CAMERA  method"""
        image = np.array(image.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        # we need to remove the alpha channel
        image = image[:, :, :3]

        mask = np.zeros(image.shape, dtype=np.uint8)

        if name == 'camera_n':
            self.n.img = image
            points = np.array([[(116, 639), (293, 100), (330, 100), (419, 639)]])  # 感兴趣的区域
            cv2.fillPoly(mask, points, (255, 255, 255))
            self.n.img_mask = cv2.bitwise_and(self.n.img, mask)
        if name == 'camera_s':
            self.s.img = image
            points = np.array([[(101, 639), (294, 100), (330, 100), (425, 639)]])  # 感兴趣的区域
            cv2.fillPoly(mask, points, (255, 255, 255))
            self.s.img_mask = cv2.bitwise_and(self.s.img, mask)
        if name == 'camera_w':
            self.w.img = image
            points = np.array([[(111, 639), (296, 100), (329, 100), (422, 639)]])  # 感兴趣的区域
            cv2.fillPoly(mask, points, (255, 255, 255))
            self.w.img_mask = cv2.bitwise_and(self.w.img, mask)
        if name == 'camera_e':
            self.e.img = image
            points = np.array([[(120, 639), (294, 100), (330, 100), (418, 639)]])  # 感兴趣的区域
            cv2.fillPoly(mask, points, (255, 255, 255))
            self.e.img_mask = cv2.bitwise_and(self.e.img, mask)

    def close(self, sensor=False):
        """
        关闭仿真连接
        :return:
        """
        settings = self.carla.world.get_settings()
        settings.max_substep_delta_time = 0.01
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)
        if sensor:

            for x in self.sensorids:
                # if x.is_listening():
                #     x.stop()
                self.carla.destroy_actor(x)
        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)


        self.sumo2carla_ids.clear()
        self.carla.close()
        self.sumo.close()
        self.sumo = None

        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.max_substep_delta_time = 0.02
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

    def reset(self):
        # 重新连接sumo仿真生成车流
        if self.run != 0:
            if self.run != self.total_time / self.episode_length - 1:
                self.close(sensor=False)
            else:
                self.close(sensor=True)
            out_csv_name = increment_path(self.out_csv_name, mkdir=True).as_posix()
            self.save_csv(out_csv_name, self.run)
            self._start_simulation()
        self.run += 1
        self.metrics = []
        self.traffic_signal = TrafficSignal(self, self.sumo.ts_id, self.delta_time, self.yellow_time, self.min_green,
                                            self.max_green, self.begin_time)

        self.n.all_number_reset()
        self.s.all_number_reset()
        self.w.all_number_reset()
        self.e.all_number_reset()

        self.n.frameCounter = 0
        self.s.frameCounter = 0
        self.w.frameCounter = 0
        self.e.frameCounter = 0

        self.next_action_time = 0

        # self.next_action_time = self.delta_time

        return self._compute_observation()

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        # 每进行一次delta_step的时候需要重新观察路面情况
        self.n.all_number_reset()
        self.s.all_number_reset()
        self.w.all_number_reset()
        self.e.all_number_reset()

        self.action = action



        if action is None or action == {}:
            for _ in range(self.delta_time / self.step_length):
                self.synchronize()
        else:
            self._apply_actions(action)
            self._run_steps()

        observation = self._compute_observation()  # 一个长度为4+6*12 + 1的np数组

        reward = self._compute_reward()

        done = self._compute_done()
        self._compute_info()

        assert self.e.frameCounter == self.n.frameCounter == self.w.frameCounter == self.s.frameCounter
        self.summary_writer.add_scalar('all_wait', self.metrics[-1]['all_wait'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('max_wait', self.metrics[-1]['max_wait'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('ave_wait', self.metrics[-1]['ave_wait'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('reward', self.metrics[-1]['reward'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('stop_time', self.metrics[-1]['stop_time'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('emergy_max_wait', self.metrics[-1]['emergy_max_wait'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('emergy_wait', self.metrics[-1]['emergy_wait'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('emergy_max_wait_0', self.metrics[-1]['emergy_max_wait_0'], self.run * 3600 + self.e.frameCounter * self.step_length)
        self.summary_writer.add_scalar('emergy_wait_0', self.metrics[-1]['emergy_wait_0'], self.run * 3600 + self.e.frameCounter * self.step_length)

        # print(self.e.frameCounter * self.step_length, done, reward, action, traci.trafficlight.getRedYellowGreenState('4'),self.metrics[-1])

        # return observation, reward, done, self.metrics[-1], self.traffic_signal.green_phase
        return observation, reward, done, {}


    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)  # 创建输出文件所在的目录，如果目录已经存在则不做任何操作
            df.to_csv(out_csv_name + '/{}'.format(run) + '.csv', index=False)

    def _start_simulation(self):
        self.sumo = SumoSimulation(cfg_file=self.sumo_cfg_file,step_length= self.step_length, host=self.sumo_host,
                                   sumo_port=self.sumo_port,sumo_gui=self.sumo_gui, client_order=self.client_order, seed=self.sumo_seed)


    # def _compute_observation(self):
    #     if self.time_to_act():
    #         phase_id = self.traffic_signal.get_phase_onehot()
    #         n_, n_max_max_wait, n_min_max_wait, n_max_all_wait, n_min_all_wait = self.n.get_all_lane_info(self.delta_time)
    #         s_, s_max_max_wait, s_min_max_wait, s_max_all_wait, s_min_all_wait = self.s.get_all_lane_info(self.delta_time)
    #         w_, w_max_max_wait, w_min_max_wait, w_max_all_wait, w_min_all_wait = self.w.get_all_lane_info(self.delta_time)
    #         e_, e_max_max_wait, e_min_max_wait, e_max_all_wait, e_min_all_wait = self.e.get_all_lane_info(self.delta_time)
    #
    #
    #         max_max_wait = max(n_max_max_wait, s_max_max_wait, w_max_max_wait, e_max_max_wait)
    #         min_max_wait = min(n_min_max_wait, s_min_max_wait, w_min_max_wait, e_min_max_wait)
    #
    #         max_all_wait = max(n_max_all_wait, s_max_all_wait, w_max_all_wait, e_max_all_wait)
    #         min_all_wait = min(n_min_all_wait, s_min_all_wait, w_min_all_wait, e_min_all_wait)
    #
    #         ret = [n_, s_, w_, e_]
    #         # ret = n_ + s_ + w_ + e_
    #
    #         # 最大等待时间和总等待时间进行归一化
    #         if max_max_wait - min_max_wait != 0:
    #             ret = [
    #                 [(value - min_max_wait) / (max_max_wait - min_max_wait) if idx == 2 or idx == 6 or idx == 10 else lst[idx] for idx, value in
    #                  enumerate(lst)] for lst in ret]
    #         if max_all_wait - min_all_wait != 0:
    #             ret = [
    #                 [(value - min_all_wait) / (max_all_wait - min_all_wait) if idx == 3 or idx == 7 or idx == 11 else lst[idx] for idx, value in
    #                  enumerate(lst)] for lst in ret]
    #
    #         min_green = [0 if self.traffic_signal.time_since_last_phase_change < self.traffic_signal.min_green else 1]
    #         ret.append(min_green)
    #         ret.append(phase_id)
    #
    #         ret = [item for sublist in ret for item in sublist]
    #
    #         return np.array(ret, dtype=np.float32)

    def _compute_observation(self):
        if self.time_to_act():
            phase_id = self.traffic_signal.get_phase_onehot()
            n_ = self.n.get_all_lane_info()
            s_ = self.s.get_all_lane_info()
            w_ = self.w.get_all_lane_info()
            e_ = self.e.get_all_lane_info()

            ret = [n_, s_, w_, e_]
            # ret = n_ + s_ + w_ + e_


            min_green = [0 if self.traffic_signal.time_since_last_phase_change < self.traffic_signal.min_green else 1]
            ret.append(min_green)
            ret.append(phase_id)

            ret = [item for sublist in ret for item in sublist]

            return np.array(ret, dtype=np.float32)


    def time_to_act(self):  # 判断现在是否需要做出动作
        # print(self.e.frameCounter, self.n.frameCounter, self.w.frameCounter, self.s.frameCounter)
        assert self.e.frameCounter == self.n.frameCounter == self.w.frameCounter == self.s.frameCounter
        return self.next_action_time == self.e.frameCounter * self.step_length   # self.e.frameCounter * self.step_length 表示当前仿真时间


    def _apply_actions(self, action):
        if self.time_to_act():
            self.traffic_signal.set_next_phase(action, self.step_length, self)
            assert self.e.frameCounter == self.n.frameCounter == self.w.frameCounter == self.s.frameCounter
            self.next_action_time = self.e.frameCounter * self.step_length + self.delta_time


    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self.synchronize()
            # print(self.n.frameCounter, self.s.frameCounter, self.w.frameCounter, self.e.frameCounter)
            self.traffic_signal.update(self.step_length)   # 假如当前信号灯是黄灯，且持续时间已经到了，需要将信号灯更新为action对应的绿灯
            if self.time_to_act():  # 一直运行到需要决策action为止
                time_to_act = True

    def _compute_reward(self):
        if self.time_to_act():
            sum_all_wait_detail = [round(self.n.get_sum_all_wait(), 2), round(self.s.get_sum_all_wait(), 2), round(self.e.get_sum_all_wait(), 2), round(self.w.get_sum_all_wait(), 2)]
            sum_all_wait = round(sum(sum_all_wait_detail), 2)
            # 计算所有路口的平均等待时间
            sum_all_wait_per_lane_ave = (self.n.get_sum_all_wait_per_lane_ave() + self.s.get_sum_all_wait_per_lane_ave() + self.e.get_sum_all_wait_per_lane_ave() + self.w.get_sum_all_wait_per_lane_ave()) / 4
            # 计算最大平均等待时间
            sum_all_wait_per_lane_ave_max = max(self.n.get_sum_all_wait_per_lane_ave_max(), self.s.get_sum_all_wait_per_lane_ave_max(),  self.e.get_sum_all_wait_per_lane_ave_max(), self.w.get_sum_all_wait_per_lane_ave_max())

            sum_max_wait = max(self.n.get_sum_max_wait(), self.s.get_sum_max_wait(), self.e.get_sum_max_wait(), self.w.get_sum_max_wait())
            sum_emergy_wait = self.n.get_sum_emergy_wait() + self.s.get_sum_emergy_wait() + self.e.get_sum_emergy_wait() + self.w.get_sum_emergy_wait()
            sum_emergy_max_wait = max(self.n.get_sum_max_emerge_wait(), self.s.get_sum_max_emerge_wait(), self.e.get_sum_max_emerge_wait(), self.w.get_sum_max_emerge_wait())
            sum_emergy_num = self.n.get_sum_emerge_num() + self.s.get_sum_emerge_num() + self.e.get_sum_emerge_num() + self.w.get_sum_emerge_num()

            sum_emergy_wait_0 = self.n.get_sum_emergy_wait_0() + self.s.get_sum_emergy_wait_0() + self.e.get_sum_emergy_wait_0() + self.w.get_sum_emergy_wait_0()
            sum_emergy_max_wait_0 = max(self.n.get_sum_max_emerge_wait_0(), self.s.get_sum_max_emerge_wait_0(),
                                      self.e.get_sum_max_emerge_wait_0(), self.w.get_sum_max_emerge_wait_0())
            sum_emergy_num_0 = self.n.get_sum_emerge_num_0() + self.s.get_sum_emerge_num_0() + self.e.get_sum_emerge_num_0() + self.w.get_sum_emerge_num_0()

            # 停车次数
            stop_time = self.n.sum_stop_time() + self.s.sum_stop_time() + self.e.sum_stop_time() + self.w.sum_stop_time()
            if sum_all_wait != 0:
                if self.last_sum_all_wait / sum_all_wait > 1:
                    reward_sum_all_wait = self.last_sum_all_wait / sum_all_wait
                elif self.last_sum_all_wait / sum_all_wait < 1:
                    reward_sum_all_wait = -self.last_sum_all_wait / sum_all_wait
                else:
                    reward_sum_all_wait = 0
            else:
                reward_sum_all_wait = self.last_sum_all_wait / (sum_all_wait + 0.0000001)

            if sum_max_wait != 0:
                if self.last_sum_max_wait / sum_max_wait > 1:
                    reward_sum_max_wait = self.last_sum_max_wait / sum_max_wait
                elif self.last_sum_max_wait / sum_max_wait < 1:
                    reward_sum_max_wait = -self.last_sum_max_wait / sum_max_wait
                else:
                    reward_sum_max_wait = 0
            else:
                reward_sum_max_wait = self.last_emergy_wait / (sum_emergy_wait + 0.0000001)
            reward_sum_all_wait_per_lane_ave = self.last_sum_all_wait_per_lane_ave - sum_all_wait_per_lane_ave
            # reward = np.around(self.w_max_wait * reward_sum_max_wait + self.w_all_wait * reward_sum_all_wait + self.w_emergy_wait * reward_emergy_wait, 2)
            # reward = np.around(self.w_max_wait * reward_sum_max_wait + self.w_all_wait * reward_sum_all_wait_per_lane_ave + self.w_emergy_wait * reward_emergy_wait, 2)
            # reward = np.around(self.w_max_wait * reward_sum_max_wait + self.w_all_wait * reward_sum_all_wait_per_lane_ave_max + self.w_emergy_wait * reward_emergy_wait, 2)
            # reward = np.around(self.w_max_wait * reward_sum_max_wait + self.w_all_wait * reward_sum_all_wait, 2)
            # print(f'time: {self.e.frameCounter*self.step_length}, last_sum_all_wait: {self.last_sum_all_wait}, sum_all_wait: {sum_all_wait}, self.last_sum_max_wait: {self.last_sum_max_wait}, sum_max_wait: {sum_max_wait}, reward: {reward}, signal: {self.traffic_signal.green_phase}', end='')

            reward = self.only_wait_all_time_reward(sum_all_wait) + self.only_wait_max_time_reward(sum_max_wait) \
                     + self.only_emerge_all_wait(sum_emergy_wait) * 2 + self.only_emergy_max_wait(sum_emergy_max_wait) * 2 + self.only_emerge_num(sum_emergy_num) * 2 \
                     + self.only_emerge_all_wait_0(sum_emergy_wait_0) + self.only_emergy_max_wait_0(sum_emergy_max_wait_0) + self.only_emerge_num_0(sum_emergy_num_0)


            if self.traffic_signal.green_phase == 0:
                ac = '南=北'
            elif self.traffic_signal.green_phase == 1:
                ac = '北-东 南-西'
            elif self.traffic_signal.green_phase == 2:
                ac = '西-北 东-南'
            else:
                ac = '东=西'
            print(f'time: {self.e.frameCounter*self.step_length},last_all_wait: {self.last_sum_all_wait}, all_wait: {sum_all_wait}, last_max_wait: {self.last_sum_max_wait}, max_wait: {sum_max_wait},reward: {reward}, { ac}', {self.action})
            self.last_sum_all_wait = sum_all_wait
            self.last_sum_all_wait_detail = sum_all_wait_detail
            self.last_sum_max_wait = sum_max_wait
            self.last_emergy_wait = sum_emergy_wait
            self.last_emergy_wait_0 = sum_emergy_wait_0
            self.last_sum_all_wait_per_lane_ave = sum_all_wait_per_lane_ave
            self.last_sum_all_wait_per_lane_ave_max = sum_all_wait_per_lane_ave_max
            self.last_reward = reward
            self.last_stop_time = stop_time
            self.last_sum_emergy_max_wait = sum_emergy_max_wait
            self.last_sum_emergy_max_wait_0 = sum_emergy_max_wait_0
            self.last_sum_emergy_num = sum_emergy_num
            self.last_sum_emergy_num_0 = sum_emergy_num_0

            return reward

    # 根据车辆等待时间总和，计算reward
    def only_wait_all_time_reward(self, sum_all_wait):
        reward = self.last_sum_all_wait - sum_all_wait
        return reward

    # 根据最大车辆等待时间计算reward
    def only_wait_max_time_reward(self, sum_max_wait):
        reward = self.last_sum_max_wait - sum_max_wait
        return reward

    # 根据所有路口的最大平均等待时间计算reward
    def only_wait_per_lane_ave_max(self, sum_all_wait_per_lane_ave_max):
        reward = self.last_sum_all_wait_per_lane_ave_max - sum_all_wait_per_lane_ave_max
        return reward

    def only_emerge_all_wait(self, sum_emergy_wait):
        return self.last_emergy_wait - sum_emergy_wait

    def only_emerge_num(self, sum_emergy_num):
        return self.last_sum_emergy_num - sum_emergy_num

    def only_emergy_max_wait(self, sum_emergy_max_wait):
        return self.last_sum_emergy_max_wait - sum_emergy_max_wait

    def only_emerge_all_wait_0(self, sum_emergy_wait_0):
        return self.last_emergy_wait_0 - sum_emergy_wait_0

    def only_emerge_num_0(self, sum_emergy_num_0):
        return self.last_sum_emergy_num_0 - sum_emergy_num_0

    def only_emergy_max_wait_0(self, sum_emergy_max_wait_0):
        return self.last_sum_emergy_max_wait_0 - sum_emergy_max_wait_0

    def _compute_done(self):
        assert self.e.frameCounter == self.n.frameCounter == self.w.frameCounter == self.s.frameCounter
        return self.e.frameCounter * self.step_length > self.episode_length

    def _compute_info(self):
        info = self._compute_step_info()
        self.metrics.append(info)

    def _compute_step_info(self):
        assert self.e.frameCounter == self.n.frameCounter == self.w.frameCounter == self.s.frameCounter

        return {
            'step_time': self.e.frameCounter * self.step_length,
            'reward': self.last_reward,
            'all_wait': self.last_sum_all_wait,  # 所有车道在上一仿真步骤中的停止车辆的总数。
            'max_wait': self.last_sum_max_wait,
            'emergy_wait': self.last_emergy_wait,
            'emergy_wait_0': self.last_emergy_wait_0,
            'all_wait_per_lane_ave': self.last_sum_all_wait_per_lane_ave,
            'stop_time': self.last_stop_time,
            'ave_wait': np.around(self.last_sum_all_wait / (self.n.sum_vehicle_of_three_lane() + self.w.sum_vehicle_of_three_lane() + self.e.sum_vehicle_of_three_lane() + self.s.sum_vehicle_of_three_lane()), 2),
            'emergy_max_wait': self.last_sum_emergy_max_wait,
            'emergy_max_wait_0': self.last_sum_emergy_max_wait_0,
        }

    def get_waiting_time_per_lane_junction(self):
        return self.n.get_waiting_time_per_lane() + self.s.get_waiting_time_per_lane() + self.w.get_waiting_time_per_lane() + self.e.get_waiting_time_per_lane()

    def render(self, mode="human"):
        pass











