import carla
import numpy as np
import torch
import traci

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from envs.lane import Lane
from utils.IDdetector import Detector
from utils.coordinate import build_projection_matrix



class Junc:
    def __init__(self, world=None, name=None, camera_blueprint=None, fps=None, deepsort=None):
        self.img = None
        self.img_mask = None
        self.name = name
        if self.name == 'n':
            self.lane_line = np.array([[-3.227272727272726, -1, 1079.0454545454536], [-7.017241379310349, -1, 2283.120689655173], [1, 0, -321], [6.309523809523795, -1, -1977.1428571428503]])
            self.camera = world.spawn_actor(camera_blueprint, carla.Transform(carla.Location(x=-3.63, y=-13.55, z=18.0), carla.Rotation(pitch=-40, yaw=-90.0, roll=0.0)))
            self.sensor_location = np.array([-3.63, -13.55, 18.0])
            self.lane_0_name = '-2_0'
            self.lane_1_name = '-2_1'
            self.lane_2_name = '-2_2'
        elif self.name == 's':
            self.lane_line = np.array([[-3.2238805970149227, -1, 1077.8955223880587], [-7.0925925925926165, -1, 2303.314814814823], [1, 0, -321], [7.166666666666665, -1, -2262.333333333333]])
            self.camera = world.spawn_actor(camera_blueprint, carla.Transform(carla.Location(x=3.63, y=13.55, z=18.0), carla.Rotation(pitch=-40, yaw=90.0, roll=0.0)))  # s
            self.sensor_location = np.array([3.63, 13.55, 18.0])
            self.lane_0_name = '3_0'
            self.lane_1_name = '3_1'
            self.lane_2_name = '3_2'
        elif self.name == 'w':
            self.lane_line = np.array([[-3.0256410256410335, -1, 1014.2820512820529], [-7.133333333333327, -1, 2317.599999999998], [60.99999999999979, -1, -19226.999999999927], [6.354838709677434, -1, -1966.7741935483934]])
            self.camera = world.spawn_actor(camera_blueprint, carla.Transform(carla.Location(x=-13.55, y=3.63, z=18.0), carla.Rotation(pitch=-40, yaw=180.0, roll=0.0)))  # w
            self.sensor_location = np.array([-13.55, 3.63, 18.0])
            self.lane_0_name = '1_0'
            self.lane_1_name = '1_1'
            self.lane_2_name = '1_2'
        else:
            self.lane_line = np.array([[-3.225352112676058, -1, 1076.2535211267605], [-7.092592592592583, -1, 2317.4999999999977], [69.50000000000138, -1, -21925.00000000046], [6.370967741935465, -1, -1965.3548387096696]])
            self.camera = world.spawn_actor(camera_blueprint,carla.Transform(carla.Location(x=13.55, y=-3.63, z=18.0), carla.Rotation(pitch=-40, yaw=0.0, roll=0.0)))  # e
            self.sensor_location = np.array([13.55, -3.63, 18.0])
            self.lane_0_name = '-0_0'
            self.lane_1_name = '-0_1'
            self.lane_2_name = '-0_2'

        self.model = Detector()
        self.lane_0 = Lane(self.lane_0_name)
        self.lane_1 = Lane(self.lane_1_name)
        self.lane_2 = Lane(self.lane_2_name)
        self.frameCounter = 0
        self.c2w = None
        self.lanes = [self.lane_0, self.lane_1, self.lane_2]

        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        image_w = camera_blueprint.get_attribute("image_size_x").as_int()
        image_h = camera_blueprint.get_attribute("image_size_y").as_int()
        fov = camera_blueprint.get_attribute("fov").as_float()
        self.K_inverse = build_projection_matrix(image_w, image_h, fov)

        self.fps = fps

    def vehicle_num_reset(self):
        self.lane_0.vehicle_num = 0
        self.lane_1.vehicle_num = 0
        self.lane_2.vehicle_num = 0

    def track_num_reset(self):
        self.lane_0.track_num = 0
        self.lane_1.track_num = 0
        self.lane_2.track_num = 0

    def max_wait_reset(self):
        self.lane_0.max_wait = 0
        self.lane_1.max_wait = 0
        self.lane_2.max_wait = 0

    def all_wait_reset(self):
        self.lane_0.all_wait = 0
        self.lane_1.all_wait = 0
        self.lane_2.all_wait = 0

    def emergy_num_reset(self):
        self.lane_0.emergy_num = 0
        self.lane_1.emergy_num = 0
        self.lane_2.emergy_num = 0

    def emergy_wait_reset(self):
        self.lane_0.emergy_wait = 0
        self.lane_1.emergy_wait = 0
        self.lane_2.emergy_wait = 0

    def stop_time_reset(self):
        self.lane_0.stop_time = 0
        self.lane_1.stop_time = 0
        self.lane_2.stop_time = 0

    def all_number_reset(self):
        self.vehicle_num_reset()
        self.track_num_reset()
        self.all_wait_reset()
        self.max_wait_reset()
        self.emergy_num_reset()
        self.emergy_max_wait_reset()
        self.emergy_wait_reset()

        self.stop_time_reset()

    def sum_stop_time(self):
        return self.lane_0.stop_time + self.lane_1.stop_time + self.lane_2.stop_time

    def get_all_lane_info(self):
        self.update_all_emerge_info()
        return list([self.lane_0.vehicle_num, self.lane_0.track_num, self.lane_0.max_wait, self.lane_0.emergy_num, self.lane_0.emergy_wait, self.lane_0.all_wait, self.lane_0.stop_time,
                     self.lane_1.vehicle_num, self.lane_1.track_num, self.lane_1.max_wait, self.lane_1.emergy_num, self.lane_1.emergy_wait, self.lane_1.all_wait, self.lane_1.stop_time,
                     self.lane_2.vehicle_num, self.lane_2.track_num, self.lane_2.max_wait, self.lane_2.emergy_num, self.lane_2.emergy_wait, self.lane_2.all_wait, self.lane_2.stop_time])

    # def get_all_lane_info(self, delta_time):
    #     vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
    #     vehicle_max_density = 200 / vehicle_size_min_gap  # 200 (lane_length)
    #     return list([min(self.lane_0.vehicle_num / vehicle_max_density, 2), min(self.lane_0.track_num / vehicle_max_density, 1), self.lane_0.max_wait, self.lane_0.all_wait,
    #                  min(self.lane_1.vehicle_num / vehicle_max_density, 1), min(self.lane_1.track_num / vehicle_max_density, 1), self.lane_1.max_wait, self.lane_1.all_wait,
    #                  min(self.lane_2.vehicle_num / vehicle_max_density, 1), min(self.lane_2.track_num / vehicle_max_density, 1), self.lane_2.max_wait,  self.lane_2.all_wait]), max(self.lane_0.max_wait, self.lane_1.max_wait, self.lane_2.max_wait), min(self.lane_0.max_wait, self.lane_1.max_wait, self.lane_2.max_wait), max(self.lane_0.all_wait, self.lane_1.all_wait, self.lane_2.all_wait), min(self.lane_0.all_wait, self.lane_1.all_wait, self.lane_2.all_wait)


    def get_sum_max_wait(self):
        return max(self.lane_0.max_wait, self.lane_1.max_wait, self.lane_2.max_wait)

    def get_sum_all_wait(self):
        return self.lane_0.all_wait + self.lane_1.all_wait + self.lane_2.all_wait

    def get_sum_emergy_wait(self):
        return self.lane_0.emergy_wait + self.lane_1.emergy_wait + self.lane_2.emergy_wait

    def get_sum_all_wait_per_lane_ave(self):
        if self.lane_0.track_num != 0 and self.lane_1.track_num != 0 and self.lane_2.track_num != 0:
            return self.lane_0.all_wait / self.lane_0.track_num + self.lane_1.all_wait / self.lane_1.track_num + self.lane_2.all_wait / self.lane_2.track_num
        else:
            return 0

    def get_sum_all_wait_per_lane_ave_max(self):
        if self.lane_0.track_num != 0 and self.lane_1.track_num != 0 and self.lane_2.track_num != 0:
            return max(self.lane_0.all_wait / self.lane_0.track_num, self.lane_1.all_wait / self.lane_1.track_num, self.lane_2.all_wait / self.lane_2.track_num)
        else:
            return 0

    # 计算每条车道上面的等待时间
    def get_waiting_time_per_lane(self):
        return [self.lane_0.all_wait, self.lane_1.all_wait, self.lane_2.all_wait]

    # 计算每条车道上的最大等待时间
    def get_max_time_per_lane(self):
        return [self.lane_0.max_wait, self.lane_1.max_wait, self.lane_2.max_wait]

    # 计算一个路口所有车道上车辆之和
    def sum_vehicle_of_three_lane(self):
        return self.lane_0.vehicle_num + self.lane_1.vehicle_num + self.lane_2.vehicle_num

    def update_all_emerge_info(self):
        for lane in self.lanes:
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane.name)
            # 获取类型为"emerge"的车辆在当前时间步的等待时间，并记录到列表中
            for vid in vehicles_on_lane:
                waiting_time = []
                vehicle_type = traci.vehicle.getTypeID(vid)
                if vehicle_type == "emerge":
                    waiting_time.append(traci.vehicle.getWaitingTime(vid))
                if waiting_time:
                    lane.emergy_num = len(waiting_time)
                    lane.emergy_wait = sum(waiting_time)
                    lane.emerge_max_wait = max(waiting_time)

    def get_sum_max_emerge_wait(self):
        return max(self.lane_0.emerge_max_wait, self.lane_1.emerge_max_wait, self.lane_2.emerge_max_wait)

    def emergy_max_wait_reset(self):
        self.lane_0.emerge_max_wait = 0
        self.lane_1.emerge_max_wait = 0
        self.lane_2.emerge_max_wait = 0

    def get_sum_emerge_num(self):
        return self.lane_0.emergy_num + self.lane_1.emergy_num + self.lane_2.emergy_num